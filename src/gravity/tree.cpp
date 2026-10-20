/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

constexpr bool kVerbose = true;

#include <cstddef>
#include <exception>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/fmm/fmm_types.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

static Vector<TreeNode> treeFetchCacheLine(int);
static void treeAllocateNodes();

HPX_PLAIN_ACTION(treeAllocateNodes, TreeAllocateNodesAction);
HPX_PLAIN_ACTION(treeCreate, TreeCreateAction);
HPX_PLAIN_ACTION(treeDestroy, TreeDestroyAction);
HPX_PLAIN_ACTION(treeFetchCacheLine, TreeFetchCacheLineAction);

class TreeAllocator {
    int next_;
    int last_;

public:
    bool ready_;
    void reset();
    bool isReady();
    TreeAllocator();
    ~TreeAllocator();
    TreeAllocator(TreeAllocator const&) = default;
    TreeAllocator& operator=(TreeAllocator const&) = default;
    TreeAllocator(TreeAllocator&&) = default;
    TreeAllocator& operator=(TreeAllocator&&) = default;
    int allocate();
};

static Vector<TreeAllocator*> allocatorList;
static thread_local TreeAllocator allocator;
static TreeNode* nodes = nullptr;
// Positioned multipole arena: one record of multipoleStride bytes per node slot,
// sized for the runtime FMM order when the arena is (re)allocated. The stride
// is the record size rounded up to 16 bytes (fmmPositionedMultipoleStride), so
// the CUDA kick stages whole records with 16-byte asynchronous copies.
static char* multis = nullptr;
static size_t multipoleStride = 0;
// Cache lines span reserved allocator slots, including holes between threads.
static std::vector<unsigned char> populatedNodes;
static Vector<Pair<ParticleInteger>> leafParticleRanges;
static MutexType leafParticleRangeMutex;
static std::atomic<int> treeThreadCount(0);
static std::atomic<int> nextId;
static std::atomic<int> nextMultId;
static Array<
    std::unordered_map<TreeId, std::shared_ptr<hpx::shared_future<Vector<TreeNode>>>, TreeIdHashHi>,
    TREE_CACHE_SIZE>
    treeCache;
static Array<SpinlockType, TREE_CACHE_SIZE> mutex;
static std::atomic<int> allocatorMtx(0);
static size_t nodesSize;

static Vector<short> leafSizes;
static SpinlockType leafSizesMutex;

long long treeNodesSize() {
    return nodesSize;
}
struct LastCacheEntryT;

static std::unordered_set<LastCacheEntryT*> lastCacheEntries;
static std::atomic<int> lastCacheEntryMtx(0);

struct LastCacheEntryT {
    TreeId line;
    TreeNode const* ptr;
    void reset() {
        line.proc = line.index = -1;
        ptr = nullptr;
    }
    LastCacheEntryT() {
        line.proc = line.index = -1;
        ptr = nullptr;
        while (lastCacheEntryMtx++ != 0) {
            lastCacheEntryMtx--;
        }
        lastCacheEntries.insert(this);
        lastCacheEntryMtx--;
    }
    ~LastCacheEntryT() {
        while (lastCacheEntryMtx++ != 0) {
            lastCacheEntryMtx--;
        }
        lastCacheEntries.erase(this);
        lastCacheEntryMtx--;
    }
};

static thread_local LastCacheEntryT lastCacheEntry;

static void resetLastCacheEntries() {
    for (auto i = lastCacheEntries.begin(); i != lastCacheEntries.end(); i++) {
        (*i)->reset();
    }
}

static TreeNode const* treeCacheRead(TreeId id);

// The coefficients follow the position header without padding for every order.
static_assert(offsetof(FmmMultiPos<kFmmMinOrder>, multi) == sizeof(MultipolePosition) &&
        offsetof(FmmMultiPos<kFmmMaxOrder>, multi) == sizeof(MultipolePosition),
    "Positioned multipole coefficients must directly follow the position");

static MultipolePosition* treeMultipoleRecord(size_t index) {
    return reinterpret_cast<MultipolePosition*>(multis + index * multipoleStride);
}

static float* treeMultipoleCoefficients(MultipolePosition* record) {
    return reinterpret_cast<float*>(reinterpret_cast<char*>(record) + sizeof(MultipolePosition));
}

static float const* treeMultipoleCoefficients(MultipolePosition const* record) {
    return reinterpret_cast<float const*>(
        reinterpret_cast<char const*>(record) + sizeof(MultipolePosition));
}

static void treeWriteMultipole(MultipolePosition* record, MultipoleBuffer const& multipole) {
    int const count = fmmMultipoleCoefficientCount(fmmOrder());
    int const slots = int((multipoleStride - sizeof(MultipolePosition)) / sizeof(float));
    float* coefficients = treeMultipoleCoefficients(record);
    for (int i = 0; i < count; i++) {
        coefficients[i] = multipole[i];
    }
    // The padding up to the record stride travels to the GPU with the record;
    // keep it defined.
    for (int i = count; i < slots; i++) {
        coefficients[i] = 0.0f;
    }
}

MultipolePosition* treeAddMultipole(
    Array<Fixed32, NDIM> const& pos, MultipoleBuffer const& multipole) {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    MultipolePosition* ptr = treeMultipoleRecord(nextMultId++ + treeAllocLineSize);
    ptr->pos = pos;
    treeWriteMultipole(ptr, multipole);
    return ptr;
}

void treeReadMultipole(MultipolePosition const* record, MultipoleBuffer& multipole) {
    int const count = fmmMultipoleCoefficientCount(fmmOrder());
    float const* coefficients = treeMultipoleCoefficients(record);
    for (int i = 0; i < count; i++) {
        multipole[i] = coefficients[i];
    }
}

void TreeAllocator::reset() {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    next_ = (nextId += treeAllocLineSize);
    nextMultId += treeAllocLineSize;
    last_ = std::min(next_ + treeAllocLineSize, (int) nodesSize);
    if (next_ >= nodesSize) {
        THROW_ERROR("%s\n", "Tree arena full");
    }
}

TreeAllocator::TreeAllocator() {
    ready_ = false;
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    allocatorList.push_back(this);
    allocatorMtx--;
}

int TreeAllocator::allocate() {
    if (next_ == last_) {
        reset();
    }
    ASSERT(next_ < nodesSize);
    return next_++;
}

TreeAllocator::~TreeAllocator() {
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    for (int i = 0; i < allocatorList.size(); i++) {
        if (allocatorList[i] == this) {
            allocatorList[i] = allocatorList.back();
            allocatorList.pop_back();
            break;
        }
    }
    allocatorMtx--;
}

TreeCreateParameters::TreeCreateParameters(int minimumRung, double openingAngle, double hmaxArg) {
    theta = openingAngle;
    minRung = minimumRung;
    minLevel = 9;
    leafPushed = false;
    doLeafSizes = false;
    parallelParticleThreshold = 0;
}

hpx::future<TreeCreateReturn> treeCreateFork(TreeCreateParameters params, size_t key,
    Pair<int, int> const& processRange, Pair<ParticleInteger> const& particleRange,
    Range<double> const& box, int const depth, bool const localRoot, bool threadme) {
    hpx::future<TreeCreateReturn> rc;
    bool remote = false;
    if (processRange.first != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme = particleRange.second - particleRange.first >= params.parallelParticleThreshold;
    }
    if (!threadme) {
        rc = hpx::make_ready_future(
            treeCreate(params, key, processRange, particleRange, box, depth, localRoot));
    } else if (remote) {
        rc = hpx::async<TreeCreateAction>(hpxLocalities()[processRange.first], params, key,
            processRange, particleRange, box, depth, localRoot);
    } else {
        rc = hpx::async([params, processRange, key, particleRange, depth, localRoot, box]() {
            auto rc = treeCreate(params, key, processRange, particleRange, box, depth, localRoot);
            return rc;
        });
    }
    return rc;
}

size_t treeAddRemote(TreeNode const& remote) {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    if (nextId + treeAllocLineSize >= nodesSize) {
        PRINT("TREE ALLOCATION EXCEEDED\n");
        abort();
    }
    size_t index = nextId++ + treeAllocLineSize;
    nodes[index] = remote;
    populatedNodes[index] = true;
    return index;
}

TreeNode* treeData() {
    return nodes;
}

static bool setGpu = false;
static bool setCpu = false;

void tree2Cpu() {
    if (!setCpu) {
#ifdef USE_CUDA
        if (setGpu && cudaEnabled()) {
            cudaSetDevice();
            CUDA_CHECK(cudaMemAdvise(nodes, nodesSize * sizeof(TreeNode),
                cudaMemAdviseUnsetReadMostly, cudaGetDevice()));
        }
#endif
        setGpu = false;
        setCpu = true;
    }
}

void tree2Gpu() {
#ifdef USE_CUDA
    if (!cudaEnabled()) {
        THROW_ERROR("GPU tree access requires compute.cuda=true\n");
    }
    if (!setGpu) {
        cudaSetDevice();
        CUDA_CHECK(cudaMemAdvise(
            nodes, nodesSize * sizeof(TreeNode), cudaMemAdviseSetReadMostly, cudaGetDevice()));
        setGpu = true;
        setCpu = false;
    }
#else
    THROW_ERROR("GPU tree access requested in a build without CUDA\n");
#endif
}

static void treeAllocateNodes() {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    static int const bucketSize = getOptions().compute.bucketSize;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TreeAllocateNodesAction>(c));
    }
    nextId = -treeAllocLineSize;
    nextMultId = -treeAllocLineSize;
    size_t const sz =
        std::max(size_t(size_t(TREE_NODE_ALLOCATION_SIZE) * particlesSize() / bucketSize),
            (size_t) NTREES_MIN);
    size_t const stride = fmmPositionedMultipoleStride(fmmOrder());
    if (nodesSize < sz || multipoleStride != stride) {
        if (nodes != nullptr) {
            freeManagedMemory(nodes);
            freeManagedMemory(multis);
        }
        nodes = static_cast<TreeNode*>(allocateManagedMemory(sz * sizeof(TreeNode)));
        multis = static_cast<char*>(allocateManagedMemory(sz * stride));
        multipoleStride = stride;
    }
    nodesSize = sz;
    populatedNodes.assign(sz, 0);
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    for (int i = 0; i < allocatorList.size(); i++) {
        allocatorList[i]->ready_ = false;
    }
    allocatorMtx--;
    hpx::wait_all(futs.begin(), futs.end());
    tree2Cpu();
}

long long treeNodesNextIndex() {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    return nextId + treeAllocLineSize;
}

TreeCreateReturn treeCreate(TreeCreateParameters params, size_t key, Pair<int, int> processRange,
    Pair<ParticleInteger> particleRange, Range<double> box, int depth, bool localRoot) {
    if (key == 1) {
        profilerEnter(__FUNCTION__);
    }
    int flops = 0;
    stackTraceActivate();
    double const h = getOptions().gravity.supportRadius;
    int bucketSize = getOptions().compute.bucketSize;
    TreeCreateReturn rc;
    if (depth >= MAX_DEPTH) {
        THROW_ERROR("%s\n", "Maximum depth exceeded\n");
    }
    if (depth == 0) {
        treeAllocateNodes();
    }
    if (localRoot) {
        particleRange = particlesCurrentRange();
        size_t cnt = particleRange.second - particleRange.first;
        int numberOfThreads = 1;
        while (numberOfThreads < 4 * hpxHardwareConcurrency()) {
            numberOfThreads *= 2;
            cnt /= 2;
        }
        params.parallelParticleThreshold = cnt;
    }
    Array<TreeId, NCHILD> children;
    Array<Fixed32, NDIM>& x = rc.pos;
    MultipoleBuffer& multi = rc.multi;
    auto& rbox = rc.box;
    float& radius = rc.radius;
    Array<double, NDIM> xcValue;
    double r;
    if (!allocator.ready_) {
        allocator.reset();
        allocator.ready_ = true;
    }
    int const index = allocator.allocate();
    bool isleaf = true;
    auto const numberOfParticles = particleRange.second - particleRange.first;
    float boxR = 0.0f;
    for (int dim = 0; dim < NDIM; dim++) {
        boxR += sqr(0.5 * (box.end[dim] - box.begin[dim]));    // 12
    }
    boxR = sqrt(boxR);    // 4
    bool ewaldSatisfied =
        (boxR < 0.25 * (params.theta / (1.0 + params.theta)) && boxR < 0.125 - 0.25 * h);    // 10
    double maxRatio = 1.0;
    flops += 26;
    static SimdFloat const toFloatFactor = kFixed2float;
    if (processRange.second - processRange.first > 1 || numberOfParticles > bucketSize ||
        (!ewaldSatisfied && numberOfParticles > 0)) {
        isleaf = false;
        int const xdim = box.longestDim();
        auto leftBox = box;
        auto rightBox = box;
        auto leftRange = processRange;
        auto rightRange = processRange;
        auto leftParticleRange = particleRange;
        auto rightParticleRange = particleRange;
        bool leftLocalRoot = false;
        bool rightLocalRoot = false;
        if (processRange.second - processRange.first > 1) {
            int const mid = (processRange.first + processRange.second) / 2;
            leftBox = domainsRange(key << 1);
            rightBox = domainsRange((key << 1) + 1);
            leftRange.second = rightRange.first = mid;
            leftLocalRoot = leftRange.second - leftRange.first == 1;
            rightLocalRoot = rightRange.second - rightRange.first == 1;
        } else {
            double const xmid = 0.5 * (box.end[xdim] + box.begin[xdim]);
            flops += 2;
            ParticleInteger const mid = particlesSort(particleRange, xmid, xdim);
            leftParticleRange.second = rightParticleRange.first = mid;
            leftBox.end[xdim] = rightBox.begin[xdim] = xmid;
        }
        auto futr = treeCreateFork(params, (key << 1) + 1, rightRange, rightParticleRange, rightBox,
            depth + 1, rightLocalRoot, true);
        auto futl = treeCreateFork(params, (key << 1), leftRange, leftParticleRange, leftBox,
            depth + 1, leftLocalRoot, false);
        auto const rcl = futl.get();
        auto const rcr = futr.get();
        rc.maxGravitySupportRadius =
            std::max(rcl.maxGravitySupportRadius, rcr.maxGravitySupportRadius);
        auto const& xl = rcl.pos;
        auto const& xr = rcr.pos;
        auto const& ml = rcl.multi;
        auto const& mr = rcr.multi;
        double const& rlValue = rcl.radius;
        double const& rrValue = rcr.radius;
        auto const& boxl = rcl.box;
        auto const& boxr = rcr.box;
        for (int dim = 0; dim < NDIM; dim++) {
            if (ml[0] != 0.0 && mr[0] != 0.0) {
                rbox.begin[dim] = std::min(boxl.begin[dim], boxr.begin[dim]);
                rbox.end[dim] = std::max(boxl.end[dim], boxr.end[dim]);
            } else if (ml[0] != 0.0) {
                rbox = boxl;
            } else if (mr[0] != 0.0) {
                rbox = boxr;
            } else {
                rbox.begin[dim] = rbox.end[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
            }
        }
        Array<double, NDIM> xlValue;
        Array<double, NDIM> xrValue;
        for (int dim = 0; dim < NDIM; dim++) {
            xlValue[dim] = xl[dim].toDouble();    // 3
            xrValue[dim] = xr[dim].toDouble();    // 3
        }
        Array<double, NDIM> xc1, xc2;
        double radius1 = 0.0, radius2;
        bool tried1 = false;
        // Global nodes have no rank-local particle range to rescan.
        if (processRange.second - processRange.first == 1 && processRange.first == hpxRank() &&
            particleRange.first >= 0 && numberOfParticles > 0 &&
            numberOfParticles <= 64 * getOptions().compute.bucketSize &&
            (rbox.end[xdim] - rbox.begin[xdim] <= 0.5)) {
            tried1 = true;
            for (int dim = 0; dim < NDIM; dim++) {
                xc1[dim] = 0.5 * (rbox.begin[dim] + rbox.end[dim]);    // 6
            }
            Array<SimdInt, NDIM> xc{};
            for (int dim = 0; dim < NDIM; dim++) {
                xc[dim] = simdBroadcast<SimdInt>(Fixed32(xc1[dim]).raw());
            }
            flops += 6;
            SimdFloat r2Max = simdBroadcast<SimdFloat>(0.0);
            for (ParticleInteger i = particleRange.first; i < particleRange.second;
                i += SIMD_FLOAT_SIZE) {
                SimdInt x{};
                Array<SimdFloat, NDIM> dx{};
                int const maxj = std::min(particleRange.second - i, SIMD_FLOAT_SIZE);
                for (int dim = 0; dim < NDIM; dim++) {
                    for (int j = 0; j < maxj; j++) {
                        simdSetLane(x, j, particlesPos(dim, i + j).raw());
                    }
                    for (int j = maxj; j < SIMD_FLOAT_SIZE; j++) {
                        simdSetLane(x, j, xc[dim][0]);
                    }
                    dx[dim] = simdCast<SimdFloat>(simdPositionDifference(x, xc[dim])) *
                        toFloatFactor;    // 6
                }
                flops += 18 + maxj * 6;
                auto const r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                r2Max = simdMax(r2Max, r2);
            }
            float r2 = 0.0f;
            for (int i = 0; i < SIMD_FLOAT_SIZE; i++) {
                r2 = std::max(r2, static_cast<float>(r2Max[i]));
            }
            flops += 4 + SIMD_FLOAT_SIZE;
            radius1 = sqrt(r2);
            //	PRINT("%e\n", radius);
        }
        double rr;
        double rl;
        Array<double, NDIM> n;
        float rValue;
        double norminv = 0.0;
        for (int dim = 0; dim < NDIM; dim++) {
            n[dim] = xlValue[dim] - xrValue[dim];    // 3
            norminv += sqr(n[dim]);                  // 6
        }
        norminv = norminv > 0.0 ? 1.0 / std::sqrt(norminv) : 0.0;    // 8
        for (int dim = 0; dim < NDIM; dim++) {
            n[dim] *= norminv;            // 3
            n[dim] = std::abs(n[dim]);    // 3
        }
        flops += 29;
        r = 0.0;
        if (mr[0] != 0.0) {
            if (ml[0] != 0.0) {
                if (norminv > 0.0) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        double const xmax = std::max(
                            xlValue[dim] + n[dim] * rlValue, xrValue[dim] + n[dim] * rrValue);
                        double const xmin = std::min(
                            xlValue[dim] - n[dim] * rlValue, xrValue[dim] - n[dim] * rrValue);
                        xc2[dim] = (xmax + xmin) * 0.5;
                        r += sqr((xmax - xmin) * 0.5);
                    }
                } else {
                    xc2 = xlValue;
                    r = std::max(sqr(rlValue), sqr(rrValue));
                }
                flops += 45;
            } else {
                xc2 = xrValue;
                r = rrValue * rrValue;
                flops++;
            }
        } else {
            if (ml[0] != 0.0) {
                xc2 = xlValue;
                r = rlValue * rlValue;
                flops++;
            } else {
                xc2 = rbox.begin;
                flops += 6;
            }
        }
        radius2 = std::sqrt(r);    // 4
        if (tried1 && radius1 < radius2) {
            xcValue = xc1;
            radius = radius1;
        } else {
            xcValue = xc2;
            radius = radius2;
        }
        r = 0.0;
        for (int dim = 0; dim < NDIM; dim++) {
            double const span = (rbox.end[dim] - rbox.begin[dim]);
            ALWAYS_ASSERT(span >= 0.0);
            r += sqr(span * 0.5);    // 12
        }
        r = std::sqrt(r);    // 4
        if (r < radius) {    // 1
            radius = r;
            for (int dim = 0; dim < NDIM; dim++) {
                xcValue[dim] = (rbox.begin[dim] + rbox.end[dim]) * 0.5;    // 6
            }
        }

        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = xcValue[dim];    // 3
        }
        flops += fmmDispatch([&](auto orderTag) {
            return treeComputeMultipole<decltype(orderTag)::value>(
                multi, ml, mr, xlValue, xrValue, xcValue);
        });
        children[LEFT] = rcl.id;
        children[RIGHT] = rcr.id;
    } else {
        // All species use the same fixed comoving gravity support. Gas smoothing
        // lengths belong only to the hydrodynamic tree and neighbor searches.
        rc.maxGravitySupportRadius = numberOfParticles > 0 ? float(h) : 0.0f;
        Array<double, NDIM>& xminValue = rbox.begin;
        Array<double, NDIM>& xmaxValue = rbox.end;
        for (int dim = 0; dim < NDIM; dim++) {
            xminValue[dim] = box.end[dim];
            xmaxValue[dim] = box.begin[dim];
            if (numberOfParticles == 0) {
                xminValue[dim] = xmaxValue[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
            }
        }
        for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
            for (int dim = 0; dim < NDIM; dim++) {
                double const x = particlesPos(dim, i).toDouble();    // 3
                xmaxValue[dim] = std::max(xmaxValue[dim], x);        // 3
                xminValue[dim] = std::min(xminValue[dim], x);        // 3
            }
        }
        for (int dim = 0; dim < NDIM; dim++) {
            xcValue[dim] = (xmaxValue[dim] + xminValue[dim]) * 0.5;    // 6
        }
        r = 0.0;
        flops += 15;
        for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
            double thisRadius = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                double const x = particlesPos(dim, i).toDouble();    // 3
                thisRadius += sqr(x - xcValue[dim]);                 // 9
            }
            r = std::max(r, thisRadius);    // 2
            flops += 14;
        }
        radius = std::sqrt(r);
        children[LEFT].index = children[RIGHT].index = -1;
        flops += fmmDispatch([&](auto orderTag) {
            return treeComputeMultipole<decltype(orderTag)::value>(multi, particleRange, xcValue);
        });
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = xcValue[dim];
        }
    }
    TreeNode node;
    node.radius = radius;
    node.maxGravitySupportRadius = rc.maxGravitySupportRadius;
    node.children = children;
    node.localRoot = localRoot;
    node.particleRange = particleRange;
    node.processRange = processRange;
    node.pos = x;
    node.mpos = treeMultipoleRecord(index);
    node.mpos->pos = x;
    treeWriteMultipole(node.mpos, multi);
    node.depth = depth;
    for (int dim = 0; dim < NDIM; dim++) {
        node.box.begin[dim] = rbox.begin[dim];
        node.box.end[dim] = rbox.end[dim];
        // Stored bounds have fewer fractional bits than particle positions.
        // Round outward so box-based acceptance cannot exclude an extremum.
        if (node.box.begin[dim].toDouble() > rbox.begin[dim]) {
            node.box.begin[dim].raw()--;
        }
        if (node.box.end[dim].toDouble() < rbox.end[dim]) {
            node.box.end[dim].raw()++;
        }
    }
    bool const global = processRange.second - processRange.first > 1;
    node.leaf = isleaf;
    if (index >= nodesSize) {
        THROW_ERROR("%s\n", "Tree arena full\n");
    }
    nodes[index] = node;
    populatedNodes[index] = true;
    rc.id.index = index;
    rc.id.proc = hpxRank();
    if (localRoot && getOptions().hydro.enabled) {
        particlesResolveWithSphParticles();
    }
    if (key == 1) {
        profilerExit();
    }
    addCpuFlops(flops);
    return rc;
}

HPX_PLAIN_ACTION(treeReset, TreeResetAction);

void treeReset() {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<TreeResetAction>(c));
    }
    treeCache = decltype(treeCache)();
    resetLastCacheEntries();
    hpx::wait_all(futs.begin(), futs.end());
}
Vector<double> treeGetLeafSizes();
HPX_PLAIN_ACTION(treeGetLeafSizes, TreeGetLeafSizesAction);

Vector<double> treeGetLeafSizes() {
    Vector<hpx::future<Vector<double>>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<TreeGetLeafSizesAction>(c));
    }
    double myAvg = 0.0;
    for (int i = 0; i < leafSizes.size(); i++) {
        myAvg += leafSizes[i];
    }
    myAvg /= leafSizes.size();
    Vector<double> avgs;
    avgs.push_back(myAvg);
    for (auto& f : futs) {
        auto tmp = f.get();
        for (auto const& m : tmp) {
            avgs.push_back(m);
        }
    }
    return avgs;
}

int treeAvgLeafSize() {
    auto sizes = treeGetLeafSizes();
    double avg = 0.0;
    for (int i = 0; i < sizes.size(); i++) {
        avg += sizes[i];
    }
    avg /= sizes.size();
    return avg;
}

void treeDestroy(bool freeTree) {
    profilerEnter(__FUNCTION__);

    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<TreeDestroyAction>(c, freeTree));
    }
    treeCache = decltype(treeCache)();
    resetLastCacheEntries();
    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}

TreeNode const* treeGetNode(TreeId id) {
    if (id.proc == hpxRank()) {
        ASSERT(id.index >= 0);
#ifndef NDEBUG
        if (id.index >= nodesSize) {
            THROW_ERROR("id.index is %li but nodes.size() is %li\n", id.index, nodesSize);
        }
#endif
        return &nodes[id.index];
    } else {
        return treeCacheRead(id);
    }
}

static TreeNode const* treeCacheRead(TreeId id) {
    int const lineSize = getOptions().compute.treeCacheLineSize;
    TreeId lineId;
    TreeNode const* ptr;
    lineId.proc = id.proc;
    ASSERT(lineId.proc >= 0 && lineId.proc < hpxSize());
    lineId.index = (id.index / lineSize) * lineSize;
    if (lineId != lastCacheEntry.line) {
        size_t const bin = TreeIdHashLo()(id);
        std::shared_ptr<hpx::shared_future<Vector<TreeNode>>> future;
        {
            std::lock_guard<SpinlockType> lock(mutex[bin]);
            auto const entry = treeCache[bin].find(lineId);
            if (entry != treeCache[bin].end()) {
                future = entry->second;
            }
        }
        if (!future) {
            auto promise = std::make_shared<hpx::promise<Vector<TreeNode>>>();
            auto candidate = std::make_shared<hpx::shared_future<Vector<TreeNode>>>(
                promise->get_future().share());
            bool inserted;
            {
                std::lock_guard<SpinlockType> lock(mutex[bin]);
                auto const entry = treeCache[bin].emplace(lineId, candidate);
                inserted = entry.second;
                future = entry.first->second;
            }
            if (inserted) {
                try {
                    hpx::post([promise, lineId]() {
                        try {
                            promise->set_value(hpx::async<TreeFetchCacheLineAction>(
                                hpxLocalities()[lineId.proc], lineId.index)
                                    .get());
                        } catch (...) {
                            promise->set_exception(std::current_exception());
                        }
                    });
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            }
        }
        ptr = future->get().data();
    } else {
        ptr = lastCacheEntry.ptr;
    }
    lastCacheEntry.ptr = ptr;
    lastCacheEntry.line = lineId;
    return ptr + id.index - lineId.index;
}

static Vector<TreeNode> treeFetchCacheLine(int index) {
    int const lineSize = getOptions().compute.treeCacheLineSize;
    ALWAYS_ASSERT(index >= 0 && size_t(index) < nodesSize);
    Vector<TreeNode> line;
    line.reserve(lineSize);
    int const begin = (index / lineSize) * lineSize;
    int const end = std::min(size_t(begin) + size_t(lineSize), nodesSize);
    // A zeroed record sized for the largest order is a zeroed record of the
    // runtime order: every per-order view reads only its leading coefficients.
    static FmmMultiPos<kFmmMaxOrder> emptyMultipole = [] {
        FmmMultiPos<kFmmMaxOrder> empty;
        empty.multi = 0.0f;
        for (int dim = 0; dim < NDIM; dim++) {
            empty.pos[dim] = 0.0;
        }
        return empty;
    }();
    for (int i = begin; i < end; i++) {
        if (populatedNodes[i]) {
            line.push_back(nodes[i]);
        } else {
            // Unused slots preserve cache offsets but never dereference arena garbage.
            TreeNode empty{};
            empty.mpos = reinterpret_cast<MultipolePosition*>(&emptyMultipole);
            empty.leaf = true;
            empty.processRange = Pair<int>(hpxRank(), hpxRank() + 1);
            for (int child = 0; child < NCHILD; child++) {
                empty.children[child].proc = hpxRank();
                empty.children[child].index = -1;
            }
            line.push_back(empty);
        }
    }
    return line;
}
