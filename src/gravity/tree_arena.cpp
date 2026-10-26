/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/cuda/cuda_memory_advice.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/tree_build.hpp>

constexpr bool kVerbose = true;

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
static std::atomic<int> allocatorMtx(0);
static size_t nodesSize;
static std::atomic<uint64_t> remoteTreeFetchCount{0};
static Vector<short> leafSizes;
static SpinlockType leafSizesMutex;

long long treeNodesSize() {
    return nodesSize;
}
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
    Array<Position, kDimensions> const& pos, MultipoleBuffer const& multipole) {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    size_t const index = nextMultId++ + treeAllocLineSize;
    if (index >= nodesSize) {
        THROW_ERROR("%s\n", "Tree multipole arena full");
    }
    MultipolePosition* ptr = treeMultipoleRecord(index);
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
            CUDA_CHECK(cudaAdviseMemory(nodes, nodesSize * sizeof(TreeNode),
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
        CUDA_CHECK(cudaAdviseMemory(
            nodes, nodesSize * sizeof(TreeNode), cudaMemAdviseSetReadMostly, cudaGetDevice()));
        setGpu = true;
        setCpu = false;
    }
#else
    THROW_ERROR("GPU tree access requested in a build without CUDA\n");
#endif
}

size_t treeNodeCapacity(size_t sourceParticleCount) {
    size_t const multiplier = kTreeNodeAllocationSize;
    if (sourceParticleCount > std::numeric_limits<size_t>::max() / multiplier) {
        throw std::overflow_error("Tree arena capacity overflows size_t");
    }
    return multiplier * sourceParticleCount / getOptions().compute.bucketSize;
}

void treeAllocateNodesLocal(size_t capacity) {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    auto const multiply = [](size_t first, size_t second) {
        if (second != 0 && first > std::numeric_limits<size_t>::max() / second) {
            throw std::overflow_error("Tree arena capacity overflows size_t");
        }
        return first * second;
    };
    size_t const sz = std::max(capacity, size_t(kMinTrees));
    if (sz > size_t(std::numeric_limits<int>::max() - treeAllocLineSize)) {
        throw std::overflow_error("Tree arena exceeds TreeId index range");
    }
    size_t const stride = fmmPositionedMultipoleStride(fmmOrder());
    std::vector<unsigned char> freshPopulation(sz, 0);
    if (nodesSize < sz || multipoleStride != stride) {
        TreeNode* freshNodes =
            static_cast<TreeNode*>(allocateManagedMemory(multiply(sz, sizeof(TreeNode))));
        char* freshMultipoles;
        try {
            freshMultipoles = static_cast<char*>(allocateManagedMemory(multiply(sz, stride)));
        } catch (...) {
            freeManagedMemory(freshNodes);
            throw;
        }
        freeManagedMemory(nodes);
        freeManagedMemory(multis);
        nodes = freshNodes;
        multis = freshMultipoles;
        multipoleStride = stride;
    }
    nodesSize = sz;
    nextId = -treeAllocLineSize;
    nextMultId = -treeAllocLineSize;
    populatedNodes.swap(freshPopulation);
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    for (int i = 0; i < allocatorList.size(); i++) {
        allocatorList[i]->ready_ = false;
    }
    allocatorMtx--;
    tree2Cpu();
}

long long treeNodesNextIndex() {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    return nextId + treeAllocLineSize;
}

namespace detail {
static TreeNode const* (*remoteReader)(TreeId) = nullptr;
static void (*remoteReset)() = nullptr;
}    // namespace detail

void treeRegisterRemoteAccess(TreeNode const* (*reader)(TreeId), void (*reset)()) {
    detail::remoteReader = reader;
    detail::remoteReset = reset;
}

void treeResetLocal() {
    if (detail::remoteReset != nullptr) {
        detail::remoteReset();
    }
}

int treeAllocateNode() {
    if (!allocator.ready_) {
        allocator.reset();
        allocator.ready_ = true;
    }
    return allocator.allocate();
}

void treeStoreNode(int index, TreeNode node, MultipoleBuffer const& multipole) {
    if (size_t(index) >= nodesSize) {
        THROW_ERROR("%s\n", "Tree arena full");
    }
    node.mpos = treeMultipoleRecord(index);
    node.mpos->pos = node.pos;
    treeWriteMultipole(node.mpos, multipole);
    nodes[index] = node;
    populatedNodes[index] = true;
}

HPX_PLAIN_ACTION(treeReset, TreeResetAction);

void treeReset() {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<TreeResetAction>(c));
    }
    treeResetLocal();
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

HPX_PLAIN_ACTION(treeDestroy, TreeDestroyAction);

void treeDestroy(bool freeTree) {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<TreeDestroyAction>(c, freeTree));
    }
    treeResetLocal();
    hpx::wait_all(futs.begin(), futs.end());
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
        if (detail::remoteReader == nullptr) {
            throw std::logic_error("Remote tree access requires an active solver adapter");
        }
        return detail::remoteReader(id);
    }
}

TreeNode const* treeLocalNode(TreeId id) {
    ALWAYS_ASSERT(id.proc == hpxRank());
    return treeGetNode(id);
}

bool treeNodeIsPopulated(int index) {
    ALWAYS_ASSERT(index >= 0 && size_t(index) < nodesSize);
    return populatedNodes[index] != 0;
}

uint64_t treeRemoteFetchCount() {
    return remoteTreeFetchCount.load(std::memory_order_relaxed);
}

void treeRecordRemoteFetch() {
    remoteTreeFetchCount.fetch_add(1, std::memory_order_relaxed);
}
