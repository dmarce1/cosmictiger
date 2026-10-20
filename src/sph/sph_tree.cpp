/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <exception>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

constexpr bool kVerbose = true;

static Vector<SphTreeNode> sphTreeFetchCacheLine(int);
static void sphTreeAllocateNodes();

HPX_PLAIN_ACTION(sphTreeAllocateNodes, SphTreeAllocateNodesAction);
HPX_PLAIN_ACTION(sphTreeCreate, SphTreeCreateAction);
HPX_PLAIN_ACTION(sphTreeDestroy, SphTreeDestroyAction);
HPX_PLAIN_ACTION(sphTreeCacheClear, SphTreeCacheClearAction);
HPX_PLAIN_ACTION(sphTreeFetchCacheLine, SphTreeFetchCacheLineAction);
HPX_PLAIN_ACTION(sphTreeFreeNeighborList, SphTreeFreeNeighborListAction);

class SphTreeAllocator {
    int next_;
    int last_;

public:
    bool ready_;
    void reset();
    bool isReady();
    SphTreeAllocator();
    ~SphTreeAllocator();
    SphTreeAllocator(SphTreeAllocator const&) = default;
    SphTreeAllocator& operator=(SphTreeAllocator const&) = default;
    SphTreeAllocator(SphTreeAllocator&&) = default;
    SphTreeAllocator& operator=(SphTreeAllocator&&) = default;
    int allocate();
};

static Vector<SphTreeAllocator*> allocatorList;
static thread_local SphTreeAllocator allocator;
static Vector<SphTreeNode> nodes;
static std::atomic<int> treeThreadCount(0);
static std::atomic<int> nextId;
static Array<std::unordered_map<TreeId, std::shared_ptr<hpx::shared_future<Vector<SphTreeNode>>>,
                 TreeIdHashHi>,
    SPH_TREE_CACHE_SIZE>
    treeCache;
static Array<SpinlockType, SPH_TREE_CACHE_SIZE> mutex;
static std::atomic<int> allocatorMtx(0);
static Vector<Pair<ParticleInteger>> leafParticleRanges;
static MutexType leafParticleRangeMutex;
static Vector<TreeId> neighborList;
static Vector<int> leaflist;
static MutexType leaflistMutex;
static MutexType neighborListMutex;

void sphTreeSetNeighborRange(TreeId id, Pair<int, int> rng) {
    nodes[id.index].neighborRange = rng;
}

void sphTreeClearNeighborRanges() {
    for (int i = 0; i < leaflist.size(); i++) {
        nodes[leaflist[i]].neighborRange.first = -1;
        nodes[leaflist[i]].neighborRange.second = -1;
    }
}

int sphTreeLeaflistSize() {
    return leaflist.size();
}

TreeId const sphTreeGetLeaf(int i) {
    TreeId id;
    id.index = leaflist[i];
    id.proc = hpxRank();
    return id;
}

void sphTreeFreeNeighborList() {
    //	PRINT("leaflist size = %i\n", leaflist.size());
    neighborList.resize(0);
}

int sphTreeAllocateNeighborList(Vector<TreeId> const& values) {
    std::lock_guard<MutexType> lock(neighborListMutex);
    if (values.size()) {
        int i = neighborList.size();
        neighborList.resize(i + values.size());
        memcpy(&neighborList[i], values.data(), sizeof(TreeId) * values.size());
        return i;
    } else {
        return 0;
    }
}

TreeId& sphTreeGetNeighbor(int i) {
    return neighborList[i];
}

long long sphTreeNodesSize() {
    return nodes.size();
}

struct LastCacheEntryT;

static std::unordered_set<LastCacheEntryT*> lastCacheEntries;
static std::atomic<int> lastCacheEntryMtx(0);

struct LastCacheEntryT {
    TreeId line;
    SphTreeNode const* ptr;
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
    while (lastCacheEntryMtx++ != 0) {
        lastCacheEntryMtx--;
    }
    for (auto* entry : lastCacheEntries) {
        entry->reset();
    }
    lastCacheEntryMtx--;
}

void sphTreeCacheClear() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<SphTreeCacheClearAction>(child));
    }
    treeCache = decltype(treeCache)();
    resetLastCacheEntries();
    for (auto& future : futures) {
        future.get();
    }
}

static SphTreeNode const* treeCacheRead(TreeId id);

void SphTreeAllocator::reset() {
    int const treeCacheLineSize = getOptions().compute.treeCacheLineSize;
    next_ = (nextId += treeCacheLineSize);
    last_ = std::min(next_ + treeCacheLineSize, (int) nodes.size());
    if (next_ >= nodes.size()) {
        THROW_ERROR("%s\n", "Tree arena full");
    }
}

SphTreeAllocator::SphTreeAllocator() {
    ready_ = false;
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    allocatorList.push_back(this);
    allocatorMtx--;
}

int SphTreeAllocator::allocate() {
    if (next_ == last_) {
        reset();
    }
    ASSERT(next_ < nodes.size());
    return next_++;
}

SphTreeAllocator::~SphTreeAllocator() {
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

hpx::future<SphTreeCreateReturn> sphTreeCreateFork(SphTreeCreateParameters params, size_t key,
    Pair<int, int> const& processRange, Pair<ParticleInteger> const& particleRange,
    Range<double> const& box, int const depth, bool const localRoot, bool threadme) {
    static std::atomic<int> numberOfThreads(0);
    hpx::future<SphTreeCreateReturn> rc;
    bool remote = false;
    if (processRange.first != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme = particleRange.second - particleRange.first > MIN_SORT_THREAD_PARTS;
        if (threadme) {
            if (numberOfThreads++ < SORT_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency() ||
                processRange.second - processRange.first > 1) {
                threadme = true;
            } else {
                threadme = false;
                numberOfThreads--;
            }
        }
    }
    if (!threadme) {
        rc = hpx::make_ready_future(
            sphTreeCreate(params, key, processRange, particleRange, box, depth, localRoot));
    } else if (remote) {
        //		PRINT( "%i calling local on %i at %li\n", hpx_rank(), proc_range.first, time(NULL));
        rc = hpx::async<SphTreeCreateAction>(hpxLocalities()[processRange.first], params, key,
            processRange, particleRange, box, depth, localRoot);
    } else {
        rc = hpx::async([params, processRange, key, particleRange, depth, localRoot, box]() {
            auto rc =
                sphTreeCreate(params, key, processRange, particleRange, box, depth, localRoot);
            numberOfThreads--;
            return rc;
        });
    }
    return rc;
}

static void sphTreeAllocateNodes() {
    int const treeCacheLineSize = getOptions().compute.treeCacheLineSize;
    static int const bucketSize = getOptions().compute.sphBucketSize;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<SphTreeAllocateNodesAction>(c));
    }
    nextId = -treeCacheLineSize;
    nodes.resize(
        std::max(size_t(size_t(SPH_TREE_NODE_ALLOCATION_SIZE) * sphParticlesSize() / bucketSize),
            (size_t) NSPH_TREES_MIN));
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    for (int i = 0; i < allocatorList.size(); i++) {
        allocatorList[i]->ready_ = false;
    }
    allocatorMtx--;
    hpx::wait_all(futs.begin(), futs.end());
}

SphTreeCreateReturn sphTreeCreate(SphTreeCreateParameters params, size_t key,
    Pair<int, int> processRange, Pair<ParticleInteger> particleRange, Range<double> box, int depth,
    bool localRoot) {
    // PRINT( "%i\n", depth);
    stackTraceActivate();
    static int const bucketSize = getOptions().compute.sphBucketSize;
    SphTreeCreateReturn rc;
    if (depth >= MAX_DEPTH) {
        THROW_ERROR("%s\n", "Maximum depth exceeded\n");
    }
    if (depth == 0) {
        sphTreeAllocateNodes();
    }
    if (localRoot) {
        leaflist.resize(0);
        particleRange.first = 0;
        particleRange.second = sphParticlesSize();
    }
    Array<TreeId, NCHILD> children;
    Array<Fixed32, NDIM> x;
    Array<double, NDIM> xc;
    size_t numberOfActiveParticles;
    int flops = 0;
    double totalFlops = 0.0;
    float radius;
    double r;
    int minDepth = depth;
    int maxDepth = depth;
    if (localRoot) {
        //		PRINT("%i\n", allocator.ready);
    }
    if (!allocator.ready_) {
        allocator.reset();
        allocator.ready_ = true;
    }
    size_t nodeCount;
    size_t activeNodes = 0;
    size_t leafNodes = 0;
    size_t activeLeafNodes = 0;
    int const index = allocator.allocate();
    if (localRoot) {
        //		PRINT("INDEX = %i\n", index);
    }
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    if (processRange.second - processRange.first > 1 ||
        particleRange.second - particleRange.first > bucketSize) {
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
            auto const dleftBox = domainsRange(key << 1);
            auto const drightBox = domainsRange((key << 1) + 1);
            leftRange.second = rightRange.first = mid;
            leftLocalRoot = leftRange.second - leftRange.first == 1;
            rightLocalRoot = rightRange.second - rightRange.first == 1;
            flops += 7;
        } else {
            double spanMax = 0.0;
            int xdim;
            for (int dim = 0; dim < NDIM; dim++) {
                auto const span = box.end[dim] - box.begin[dim];
                if (span > spanMax) {
                    spanMax = span;
                    xdim = dim;
                }
            }
            double const xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
            ParticleInteger const mid = sphParticlesSort(particleRange, xmid, xdim);
            leftParticleRange.second = rightParticleRange.first = mid;
            leftBox.end[xdim] = rightBox.begin[xdim] = xmid;
            flops += 2;
        }
        auto futr = sphTreeCreateFork(params, (key << 1) + 1, rightRange, rightParticleRange,
            rightBox, depth + 1, rightLocalRoot, true);
        auto futl = sphTreeCreateFork(params, (key << 1), leftRange, leftParticleRange, leftBox,
            depth + 1, leftLocalRoot, false);
        SphTreeCreateReturn const rcl = futl.get();
        SphTreeCreateReturn const rcr = futr.get();
        auto const iboxl = rcl.innerBox;
        auto const iboxr = rcr.innerBox;
        auto const oboxl = rcl.outerBox;
        auto const oboxr = rcr.outerBox;
        minDepth = std::min(rcl.minDepth, rcr.minDepth);
        maxDepth = std::max(rcl.maxDepth, rcr.maxDepth);
        totalFlops += rcl.floatingPointOperations + rcr.floatingPointOperations;
        numberOfActiveParticles = rcl.numberOfActiveParticles + rcr.numberOfActiveParticles;
        children[LEFT] = rcl.id;
        children[RIGHT] = rcr.id;
        nodeCount = 1 + rcl.nodeCount + rcr.nodeCount;
        leafNodes = rcl.leafNodes + rcr.leafNodes;
        activeLeafNodes = rcl.activeLeafNodes + rcr.activeLeafNodes;
        if (rcl.numberOfActiveParticles || rcr.numberOfActiveParticles) {
            activeNodes += 1 + rcl.activeNodes + rcr.activeNodes;
        }
        for (int dim = 0; dim < NDIM; dim++) {
            innerBox.begin[dim] = std::min(iboxl.begin[dim], iboxr.begin[dim]);
            innerBox.end[dim] = std::max(iboxl.end[dim], iboxr.end[dim]);
            outerBox.begin[dim] = std::min(oboxl.begin[dim], oboxr.begin[dim]);
            outerBox.end[dim] = std::max(oboxl.end[dim], oboxr.end[dim]);
        }
        //		PRINT("%i %i %i %i | %e %e %e %e |%e %e %e %e \n", iboxr.valid, iboxl.valid,
        // oboxr.valid, oboxl.valid, iboxl.begin[XDIM], iboxl.end[XDIM],
        // iboxr.begin[XDIM], iboxr.end[XDIM], oboxl.begin[XDIM], oboxl.end[XDIM],
        // oboxr.begin[XDIM], oboxr.end[XDIM]);
    } else {
        children[LEFT].index = children[RIGHT].index = -1;
        children[LEFT].proc = children[RIGHT].proc = -1;
        numberOfActiveParticles = 0;
        for (int dim = 0; dim < NDIM; dim++) {
            innerBox.begin[dim] = 1.9;
            innerBox.end[dim] = -0.9;
            outerBox.begin[dim] = 1.9;
            outerBox.end[dim] = -0.9;
        }
        auto const tiny = 2.0 * RangeFixed::min().toDouble();
        for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
            float const h = params.supportRadiusMultiplier * sphParticlesSupportRadius(i);
            Array<Fixed32, NDIM> xValue;
            for (int dim = 0; dim < NDIM; dim++) {
                xValue[dim] = sphParticlesPosition(dim, i);
            }
            for (int dim = 0; dim < NDIM; dim++) {
                double const x = xValue[dim].toDouble();
                innerBox.begin[dim] = std::min(innerBox.begin[dim].toDouble(), x - tiny);
                innerBox.end[dim] = std::max(innerBox.end[dim].toDouble(), x + tiny);
                if (sphParticlesRung(i) >= params.minimumRung) {
                    outerBox.begin[dim] = std::min(outerBox.begin[dim].toDouble(), x - h - tiny);
                    outerBox.end[dim] = std::max(outerBox.end[dim].toDouble(), x + h + tiny);
                }
            }
            if (sphParticlesRung(i) >= params.minimumRung) {
                numberOfActiveParticles++;
            }
        }

        /*	double small_span = 1e30;
         double hmax = 0.0;
         for( int dim = 0; dim < NDIM; dim++) {
         double span = box.end[dim] - box.begin[dim];
         small_span = std::min(small_span, span);
         hmax += sqr(span);
         }
         //	hmax += 12.0 * sqr(small_span);
         hmax = 1.5 * sqrt(hmax + 3.0 * sqr(small_span));
         for( int dim = 0; dim < NDIM; dim++) {
         inner_box.begin[dim] = box.begin[dim];
         inner_box.end[dim] = box.end[dim];
         outer_box.begin[dim] = box.begin[dim] - hmax;
         outer_box.end[dim] = box.begin[dim] + hmax;
         }*/

        nodeCount = 1;
        leafNodes = 1;
        if (numberOfActiveParticles) {
            activeLeafNodes++;
            activeNodes++;
        }
        std::lock_guard<MutexType> lock(leaflistMutex);
        leaflist.push_back(index);
    }
    SphTreeNode node;
    node = nodes[index];
    node.nodeCount = nodeCount;
    node.innerBox = innerBox;
    node.outerBox = outerBox;

    node.children = children;
    node.localRoot = localRoot;
    node.particleRange = particleRange;
    node.processRange = processRange;
    node.numberOfActiveParticles = numberOfActiveParticles;
    node.activeNodes = activeNodes;
    node.neighborRange.first = -1;
    node.neighborRange.second = -1;
    node.depth = depth;
    node.sinkParticleRange = particleRange;
    ParticleInteger const numberOfParticles = particleRange.second - particleRange.first;
    bool const global = processRange.second - processRange.first > 1;
    node.leaf = !global && (numberOfParticles <= getOptions().compute.sphBucketSize);
    /*	if ( BUCKET_SIZE <= get_options().sph_bucket_size) {
     if (node.leaf) {
     std::lock_guard<mutex_type> lock(leaf_part_range_mutex);
     leaf_part_ranges.push_back(part_range);
     }
     }*/
    if (index >= nodes.size()) {
        THROW_ERROR("%s\n", "Tree arena full\n");
    }
    rc.activeNodes = activeNodes;
    rc.id.index = index;
    rc.id.proc = hpxRank();
    rc.innerBox = node.innerBox;
    rc.outerBox = node.outerBox;
    for (int dim = 0; dim < NDIM; dim++) {
        node.box.begin[dim] = box.begin[dim];
        node.box.end[dim] =
            box.end[dim];    // == 1.0 ? fixed32::max() : fixed32(box.end[dim]);
                             //	PRINT( "---------%e %e\n", node.box.begin[dim].to_float(),
                             // node.box.end[dim].to_float());
    }
    nodes[index] = node;
    rc.leafNodes = leafNodes;
    rc.activeLeafNodes = activeLeafNodes;
    rc.nodeCount = node.nodeCount;
    rc.numberOfActiveParticles = numberOfActiveParticles;
    totalFlops += flops;
    rc.floatingPointOperations = totalFlops;
    rc.minDepth = minDepth;
    rc.maxDepth = maxDepth;
    if (localRoot) {
        sphParticlesResolveWithParticles();
    }
    return rc;
}

void sphTreeDestroy(bool freeSphTree) {
    profilerEnter(__FUNCTION__);

    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<SphTreeDestroyAction>(c, freeSphTree));
    }
    if (freeSphTree) {
        nodes = decltype(nodes)();
    } else {
        int const numberOfThreads = hpxHardwareConcurrency();
        for (int proc = 0; proc < numberOfThreads; proc++) {
            futs.push_back(hpx::async([proc, numberOfThreads]() {
                int const b = (size_t) proc * nodes.size() / numberOfThreads;
                int const e = (size_t) (proc + 1) * nodes.size() / numberOfThreads;
                for (int i = b; i < e; i++) {
                    nodes[i].innerBox = Fixed32Range();
                    nodes[i].outerBox = Fixed32Range();
                }
            }));
        }
    }
    treeCache = decltype(treeCache)();
    resetLastCacheEntries();
    hpx::wait_all(futs.begin(), futs.end());
    profilerExit();
}

void sphTreeSetNumberOfActiveParticles(TreeId id, ParticleInteger i) {
    nodes[id.index].numberOfActiveParticles = i;
}

void sphTreeSetConverged(TreeId id, bool c) {
    nodes[id.index].converged = c;
}

SphTreeNode const* sphTreeGetNode(TreeId id) {
    if (id.proc == hpxRank()) {
        ASSERT(id.index >= 0);
#ifndef NDEBUG
        if (id.index >= nodes.size()) {
            THROW_ERROR("id.index is %li but nodes.size() is %li\n", id.index, nodes.size());
        }
#endif
        return &nodes[id.index];
    } else {
        return treeCacheRead(id);
    }
}

void sphTreeSetBoxes(TreeId id, Fixed32Range const& ibox, Fixed32Range const& obox) {
    nodes[id.index].outerBox = obox;
    nodes[id.index].innerBox = ibox;
}

static SphTreeNode const* treeCacheRead(TreeId id) {
    int const lineSize = getOptions().compute.treeCacheLineSize;
    TreeId lineId;
    lineId.proc = id.proc;
    ASSERT(lineId.proc >= 0 && lineId.proc < hpxSize());
    lineId.index = (id.index / lineSize) * lineSize;
    SphTreeNode const* ptr;
    if (lineId != lastCacheEntry.line) {
        size_t const bin = TreeIdHashLo()(lineId);
        std::shared_ptr<hpx::shared_future<Vector<SphTreeNode>>> future;
        {
            std::lock_guard<SpinlockType> lock(mutex[bin]);
            auto const entry = treeCache[bin].find(lineId);
            if (entry != treeCache[bin].end()) {
                future = entry->second;
            }
        }
        if (!future) {
            auto promise = std::make_shared<hpx::promise<Vector<SphTreeNode>>>();
            auto candidate = std::make_shared<hpx::shared_future<Vector<SphTreeNode>>>(
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
                    hpx::apply([promise, lineId]() {
                        try {
                            auto remote = hpx::async<SphTreeFetchCacheLineAction>(
                                hpxLocalities()[lineId.proc], lineId.index);
                            promise->set_value(remote.get());
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

static Vector<SphTreeNode> sphTreeFetchCacheLine(int index) {
    int const lineSize = getOptions().compute.treeCacheLineSize;
    Vector<SphTreeNode> line(lineSize);
    int const begin = (index / lineSize) * lineSize;
    int const end = std::min(begin + lineSize, int(nodes.size()));
    for (int i = begin; i < end; i++) {
        line[i - begin] = nodes[i];
    }
    return line;
}
