/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>

constexpr bool kVerbose = true;

static Vector<SphTreeNode> sphTreeFetchCacheLine(int);
static void sphTreeAllocateNodes();
static SphTreeCreateReturn sphTreeRefreshNode(
    TreeId id, SphTreeCreateParameters params, double driftFactor);

HPX_PLAIN_ACTION(sphTreeAllocateNodes, SphTreeAllocateNodesAction);
HPX_PLAIN_ACTION(sphTreeCreate, SphTreeCreateAction);
HPX_PLAIN_ACTION(sphTreeDestroy, SphTreeDestroyAction);
HPX_PLAIN_ACTION(sphTreeCacheClear, SphTreeCacheClearAction);
HPX_PLAIN_ACTION(sphTreeFetchCacheLine, SphTreeFetchCacheLineAction);
HPX_PLAIN_ACTION(sphTreeFreeNeighborList, SphTreeFreeNeighborListAction);
HPX_PLAIN_ACTION(sphTreeAllValid, SphTreeAllValidAction);
HPX_PLAIN_ACTION(sphTreeRefreshNode, SphTreeRefreshNodeAction);

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
    kSphTreeCacheSize>
    treeCache;
static Array<SpinlockType, kSphTreeCacheSize> mutex;
static std::atomic<int> allocatorMtx(0);
static Vector<Pair<ParticleInteger>> leafParticleRanges;
static MutexType leafParticleRangeMutex;
static Vector<TreeId> neighborList;
static Vector<int> leaflist;
static MutexType leaflistMutex;
static MutexType neighborListMutex;
static std::atomic<bool> treeValid(false);
static TreeId retainedRoot{-1, -1};
static TreeId localRootId{-1, -1};
static ParticleInteger retainedParticleCount = -1;
static int retainedMinimumRung = kMaxRung;

void sphTreeInvalidate() {
    treeValid = false;
}

bool sphTreeIsValid() {
    return treeValid && retainedParticleCount == sphParticlesSize();
}

bool sphTreeAllValid() {
    std::vector<hpx::future<bool>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<SphTreeAllValidAction>(child));
    }
    bool valid = sphTreeIsValid();
    for (auto& future : futures) {
        valid = future.get() && valid;
    }
    return valid;
}

namespace detail {

void mergeSphVolumeRatios(SphTreeCreateReturn& result, double minimum, double maximum) {
    // Empty subtrees return zero; every occupied box has positive padded volume.
    if (maximum > 0.0) {
        result.minVolumeRatio =
            result.maxVolumeRatio > 0.0 ? std::min(result.minVolumeRatio, minimum) : minimum;
        result.maxVolumeRatio = std::max(result.maxVolumeRatio, maximum);
    }
}

double sphTreeVolumeRatio(SphTreeNode const& node, sph_flops::Counter& flops) {
    if (!node.isLocal() || node.numberOfParticles() == 0) {
        return 0.0;
    }
    // clang-format off
    /*
     * Derived diagnostic (no external equation):
     *   R = V_swept / V_partition
     *     = ∏_d (e_swept,d − b_swept,d) / ∏_d (e_partition,d − b_partition,d).
     * Both boxes use the same comoving periodic unit-box coordinates. The swept
     * bounds include motion since the most recent rebuild, including conservative
     * expansion at periodic seams; kernel support is excluded. Use double spans
     * and products: fixed-point volume products can underflow in refined cells.
     * Symbol directory: R -> returned ratio; b/e_swept -> swept box begin, end;
     * b/e_partition -> partition box begin, end; V_swept -> swept volume;
     * V_partition -> partition volume.
     */
    // clang-format on
    double V_partition = 1.0;
    double V_swept = 1.0;
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        double const partitionSpan = node.box.end[dim].toDouble() - node.box.begin[dim].toDouble();
        double const sweptSpan =
            node.driftBox.end[dim].toDouble() - node.driftBox.begin[dim].toDouble();
        ALWAYS_ASSERT(partitionSpan > 0.0 && sweptSpan >= 0.0);
        V_partition *= partitionSpan;
        V_swept *= sweptSpan;
    }
    flops.doublePrecision += 8 * kSpatialDimensions + 1;
    return V_swept / V_partition;
}

void mergeSphBounds(Fixed32Range& target, Fixed32Range const& source) {
    for (int dim = 0; dim < kDimensions; ++dim) {
        target.begin[dim] = std::min(target.begin[dim], source.begin[dim]);
        target.end[dim] = std::max(target.end[dim], source.end[dim]);
    }
}

Fixed32Range sweptSphBounds(Fixed32Range const& positions, Vector3<double> const& v_minus,
    Vector3<double> const& v_plus, double D) {
    Fixed32Range result = positions;
    double const padding = 2.0 * RangeFixed::min().toDouble();
    // clang-format off
    /*
     * Derived conservative bound for the implemented linear drift x' = x + v D:
     *   b'_d = b_d + min(0, D v⁻_d, D v⁺_d),
     *   e'_d = e_d + max(0, D v⁻_d, D v⁺_d).
     * This is a local derivation, not a numbered source equation. D = Δτ/a uses
     * the same stored velocities and periodic unit-box coordinates as drift().
     * Bounds include every intermediate position and permit negative D. A span
     * of at least one periodic box covers that entire axis; saturating before
     * fixed-point conversion avoids wrapping the signed bound representation.
     * Symbol directory: b_d/e_d -> position box begin, end; D -> drift factor;
     * v⁻_d/v⁺_d -> minimum, maximum velocity; b'_d/e'_d -> result.
     * Code names: v_minus = v⁻, v_plus = v⁺.
     */
    // clang-format on
    for (int dim = 0; dim < kDimensions; ++dim) {
        if (positions.begin[dim] > positions.end[dim]) {
            continue;
        }
        double const firstDisplacement = D * v_minus[dim];
        double const secondDisplacement = D * v_plus[dim];
        double const lower = positions.begin[dim].toDouble() +
            std::min({0.0, firstDisplacement, secondDisplacement}) - padding;
        double const upper = positions.end[dim].toDouble() +
            std::max({0.0, firstDisplacement, secondDisplacement}) + padding;
        if (!std::isfinite(lower) || !std::isfinite(upper)) {
            throw std::runtime_error("Nonfinite SPH drift bound");
        }
        if (upper - lower >= 1.0) {
            result.begin[dim] = -padding;
            result.end[dim] = 1.0 + padding;
        } else {
            result.begin[dim] = lower;
            result.end[dim] = upper;
        }
    }
    return result;
}

void updateLocalSphDriftBounds(TreeId id, double driftFactor, int rung) {
    auto& node = nodes[id.index];
    // Only particles active at the last build/refit can change rung before the
    // next refit. A drift on that rung or a finer one cannot move this subtree.
    // Unrestricted drifts and coarser-rung requests still visit every particle.
    if (rung >= retainedMinimumRung && node.numberOfActiveParticles == 0) {
        return;
    }
    if (node.leaf) {
        auto positions = emptySphBounds();
        Vector3<double> minimumVelocity;
        Vector3<double> maximumVelocity;
        for (int dim = 0; dim < kDimensions; ++dim) {
            minimumVelocity[dim] = 0.0;
            maximumVelocity[dim] = 0.0;
        }
        double const padding = 2.0 * RangeFixed::min().toDouble();
        for (ParticleInteger particle = node.particleRange.first;
            particle < node.particleRange.second; ++particle) {
            bool const moving = rung < 0 || sphParticlesRung(particle) == rung;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const position = sphParticlesPositionValue(dim, particle).toDouble();
                positions.begin[dim] =
                    std::min(positions.begin[dim].toDouble(), position - padding);
                positions.end[dim] = std::max(positions.end[dim].toDouble(), position + padding);
                double const velocity = moving ? sphParticlesVelocityValue(dim, particle) : 0.0;
                minimumVelocity[dim] = std::min(minimumVelocity[dim], velocity);
                maximumVelocity[dim] = std::max(maximumVelocity[dim], velocity);
            }
        }
        mergeSphBounds(node.driftBox,
            sweptSphBounds(positions, minimumVelocity, maximumVelocity, driftFactor));
    } else {
        for (auto const& child : node.children) {
            if (child.index >= 0) {
                ALWAYS_ASSERT(child.proc == hpxRank());
                updateLocalSphDriftBounds(child, driftFactor, rung);
                mergeSphBounds(node.driftBox, nodes[child.index].driftBox);
            }
        }
    }
}

}    // namespace detail

void sphTreeDriftBounds(double driftFactor, int rung) {
    if (treeValid && localRootId.index >= 0 && retainedParticleCount == sphParticlesSize()) {
        detail::updateLocalSphDriftBounds(localRootId, driftFactor, rung);
    }
}

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
        threadme = particleRange.second - particleRange.first > kMinSortThreadParts;
        if (threadme) {
            if (numberOfThreads++ < kSortOversubscription * hpx::thread::hardware_concurrency() ||
                processRange.second - processRange.first > 1) {
                threadme = true;
            } else {
                threadme = false;
                numberOfThreads--;
            }
        }
    }
    if (!threadme) {
        try {
            rc = hpx::make_ready_future(
                sphTreeCreate(params, key, processRange, particleRange, box, depth, localRoot));
        } catch (...) {
            rc = hpx::make_exceptional_future<SphTreeCreateReturn>(std::current_exception());
        }
    } else if (remote) {
        rc = hpx::async<SphTreeCreateAction>(hpxLocalities()[processRange.first], params, key,
            processRange, particleRange, box, depth, localRoot);
    } else {
        rc = hpx::async(hpx::annotated_function(
            [params, processRange, key, particleRange, depth, localRoot, box]() {
                try {
                    auto rc = sphTreeCreate(
                        params, key, processRange, particleRange, box, depth, localRoot);
                    numberOfThreads--;
                    return rc;
                } catch (...) {
                    numberOfThreads--;
                    throw;
                }
            },
            "sph.tree.create"));
    }
    return rc;
}

static void sphTreeAllocateNodes() {
    sphTreeInvalidate();
    int const treeCacheLineSize = getOptions().compute.treeCacheLineSize;
    int const bucketSize = getOptions().compute.sphBucketSize;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<SphTreeAllocateNodesAction>(c));
    }
    size_t const requiredNodes =
        std::max(size_t(size_t(kSphTreeNodeAllocationSize) * sphParticlesSize() / bucketSize),
            (size_t) kMinSphTrees);
    // Rebuilds reuse the constructed arena. Grow only at this completed-work
    // boundary, before any tree tasks can retain pointers into the vector.
    if (requiredNodes > nodes.size()) {
        nodes.resize(requiredNodes);
    }
    // Each OS worker must acquire a fresh allocation chunk for this tree,
    // even when the underlying storage did not move or change size.
    nextId = -treeCacheLineSize;
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    for (int i = 0; i < allocatorList.size(); i++) {
        allocatorList[i]->ready_ = false;
    }
    allocatorMtx--;
    for (auto& future : futs) {
        future.get();
    }
}

SphTreeCreateReturn sphTreeCreate(SphTreeCreateParameters params, size_t key,
    Pair<int, int> processRange, Pair<ParticleInteger> particleRange, Range<double> box, int depth,
    bool localRoot) {
    int const bucketSize = getOptions().compute.sphBucketSize;
    SphTreeCreateReturn rc;
    if (depth < 0 || depth > SphTreeNode::kMaximumDepth) {
        throw std::out_of_range("SPH tree depth must be between 0 and 15");
    }
    if (depth == 0) {
        sphTreeAllocateNodes();
    }
    if (localRoot) {
        leaflist.resize(0);
        particleRange.first = 0;
        particleRange.second = sphParticlesSize();
        // A retained hierarchical owner can hold inactive gas outside its rank
        // domain. Rank-routing nodes still follow the binary domain keys, but
        // each local octree partitions the full periodic box so a later rebuild
        // can subdivide those particles rather than chasing a domain boundary.
        box = unitBox<double>();
    }
    Array<TreeId, kSphChildCount> children;
    for (auto& child : children) {
        child = TreeId{-1, -1};
    }
    Vector3<Position> nodePosition;
    Vector3<double> xc;
    size_t numberOfActiveParticles;
    sph_flops::Counter flops;
    double totalFlops = 0.0;
    float radius;
    double r;
    int minDepth = depth;
    int maxDepth = depth;
    if (!allocator.ready_) {
        allocator.reset();
        allocator.ready_ = true;
    }
    size_t nodeCount;
    size_t activeNodes = 0;
    size_t leafNodes = 0;
    size_t activeLeafNodes = 0;
    int const index = allocator.allocate();
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    Fixed32Range driftBox;
    if (processRange.second - processRange.first > 1 ||
        particleRange.second - particleRange.first > bucketSize) {
        if (depth == SphTreeNode::kMaximumDepth) {
            throw std::out_of_range("SPH octree subdivision would require depth 16");
        }
        std::array<hpx::future<SphTreeCreateReturn>, kSphChildCount> childFutures;
        if (processRange.second - processRange.first > 1) {
            // Rank ownership still follows the domain decomposition's binary keys.
            // Only the spatial particle trees below these routing nodes are hybrid trees.
            int const middleRank = (processRange.first + processRange.second) / 2;
            for (int childIndex = 0; childIndex < 2; ++childIndex) {
                size_t const childKey = (key << 1) + childIndex;
                Pair<int> const childRanks = childIndex == 0 ?
                    Pair<int>(processRange.first, middleRank) :
                    Pair<int>(middleRank, processRange.second);
                childFutures[childIndex] = sphTreeCreateFork(params, childKey, childRanks,
                    particleRange, domainsRange(childKey), depth + 1,
                    childRanks.second - childRanks.first == 1, true);
            }
        } else {
            // Reduce fanout only near the bucket threshold. Widen before
            // multiplying so large configured bucket sizes cannot overflow int.
            auto const particleCount = std::uint64_t(particleRange.second - particleRange.first);
            auto const bucketCount = std::uint64_t(bucketSize);
            int splitDimensions = 1;
            if (particleCount > 4 * bucketCount) {
                splitDimensions = 3;
            } else if (particleCount > 2 * bucketCount) {
                splitDimensions = 2;
            }
            splitDimensions = std::min(splitDimensions, kSpatialDimensions);
            int splitMask = 0;
            for (int split = 0; split < splitDimensions; ++split) {
                int longestDimension = -1;
                double longestSpan = -1.0;
                for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                    double const span = box.end[dim] - box.begin[dim];
                    flops.doublePrecision += 1;
                    if ((splitMask & (1 << dim)) == 0 && span > longestSpan) {
                        longestDimension = dim;
                        longestSpan = span;
                    }
                }
                splitMask |= 1 << longestDimension;
            }
            // Child bit d selects the upper half along dimension d. Partition
            // selected axes in z/y/x order so occupied slots have contiguous ranges.
            // Longest-axis selection limits cell aspect ratios after partial splits.
            // Derived geometric split: m_d = (b_d + e_d)/2.
            // b_d -> box begin, e_d -> box end, m_d -> split midpoint.
            // Code names: m = m_d, indexed by dim.
            std::array<Pair<ParticleInteger>, kSphChildCount> childParticleRanges;
            childParticleRanges[0] = particleRange;
            Vector3<double> m;
            int partitionedMask = 0;
            for (int dim = kDimensions - 1; dim >= 0; --dim) {
                int const upperBit = 1 << dim;
                if ((splitMask & upperBit) == 0) {
                    continue;
                }
                m[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
                Position const splitPosition(m[dim]);
                flops.doublePrecision += 2 + sph_flops::positionConstructionFlops();
                for (int childIndex = 0; childIndex < kSphChildCount; ++childIndex) {
                    if ((childIndex & ~partitionedMask) != 0) {
                        continue;
                    }
                    auto& lowerRange = childParticleRanges[childIndex];
                    ParticleInteger const split = sphParticlesSort(lowerRange, splitPosition, dim);
                    childParticleRanges[childIndex + upperBit] =
                        Pair<ParticleInteger>(split, lowerRange.second);
                    lowerRange.second = split;
                }
                partitionedMask |= upperBit;
            }
            for (int childIndex = kSphChildCount - 1; childIndex >= 0; --childIndex) {
                if ((childIndex & ~splitMask) != 0) {
                    continue;
                }
                auto const childRange = childParticleRanges[childIndex];
                if (childRange.first == childRange.second) {
                    continue;
                }
                auto childBox = box;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    if ((splitMask & (1 << dim)) == 0) {
                        continue;
                    }
                    if ((childIndex & (1 << dim)) != 0) {
                        childBox.begin[dim] = m[dim];
                    } else {
                        childBox.end[dim] = m[dim];
                    }
                }
                childFutures[childIndex] = sphTreeCreateFork(params, key, processRange, childRange,
                    childBox, depth + 1, false, childIndex != 0);
            }
        }
        // Finish every child before propagating an exception: callers may tear
        // down the tree arena as soon as this build returns or throws.
        for (auto& future : childFutures) {
            if (future.valid()) {
                future.wait();
            }
        }
        nodeCount = 1;
        numberOfActiveParticles = 0;
        bool firstChild = true;
        for (int childIndex = 0; childIndex < kSphChildCount; ++childIndex) {
            if (!childFutures[childIndex].valid()) {
                continue;
            }
            auto const child = childFutures[childIndex].get();
            children[childIndex] = child.id;
            nodeCount += child.nodeCount;
            leafNodes += child.leafNodes;
            activeLeafNodes += child.activeLeafNodes;
            activeNodes += child.activeNodes;
            numberOfActiveParticles += child.numberOfActiveParticles;
            totalFlops += child.floatingPointOperations;
            detail::mergeSphVolumeRatios(rc, child.minVolumeRatio, child.maxVolumeRatio);
            if (firstChild) {
                innerBox = child.innerBox;
                outerBox = child.outerBox;
                driftBox = child.driftBox;
                minDepth = child.minDepth;
                maxDepth = child.maxDepth;
                firstChild = false;
            } else {
                minDepth = std::min(minDepth, child.minDepth);
                maxDepth = std::max(maxDepth, child.maxDepth);
                for (int dim = 0; dim < kDimensions; ++dim) {
                    innerBox.begin[dim] = std::min(innerBox.begin[dim], child.innerBox.begin[dim]);
                    innerBox.end[dim] = std::max(innerBox.end[dim], child.innerBox.end[dim]);
                    outerBox.begin[dim] = std::min(outerBox.begin[dim], child.outerBox.begin[dim]);
                    outerBox.end[dim] = std::max(outerBox.end[dim], child.outerBox.end[dim]);
                    driftBox.begin[dim] = std::min(driftBox.begin[dim], child.driftBox.begin[dim]);
                    driftBox.end[dim] = std::max(driftBox.end[dim], child.driftBox.end[dim]);
                }
            }
        }
        activeNodes += numberOfActiveParticles != 0;
    } else {
        numberOfActiveParticles = 0;
        innerBox = emptySphBounds();
        outerBox = emptySphBounds();
        auto const tiny = 2.0 * RangeFixed::min().toDouble();
        flops.doublePrecision += 4 * kDimensions + 2;
        for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
            float const h = params.smoothingLengthMultiplier * sphParticlesSmoothingLength(i);
            flops.singlePrecision += 1;
            Vector3<Position> xValue;
            for (int dim = 0; dim < kDimensions; dim++) {
                xValue[dim] = sphParticlesPositionValue(dim, i);
            }
            for (int dim = 0; dim < kDimensions; dim++) {
                double const position = xValue[dim].toDouble();
                innerBox.begin[dim] = std::min(innerBox.begin[dim].toDouble(), position - tiny);
                innerBox.end[dim] = std::max(innerBox.end[dim].toDouble(), position + tiny);
                // Position and old-bound conversions, two offsets and two new bounds.
                flops.doublePrecision += 7;
                if (sphParticlesRung(i) >= params.minimumRung) {
                    outerBox.begin[dim] =
                        std::min(outerBox.begin[dim].toDouble(), position - h - tiny);
                    outerBox.end[dim] = std::max(outerBox.end[dim].toDouble(), position + h + tiny);
                    flops.doublePrecision += 10;    // Includes both support promotions.
                }
            }
            if (sphParticlesRung(i) >= params.minimumRung) {
                numberOfActiveParticles++;
            }
        }

        driftBox = innerBox;

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
    // Initialize only nodes used by this build. Retained slots can contain the
    // previous tree's convergence flags, particle ranges and neighbor indices.
    SphTreeNode node;
    node.nodeCount = nodeCount;
    node.innerBox = innerBox;
    node.outerBox = outerBox;
    node.driftBox = driftBox;

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
    node.leaf = !global && (numberOfParticles <= bucketSize);
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
    rc.driftBox = node.driftBox;
    for (int dim = 0; dim < kDimensions; dim++) {
        node.box.begin[dim] = box.begin[dim];
        node.box.end[dim] = box.end[dim];    // == 1.0 ? fixed32::max() : fixed32(box.end[dim]);
    }
    double const volumeRatio = detail::sphTreeVolumeRatio(node, flops);
    detail::mergeSphVolumeRatios(rc, volumeRatio, volumeRatio);
    nodes[index] = node;
    rc.leafNodes = leafNodes;
    rc.activeLeafNodes = activeLeafNodes;
    rc.nodeCount = node.nodeCount;
    rc.numberOfActiveParticles = numberOfActiveParticles;
    flops.doublePrecision +=
        2 * kDimensions;    // Store the double spatial box as fixed-point bounds.
    addCpuFlops(
        flops.counts());    // Children submit their own work; do not submit their totals twice.
    totalFlops += FlopCount(flops);
    rc.floatingPointOperations = totalFlops;
    rc.minDepth = minDepth;
    rc.maxDepth = maxDepth;
    if (localRoot) {
        sphParticlesResolveWithParticles();
        localRootId = rc.id;
        retainedParticleCount = sphParticlesSize();
        retainedMinimumRung = params.minimumRung;
        treeValid = true;
    }
    if (depth == 0) {
        retainedRoot = rc.id;
        treeValid = true;
    }
    return rc;
}

static SphTreeCreateReturn sphTreeRefreshNode(
    TreeId id, SphTreeCreateParameters params, double driftFactor) {
    ALWAYS_ASSERT(id.proc == hpxRank());
    auto& node = nodes[id.index];
    if (node.localRoot) {
        sphTreeFreeNeighborList();
        retainedMinimumRung = params.minimumRung;
    }
    SphTreeCreateReturn result{};
    result.id = id;
    result.innerBox = emptySphBounds();
    result.outerBox = emptySphBounds();
    result.driftBox = node.driftBox;
    result.nodeCount = 1;
    result.minDepth = node.depth;
    result.maxDepth = node.depth;
    sph_flops::Counter flops;
    if (node.leaf) {
        Vector3<double> minimumVelocity;
        Vector3<double> maximumVelocity;
        for (int dim = 0; dim < kDimensions; ++dim) {
            minimumVelocity[dim] = 0.0;
            maximumVelocity[dim] = 0.0;
        }
        double const padding = 2.0 * RangeFixed::min().toDouble();
        for (ParticleInteger particle = node.particleRange.first;
            particle < node.particleRange.second; ++particle) {
            bool const active = sphParticlesRung(particle) >= params.minimumRung;
            result.numberOfActiveParticles += active;
            float const support =
                params.smoothingLengthMultiplier * sphParticlesSmoothingLength(particle);
            flops.singlePrecision += 1;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const position = sphParticlesPositionValue(dim, particle).toDouble();
                double const velocity = sphParticlesVelocityValue(dim, particle);
                minimumVelocity[dim] = std::min(minimumVelocity[dim], velocity);
                maximumVelocity[dim] = std::max(maximumVelocity[dim], velocity);
                result.innerBox.begin[dim] =
                    std::min(result.innerBox.begin[dim].toDouble(), position - padding);
                result.innerBox.end[dim] =
                    std::max(result.innerBox.end[dim].toDouble(), position + padding);
                flops.doublePrecision += 7;
                if (active) {
                    result.outerBox.begin[dim] = std::min(
                        result.outerBox.begin[dim].toDouble(), position - support - padding);
                    result.outerBox.end[dim] =
                        std::max(result.outerBox.end[dim].toDouble(), position + support + padding);
                    flops.doublePrecision += 10;
                }
            }
        }
        detail::mergeSphBounds(result.driftBox,
            detail::sweptSphBounds(result.innerBox, minimumVelocity, maximumVelocity, driftFactor));
        result.leafNodes = 1;
        result.activeLeafNodes = result.numberOfActiveParticles != 0;
    } else {
        std::array<hpx::future<SphTreeCreateReturn>, kSphChildCount> futures;
        for (int childIndex = 0; childIndex < kSphChildCount; ++childIndex) {
            auto const child = node.children[childIndex];
            if (child.index < 0) {
                continue;
            }
            if (child.proc != hpxRank()) {
                futures[childIndex] = hpx::async<SphTreeRefreshNodeAction>(
                    hpxLocalities()[child.proc], child, params, driftFactor);
            } else if (node.numberOfParticles() >= kMinSortThreadParts || !node.isLocal()) {
                futures[childIndex] = hpx::async(hpx::annotated_function(
                    [child, params, driftFactor]() {
                        return sphTreeRefreshNode(child, params, driftFactor);
                    },
                    "sph.tree.refresh"));
            } else {
                try {
                    futures[childIndex] =
                        hpx::make_ready_future(sphTreeRefreshNode(child, params, driftFactor));
                } catch (...) {
                    futures[childIndex] =
                        hpx::make_exceptional_future<SphTreeCreateReturn>(std::current_exception());
                }
            }
        }
        for (auto& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
        result.minDepth = SphTreeNode::kMaximumDepth;
        for (auto& future : futures) {
            if (!future.valid()) {
                continue;
            }
            auto const child = future.get();
            detail::mergeSphBounds(result.innerBox, child.innerBox);
            detail::mergeSphBounds(result.outerBox, child.outerBox);
            detail::mergeSphBounds(result.driftBox, child.driftBox);
            result.numberOfActiveParticles += child.numberOfActiveParticles;
            result.nodeCount += child.nodeCount;
            result.activeNodes += child.activeNodes;
            result.leafNodes += child.leafNodes;
            result.activeLeafNodes += child.activeLeafNodes;
            result.minDepth = std::min(result.minDepth, child.minDepth);
            result.maxDepth = std::max(result.maxDepth, child.maxDepth);
            result.floatingPointOperations += child.floatingPointOperations;
            detail::mergeSphVolumeRatios(result, child.minVolumeRatio, child.maxVolumeRatio);
        }
    }
    result.activeNodes += result.numberOfActiveParticles != 0;
    node.innerBox = result.innerBox;
    node.outerBox = result.outerBox;
    node.driftBox = result.driftBox;
    double const volumeRatio = detail::sphTreeVolumeRatio(node, flops);
    detail::mergeSphVolumeRatios(result, volumeRatio, volumeRatio);
    node.numberOfActiveParticles = result.numberOfActiveParticles;
    node.activeNodes = result.activeNodes;
    node.converged = false;
    node.neighborRange = Pair<int>(-1, -1);
    addCpuFlops(flops.counts());
    result.floatingPointOperations += FlopCount(flops);
    return result;
}

SphTreeCreateReturn sphTreeRefresh(SphTreeCreateParameters params, double driftFactor) {
    if (!sphTreeAllValid() || retainedRoot.index < 0) {
        throw std::runtime_error("Cannot refresh an invalid SPH tree; rebuild it first");
    }
    sphTreeCacheClear();
    auto result = sphTreeRefreshNode(retainedRoot, params, driftFactor);
    // Refresh changes remotely cached node contents and every leaf's neighbor list.
    sphTreeCacheClear();
    return result;
}

void sphTreeDestroy(bool freeSphTree) {
    sphTreeInvalidate();
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<SphTreeDestroyAction>(c, freeSphTree));
    }
    if (freeSphTree) {
        nodes = decltype(nodes)();
    }
    // Normal teardown retains the arena without traversing its unused slots.
    // Cached remote lines and thread-local pointers still belong to the old
    // tree and must be invalidated on every locality before the next rebuild.
    treeCache = decltype(treeCache)();
    resetLastCacheEntries();
    hpx::wait_all(futs.begin(), futs.end());
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
                    hpx::post(hpx::annotated_function(
                        [promise, lineId]() {
                            try {
                                auto remote = hpx::async<SphTreeFetchCacheLineAction>(
                                    hpxLocalities()[lineId.proc], lineId.index);
                                promise->set_value(remote.get());
                            } catch (...) {
                                promise->set_exception(std::current_exception());
                            }
                        },
                        "sph.tree.fetchCacheLine"));
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
