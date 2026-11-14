/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <format>
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
#include <cosmictiger/sph/ghosts.hpp>
#include <cosmictiger/sph/parent_tree.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>

constexpr bool kVerbose = true;

static void sphTreeAllocateNodes();
static SphTreeCreateReturn sphTreeRefreshNode(
    TreeId id, SphTreeCreateParameters params, double driftFactor);
static SphTreeCreateReturn sphTreeBuild(SphTreeCreateParameters params, size_t key,
    Pair<int, int> processRange, Pair<ParticleInteger> particleRange, Range<double> box, int depth,
    bool localRoot);

static SphTreeCreateReturn sphTreeCreateLocalRoots(SphTreeCreateParameters params);
static SphTreeCreateReturn sphTreeRefreshLocalRoots(
    SphTreeCreateParameters params, double driftFactor);

HPX_PLAIN_ACTION(sphTreeAllocateNodes, SphTreeAllocateNodesAction);
HPX_PLAIN_ACTION(sphTreeCreateLocalRoots, SphTreeCreateLocalRootsAction);
HPX_PLAIN_ACTION(sphTreeRefreshLocalRoots, SphTreeRefreshLocalRootsAction);
HPX_PLAIN_ACTION(sphTreeDestroy, SphTreeDestroyAction);
HPX_PLAIN_ACTION(sphTreeFreeNeighborList, SphTreeFreeNeighborListAction);
HPX_PLAIN_ACTION(sphTreeAllValid, SphTreeAllValidAction);

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

/// Mark retained gas-tree topology and geometry unusable until rebuilt.
void sphTreeInvalidate() {
    treeValid = false;
}

/// Check the retained validity flag and gas-particle count.
bool sphTreeIsValid() {
    return treeValid && retainedParticleCount == sphParticlesSize();
}

/// Require valid retained gas trees throughout the SPH collective topology.
bool sphTreeAllValid() {
    std::vector<hpx::future<bool>> futures;
    for (auto const& child : sphCollectiveChildren()) {
        futures.push_back(hpx::async<SphTreeAllValidAction>(child));
    }
    bool valid = sphTreeIsValid();
    for (auto& future : futures) {
        valid = future.get() && valid;
    }
    return valid;
}

namespace detail {

/// Merge nonempty subtree extrema of swept-to-partition volume ratios.
void mergeSphVolumeRatios(SphTreeCreateReturn& result, double minimum, double maximum) {
    // Empty subtrees return zero; every occupied box has positive padded volume.
    if (maximum > 0.0) {
        result.minVolumeRatio =
            result.maxVolumeRatio > 0.0 ? std::min(result.minVolumeRatio, minimum) : minimum;
        result.maxVolumeRatio = std::max(result.maxVolumeRatio, maximum);
    }
}

/// Measure a local occupied node's swept-box volume relative to its original partition
/// volume.
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

/// Expand a bounding box to include another box component by component.
void mergeSphBounds(SphBox& target, SphBox const& source) {
    for (int dim = 0; dim < kDimensions; ++dim) {
        target.begin[dim] = std::min(target.begin[dim], source.begin[dim]);
        target.end[dim] = std::max(target.end[dim], source.end[dim]);
    }
}

/// Conservatively expand position bounds over a signed drift interval, accounting for
/// periodic seams.
SphBox sweptSphBounds(SphBox const& positions, Vector3<double> const& v_minus,
    Vector3<double> const& v_plus, double D) {
    SphBox result = positions;
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
            std::min({0.0, firstDisplacement, secondDisplacement}) - kSphBoxPadding;
        double const upper = positions.end[dim].toDouble() +
            std::max({0.0, firstDisplacement, secondDisplacement}) + kSphBoxPadding;
        if (!std::isfinite(lower) || !std::isfinite(upper)) {
            throw std::runtime_error("Nonfinite SPH drift bound");
        }
        if (upper - lower >= 1.0) {
            result.begin[dim] = -kSphBoxPadding;
            result.end[dim] = 1.0 + kSphBoxPadding;
        } else {
            result.begin[dim] = lower;
            result.end[dim] = upper;
        }
    }
    return result;
}

/// Update affected leaf swept bounds from velocities and propagate their union to local
/// ancestors.
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
        for (ParticleInteger particle = node.particleRange.first;
            particle < node.particleRange.second; ++particle) {
            bool const moving = rung < 0 || sphParticlesRung(particle) == rung;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const position = sphParticlesPositionValue(dim, particle).toDouble();
                positions.begin[dim] =
                    std::min(positions.begin[dim].toDouble(), position - kSphBoxPadding);
                positions.end[dim] =
                    std::max(positions.end[dim].toDouble(), position + kSphBoxPadding);
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

/// Expand retained local-tree swept bounds when the topology and gas count remain valid.
void sphTreeDriftBounds(double driftFactor, int rung) {
    if (treeValid && localRootId.index >= 0 && retainedParticleCount == sphParticlesSize()) {
        detail::updateLocalSphDriftBounds(localRootId, driftFactor, rung);
    }
}

/// Assign the range of staged neighbor IDs belonging to a local node.
void sphTreeSetNeighborRange(TreeId id, Pair<int, int> rng) {
    nodes[id.index].neighborRange = rng;
}

/// Invalidate every local leaf's stored neighbor-list range.
void sphTreeClearNeighborRanges() {
    for (int i = 0; i < leaflist.size(); i++) {
        nodes[leaflist[i]].neighborRange.first = -1;
        nodes[leaflist[i]].neighborRange.second = -1;
    }
}

/// Return the number of local gas-tree leaves.
int sphTreeLeaflistSize() {
    return leaflist.size();
}

/// Return the local TreeId for a leaf-list entry.
TreeId const sphTreeGetLeaf(int i) {
    TreeId id;
    id.index = leaflist[i];
    id.proc = hpxRank();
    return id;
}

/// Clear the active neighbor-ID list while retaining its allocation.
void sphTreeFreeNeighborList() {
    neighborList.resize(0);
}

/// Append neighbor IDs under the list mutex and return their starting offset.
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

/// Read a neighbor TreeId from the shared list by offset.
TreeId& sphTreeGetNeighbor(int i) {
    return neighborList[i];
}

/// Return the current gas-tree node-arena size.
long long sphTreeNodesSize() {
    return nodes.size();
}

/// Reserve another block of gas-tree node indices, failing if the arena is exhausted.
void SphTreeAllocator::reset() {
    auto const& opts = getOptions();
    int const treeCacheLineSize = opts.compute.treeCacheLineSize;
    next_ = (nextId += treeCacheLineSize);
    last_ = std::min(next_ + treeCacheLineSize, (int) nodes.size());
    if (next_ >= nodes.size()) {
        throw std::length_error("Tree arena full");
    }
}

/// Register a thread-local node-block allocator for coordinated arena resets.
SphTreeAllocator::SphTreeAllocator() {
    ready_ = false;
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    allocatorList.push_back(this);
    allocatorMtx--;
}

/// Return the next gas-tree node index, reserving a new block when necessary.
int SphTreeAllocator::allocate() {
    if (next_ == last_) {
        reset();
    }
    ASSERT(next_ < nodes.size());
    return next_++;
}

/// Remove this allocator from the shared registration list.
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

/// Build a local subtree synchronously or with bounded asynchronous branching.
hpx::future<SphTreeCreateReturn> sphTreeCreateFork(SphTreeCreateParameters params, size_t key,
    Pair<int, int> const& processRange, Pair<ParticleInteger> const& particleRange,
    Range<double> const& box, int const depth, bool const localRoot, bool threadme) {
    static std::atomic<int> numberOfThreads(0);
    hpx::future<SphTreeCreateReturn> rc;
    // Every node below a local root lives on that root's locality.
    ALWAYS_ASSERT(processRange.first == hpxRank());
    if (threadme) {
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
                sphTreeBuild(params, key, processRange, particleRange, box, depth, localRoot));
        } catch (...) {
            rc = hpx::make_exceptional_future<SphTreeCreateReturn>(std::current_exception());
        }
    } else {
        rc = hpx::async(hpx::annotated_function(
            [params, processRange, key, particleRange, depth, localRoot, box]() {
                try {
                    auto rc = sphTreeBuild(
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

/// Prepare reusable gas-tree arenas and reset allocator blocks throughout the SPH topology.
static void sphTreeAllocateNodes() {
    auto const& opts = getOptions();
    sphTreeInvalidate();
    int const treeCacheLineSize = opts.compute.treeCacheLineSize;
    int const bucketSize = opts.compute.sphBucketSize;
    Vector<hpx::future<void>> futs;
    for (auto const& c : sphCollectiveChildren()) {
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

namespace detail {

// Merge one locality's tree summary into a running total (first: copy).
void mergeSphTreeReturn(SphTreeCreateReturn& total, SphTreeCreateReturn const& part, bool first) {
    if (first) {
        total = part;
        return;
    }
    mergeSphBounds(total.innerBox, part.innerBox);
    mergeSphBounds(total.outerBox, part.outerBox);
    mergeSphBounds(total.driftBox, part.driftBox);
    total.numberOfActiveParticles += part.numberOfActiveParticles;
    total.activeNodes += part.activeNodes;
    total.nodeCount += part.nodeCount;
    total.activeLeafNodes += part.activeLeafNodes;
    total.leafNodes += part.leafNodes;
    total.minDepth = std::min(total.minDepth, part.minDepth);
    total.maxDepth = std::max(total.maxDepth, part.maxDepth);
    total.floatingPointOperations += part.floatingPointOperations;
    mergeSphVolumeRatios(total, part.minVolumeRatio, part.maxVolumeRatio);
}

}    // namespace detail

// Collective: every locality builds its own octree at depth 0 over the whole periodic box, in
// parallel, and the summaries are reduced. No locality needs another's root: remote leaves
// arrive as ghosts.
static SphTreeCreateReturn sphTreeCreateLocalRoots(SphTreeCreateParameters params) {
    Vector<hpx::future<SphTreeCreateReturn>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphTreeCreateLocalRootsAction>(child, params));
    }
    SphTreeCreateReturn result;
    std::exception_ptr failure;
    try {
        result = sphTreeBuild(params, sphDomainLeafKey(hpxRank(), hpxSize()),
            Pair<int>(hpxRank(), hpxRank() + 1), Pair<ParticleInteger>(-1, -1), unitBox<double>(),
            0, true);
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& child : children) {
        try {
            auto const part = child.get();
            if (!failure) {
                detail::mergeSphTreeReturn(result, part, false);
            }
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return result;
}

/// Create the distributed forest or a validated single-locality subtree at the requested
/// depth.
SphTreeCreateReturn sphTreeCreate(SphTreeCreateParameters params, size_t key,
    Pair<int, int> processRange, Pair<ParticleInteger> particleRange, Range<double> box, int depth,
    bool localRoot) {
    if (depth < 0 || depth > SphTreeNode::kMaximumDepth) {
        throw std::out_of_range("SPH tree depth must be between 0 and 15");
    }
    if (depth == 0 && processRange.first == 0 && processRange.second == hpxSize()) {
        // The whole tree: one octree per locality, built in parallel; no rank-routing nodes.
        ALWAYS_ASSERT(hpxRank() == 0);
        sphTreeAllocateNodes();
        auto result = sphTreeCreateLocalRoots(params);
        retainedRoot = result.id;
        return result;
    }
    if (processRange.second - processRange.first != 1 || processRange.first != hpxRank()) {
        throw std::invalid_argument("An SPH subtree build covers this locality alone");
    }
    if (depth == 0) {
        sphTreeAllocateNodes();
    }
    return sphTreeBuild(params, key, processRange, particleRange, box, depth, localRoot);
}

/// Partition gas recursively and build local support bounds, active counts, and retained
/// topology metadata.
static SphTreeCreateReturn sphTreeBuild(SphTreeCreateParameters params, size_t key,
    Pair<int, int> processRange, Pair<ParticleInteger> particleRange, Range<double> box, int depth,
    bool localRoot) {
    auto const& opts = getOptions();
    int const bucketSize = opts.compute.sphBucketSize;
    SphTreeCreateReturn rc;
    if (depth < 0 || depth > SphTreeNode::kMaximumDepth) {
        throw std::out_of_range("SPH tree depth must be between 0 and 15");
    }
    ALWAYS_ASSERT(processRange.second - processRange.first == 1 && processRange.first == hpxRank());
    if (localRoot) {
        leaflist.resize(0);
        particleRange.first = 0;
        particleRange.second = sphParticlesSize();
        // A retained hierarchical owner can hold inactive gas outside its rank
        // domain, so each local octree partitions the full periodic box and a later
        // rebuild can subdivide those particles rather than chasing a domain boundary.
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
    SphBox innerBox;
    SphBox outerBox;
    SphBox driftBox;
    if (particleRange.second - particleRange.first > bucketSize) {
        if (depth == SphTreeNode::kMaximumDepth) {
            throw std::out_of_range("SPH octree subdivision would require depth 16");
        }
        std::array<hpx::future<SphTreeCreateReturn>, kSphChildCount> childFutures;
        {
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
        flops.doublePrecision += 4 * kDimensions;    // Empty-bound conversions.
        for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
            float const h = params.smoothingLengthMultiplier * sphParticlesSmoothingLength(i);
            flops.singlePrecision += 1;
            Vector3<Position> xValue;
            for (int dim = 0; dim < kDimensions; dim++) {
                xValue[dim] = sphParticlesPositionValue(dim, i);
            }
            for (int dim = 0; dim < kDimensions; dim++) {
                double const position = xValue[dim].toDouble();
                innerBox.begin[dim] =
                    std::min(innerBox.begin[dim].toDouble(), position - kSphBoxPadding);
                innerBox.end[dim] =
                    std::max(innerBox.end[dim].toDouble(), position + kSphBoxPadding);
                // Position and old-bound conversions, two offsets and two new bounds.
                flops.doublePrecision += 7;
                if (sphParticlesRung(i) >= params.minimumRung) {
                    outerBox.begin[dim] =
                        std::min(outerBox.begin[dim].toDouble(), position - h - kSphBoxPadding);
                    outerBox.end[dim] =
                        std::max(outerBox.end[dim].toDouble(), position + h + kSphBoxPadding);
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
        throw std::length_error("Tree arena full");
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

/// Refit a retained local node's support and motion bounds without rebuilding its topology.
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
                    std::min(result.innerBox.begin[dim].toDouble(), position - kSphBoxPadding);
                result.innerBox.end[dim] =
                    std::max(result.innerBox.end[dim].toDouble(), position + kSphBoxPadding);
                flops.doublePrecision += 7;
                if (active) {
                    result.outerBox.begin[dim] = std::min(
                        result.outerBox.begin[dim].toDouble(), position - support - kSphBoxPadding);
                    result.outerBox.end[dim] = std::max(
                        result.outerBox.end[dim].toDouble(), position + support + kSphBoxPadding);
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
            ALWAYS_ASSERT(child.proc == hpxRank());
            if (node.numberOfParticles() >= kMinSortThreadParts) {
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

// Collective: every locality refreshes its own octree in parallel.
static SphTreeCreateReturn sphTreeRefreshLocalRoots(
    SphTreeCreateParameters params, double driftFactor) {
    Vector<hpx::future<SphTreeCreateReturn>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphTreeRefreshLocalRootsAction>(child, params, driftFactor));
    }
    SphTreeCreateReturn result;
    std::exception_ptr failure;
    try {
        result = sphTreeRefreshNode(localRootId, params, driftFactor);
        result.id = localRootId;
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& child : children) {
        try {
            auto const part = child.get();
            if (!failure) {
                detail::mergeSphTreeReturn(result, part, false);
            }
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return result;
}

/// Refit all local roots, requiring valid retained topology on every locality.
SphTreeCreateReturn sphTreeRefresh(SphTreeCreateParameters params, double driftFactor) {
    if (!sphTreeAllValid() || retainedRoot.index < 0) {
        throw std::runtime_error("Cannot refresh an invalid SPH tree; rebuild it first");
    }
    return sphTreeRefreshLocalRoots(params, driftFactor);
}

/// Invalidate distributed gas trees and optionally release their node arenas.
void sphTreeDestroy(bool freeSphTree) {
    sphTreeInvalidate();
    Vector<hpx::future<void>> futs;
    auto const children = sphCollectiveChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<SphTreeDestroyAction>(c, freeSphTree));
    }
    if (freeSphTree) {
        nodes = decltype(nodes)();
    }
    hpx::wait_all(futs.begin(), futs.end());
}

/// Set the stored active-particle count of a local node.
void sphTreeSetNumberOfActiveParticles(TreeId id, ParticleInteger i) {
    nodes[id.index].numberOfActiveParticles = i;
}

/// Set a local node's density-convergence flag.
void sphTreeSetConverged(TreeId id, bool c) {
    nodes[id.index].converged = c;
}

/// Return this locality's retained gas-tree root ID.
TreeId sphTreeLocalRoot() {
    return localRootId;
}

/// Return a local node or staged remote ghost leaf, rejecting unstaged remote access.
SphTreeNode const* sphTreeGetNode(TreeId id) {
    if (id.proc == hpxRank()) {
        ASSERT(id.index >= 0);
#ifndef NDEBUG
        if (id.index >= nodes.size()) {
            throw std::out_of_range(
                std::format("id.index is {} but nodes.size() is {}", id.index, nodes.size()));
        }
#endif
        return &nodes[id.index];
    }
    // A remote node is always a ghost leaf: there are no rank-routing nodes and no node cache.
    if (auto const* ghost = sphGhostsFindNode(id)) {
        return ghost;
    }
    throw std::logic_error(
        std::format("SPH tree: rank {} reads node ({}, {}), which is not a ghost leaf here",
            hpxRank(), id.proc, id.index));
}

/// Replace a local node's inner position and outer support bounds.
void sphTreeSetBoxes(TreeId id, SphBox const& ibox, SphBox const& obox) {
    nodes[id.index].outerBox = obox;
    nodes[id.index].innerBox = ibox;
}

