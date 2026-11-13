/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_TREE_HPP_
#define SPH_TREE_HPP_

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <hpx/serialization/input_archive.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/tree_id.hpp>
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

// Child bit d selects the upper half of spatial dimension d. Hybrid spatial
// nodes bisect the longest one, two or three axes; unselected bits stay zero.
// Unoccupied slots use {-1, -1}.
inline constexpr int kSphChildCount = 1 << kDimensions;

struct SphTreeNode {
    static constexpr int kMaximumDepth = 15;
    /*
     * Variable directory
     *
     * Expected total: 224 bytes, including 7 bytes of alignment padding.
     * Assumes 4-byte int/ParticleInteger, 8-byte size_t, 1-byte bool,
     * kSphChildCount == 8, and kDimensions == 3.
     *
     * children               - 64 bytes - Child node IDs (process rank and node index).
     * processRange           -  8 bytes - Process ranks covered by this subtree [begin, end).
     * particleRange          -  8 bytes - Source particle indices [begin, end).
     * sinkParticleRange      -  8 bytes - Corresponding sink/output particle indices [begin, end).
     * innerBox               - 24 bytes - Bounds enclosing all particle positions in this subtree.
     * outerBox               - 24 bytes - Bounds enclosing active particles' kernel supports,
     *                                    expanded by the configured smoothing-length multiplier
     *                                    (kSphSupportSearchMultiplier while supports may change,
     *                                    kSphConvergedSearchMultiplier after convergence).
     * box                    - 24 bytes - Spatial partition assigned to this node.
     * driftBox               - 24 bytes - Conservative swept bounds since the last rebuild.
     * numberOfActiveParticles-  8 bytes - Number of active particles in this subtree.
     * depth                  -  5 bits - Tree level (0-15), packed with the three flags below.
     * localRoot              -  1 bit   - Whether this node roots a process-local subtree.
     * leaf                   -  1 bit   - Whether this node is a leaf particle bucket.
     * converged              -  1 bit   - Node convergence flag; currently unused by SPH consumers.
     * nodeCount              -  8 bytes - Total nodes in this subtree, including this node.
     * activeNodes            -  8 bytes - Nodes in this subtree containing active particles.
     * neighborRange          -  8 bytes - Neighbor-list indices [begin, end); (-1, -1) if unset.
     *
     * With LONG_LONG_PART_INT, both particle ranges occupy 16 bytes each,
     * increasing the expected total to 240 bytes.
     * Sizes describe the in-memory layout, not the serialized representation.
     */
    Array<TreeId, kSphChildCount> children;
    Pair<int, int> processRange;
    Pair<ParticleInteger> particleRange;
    Pair<ParticleInteger> sinkParticleRange;
    SphBox innerBox;
    SphBox outerBox;
    SphBox box;
    SphBox driftBox;
    size_t numberOfActiveParticles;
    struct {
        unsigned char depth : 5;
        unsigned char localRoot : 1;
        unsigned char leaf : 1;
        unsigned char converged : 1;
    };
    size_t nodeCount;
    size_t activeNodes;
    Pair<int, int> neighborRange;
    /// Initialize an empty leaf with invalid children, zero bounds, and no active
    /// particles or neighbors.
    SphTreeNode()
      : processRange(0, 0)
      , particleRange(0, 0)
      , sinkParticleRange(0, 0)
      , numberOfActiveParticles(0)
      , depth(0)
      , localRoot(false)
      , leaf(true)
      , converged(false)
      , nodeCount(0)
      , activeNodes(0)
      , neighborRange(0, 0) {
        for (int child = 0; child < kSphChildCount; child++) {
            children[child].proc = -1;
            children[child].index = -1;
        }
        for (int dim = 0; dim < kDimensions; dim++) {
            innerBox.begin[dim] = innerBox.end[dim] = 0.0;
            outerBox.begin[dim] = outerBox.end[dim] = 0.0;
            box.begin[dim] = box.end[dim] = 0.0;
            driftBox.begin[dim] = driftBox.end[dim] = 0.0;
        }
    }
    /// Return the size of this node's gas-particle interval.
    inline ParticleInteger numberOfParticles() const {
        return particleRange.second - particleRange.first;
    }
    /// Combine the node's particle interval with its owning locality.
    inline ParticleGlobalRange globalParticleRange() const {
        ParticleGlobalRange r;
        r.proc = processRange.first;
        r.range = particleRange;
        return r;
    }
    /// Test whether this node's process interval contains exactly one locality.
    inline bool isLocal() const {
        return processRange.second - processRange.first == 1;
    }
    /// Test whether the node is wholly owned by the current locality.
    inline bool isLocalHere() const {
        return isLocal() && processRange.first == hpxRank();
    }
    /// Archive `archivedConverged`, `neighborRange`, `box`, `sinkParticleRange`,
    /// `innerBox`, `outerBox` and the remaining fields in transport order.
    template <typename A>
    void serialize(A&& arc, unsigned) {
        // Preserve the archive's bool/int fields; bit-fields cannot bind to references.
        bool archivedConverged = converged;
        bool archivedLocalRoot = localRoot;
        bool archivedLeaf = leaf;
        int archivedDepth = depth;
        arc & archivedConverged;
        arc & neighborRange;
        arc & box;
        arc & sinkParticleRange;
        arc & innerBox;
        arc & outerBox;
        arc & children;
        arc & processRange;
        arc & particleRange;
        arc & numberOfActiveParticles;
        arc & archivedLocalRoot;
        arc & archivedLeaf;
        arc & activeNodes;
        arc & nodeCount;
        arc & archivedDepth;
        arc & driftBox;
        if constexpr (std::is_same_v<std::remove_cvref_t<A>, hpx::serialization::input_archive>) {
            if (archivedDepth < 0 || archivedDepth > kMaximumDepth) {
                throw std::out_of_range("SPH tree depth must be between 0 and 15");
            }
            converged = archivedConverged;
            localRoot = archivedLocalRoot;
            leaf = archivedLeaf;
            depth = archivedDepth;
        }
    }
};

struct SphTreeCreateReturn {
    SphBox innerBox;
    SphBox outerBox;
    SphBox driftBox;
    TreeId id;
    size_t numberOfActiveParticles;
    size_t activeNodes;
    size_t nodeCount;
    size_t activeLeafNodes;
    size_t leafNodes;
    int maxDepth;
    int minDepth;
    double floatingPointOperations;
    // Extrema of swept volume since rebuild / original partition volume over
    // nonempty spatial nodes. Excludes rank routing; zero for an empty tree.
    // Fresh tight bounds can occupy less than the full partition (ratio < 1).
    double maxVolumeRatio = 0.0;
    double minVolumeRatio = 0.0;
    /// Archive this record in the field order required by distributed transport.
    template <typename A>
    void serialize(A&& a, unsigned) {
        a & activeLeafNodes;
        a & leafNodes;
        a & activeNodes;
        a & innerBox;
        a & outerBox;
        a & id;
        a & numberOfActiveParticles;
        a & nodeCount;
        a & floatingPointOperations;
        a & maxDepth;
        a & minDepth;
        a & driftBox;
        a & maxVolumeRatio;
        a & minVolumeRatio;
    }
};

struct SphTreeCreateParameters {
    int minimumRung;
    float smoothingLengthMultiplier;    // Support multiplier, normally kSphSupportSearchMultiplier.
    SphTreeCreateParameters() = default;
    SphTreeCreateParameters(int minimumRung, float smoothingLengthMultiplier);
    /// Archive `minimumRung`, `smoothingLengthMultiplier` in transport order.
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & minimumRung;
        arc & smoothingLengthMultiplier;
    }
};

SphTreeCreateReturn sphTreeCreate(SphTreeCreateParameters params, size_t key = 1,
    Pair<int, int> processRange = Pair<int>(0, hpxSize()),
    Pair<ParticleInteger> particleRange = Pair<ParticleInteger>(-1, -1),
    Range<double> box = unitBox<double>(), int depth = 0, bool localRoot = (hpxSize() == 1));
// The threshold counts all active gravity particles, including collisionless ones.
// The default uses an exact integer comparison, without multiplying counts or
// rounding one third of a large population through double precision.
constexpr bool sphTreeShouldRebuild(
    size_t activeParticles, size_t totalParticles, double fraction = 1.0 / 3.0) {
    if (totalParticles == 0) {
        return false;
    }
    if (fraction == 1.0) {
        return activeParticles >= totalParticles;
    }
    if (fraction == 1.0 / 3.0) {
        return activeParticles >= totalParticles / 3 + (totalParticles % 3 != 0);
    }
    // Match the precision of the configured fraction when the counts are exact
    // doubles: decimal inputs such as 0.07 include their boundary (7/100).
    if (totalParticles <= (std::uint64_t{1} << std::numeric_limits<double>::digits)) {
        return static_cast<double>(activeParticles) / static_cast<double>(totalParticles) >=
            fraction;
    }
    // Avoid rounding enormous integer counts to double before comparing them.
    return static_cast<long double>(activeParticles) >=
        static_cast<long double>(fraction) * static_cast<long double>(totalParticles);
}
// Refresh active/support bounds without sorting or changing particle membership.
// driftFactor has units of displacement / stored velocity (dt / a for drift()).
SphTreeCreateReturn sphTreeRefresh(SphTreeCreateParameters params, double driftFactor = 0.0);
// Local invalidation is cheap enough for particle resize/swap hooks. The all-valid
// query reduces over all localities and must be called from the coordinator.
void sphTreeInvalidate();
bool sphTreeIsValid();
bool sphTreeAllValid();
// Called on each locality before its particle drift. Rung < 0 includes every
// particle; otherwise only that exact rung moves. Refresh before global queries.
void sphTreeDriftBounds(double driftFactor, int rung = -1);
// This locality's local root, {-1, -1} when no tree is built here. Each locality builds and
// refreshes its own octree; there are no rank-routing nodes above the local roots, and no
// locality reads another's tree except its exported leaves, held as ghosts.
TreeId sphTreeLocalRoot();
// Call after all tree consumers finish. Normal teardown retains the constructed arena for
// rebuilding; true explicitly releases it.
void sphTreeDestroy(bool freeSphTree = false);
// A local node, or a remote leaf held as a ghost here; any other remote id throws.
SphTreeNode const* sphTreeGetNode(TreeId);
void sphTreeSortSphParticlesByParticles();
void sphTreeSetConverged(TreeId id, bool = true);
void sphTreeSetNumberOfActiveParticles(TreeId id, ParticleInteger i);
void sphTreeSetBoxes(TreeId, SphBox const&, SphBox const&);
void sphTreeFreeNeighborList();
int sphTreeAllocateNeighborList(Vector<TreeId> const&);
TreeId& sphTreeGetNeighbor(int i);
long long sphTreeNodesSize();
void sphTreeSetNeighborRange(TreeId id, Pair<int, int> rng);
int sphTreeLeaflistSize();
TreeId const sphTreeGetLeaf(int i);
void sphTreeClearNeighborRanges();

#endif /* SPH_TREE_HPP_ */
