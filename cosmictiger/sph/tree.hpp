/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_TREE_HPP_
#define SPH_TREE_HPP_

#include <stdexcept>
#include <type_traits>

#include <hpx/serialization/input_archive.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/tree_id.hpp>
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

// Octant bit d selects the upper half of spatial dimension d. Unoccupied
// octants use {-1, -1}; rank-routing nodes use only slots 0 and 1.
inline constexpr int kSphChildCount = 1 << kDimensions;

struct SphTreeNode {
    static constexpr int kMaximumDepth = 15;
    /*
     * Variable directory
     *
     * Expected total: 200 bytes, including 7 bytes of alignment padding.
     * Assumes 4-byte int/ParticleInteger, 8-byte size_t, 1-byte bool,
     * kSphChildCount == 8, and kDimensions == 3.
     *
     * children               - 64 bytes - Child node IDs (process rank and node index).
     * processRange           -  8 bytes - Process ranks covered by this subtree [begin, end).
     * particleRange          -  8 bytes - Source particle indices [begin, end).
     * sinkParticleRange      -  8 bytes - Corresponding sink/output particle indices [begin, end).
     * innerBox               - 24 bytes - Bounds enclosing all particle positions in this subtree.
     * outerBox               - 24 bytes - Bounds enclosing active particles' kernel supports,
     *                                    expanded by the configured support-radius multiplier.
     * box                    - 24 bytes - Spatial partition assigned to this node.
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
     * increasing the expected total to 216 bytes.
     * Sizes describe the in-memory layout, not the serialized representation.
     */
    Array<TreeId, kSphChildCount> children;
    Pair<int, int> processRange;
    Pair<ParticleInteger> particleRange;
    Pair<ParticleInteger> sinkParticleRange;
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    Fixed32Range box;
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
        }
    }
    inline ParticleInteger numberOfParticles() const {
        return particleRange.second - particleRange.first;
    }
    inline ParticleGlobalRange globalParticleRange() const {
        ParticleGlobalRange r;
        r.proc = processRange.first;
        r.range = particleRange;
        return r;
    }
    inline bool isLocal() const {
        return processRange.second - processRange.first == 1;
    }
    inline bool isLocalHere() const {
        return isLocal() && processRange.first == hpxRank();
    }
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
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    TreeId id;
    size_t numberOfActiveParticles;
    size_t activeNodes;
    size_t nodeCount;
    size_t activeLeafNodes;
    size_t leafNodes;
    int maxDepth;
    int minDepth;
    double floatingPointOperations;
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
    }
};

struct SphTreeCreateParameters {
    int minimumRung;
    float supportRadiusMultiplier;
    SphTreeCreateParameters() = default;
    SphTreeCreateParameters(int minimumRung, float supportRadiusMultiplier);
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & minimumRung;
        arc & supportRadiusMultiplier;
    }
};

SphTreeCreateReturn sphTreeCreate(SphTreeCreateParameters params, size_t key = 1,
    Pair<int, int> processRange = Pair<int>(0, hpxSize()),
    Pair<ParticleInteger> particleRange = Pair<ParticleInteger>(-1, -1),
    Range<double> box = unitBox<double>(), int depth = 0, bool localRoot = (hpxSize() == 1));
void sphTreeCacheClear();
// Call after all tree consumers finish. Normal teardown invalidates caches but
// retains the constructed arena for rebuilding; true explicitly releases it.
void sphTreeDestroy(bool freeSphTree = false);
SphTreeNode const* sphTreeGetNode(TreeId);
void sphTreeSortSphParticlesByParticles();
void sphTreeSetConverged(TreeId id, bool = true);
void sphTreeSetNumberOfActiveParticles(TreeId id, ParticleInteger i);
void sphTreeSetBoxes(TreeId, Fixed32Range const&, Fixed32Range const&);
void sphTreeFreeNeighborList();
int sphTreeAllocateNeighborList(Vector<TreeId> const&);
TreeId& sphTreeGetNeighbor(int i);
long long sphTreeNodesSize();
void sphTreeSetNeighborRange(TreeId id, Pair<int, int> rng);
int sphTreeLeaflistSize();
TreeId const sphTreeGetLeaf(int i);
void sphTreeClearNeighborRanges();

#endif /* SPH_TREE_HPP_ */
