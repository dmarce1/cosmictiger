/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_TREE_HPP_
#define SPH_TREE_HPP_

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/tree_id.hpp>
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

struct SphTreeNode {
    Array<TreeId, kChildCount> children;
    Pair<int, int> processRange;
    Pair<ParticleInteger> particleRange;
    Pair<ParticleInteger> sinkParticleRange;
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    Fixed32Range box;
    size_t numberOfActiveParticles;
    bool localRoot;
    bool leaf;
    bool converged;
    size_t nodeCount;
    size_t activeNodes;
    int depth;
    Pair<int, int> neighborRange;
    SphTreeNode()
      : processRange(0, 0)
      , particleRange(0, 0)
      , sinkParticleRange(0, 0)
      , numberOfActiveParticles(0)
      , localRoot(false)
      , leaf(true)
      , converged(false)
      , nodeCount(0)
      , activeNodes(0)
      , depth(0)
      , neighborRange(0, 0) {
        for (int child = 0; child < kChildCount; child++) {
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
        arc & converged;
        arc & neighborRange;
        arc & box;
        arc & sinkParticleRange;
        arc & innerBox;
        arc & outerBox;
        arc & children;
        arc & processRange;
        arc & particleRange;
        arc & numberOfActiveParticles;
        arc & localRoot;
        arc & leaf;
        arc & activeNodes;
        arc & nodeCount;
        arc & depth;
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
