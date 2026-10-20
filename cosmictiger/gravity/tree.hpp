/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <hpx/serialization/serialization_fwd.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/fmm/fmm_types.hpp>
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

struct TreeId {
    int proc;
    int index;
    inline bool operator==(TreeId other) const {
        return proc == other.proc && index == other.index;
    }
    inline bool operator<(TreeId other) const {
        if (proc < other.proc) {
            return true;
        } else if (proc > other.proc) {
            return false;
        } else if (index < other.index) {
            return true;
        } else {
            return false;
        }
    }
    inline bool operator!=(TreeId other) const {
        return proc != other.proc || index != other.index;
    }
    template <class A>
    void serialize(A&& a, unsigned) {
        a & proc;
        a & index;
    }
};

struct TreeIdHash {
    inline size_t operator()(TreeId id) const {
        int const lineSize = getOptions().compute.treeCacheLineSize;
        int const i = id.index / lineSize;
        return i * (hpxSize() - 1) + ((id.proc < hpxRank()) ? id.proc : id.proc - 1);
    }
};

struct TreeIdHashLo {
    inline size_t operator()(TreeId id) const {
        TreeIdHash hash;
        return hash(id) % TREE_CACHE_SIZE;
    }
};

struct TreeIdHashHi {
    inline size_t operator()(TreeId id) const {
        TreeIdHash hash;
        return hash(id) / TREE_CACHE_SIZE;
    }
};

// Tree multipoles live in an arena of runtime-order positioned records
// (fmmPositionedMultipoleBytes(fmmOrder()) bytes each, stored
// fmmPositionedMultipoleStride(fmmOrder()) bytes apart). Nodes point at the
// order-agnostic position header; per-order code views the moments through
// fmmMultiPos<order>(node.mpos).
MultipolePosition* treeAddMultipole(
    Array<Fixed32, NDIM> const& pos, MultipoleBuffer const& multipole);
void treeReadMultipole(MultipolePosition const* record, MultipoleBuffer& multipole);

struct TreeNode {
    Fixed32Range box;
    Array<TreeId, NCHILD> children;
    Pair<ParticleInteger> particleRange;
    Pair<int, int> processRange;
    Array<Fixed32, NDIM> pos;
    MultipolePosition* mpos;
    float radius;
    // Maximum actual compact gravity support among represented particles, not an
    // effective smoothing length. Rebuild/refit after those supports change.
    float maxGravitySupportRadius = 0.0f;
    struct {
        unsigned short depth : 14;
        unsigned short localRoot : 1;
        unsigned short leaf : 1;
    };
    CUDA_EXPORT

    CUDA_EXPORT
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
    HPX_SERIALIZATION_SPLIT_MEMBER();
    template <class A>
    void load(A&& arc, unsigned) {
        Array<Fixed32, NDIM> multipolePosition;
        MultipoleBuffer multipole;
        arc & multipolePosition;
        arc & multipole;
        mpos = treeAddMultipole(multipolePosition, multipole);
        arc & box;
        arc & pos;
        arc & children;
        arc & processRange;
        arc & particleRange;
        arc & radius;
        arc & maxGravitySupportRadius;
        int localRoot0;
        int leaf0;
        int depth0;
        arc & localRoot0;
        arc & leaf0;
        arc & depth0;
        localRoot = localRoot0;
        depth = depth0;
        leaf = leaf0;
    }
    template <class A>
    void save(A&& arc, unsigned) const {
        MultipoleBuffer multipole;
        treeReadMultipole(mpos, multipole);
        arc & mpos->pos;
        arc & multipole;
        arc & box;
        arc & pos;
        arc & children;
        arc & processRange;
        arc & particleRange;
        arc & radius;
        arc & maxGravitySupportRadius;
        int localRoot0;
        int leaf0;
        int depth0;
        localRoot0 = localRoot;
        leaf0 = leaf;
        depth0 = depth;
        arc & localRoot0;
        arc & leaf0;
        arc & depth0;
    }
};

struct TreeCreateReturn {
    MultipoleBuffer multi;
    Array<Fixed32, NDIM> pos;
    TreeId id;
    float radius;
    float maxGravitySupportRadius = 0.0f;
    Range<double> box;
    template <class A>
    void serialize(A&& a, unsigned) {
        a & multi;
        a & id;
        a & pos;
        a & radius;
        a & maxGravitySupportRadius;
        a & box;
    }
};

struct TreeCreateParameters {
    int minRung;
    double theta;
    ParticleInteger parallelParticleThreshold;
    int minLevel;
    bool leafPushed;
    bool doLeafSizes;
    TreeCreateParameters() {
        leafPushed = false;
        doLeafSizes = false;
    }
    TreeCreateParameters(int minRung, double theta, double hmax);
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & parallelParticleThreshold;
        arc & doLeafSizes;
        arc & leafPushed;
        arc & minRung;
        arc & theta;
        arc & minLevel;
    }
};

TreeCreateReturn treeCreate(TreeCreateParameters params, size_t key = 1,
    Pair<int, int> processRange = Pair<int>(0, hpxSize()),
    Pair<ParticleInteger> particleRange = Pair<ParticleInteger>(-1, -1),
    Range<double> box = unitBox<double>(), int depth = 0, bool localRoot = (hpxSize() == 1));
void treeReset();
void treeDestroy(bool freeTree = false);
int treeMinLevel(double theta, double hsoft);
TreeNode const* treeGetNode(TreeId);
void treeSortParticlesBySphParticles();
void treeFreeNeighborList();
long long treeNodesSize();
long long treeNodesNextIndex();
void treeClearNeighborRanges();
int treeAvgLeafSize();
int treeAllocateNeighborList(Vector<TreeId> const& values);
void treeSetNeighborRange(TreeId id, Pair<int, int> rng);
void treeSetBoxes(TreeId id, Fixed32Range const& ibox, Fixed32Range const& obox, float hmax);
TreeId const treeGetLeaf(int i);
size_t treeAddRemote(TreeNode const& remote);
TreeNode* treeData();
TreeId& treeGetNeighbor(int i);
void tree2Cpu();
void tree2Gpu();

#endif /* TREE_HPP_ */
