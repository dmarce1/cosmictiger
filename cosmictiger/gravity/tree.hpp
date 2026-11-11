/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <cstdint>

#include <hpx/serialization/serialization_fwd.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/tree_id.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree_source_moments.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

// Tree multipoles live in an arena of build-order positioned records
// (fmmPositionedMultipoleBytes(kFmmOrder) bytes each, stored
// fmmPositionedMultipoleStride(kFmmOrder) bytes apart). Nodes point at the
// order-agnostic position header; per-order code views the moments through
// fmmMultiPos(node.mpos).
MultipolePosition* treeAddMultipole(Vector3<Position> const& pos, MultipoleBuffer const& multipole);
void treeReadMultipole(MultipolePosition const* record, MultipoleBuffer& multipole);

struct TreeNode {
    Fixed32Range box;
    Array<TreeId, kChildCount> children;
    Pair<ParticleInteger> particleRange;
    FmmParticleStore sourceStore = FmmParticleStore::Owned;
    uint64_t meshCellKey = 0;
    Pair<int, int> processRange;
    Vector3<Position> pos;
    MultipolePosition* mpos;
    float radius;
    struct {
        unsigned short depth : 14;
        unsigned short localRoot : 1;
        unsigned short leaf : 1;
    };
    CUDA_EXPORT

    /// Return the number of particles in this node's half-open interval.
    CUDA_EXPORT
    inline ParticleInteger numberOfParticles() const {
        return particleRange.second - particleRange.first;
    }
    /// Combine the node's particle interval with its owning process for source reads.
    inline ParticleGlobalRange globalParticleRange() const {
        ParticleGlobalRange r;
        r.proc = processRange.first;
        r.range = particleRange;
        return r;
    }
    /// Test whether the node spans exactly one locality.
    inline bool isLocal() const {
        return processRange.second - processRange.first == 1;
    }
    /// Test whether the node is wholly owned by the current locality.
    inline bool isLocalHere() const {
        return isLocal() && processRange.first == hpxRank();
    }
    HPX_SERIALIZATION_SPLIT_MEMBER();
    /// Restore node metadata and allocate local transport storage for the received
    /// positioned multipole.
    template <class A>
    void load(A&& arc, unsigned) {
        Vector3<Position> multipolePosition;
        MultipoleBuffer multipole;
        arc & multipolePosition;
        arc & multipole;
        mpos = treeAddMultipole(multipolePosition, multipole);
        arc & box;
        arc & pos;
        arc & children;
        arc & processRange;
        arc & particleRange;
        arc & sourceStore;
        arc & meshCellKey;
        arc & radius;
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
    /// Archive the node and its pointed-to multipole contents rather than serializing
    /// the pointer value.
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
        arc & sourceStore;
        arc & meshCellKey;
        arc & radius;
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

// Only local source nodes can be resolved against this locality's arrays.
inline FmmSourceView treeSourceView(TreeNode const& node) {
    return fmmSourceView(node.sourceStore);
}

struct TreeCreateReturn {
    TreeBuildMoments multi;
    Vector3<Position> pos;
    TreeId id;
    float radius;
    Range<double> box;
    /// Archive this record in the field order required by distributed transport.
    template <class A>
    void serialize(A&& a, unsigned) {
        a & multi;
        a & id;
        a & pos;
        a & radius;
        a & box;
    }
};

struct TreeCreateParameters {
    int minRung{};
    double theta{};
    ParticleInteger parallelParticleThreshold{};
    int minLevel{};
    bool leafPushed;
    bool doLeafSizes;
    /// Initialize leaf-processing and leaf-size diagnostics flags for tree
    /// construction.
    TreeCreateParameters() {
        leafPushed = false;
        doLeafSizes = false;
    }
    TreeCreateParameters(int minRung, double theta);
    /// Archive `parallelParticleThreshold`, `doLeafSizes`, `leafPushed`, `minRung`,
    /// `theta`, `minLevel` in transport order.
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

void treeReset();
void treeDestroy(bool freeTree = false);
TreeNode const* treeGetNode(TreeId);
// Cumulative remote tree reads issued by this locality, never reset.
uint64_t treeRemoteFetchCount();
void treeSortParticlesBySphParticles();
void treeFreeNeighborList();
long long treeNodesSize();
long long treeNodesNextIndex();
void treeClearNeighborRanges();
int treeAvgLeafSize();
int treeAllocateNeighborList(Vector<TreeId> const& values);
void treeSetNeighborRange(TreeId id, Pair<int, int> rng);
TreeId const treeGetLeaf(int i);
size_t treeAddRemote(TreeNode const& remote);
TreeNode* treeData();
TreeId& treeGetNeighbor(int i);
void tree2Cpu();
void tree2Gpu();

#endif /* TREE_HPP_ */
