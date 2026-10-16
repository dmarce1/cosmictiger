/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <hpx/serialization/serialization_fwd.hpp>

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>

struct MultipolePos {
    Multipole<float> m;
    Array<Fixed32, NDIM> pos;
};

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
        int const lineSize = getOptions().treeCacheLineSize;
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

struct MultiPos {
    Multipole<float> multi;
    Array<Fixed32, NDIM> pos;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & multi;
        arc & pos;
    }
};

MultiPos* treeAddMultipole(MultiPos const& mpos);

struct TreeNode {
    Fixed32Range box;
    Array<TreeId, NCHILD> children;
    Pair<ParticleInteger> particleRange;
    Pair<int, int> processRange;
    Array<Fixed32, NDIM> pos;
    MultiPos* mpos;
    float radius;
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
        MultiPos thisMpos;
        arc & thisMpos;
        mpos = treeAddMultipole(thisMpos);
        arc & box;
        arc & pos;
        arc & children;
        arc & processRange;
        arc & particleRange;
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
    template <class A>
    void save(A&& arc, unsigned) const {
        arc&* mpos;
        arc & box;
        arc & pos;
        arc & children;
        arc & processRange;
        arc & particleRange;
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

struct TreeCreateReturn {
    Multipole<float> multi;
    Array<Fixed32, NDIM> pos;
    TreeId id;
    float radius;
    Range<double> box;
    template <class A>
    void serialize(A&& a, unsigned) {
        a & multi;
        a & id;
        a & pos;
        a & radius;
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
