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

#ifndef SPH_TREE_HPP_
#define SPH_TREE_HPP_

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/tree.hpp>

struct SphTreeNode {
    Array<TreeId, NCHILD> children;
    Pair<int, int> procRange;
    Pair<ParticleInteger> partRange;
    Pair<ParticleInteger> sinkPartRange;
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    Fixed32Range box;
    size_t numberActive;
    bool localRoot;
    bool leaf;
    bool converged;
    size_t nodeCount;
    size_t activeNodes;
    int depth;
    Pair<int, int> neighborRange;
    SphTreeNode()
      : procRange(0, 0)
      , partRange(0, 0)
      , sinkPartRange(0, 0)
      , numberActive(0)
      , localRoot(false)
      , leaf(true)
      , converged(false)
      , nodeCount(0)
      , activeNodes(0)
      , depth(0)
      , neighborRange(0, 0) {
        for (int child = 0; child < NCHILD; child++) {
            children[child].proc = -1;
            children[child].index = -1;
        }
        for (int dim = 0; dim < NDIM; dim++) {
            innerBox.begin[dim] = innerBox.end[dim] = 0.0;
            outerBox.begin[dim] = outerBox.end[dim] = 0.0;
            box.begin[dim] = box.end[dim] = 0.0;
        }
    }
    inline ParticleInteger numberParticles() const {
        return partRange.second - partRange.first;
    }
    inline ParticleGlobalRange globalPartRange() const {
        ParticleGlobalRange r;
        r.proc = procRange.first;
        r.range = partRange;
        return r;
    }
    inline bool isLocal() const {
        return procRange.second - procRange.first == 1;
    }
    inline bool isLocalHere() const {
        return isLocal() && procRange.first == hpxRank();
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & converged;
        arc & neighborRange;
        arc & box;
        arc & sinkPartRange;
        arc & innerBox;
        arc & outerBox;
        arc & children;
        arc & procRange;
        arc & partRange;
        arc & numberActive;
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
    size_t numberActive;
    size_t activeNodes;
    size_t nodeCount;
    size_t activeLeafNodes;
    size_t leafNodes;
    int maxDepth;
    int minDepth;
    double flops;
    template <typename A>
    void serialize(A&& a, unsigned) {
        a & activeLeafNodes;
        a & leafNodes;
        a & activeNodes;
        a & innerBox;
        a & outerBox;
        a & id;
        a & numberActive;
        a & nodeCount;
        a & flops;
        a & maxDepth;
        a & minDepth;
    }
};

struct SphTreeCreateParameters {
    int minRung;
    float hWt;
    SphTreeCreateParameters() = default;
    SphTreeCreateParameters(int minRung, float hWt);
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & minRung;
        arc & hWt;
    }
};

SphTreeCreateReturn sphTreeCreate(SphTreeCreateParameters params, size_t key = 1,
    Pair<int, int> procRange = Pair<int>(0, hpxSize()),
    Pair<ParticleInteger> partRange = Pair<ParticleInteger>(-1, -1),
    Range<double> box = unitBox<double>(), int depth = 0, bool localRoot = (hpxSize() == 1));
void sphTreeCacheClear();
void sphTreeDestroy(bool freeSphTree = false);
SphTreeNode const* sphTreeGetNode(TreeId);
void sphTreeSortSphParticlesByParticles();
void sphTreeSetConverged(TreeId id, bool = true);
void sphTreeSetNumberActive(TreeId id, ParticleInteger i);
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
