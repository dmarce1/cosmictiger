/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_PARENT_TREE_HPP_
#define SPH_PARENT_TREE_HPP_

#include <cstdint>
#include <vector>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>

/*
 * The SPH parent tree (docs/sph_parent_tree_plan.md, section 3.1): one node per group of
 * localities, whose leaves are exactly the locality domains of the domain decomposition.
 *
 * Topology. The domain decomposition bisects the rank range [first, second) at
 * (first + second)/2 with child keys (key << 1) + i. A parent node regroups
 * kSpatialDimensions binary levels into one node of 2^kSpatialDimensions children while
 * it still covers that many ranks, and keeps single binary levels below that, so a node
 * of n ranks has 2^D children for n >= 2^D, 2 children for 2 <= n < 2^D, and is a leaf
 * for n = 1. Every child keeps its own domain key, so domainsRange(child.key) is its box.
 *
 * Placement. A node lives on the first rank of its range. Its first child covers the same
 * first rank, so every interior node shares its locality with exactly one child, and the
 * root lives on locality 0.
 */

struct SphParentChild {
    std::uint64_t key;
    Pair<int> rankRange;
};

// Number of children of a node covering rankCount ranks: 2^D, 2, or 0 for a leaf.
constexpr int sphParentArity(int rankCount) {
    int const fullArity = 1 << kSpatialDimensions;
    return rankCount >= fullArity ? fullArity : (rankCount >= 2 ? 2 : 0);
}

// Children of the node with domain key `key` covering `rankRange`, in rank order.
std::vector<SphParentChild> sphParentChildren(std::uint64_t key, Pair<int> rankRange);

// Domain key of the leaf that holds `rank` among `rankCount` ranks: the binary walk of the
// domain decomposition (src/gravity/fmm/cell_tree.cpp, treeCreateCell).
std::uint64_t sphDomainLeafKey(int rank, int rankCount);

// Census of a subtree, reduced from the leaves to the node that asked.
struct SphParentTreeCensus {
    int leafCount = 0;
    int nodeCount = 0;
    int maximumDepth = 0;
    std::int64_t rankSum = 0;
    // Every node sat on the first rank of its range, every leaf on its own rank with the
    // domain key of that rank, and every leaf was this locality's registered leaf.
    bool consistent = true;

    SphParentTreeCensus& operator+=(SphParentTreeCensus const& other);

    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & leafCount;
        archive & nodeCount;
        archive & maximumDepth;
        archive & rankSum;
        archive & consistent;
    }
};

// Construction record of one node.
struct SphParentNodeRecord {
    std::uint64_t key = 1;
    Pair<int> rankRange{0, 1};
    int depth = 0;

    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & key;
        archive & rankRange;
        archive & depth;
    }
};

// Build the tree for the current locality set and domain decomposition. Rank 0 only.
void sphParentTreeCreate();

// Release the tree. Rank 0 only; a no-op when no tree exists.
void sphParentTreeDestroy();

bool sphParentTreeExists();

// Visit every node from the root and reduce its census. Rank 0 only.
SphParentTreeCensus sphParentTreeCensus();

// This locality's leaf record, or nullptr when no tree exists.
SphParentNodeRecord const* sphParentTreeLocalLeaf();

// HPX is hidden from nvcc by cosmictiger/core/hpx.hpp; the component is host only.
#ifndef __CUDACC__

#include <cosmictiger/core/hpx.hpp>

// One node of the parent tree, an HPX component that lives on the first rank of its range.
// It owns managed ids of its children; nothing refers back to a parent, so releasing the
// root releases the tree. Actions run on HPX worker threads of the hosting locality.
class SphParentNode : public hpx::components::component_base<SphParentNode> {
    SphParentNodeRecord record_;
    std::vector<hpx::id_type> children_;

public:
    SphParentNode() = default;
    explicit SphParentNode(SphParentNodeRecord const& record);

    // Create the children on their first ranks and build them; a leaf registers itself as
    // its locality's leaf. Returns when the whole subtree exists.
    void build();
    // Release the subtree: children first, then this node's ids; a leaf unregisters.
    void release();
    SphParentTreeCensus census() const;

    HPX_DEFINE_COMPONENT_ACTION(SphParentNode, build, BuildAction);
    HPX_DEFINE_COMPONENT_ACTION(SphParentNode, release, ReleaseAction);
    HPX_DEFINE_COMPONENT_ACTION(SphParentNode, census, CensusAction);
};

HPX_REGISTER_ACTION_DECLARATION(SphParentNode::BuildAction, SphParentNodeBuildAction);
HPX_REGISTER_ACTION_DECLARATION(SphParentNode::ReleaseAction, SphParentNodeReleaseAction);
HPX_REGISTER_ACTION_DECLARATION(SphParentNode::CensusAction, SphParentNodeCensusAction);

#endif    // __CUDACC__

#endif /* SPH_PARENT_TREE_HPP_ */
