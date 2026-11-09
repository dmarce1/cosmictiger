/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cstdint>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/sph/parent_tree.hpp>

namespace detail {

void requireParentTree(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("SPH parent tree regression: " + message);
    }
}

// Depth of a key in the binary domain tree: the root key 1 has depth 0.
int binaryDepth(std::uint64_t key) {
    int depth = 0;
    while (key > 1) {
        key >>= 1;
        depth++;
    }
    return depth;
}

struct TopologyWalk {
    std::vector<int> leafRanks;
    std::vector<std::uint64_t> leafKeys;
    int maximumDepth = 0;
};

// Check one node of the pure topology and recurse.
void walkTopology(
    std::uint64_t key, Pair<int> rankRange, int depth, int rankCount, TopologyWalk& walk) {
    int const count = rankRange.second - rankRange.first;
    requireParentTree(count >= 1, "a node covers no rank");
    walk.maximumDepth = std::max(walk.maximumDepth, depth);
    auto const children = sphParentChildren(key, rankRange);
    requireParentTree(int(children.size()) == sphParentArity(count),
        std::format("node of {} ranks has {} children", count, children.size()));
    if (children.empty()) {
        walk.leafRanks.push_back(rankRange.first);
        walk.leafKeys.push_back(key);
        return;
    }
    int const levels = children.size() == 2 ? 1 : kSpatialDimensions;
    int expectedFirst = rankRange.first;
    for (auto const& child : children) {
        requireParentTree(child.rankRange.first == expectedFirst &&
                child.rankRange.second > child.rankRange.first,
            "children do not partition the parent's ranks in order");
        requireParentTree(
            binaryDepth(child.key) == binaryDepth(key) + levels && (child.key >> levels) == key,
            "child key is not a descendant of the parent key at the regrouped depth");
        expectedFirst = child.rankRange.second;
        walkTopology(child.key, child.rankRange, depth + 1, rankCount, walk);
    }
    requireParentTree(expectedFirst == rankRange.second, "children leave ranks uncovered");
    // Placement rule: the parent lives on its first rank, which only its first child shares.
    requireParentTree(children.front().rankRange.first == rankRange.first,
        "the first child does not share the parent's locality");
}

// Topology for every rank count up to 130, against the binary domain decomposition.
void checkParentTopology() {
    for (int rankCount = 1; rankCount <= 130; rankCount++) {
        TopologyWalk walk;
        walkTopology(1, Pair<int>(0, rankCount), 0, rankCount, walk);
        requireParentTree(int(walk.leafRanks.size()) == rankCount,
            std::format("{} ranks gave {} leaves", rankCount, walk.leafRanks.size()));
        for (int rank = 0; rank < rankCount; rank++) {
            requireParentTree(walk.leafRanks[rank] == rank, "leaves are not in rank order");
            requireParentTree(walk.leafKeys[rank] == sphDomainLeafKey(rank, rankCount),
                std::format("leaf of rank {} of {} has key {} instead of the domain key {}", rank,
                    rankCount, walk.leafKeys[rank], sphDomainLeafKey(rank, rankCount)));
        }
        // One full level per factor 2^D, then at most D - 1 binary levels.
        int const fullArity = 1 << kSpatialDimensions;
        int fullLevels = 0;
        int remaining = rankCount;
        while (remaining >= fullArity) {
            remaining = (remaining + fullArity - 1) / fullArity;
            fullLevels++;
        }
        requireParentTree(walk.maximumDepth <= fullLevels + kSpatialDimensions,
            std::format("{} ranks reach depth {}", rankCount, walk.maximumDepth));
    }
    if constexpr (kSpatialDimensions == 3) {
        // 64 ranks are two full 8-way levels; 8 ranks one; 7 ranks three binary levels.
        TopologyWalk eight;
        walkTopology(1, Pair<int>(0, 8), 0, 8, eight);
        requireParentTree(eight.maximumDepth == 1, "8 ranks are not one 8-way level");
        TopologyWalk sixtyFour;
        walkTopology(1, Pair<int>(0, 64), 0, 64, sixtyFour);
        requireParentTree(sixtyFour.maximumDepth == 2, "64 ranks are not two 8-way levels");
        TopologyWalk seven;
        walkTopology(1, Pair<int>(0, 7), 0, 7, seven);
        requireParentTree(seven.maximumDepth == 3, "7 ranks are not three binary levels");
    }
}

// Build the distributed tree for this run, check it, release it and build it again.
void checkDistributedParentTree() {
    TopologyWalk expected;
    walkTopology(1, Pair<int>(0, hpxSize()), 0, hpxSize(), expected);
    for (int round = 0; round < 2; round++) {
        sphParentTreeCreate();
        requireParentTree(sphParentTreeExists(), "tree does not exist after create");
        auto const census = sphParentTreeCensus();
        std::int64_t const rankCount = hpxSize();
        requireParentTree(census.consistent, "a node or leaf is misplaced or unregistered");
        requireParentTree(census.leafCount == hpxSize(), "leaf count differs from the ranks");
        requireParentTree(census.rankSum == rankCount * (rankCount - 1) / 2,
            "leaves do not visit every rank exactly once");
        requireParentTree(census.maximumDepth == expected.maximumDepth,
            "distributed depth differs from the topology");
        requireParentTree(sphParentTreeLocalLeaf() != nullptr &&
                sphParentTreeLocalLeaf()->key == sphDomainLeafKey(0, hpxSize()),
            "rank 0 does not hold its own leaf");
        std::cout << std::format("SPH parent tree round {}: ranks={} nodes={} depth={}\n", round,
                         hpxSize(), census.nodeCount, census.maximumDepth)
                  << std::flush;
        sphParentTreeDestroy();
        requireParentTree(
            !sphParentTreeExists() && sphParentTreeLocalLeaf() == nullptr, "tree survived destroy");
    }
}

}    // namespace detail

void sphParentTreeTest() {
    detail::checkParentTopology();
    detail::checkDistributedParentTree();
    std::cout << "SPH parent tree test passed\n" << std::flush;
}
