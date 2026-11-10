/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/sph/parent_tree.hpp>

// Registration, exactly once in the program and at global scope. The component lives in
// the cosmictiger_objects library, so it registers through static initializers like the
// plain actions; HPX_REGISTER_COMPONENT_MODULE must not be used outside src/main.cpp.
using SphParentNodeComponent = hpx::components::component<SphParentNode>;
HPX_REGISTER_COMPONENT(SphParentNodeComponent, SphParentNode);

HPX_REGISTER_ACTION(SphParentNode::BuildAction, SphParentNodeBuildAction);
HPX_REGISTER_ACTION(SphParentNode::ReleaseAction, SphParentNodeReleaseAction);
HPX_REGISTER_ACTION(SphParentNode::CensusAction, SphParentNodeCensusAction);

namespace detail {

// The root's managed id, held on rank 0 only. Interior nodes hold their children's ids.
static hpx::id_type parentTreeRoot;

// This locality's leaf, registered by the leaf's build() and cleared by its release().
static MutexType localLeafMutex;
static std::optional<SphParentNodeRecord> localLeaf;

// Join every future, rethrowing the first failure after all have finished.
template <class T>
std::vector<T> joinParentTreeFutures(std::vector<hpx::future<T>>& futures) {
    std::vector<T> results;
    results.reserve(futures.size());
    std::exception_ptr failure;
    for (auto& future : futures) {
        try {
            results.push_back(future.get());
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return results;
}

/// Join all child operations and rethrow the first failure only after every future has been
/// consumed.
void joinParentTreeFutures(std::vector<hpx::future<void>>& futures) {
    std::exception_ptr failure;
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}    // namespace detail

/// Partition a rank interval into dimension-aware parent-tree children with nonempty rank
/// groups.
std::vector<SphParentChild> sphParentChildren(std::uint64_t key, Pair<int> rankRange) {
    std::vector<SphParentChild> children;
    int const arity = sphParentArity(rankRange.second - rankRange.first);
    if (arity == 0) {
        return children;
    }
    // One binary level for a binary node, kSpatialDimensions levels for a full node. Every
    // group still holds at least one rank, because a full node covers at least 2^D ranks.
    int const levels = arity == 2 ? 1 : kSpatialDimensions;
    children.push_back(SphParentChild{key, rankRange});
    for (int level = 0; level < levels; level++) {
        std::vector<SphParentChild> next;
        next.reserve(2 * children.size());
        for (auto const& child : children) {
            int const middleRank = (child.rankRange.first + child.rankRange.second) / 2;
            next.push_back(
                SphParentChild{child.key << 1, Pair<int>(child.rankRange.first, middleRank)});
            next.push_back(SphParentChild{
                (child.key << 1) + 1, Pair<int>(middleRank, child.rankRange.second)});
        }
        children = std::move(next);
    }
    return children;
}

/// Compute the binary domain-tree key assigned to a rank.
std::uint64_t sphDomainLeafKey(int rank, int rankCount) {
    std::uint64_t key = 1;
    Pair<int> rankRange(0, rankCount);
    while (rankRange.second - rankRange.first > 1) {
        int const middleRank = (rankRange.first + rankRange.second) / 2;
        if (rank < middleRank) {
            rankRange.second = middleRank;
            key <<= 1;
        } else {
            rankRange.first = middleRank;
            key = (key << 1) + 1;
        }
    }
    return key;
}

namespace detail {

// Collect the remote children of every node that lives on `rank`, below `key`/`rankRange`.
void collectLocalityChildren(
    std::uint64_t key, Pair<int> rankRange, int rank, std::vector<int>& children) {
    for (auto const& child : sphParentChildren(key, rankRange)) {
        if (rank < child.rankRange.first || rank >= child.rankRange.second) {
            // A sibling subtree: forwarded to if this node lives on `rank`.
            if (rankRange.first == rank) {
                children.push_back(child.rankRange.first);
            }
            continue;
        }
        collectLocalityChildren(child.key, child.rankRange, rank, children);
    }
}

}    // namespace detail

/// Return the locality children induced by the dimension-aware parent-tree topology.
std::vector<int> sphParentLocalityChildren(int rank, int rankCount) {
    std::vector<int> children;
    detail::collectLocalityChildren(1, Pair<int>(0, rankCount), rank, children);
    return children;
}

/// Return cached HPX child-locality IDs used by SPH collective operations.
Vector<hpx::id_type> const& sphCollectiveChildren() {
    // The topology depends only on the rank count, fixed for a run.
    static Vector<hpx::id_type> const children = [] {
        Vector<hpx::id_type> result;
        for (int const child : sphParentLocalityChildren(hpxRank(), hpxSize())) {
            result.push_back(hpxLocalities()[child]);
        }
        return result;
    }();
    return children;
}

/// Merge tree census counts, maximum depth, rank sum, and consistency flags.
SphParentTreeCensus& SphParentTreeCensus::operator+=(SphParentTreeCensus const& other) {
    leafCount += other.leafCount;
    nodeCount += other.nodeCount;
    maximumDepth = std::max(maximumDepth, other.maximumDepth);
    rankSum += other.rankSum;
    consistent = consistent && other.consistent;
    return *this;
}

/// Retain the node's rank interval, key, and depth for distributed parent-tree
/// construction.
SphParentNode::SphParentNode(SphParentNodeRecord const& record)
  : record_(record) {}

/// Create child components recursively or register this locality's unique parent-tree leaf.
void SphParentNode::build() {
    auto const children = sphParentChildren(record_.key, record_.rankRange);
    if (children.empty()) {
        std::lock_guard<MutexType> lock(detail::localLeafMutex);
        if (detail::localLeaf) {
            throw std::logic_error("SPH parent tree: a locality received a second leaf");
        }
        detail::localLeaf = record_;
        return;
    }
    std::vector<hpx::future<hpx::id_type>> created;
    created.reserve(children.size());
    for (auto const& child : children) {
        SphParentNodeRecord childRecord;
        childRecord.key = child.key;
        childRecord.rankRange = child.rankRange;
        childRecord.depth = record_.depth + 1;
        created.push_back(
            hpx::new_<SphParentNode>(hpxLocalities()[child.rankRange.first], childRecord));
    }
    children_ = detail::joinParentTreeFutures(created);
    std::vector<hpx::future<void>> built;
    built.reserve(children_.size());
    for (auto const& child : children_) {
        built.push_back(hpx::async<SphParentNode::BuildAction>(child));
    }
    detail::joinParentTreeFutures(built);
}

/// Release descendants and clear the local leaf registration before dropping managed child
/// IDs.
void SphParentNode::release() {
    std::vector<hpx::future<void>> released;
    released.reserve(children_.size());
    for (auto const& child : children_) {
        released.push_back(hpx::async<SphParentNode::ReleaseAction>(child));
    }
    detail::joinParentTreeFutures(released);
    // Dropping the managed ids lets AGAS destroy the children on their localities.
    children_.clear();
    if (record_.rankRange.second - record_.rankRange.first == 1) {
        std::lock_guard<MutexType> lock(detail::localLeafMutex);
        detail::localLeaf.reset();
    }
}

/// Recursively count and validate component ownership, topology, and leaf registration.
SphParentTreeCensus SphParentNode::census() const {
    SphParentTreeCensus result;
    result.nodeCount = 1;
    result.maximumDepth = record_.depth;
    result.consistent = hpxRank() == record_.rankRange.first;
    auto const children = sphParentChildren(record_.key, record_.rankRange);
    if (children.empty()) {
        result.leafCount = 1;
        result.rankSum = hpxRank();
        std::lock_guard<MutexType> lock(detail::localLeafMutex);
        result.consistent = result.consistent &&
            record_.key == sphDomainLeafKey(hpxRank(), hpxSize()) && detail::localLeaf &&
            detail::localLeaf->key == record_.key;
        return result;
    }
    result.consistent = result.consistent && children.size() == children_.size();
    std::vector<hpx::future<SphParentTreeCensus>> censuses;
    censuses.reserve(children_.size());
    for (size_t index = 0; index < children_.size(); index++) {
        // Each child must live on the first rank of its own range.
        result.consistent = result.consistent && index < children.size() &&
            int(hpx::naming::get_locality_id_from_id(children_[index])) ==
                children[index].rankRange.first;
        censuses.push_back(hpx::async<SphParentNode::CensusAction>(children_[index]));
    }
    for (auto const& child : detail::joinParentTreeFutures(censuses)) {
        result += child;
    }
    return result;
}

/// Replace the distributed parent tree from rank zero using the current locality count.
void sphParentTreeCreate() {
    if (hpxRank() != 0) {
        throw std::logic_error("SPH parent tree: create runs on rank 0");
    }
    sphParentTreeDestroy();
    SphParentNodeRecord rootRecord;
    rootRecord.key = 1;
    rootRecord.rankRange = Pair<int>(0, hpxSize());
    rootRecord.depth = 0;
    detail::parentTreeRoot = hpx::new_<SphParentNode>(hpxLocalities()[0], rootRecord).get();
    hpx::async<SphParentNode::BuildAction>(detail::parentTreeRoot).get();
}

/// Release the distributed parent tree if one exists.
void sphParentTreeDestroy() {
    if (!detail::parentTreeRoot) {
        return;
    }
    hpx::async<SphParentNode::ReleaseAction>(detail::parentTreeRoot).get();
    detail::parentTreeRoot = hpx::invalid_id;
}

/// Report whether this process holds the parent-tree root ID.
bool sphParentTreeExists() {
    return bool(detail::parentTreeRoot);
}

/// Return a distributed topology census, requiring an existing root.
SphParentTreeCensus sphParentTreeCensus() {
    if (!detail::parentTreeRoot) {
        throw std::logic_error("SPH parent tree: census without a tree");
    }
    return hpx::async<SphParentNode::CensusAction>(detail::parentTreeRoot).get();
}

/// Return the registered local leaf record, or null when no leaf is registered.
SphParentNodeRecord const* sphParentTreeLocalLeaf() {
    std::lock_guard<MutexType> lock(detail::localLeafMutex);
    return detail::localLeaf ? &*detail::localLeaf : nullptr;
}
