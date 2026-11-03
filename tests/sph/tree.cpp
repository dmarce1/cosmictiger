/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>

namespace detail {
constexpr int kSphTreeMinimumRung = 1;
constexpr float kSphTreeSupportMultiplier = 1.5f;

void requireSphTree(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("SPH tree regression: ") + message);
    }
}

void verifyEmptySphBounds() {
    auto const empty = emptySphBounds();
    SphTreeNeighborReturn bounds;
    SphTreeNeighborReturn populated;
    for (int dim = 0; dim < kDimensions; ++dim) {
        populated.innerBox.begin[dim] = -0.25;
        populated.innerBox.end[dim] = 1.25;
        populated.outerBox.begin[dim] = -1.5;
        populated.outerBox.end[dim] = 2.5;
        requireSphTree(empty.begin[dim] > empty.end[dim], "empty bounds are not inverted");
    }
    requireSphTree(!empty.periodicIntersects(empty) &&
            !empty.periodicIntersects(populated.outerBox) &&
            !populated.outerBox.periodicIntersects(empty),
        "empty bounds intersect a periodic box");
    bounds += populated;
    bounds += SphTreeNeighborReturn{};
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireSphTree(bounds.innerBox.begin[dim] == populated.innerBox.begin[dim] &&
                bounds.innerBox.end[dim] == populated.innerBox.end[dim] &&
                bounds.outerBox.begin[dim] == populated.outerBox.begin[dim] &&
                bounds.outerBox.end[dim] == populated.outerBox.end[dim],
            "empty bounds changed accumulated particle/support bounds");
    }
}

std::vector<char> legacySphTreeArchive(SphTreeNode const& node, int depth) {
    // The scalar metadata keeps three bools and an int; child arity is the current octree arity.
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << bool(node.converged);
        archive << node.neighborRange;
        archive << node.box;
        archive << node.sinkParticleRange;
        archive << node.innerBox;
        archive << node.outerBox;
        archive << node.children;
        archive << node.processRange;
        archive << node.particleRange;
        archive << node.numberOfActiveParticles;
        archive << bool(node.localRoot);
        archive << bool(node.leaf);
        archive << node.activeNodes;
        archive << node.nodeCount;
        archive << depth;
    }
    return bytes;
}

void verifySphTreePackedMetadata() {
    for (int depth = 0; depth <= SphTreeNode::kMaximumDepth; ++depth) {
        for (int flags = 0; flags < 8; ++flags) {
            SphTreeNode expected;
            expected.depth = depth;
            expected.localRoot = (flags & 1) != 0;
            expected.leaf = (flags & 2) != 0;
            expected.converged = (flags & 4) != 0;
            std::vector<char> bytes;
            {
                hpx::serialization::output_archive archive(bytes);
                archive << expected;
            }
            requireSphTree(bytes == legacySphTreeArchive(expected, depth),
                "packed metadata changed the scalar archive fields");
            SphTreeNode actual;
            {
                hpx::serialization::input_archive archive(bytes);
                archive >> actual;
            }
            requireSphTree(actual.depth == depth && actual.localRoot == expected.localRoot &&
                    actual.leaf == expected.leaf && actual.converged == expected.converged,
                "packed metadata did not survive the HPX archive");
        }
    }
    for (int depth : {-1, SphTreeNode::kMaximumDepth + 1, kMaxDepth - 1}) {
        bool rejected = false;
        try {
            sphTreeCreate(SphTreeCreateParameters{}, 1, Pair<int>(0, 1),
                Pair<ParticleInteger>(0, 0), unitBox<double>(), depth, false);
        } catch (std::out_of_range const&) {
            rejected = true;
        }
        requireSphTree(rejected, "unrepresentable tree depth was not rejected");
        auto bytes = legacySphTreeArchive(SphTreeNode{}, depth);
        rejected = false;
        try {
            SphTreeNode node;
            hpx::serialization::input_archive archive(bytes);
            archive >> node;
        } catch (std::out_of_range const&) {
            rejected = true;
        }
        requireSphTree(rejected, "unrepresentable archived depth was not rejected");
    }
}

void initializeSphTreeParticles(ParticleInteger count, int generation) {
    particlesResize(count);
    sphParticlesResize(count, false);
    double const offset = generation == 1 ? 0.5625 : 0.125;
    for (ParticleInteger i = 0; i < count; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = offset + 0.25 * ((i >> dim) & 1);
            particlesVel(dim, i) = 0.0f;
        }
        particlesRung(i) = (i + generation) % 3 == 0 ? 0 : 2;
        particlesType(i) = kSphType;
        particlesCatIndex(i) = i;
        sphParticlesParticleIndex(i) = i;
        sphParticlesSupportRadius(i) = float(generation + 1) / 64.0f;
        sphParticlesSetInternalEnergy(i, 1.0f);
    }
}

struct SphTreeSummary {
    size_t nodes = 0;
    size_t activeNodes = 0;
    size_t leaves = 0;
    size_t activeLeaves = 0;
};

SphTreeSummary verifySphTreeNode(TreeId id, int depth, std::set<int>& leaves) {
    auto const& node = *sphTreeGetNode(id);
    requireSphTree(id.proc == hpxRank(), "single-locality tree contains a remote node");
    requireSphTree(node.depth == depth && node.localRoot == (depth == 0),
        "node depth or local-root state was not refreshed");
    requireSphTree(node.processRange.first == 0 && node.processRange.second == 1,
        "node process range is invalid");
    requireSphTree(!node.converged, "rebuilt node retained its old convergence flag");
    requireSphTree(node.neighborRange.first == -1 && node.neighborRange.second == -1,
        "rebuilt node retained its old neighbor range");
    requireSphTree(node.particleRange.first >= 0 &&
            node.particleRange.first <= node.particleRange.second &&
            node.particleRange.second <= sphParticlesSize(),
        "node particle range lies outside the new population");
    requireSphTree(node.sinkParticleRange == node.particleRange,
        "node sink range differs from the rebuilt particle range");

    size_t activeParticles = 0;
    for (ParticleInteger i = node.particleRange.first; i < node.particleRange.second; ++i) {
        activeParticles += sphParticlesRung(i) >= kSphTreeMinimumRung;
        requireSphTree(particlesCatIndex(sphParticlesParticleIndex(i)) == i,
            "tree sorting did not refresh the gas-to-particle mapping");
    }
    requireSphTree(node.numberOfActiveParticles == activeParticles,
        "node retained an old active-particle count");
    double const quantum = RangeFixed::kToFloatFactor_;
    for (int dim = 0; dim < kDimensions; ++dim) {
        double lower = std::numeric_limits<double>::infinity();
        double upper = -std::numeric_limits<double>::infinity();
        double supportLower = lower;
        double supportUpper = upper;
        for (ParticleInteger i = node.particleRange.first; i < node.particleRange.second; ++i) {
            double const position = sphParticlesPosition(dim, i).toDouble();
            lower = std::min(lower, position);
            upper = std::max(upper, position);
            requireSphTree(node.box.begin[dim].toDouble() <= position &&
                    position <= node.box.end[dim].toDouble(),
                "node spatial box does not enclose its new particles");
            if (sphParticlesRung(i) >= kSphTreeMinimumRung) {
                double const support = kSphTreeSupportMultiplier * sphParticlesSupportRadius(i);
                supportLower = std::min(supportLower, position - support);
                supportUpper = std::max(supportUpper, position + support);
            }
        }
        if (node.numberOfParticles() != 0) {
            requireSphTree(
                std::abs(node.innerBox.begin[dim].toDouble() - (lower - 2 * quantum)) <= quantum &&
                    std::abs(node.innerBox.end[dim].toDouble() - (upper + 2 * quantum)) <= quantum,
                "inner bounds do not match the new particle extrema");
        } else {
            requireSphTree(node.innerBox.begin[dim] > node.innerBox.end[dim],
                "empty node retained populated inner bounds");
        }
        if (activeParticles != 0) {
            requireSphTree(std::abs(node.outerBox.begin[dim].toDouble() -
                               (supportLower - 2 * quantum)) <= quantum &&
                    std::abs(node.outerBox.end[dim].toDouble() - (supportUpper + 2 * quantum)) <=
                        quantum,
                "outer bounds do not match the new active particles and support radii");
        } else {
            requireSphTree(node.outerBox.begin[dim] > node.outerBox.end[dim],
                "inactive node retained active support bounds");
        }
    }

    SphTreeSummary summary{1, activeParticles != 0 ? size_t(1) : size_t(0), 0, 0};
    requireSphTree(node.leaf == (node.numberOfParticles() <= getOptions().compute.sphBucketSize),
        "leaf state does not match the rebuilt population");
    if (node.leaf) {
        requireSphTree(leaves.insert(id.index).second, "tree visits the same leaf twice");
        summary.leaves = 1;
        summary.activeLeaves = activeParticles != 0;
        for (auto const& child : node.children) {
            requireSphTree(child.index == -1 && child.proc == -1,
                "leaf retained children from an old internal node");
        }
    } else {
        ParticleInteger nextParticle = node.particleRange.first;
        int populatedChildren = 0;
        for (int childIndex = 0; childIndex < kSphChildCount; ++childIndex) {
            TreeId const childId = node.children[childIndex];
            if (childId.index < 0) {
                requireSphTree(childId.proc == -1, "empty octant has a valid process rank");
                continue;
            }
            ++populatedChildren;
            auto const& child = *sphTreeGetNode(childId);
            requireSphTree(child.particleRange.first == nextParticle &&
                    child.particleRange.second > nextParticle,
                "occupied octants do not partition their parent's particle range");
            nextParticle = child.particleRange.second;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const midpoint =
                    0.5 * (node.box.begin[dim].toDouble() + node.box.end[dim].toDouble());
                double const lower =
                    (childIndex & (1 << dim)) ? midpoint : node.box.begin[dim].toDouble();
                double const upper =
                    (childIndex & (1 << dim)) ? node.box.end[dim].toDouble() : midpoint;
                requireSphTree(std::abs(child.box.begin[dim].toDouble() - lower) <= quantum &&
                        std::abs(child.box.end[dim].toDouble() - upper) <= quantum,
                    "child geometry is not the corresponding octant");
            }
            auto const childSummary = verifySphTreeNode(childId, depth + 1, leaves);
            summary.nodes += childSummary.nodes;
            summary.activeNodes += childSummary.activeNodes;
            summary.leaves += childSummary.leaves;
            summary.activeLeaves += childSummary.activeLeaves;
        }
        requireSphTree(populatedChildren > 0 && nextParticle == node.particleRange.second,
            "internal node has missing particles or no occupied octants");
    }
    requireSphTree(node.nodeCount == summary.nodes && node.activeNodes == summary.activeNodes,
        "node subtree counts were not refreshed");
    return summary;
}

void verifySphTreeBuild(SphTreeCreateReturn const& built, ParticleInteger count) {
    auto const& root = *sphTreeGetNode(built.id);
    requireSphTree(root.particleRange.first == 0 && root.particleRange.second == count,
        "root retained the previous population range");
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireSphTree(root.box.begin[dim].toDouble() == 0.0 && root.box.end[dim].toDouble() == 1.0,
            "root spatial box was not initialized");
        requireSphTree(built.innerBox.begin[dim] == root.innerBox.begin[dim] &&
                built.innerBox.end[dim] == root.innerBox.end[dim] &&
                built.outerBox.begin[dim] == root.outerBox.begin[dim] &&
                built.outerBox.end[dim] == root.outerBox.end[dim],
            "tree return disagrees with the stored bounds");
    }
    std::set<int> leaves;
    auto const summary = verifySphTreeNode(built.id, 0, leaves);
    requireSphTree(built.nodeCount == summary.nodes && built.activeNodes == summary.activeNodes &&
            built.leafNodes == summary.leaves && built.activeLeafNodes == summary.activeLeaves &&
            built.numberOfActiveParticles == root.numberOfActiveParticles,
        "tree return contains stale population or subtree counts");
    requireSphTree(size_t(sphTreeLeaflistSize()) == leaves.size(),
        "leaf list retained entries from the old tree");
    for (int i = 0; i < sphTreeLeaflistSize(); ++i) {
        auto const leaf = sphTreeGetLeaf(i);
        requireSphTree(leaf.proc == hpxRank() && leaves.erase(leaf.index) == 1,
            "leaf list contains a duplicate or obsolete entry");
    }
}

void seedOldSphTreeState(TreeId id) {
    auto const node = *sphTreeGetNode(id);
    sphTreeSetConverged(id);
    sphTreeSetNumberOfActiveParticles(id, 1000);
    sphTreeSetNeighborRange(id, Pair<int>(123, 456));
    Fixed32Range staleBox;
    for (int dim = 0; dim < kDimensions; ++dim) {
        staleBox.begin[dim] = 1.5;
        staleBox.end[dim] = 1.75;
    }
    sphTreeSetBoxes(id, staleBox, staleBox);
    if (!node.leaf) {
        for (auto const& child : node.children) {
            if (child.index >= 0) {
                seedOldSphTreeState(child);
            }
        }
    }
}
void verifyOctreeNeighbors(TreeId root) {
    SphTreeNeighborParameters params;
    params.runType = kSphTreeNeighborNeighbors;
    params.innerSelection = kSphInteractionsIJ;
    Vector<TreeId> checklist;
    checklist.push_back(root);
    sphTreeNeighbor(params, root, checklist).get();
    for (int targetIndex = 0; targetIndex < sphTreeLeaflistSize(); ++targetIndex) {
        auto const& target = *sphTreeGetNode(sphTreeGetLeaf(targetIndex));
        std::set<TreeId> expected;
        for (int sourceIndex = 0; sourceIndex < sphTreeLeaflistSize(); ++sourceIndex) {
            auto const sourceId = sphTreeGetLeaf(sourceIndex);
            auto const& source = *sphTreeGetNode(sourceId);
            if (target.outerBox.periodicIntersects(source.innerBox) ||
                target.innerBox.periodicIntersects(source.outerBox)) {
                expected.insert(sourceId);
            }
        }
        std::set<TreeId> actual;
        for (int index = target.neighborRange.first; index < target.neighborRange.second; ++index) {
            requireSphTree(actual.insert(sphTreeGetNeighbor(index)).second,
                "neighbor walk visited an octant twice");
        }
        requireSphTree(
            actual == expected, "neighbor walk disagrees with exhaustive leaf intersections");
    }
}

void verifyRefreshedOctreeBoxes(TreeId root) {
    for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
        sphParticlesSupportRadius(i) = 0.2f + float(i % 3) * 0.05f;
    }
    SphTreeNeighborParameters params;
    params.runType = kSphTreeNeighborBoxes;
    params.innerSelection = kSphSetAll;
    params.outerSelection = kSphSetActive;
    params.minimumRung = kSphTreeMinimumRung;
    params.supportRadiusMultiplier = kSphTreeSupportMultiplier;
    sphTreeNeighbor(params, root, {}).get();
    SphTreeNeighborReturn combined;
    for (int index = 0; index < sphTreeLeaflistSize(); ++index) {
        auto const& leaf = *sphTreeGetNode(sphTreeGetLeaf(index));
        SphTreeNeighborReturn bounds;
        bounds.innerBox = leaf.innerBox;
        bounds.outerBox = leaf.outerBox;
        combined += bounds;
    }
    auto const& node = *sphTreeGetNode(root);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireSphTree(node.innerBox.begin[dim] == combined.innerBox.begin[dim] &&
                node.innerBox.end[dim] == combined.innerBox.end[dim] &&
                node.outerBox.begin[dim] == combined.outerBox.begin[dim] &&
                node.outerBox.end[dim] == combined.outerBox.end[dim],
            "box refresh omitted an octant");
    }
    verifyOctreeNeighbors(root);
}

}    // namespace detail

void sphTreeTest() {
    detail::verifySphTreePackedMetadata();
    detail::verifyEmptySphBounds();
    // This fixture exercises the real local allocator and rebuild path. Remote
    // cache invalidation and growth beyond the arena minimum need separate tests.
    detail::requireSphTree(hpxSize() == 1, "arena regression requires one locality");
    auto options = getOptions();
    options.hydro.enabled = true;
    options.chemistry.enabled = false;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.compute.bucketSize = 8;
    options.compute.sphBucketSize = 2;
    options.initial.numberOfParticles = 8;
    setOptions(options);
    sphTreeDestroy(true);
    SphTreeCreateParameters params{};
    params.minimumRung = detail::kSphTreeMinimumRung;
    params.supportRadiusMultiplier = detail::kSphTreeSupportMultiplier;

    detail::initializeSphTreeParticles(8, 0);
    auto const first = sphTreeCreate(params);
    detail::verifySphTreeBuild(first, 8);
    auto const arenaSize = sphTreeNodesSize();
    auto const* firstRoot = sphTreeGetNode(first.id);
    detail::requireSphTree(
        arenaSize > 0 && first.id.index == 0, "initial tree did not begin at the arena origin");
    detail::requireSphTree(
        !firstRoot->leaf, "SPH tree used the larger gravity bucket instead of its own bucket");
    detail::seedOldSphTreeState(first.id);

    sphTreeDestroy();
    detail::requireSphTree(sphTreeNodesSize() == arenaSize && sphTreeGetNode(first.id) == firstRoot,
        "normal teardown released the constructed arena");
    options.compute.bucketSize = 1;
    options.compute.sphBucketSize = 4;
    setOptions(options);
    detail::initializeSphTreeParticles(3, 1);
    auto const second = sphTreeCreate(params);
    detail::requireSphTree(sphTreeNodesSize() == arenaSize && second.id == first.id &&
            sphTreeGetNode(second.id) == firstRoot,
        "rebuild moved the retained arena or failed to reset its allocation cursor");
    detail::verifySphTreeBuild(second, 3);
    detail::requireSphTree(sphTreeGetNode(second.id)->leaf,
        "rebuilt SPH tree ignored its increased bucket or used the gravity bucket");

    detail::seedOldSphTreeState(second.id);
    sphTreeDestroy();
    options.compute.bucketSize = 8;
    options.compute.sphBucketSize = 1;
    setOptions(options);
    detail::initializeSphTreeParticles(8, 2);
    auto const regrown = sphTreeCreate(params);
    detail::requireSphTree(sphTreeNodesSize() == arenaSize && regrown.id == first.id &&
            sphTreeGetNode(regrown.id) == firstRoot,
        "population regrowth moved the retained arena or failed to reset its allocation cursor");
    detail::verifySphTreeBuild(regrown, 8);
    detail::requireSphTree(regrown.leafNodes >= 8, "rebuilt SPH tree ignored its decreased bucket");

    sphTreeDestroy(true);
    detail::requireSphTree(sphTreeNodesSize() == 0, "explicit teardown did not release the arena");
    detail::initializeSphTreeParticles(5, 3);
    auto const afterRelease = sphTreeCreate(params);
    detail::requireSphTree(sphTreeNodesSize() == arenaSize && afterRelease.id.index == 0,
        "tree could not allocate a fresh arena after explicit release");
    detail::verifySphTreeBuild(afterRelease, 5);
    sphTreeDestroy();
    detail::initializeSphTreeParticles(64, 0);
    for (ParticleInteger i = 0; i < 64; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = 0.125 + 0.25 * ((i >> (2 * dim)) & 3);
        }
    }
    auto const fullOctree = sphTreeCreate(params);
    detail::verifySphTreeBuild(fullOctree, 64);
    detail::requireSphTree(fullOctree.nodeCount == 73 && fullOctree.leafNodes == 64 &&
            fullOctree.minDepth == 2 && fullOctree.maxDepth == 2,
        "uniform 4^3 fixture did not produce two levels of eight-way spatial subdivision");
    detail::verifyOctreeNeighbors(fullOctree.id);
    detail::verifyRefreshedOctreeBoxes(fullOctree.id);

    sphTreeDestroy();
    detail::initializeSphTreeParticles(8, 0);
    for (ParticleInteger i = 0; i < 8; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = ((i >> dim) & 1) ? 0.5 : 0.0;
        }
    }
    auto const boundaryOctree = sphTreeCreate(params);
    detail::verifySphTreeBuild(boundaryOctree, 8);
    detail::requireSphTree(
        boundaryOctree.nodeCount == 9, "split-plane particles did not select the upper octants");
    detail::verifyOctreeNeighbors(boundaryOctree.id);

    sphTreeDestroy();
    detail::initializeSphTreeParticles(2, 0);
    for (int dim = 0; dim < kDimensions; ++dim) {
        particlesPos(dim, 0) = particlesPos(dim, 1) = 0.0;
    }
    particlesPos(0, 1) = std::ldexp(1.0, -SphTreeNode::kMaximumDepth);
    auto const deepestAllowed = sphTreeCreate(params);
    detail::verifySphTreeBuild(deepestAllowed, 2);
    detail::requireSphTree(deepestAllowed.maxDepth == SphTreeNode::kMaximumDepth,
        "valid depth-15 leaves were rejected or stopped early");
    detail::verifyOctreeNeighbors(deepestAllowed.id);

    sphTreeDestroy();
    detail::initializeSphTreeParticles(2, 0);
    for (int dim = 0; dim < kDimensions; ++dim) {
        particlesPos(dim, 0) = particlesPos(dim, 1) = 0.25;
    }
    bool rejectedDepth16 = false;
    try {
        sphTreeCreate(params);
    } catch (std::out_of_range const& error) {
        rejectedDepth16 = std::string(error.what()).find("depth 16") != std::string::npos;
    }
    detail::requireSphTree(
        rejectedDepth16, "coincident particles needing depth 16 did not throw an exception");

    // Exercise an asynchronous failing octant beside a successful sibling.
    sphTreeDestroy();
    ParticleInteger const parallelCount = kMinSortThreadParts + 2;
    detail::initializeSphTreeParticles(parallelCount, 0);
    for (ParticleInteger i = 0; i < parallelCount; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = i == 0 ? 0.125 : 0.875;
        }
    }
    rejectedDepth16 = false;
    try {
        sphTreeCreate(params);
    } catch (std::out_of_range const& error) {
        rejectedDepth16 = std::string(error.what()).find("depth 16") != std::string::npos;
    }
    detail::requireSphTree(
        rejectedDepth16, "asynchronous overfull octant did not propagate the depth-16 exception");

    // Failure must leave no outstanding child tasks or stale allocator state.
    sphTreeDestroy();
    detail::initializeSphTreeParticles(8, 0);
    auto const recovered = sphTreeCreate(params);
    detail::verifySphTreeBuild(recovered, 8);
    detail::verifyOctreeNeighbors(recovered.id);
    sphTreeDestroy();
    detail::initializeSphTreeParticles(0, 0);
    auto const empty = sphTreeCreate(params);
    detail::verifySphTreeBuild(empty, 0);
    sphTreeDestroy(true);
    std::cout << "SPH tree arena regression passed (CPU, one locality)\n" << std::flush;
}
