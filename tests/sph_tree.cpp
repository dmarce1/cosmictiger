/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

namespace detail {
constexpr int kSphTreeMinimumRung = 1;
constexpr float kSphTreeSupportMultiplier = 1.5f;

void requireSphTree(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("SPH tree regression: ") + message);
    }
}

void initializeSphTreeParticles(ParticleInteger count, int generation) {
    particlesResize(count);
    sphParticlesResize(count, false);
    double const offset = generation == 1 ? 0.5625 : 0.125;
    for (ParticleInteger i = 0; i < count; ++i) {
        for (int dim = 0; dim < NDIM; ++dim) {
            particlesPos(dim, i) = offset + 0.25 * ((i >> dim) & 1);
            particlesVel(dim, i) = 0.0f;
        }
        particlesRung(i) = (i + generation) % 3 == 0 ? 0 : 2;
        particlesType(i) = SPH_TYPE;
        particlesCatIndex(i) = i;
        sphParticlesParticleIndex(i) = i;
        sphParticlesSupportRadius(i) = float(generation + 1) / 64.0f;
        sphParticlesInternalEnergy(i) = 1.0f;
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
    for (int dim = 0; dim < NDIM; ++dim) {
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
        auto const& left = *sphTreeGetNode(node.children[LEFT]);
        auto const& right = *sphTreeGetNode(node.children[RIGHT]);
        requireSphTree(left.particleRange.first == node.particleRange.first &&
                left.particleRange.second == right.particleRange.first &&
                right.particleRange.second == node.particleRange.second,
            "child particle ranges do not partition their parent");
        for (auto const& child : node.children) {
            auto const childSummary = verifySphTreeNode(child, depth + 1, leaves);
            summary.nodes += childSummary.nodes;
            summary.activeNodes += childSummary.activeNodes;
            summary.leaves += childSummary.leaves;
            summary.activeLeaves += childSummary.activeLeaves;
        }
    }
    requireSphTree(node.nodeCount == summary.nodes && node.activeNodes == summary.activeNodes,
        "node subtree counts were not refreshed");
    return summary;
}

void verifySphTreeBuild(SphTreeCreateReturn const& built, ParticleInteger count) {
    auto const& root = *sphTreeGetNode(built.id);
    requireSphTree(root.particleRange.first == 0 && root.particleRange.second == count,
        "root retained the previous population range");
    for (int dim = 0; dim < NDIM; ++dim) {
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
    for (int dim = 0; dim < NDIM; ++dim) {
        staleBox.begin[dim] = 1.5;
        staleBox.end[dim] = 1.75;
    }
    sphTreeSetBoxes(id, staleBox, staleBox);
    if (!node.leaf) {
        for (auto const& child : node.children) {
            seedOldSphTreeState(child);
        }
    }
}
}    // namespace detail

void sphTreeTest() {
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
    sphTreeDestroy(true);
    PRINT("SPH tree arena regression passed (CPU, one locality)\n");
}
