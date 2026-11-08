/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
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
#include <cosmictiger/core/sph_midpoint.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
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
        archive << node.driftBox;
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
            for (int dim = 0; dim < kDimensions; ++dim) {
                expected.driftBox.begin[dim] = -0.125 * (dim + 1);
                expected.driftBox.end[dim] = 1.125 + 0.125 * dim;
            }
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
            for (int dim = 0; dim < kDimensions; ++dim) {
                requireSphTree(actual.driftBox.begin[dim] == expected.driftBox.begin[dim] &&
                        actual.driftBox.end[dim] == expected.driftBox.end[dim],
                    "drift bounds did not survive the remote-node archive");
            }
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
        sphParticlesSmoothingLength(i) = float(generation + 1) / 64.0f;
        sphParticlesSetInternalEnergy(i, 1.0f);
    }
}

int expectedSphSplitMask(SphTreeNode const& node) {
    auto const count = node.numberOfParticles();
    auto const bucket = getOptions().compute.sphBucketSize;
    int const dimensions = count > 4 * bucket ? 3 : count > 2 * bucket ? 2 : 1;
    std::array<int, kDimensions> axes{0, 1, 2};
    std::stable_sort(axes.begin(), axes.end(), [&](int first, int second) {
        return node.box.end[first].toDouble() - node.box.begin[first].toDouble() >
            node.box.end[second].toDouble() - node.box.begin[second].toDouble();
    });
    int mask = 0;
    for (int axis = 0; axis < dimensions; ++axis) {
        mask |= 1 << axes[axis];
    }
    return mask;
}

void verifySphTreeVolumeRatios(SphTreeCreateReturn const& tree) {
    std::vector<double> ratios;
    std::vector<TreeId> pending{tree.id};
    while (!pending.empty()) {
        auto const id = pending.back();
        pending.pop_back();
        auto const& node = *sphTreeGetNode(id);
        if (node.isLocal() && node.numberOfParticles() > 0) {
            // Independent reference: multiply dimensionless side-length ratios.
            double ratio = 1.0;
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                double const originalSpan =
                    node.box.end[dimension].toDouble() - node.box.begin[dimension].toDouble();
                double const evolvedSpan = node.driftBox.end[dimension].toDouble() -
                    node.driftBox.begin[dimension].toDouble();
                requireSphTree(originalSpan > 0.0 && evolvedSpan > 0.0,
                    "occupied spatial node has invalid volume bounds");
                ratio *= evolvedSpan / originalSpan;
            }
            ratios.push_back(ratio);
        }
        for (auto const child : node.children) {
            if (child.index >= 0) {
                pending.push_back(child);
            }
        }
    }
    double expectedMinimum = 0.0;
    double expectedMaximum = 0.0;
    if (!ratios.empty()) {
        auto const extrema = std::minmax_element(ratios.begin(), ratios.end());
        expectedMinimum = *extrema.first;
        expectedMaximum = *extrema.second;
    }
    requireSphTree(std::abs(tree.minVolumeRatio - expectedMinimum) <= 1.0e-12 * expectedMinimum &&
            std::abs(tree.maxVolumeRatio - expectedMaximum) <= 1.0e-12 * expectedMaximum,
        "tree volume extrema disagree with occupied spatial nodes");
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
                double const support = kSphTreeSupportMultiplier * sphParticlesSmoothingLength(i);
                supportLower = std::min(supportLower, position - support);
                supportUpper = std::max(supportUpper, position + support);
            }
        }
        if (node.numberOfParticles() != 0) {
            requireSphTree(std::abs(node.innerBox.begin[dim].toDouble() -
                               (lower - kSphBoxPadding)) <= quantum &&
                    std::abs(node.innerBox.end[dim].toDouble() - (upper + kSphBoxPadding)) <=
                        quantum,
                "inner bounds do not match the new particle extrema");
        } else {
            requireSphTree(node.innerBox.begin[dim] > node.innerBox.end[dim],
                "empty node retained populated inner bounds");
        }
        if (activeParticles != 0) {
            requireSphTree(std::abs(node.outerBox.begin[dim].toDouble() -
                               (supportLower - kSphBoxPadding)) <= quantum &&
                    std::abs(node.outerBox.end[dim].toDouble() - (supportUpper + kSphBoxPadding)) <=
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
        int const splitMask = expectedSphSplitMask(node);
        for (int childIndex = 0; childIndex < kSphChildCount; ++childIndex) {
            TreeId const childId = node.children[childIndex];
            if (childId.index < 0) {
                requireSphTree(childId.proc == -1, "empty octant has a valid process rank");
                continue;
            }
            requireSphTree(
                (childIndex & ~splitMask) == 0, "child occupies a slot on an unsplit axis");
            ++populatedChildren;
            auto const& child = *sphTreeGetNode(childId);
            requireSphTree(child.particleRange.first == nextParticle &&
                    child.particleRange.second > nextParticle,
                "occupied octants do not partition their parent's particle range");
            nextParticle = child.particleRange.second;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const midpoint =
                    0.5 * (node.box.begin[dim].toDouble() + node.box.end[dim].toDouble());
                bool const split = (splitMask & (1 << dim)) != 0;
                double const lower = split && (childIndex & (1 << dim)) != 0 ?
                    midpoint :
                    node.box.begin[dim].toDouble();
                double const upper = split && (childIndex & (1 << dim)) == 0 ?
                    midpoint :
                    node.box.end[dim].toDouble();
                requireSphTree(std::abs(child.box.begin[dim].toDouble() - lower) <= quantum &&
                        std::abs(child.box.end[dim].toDouble() - upper) <= quantum,
                    "child geometry does not match the selected longest-axis split");
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
    verifySphTreeVolumeRatios(built);
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
    for (auto const direction : {SphInteractionDirection::TargetSupport,
             SphInteractionDirection::SourceSupport, SphInteractionDirection::EitherSupport}) {
        sphTreeFreeNeighborList();
        SphTreeNeighborParameters params;
        params.operation = SphNeighborOperation::FindNeighbors;
        params.interactionDirection = direction;
        Vector<TreeId> checklist;
        checklist.push_back(root);
        sphTreeNeighbor(params, root, checklist).get();
        for (int targetIndex = 0; targetIndex < sphTreeLeaflistSize(); ++targetIndex) {
            auto const& target = *sphTreeGetNode(sphTreeGetLeaf(targetIndex));
            std::set<TreeId> expected;
            for (int sourceIndex = 0; sourceIndex < sphTreeLeaflistSize(); ++sourceIndex) {
                auto const sourceId = sphTreeGetLeaf(sourceIndex);
                auto const& source = *sphTreeGetNode(sourceId);
                bool const targetSupport = direction != SphInteractionDirection::SourceSupport &&
                    target.outerBox.periodicIntersects(source.innerBox);
                bool const sourceSupport = direction != SphInteractionDirection::TargetSupport &&
                    target.innerBox.periodicIntersects(source.outerBox);
                if (targetSupport || sourceSupport) {
                    expected.insert(sourceId);
                }
            }
            std::set<TreeId> actual;
            for (int index = target.neighborRange.first; index < target.neighborRange.second;
                ++index) {
                requireSphTree(actual.insert(sphTreeGetNeighbor(index)).second,
                    "neighbor walk visited a child twice");
            }
            requireSphTree(actual == expected,
                "directional neighbor walk disagrees with exhaustive leaf intersections");
        }
    }
}

void verifyHybridSphTree(SphTreeCreateParameters const& params) {
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.compute.sphBucketSize = 2;
    setOptions(options);
    for (ParticleInteger count : {0, 2, 3, 4, 5, 8, 9}) {
        sphTreeDestroy();
        initializeSphTreeParticles(count, 0);
        for (ParticleInteger particle = 0; particle < count; ++particle) {
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                particlesPos(dimension, particle) = 0.25 + 0.5 * ((particle >> dimension) & 1);
            }
        }
        auto const built = sphTreeCreate(params);
        verifySphTreeBuild(built, count);
        auto const& root = *sphTreeGetNode(built.id);
        int occupied = 0;
        for (auto const child : root.children) {
            occupied += child.index >= 0;
        }
        int const expected = count > 8 ? 8 : count > 4 ? 4 : count > 2 ? 2 : 0;
        requireSphTree(occupied == expected, "hybrid fanout changed at the wrong bucket threshold");
        verifyOctreeNeighbors(built.id);
    }

    options.compute.sphBucketSize = 1;
    setOptions(options);
    sphTreeDestroy();
    initializeSphTreeParticles(12, 0);
    ParticleInteger particle = 0;
    for (int octant = 0; octant < kSphChildCount; ++octant) {
        int const count = octant == 0 ? 4 : octant == 1 ? 2 : 1;
        for (int member = 0; member < count; ++member, ++particle) {
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                double const origin = 0.5 * ((octant >> dimension) & 1);
                particlesPos(dimension, particle) =
                    origin + 0.125 + 0.25 * ((member >> dimension) & 1);
            }
        }
    }
    auto const mixed = sphTreeCreate(params);
    verifySphTreeBuild(mixed, 12);
    requireSphTree(mixed.nodeCount == 15 && mixed.leafNodes == 12 && mixed.minDepth == 1 &&
            mixed.maxDepth == 2,
        "mixed tree did not combine eight-, four-, and two-child nodes");
    verifyOctreeNeighbors(mixed.id);

    // A first x split leaves y and z longer. Sparse physical child bits must
    // retain their axis meaning when the next split selects y or z.
    for (int longAxis : {1, 2}) {
        sphTreeDestroy();
        ParticleInteger const count = longAxis == 1 ? 2 : 3;
        initializeSphTreeParticles(count, 0);
        for (ParticleInteger member = 0; member < count; ++member) {
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                particlesPos(dimension, member) = member == 2 ? 0.75 :
                    dimension == longAxis && member == 1      ? 0.75 :
                                                                0.25;
            }
        }
        auto const anisotropic = sphTreeCreate(params);
        verifySphTreeBuild(anisotropic, count);
        auto const& root = *sphTreeGetNode(anisotropic.id);
        auto const& refined = *sphTreeGetNode(root.children[0]);
        requireSphTree(refined.children[0].index >= 0 &&
                refined.children[1 << longAxis].index >= 0 && refined.nodeCount == 3,
            "anisotropic refinement did not split its longest axis");
        verifyOctreeNeighbors(anisotropic.id);
    }
    sphTreeDestroy();
    setOptions(savedOptions);
}

void verifySphTreeVolumeEvolution(SphTreeCreateParameters const& params) {
    SphTreeCreateReturn defaultResult;
    requireSphTree(defaultResult.minVolumeRatio == 0.0 && defaultResult.maxVolumeRatio == 0.0,
        "volume extrema are not initialized for an empty result");
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.compute.sphBucketSize = 2;
    setOptions(options);
    sphTreeDestroy();
    initializeSphTreeParticles(2, 0);
    for (ParticleInteger particle = 0; particle < 2; ++particle) {
        particlesRung(particle) = 2;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            particlesPos(dimension, particle) = particle == 0 ? 0.25 : 0.5;
            particlesVel(dimension, particle) = particle == 0 ? -0.25f : 0.25f;
        }
    }
    auto const built = sphTreeCreate(params);
    verifySphTreeVolumeRatios(built);
    requireSphTree(std::abs(built.maxVolumeRatio - 1.0 / 64.0) < 1.0e-7 &&
            built.minVolumeRatio == built.maxVolumeRatio,
        "fresh volume diagnostic did not use the original partition denominator");
    auto const evolved = sphTreeRefresh(params, 0.5);
    verifySphTreeVolumeRatios(evolved);
    requireSphTree(std::abs(evolved.maxVolumeRatio - 1.0 / 8.0) < 1.0e-7 &&
            evolved.minVolumeRatio == evolved.maxVolumeRatio,
        "known half-step drift has the wrong volume ratio");
    sphTreeDestroy();
    auto const rebuilt = sphTreeCreate(params);
    verifySphTreeVolumeRatios(rebuilt);
    requireSphTree(rebuilt.minVolumeRatio == built.minVolumeRatio &&
            rebuilt.maxVolumeRatio == built.maxVolumeRatio,
        "rebuild retained historical swept-volume extrema");
    sphTreeDestroy();
    initializeSphTreeParticles(0, 0);
    auto const empty = sphTreeCreate(params);
    auto const emptyRefresh = sphTreeRefresh(params);
    requireSphTree(empty.minVolumeRatio == 0.0 && empty.maxVolumeRatio == 0.0 &&
            emptyRefresh.minVolumeRatio == 0.0 && emptyRefresh.maxVolumeRatio == 0.0,
        "empty or inverted bounds contributed a volume ratio");
    sphTreeDestroy();
    setOptions(savedOptions);
}

void verifySphNeighborParameterArchive() {
    SphTreeNeighborParameters expected;
    expected.operation = SphNeighborOperation::FindNeighbors;
    expected.interactionDirection = SphInteractionDirection::SourceSupport;
    expected.innerParticleSelection = SphParticleSelection::Active;
    expected.outerParticleSelection = SphParticleSelection::All;
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << expected;
    }
    SphTreeNeighborParameters restored;
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> restored;
    }
    requireSphTree(restored.operation == expected.operation &&
            restored.interactionDirection == expected.interactionDirection &&
            restored.innerParticleSelection == expected.innerParticleSelection &&
            restored.outerParticleSelection == expected.outerParticleSelection,
        "neighbor parameter archive lost an independent selection control");
}

void verifyRefreshedOctreeBoxes(TreeId root) {
    for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
        sphParticlesSmoothingLength(i) = 0.2f + float(i % 3) * 0.05f;
    }
    SphTreeNeighborParameters params;
    params.operation = SphNeighborOperation::RefreshBounds;
    params.innerParticleSelection = SphParticleSelection::All;
    params.outerParticleSelection = SphParticleSelection::Active;
    params.minimumRung = kSphTreeMinimumRung;
    params.smoothingLengthMultiplier = kSphTreeSupportMultiplier;
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

void verifySphRebuildThreshold() {
    requireSphTree(!sphTreeShouldRebuild(0, 0), "empty population requests a threshold rebuild");
    requireSphTree(
        !sphTreeShouldRebuild(2, 9) && sphTreeShouldRebuild(3, 9) && sphTreeShouldRebuild(4, 9),
        "one-third rebuild threshold is not inclusive");
    requireSphTree(!sphTreeShouldRebuild(3, 10) && sphTreeShouldRebuild(4, 10),
        "one-third rebuild threshold was rounded down");
    size_t const maximum = std::numeric_limits<size_t>::max();
    requireSphTree(!sphTreeShouldRebuild(maximum / 3 - 1, maximum) &&
            sphTreeShouldRebuild(maximum / 3 + (maximum % 3 != 0), maximum),
        "large population overflowed the rebuild threshold");
    requireSphTree(!sphTreeShouldRebuild(0, 0, 0.0) && sphTreeShouldRebuild(0, 10, 0.0),
        "zero fraction must rebuild every nonempty population");
    requireSphTree(!sphTreeShouldRebuild(9, 10, 1.0) && sphTreeShouldRebuild(10, 10, 1.0) &&
            !sphTreeShouldRebuild(maximum - 1, maximum, 1.0),
        "unit fraction must require all particles active");
    requireSphTree(!sphTreeShouldRebuild(4, 10, 0.5) && sphTreeShouldRebuild(5, 10, 0.5) &&
            !sphTreeShouldRebuild(5, 11, 0.5) && sphTreeShouldRebuild(6, 11, 0.5),
        "custom fraction must use the inclusive ceiling threshold");
    requireSphTree(!sphTreeShouldRebuild(maximum / 2, maximum, 0.5) &&
            sphTreeShouldRebuild(maximum / 2 + 1, maximum, 0.5),
        "custom fraction lost precision at large population counts");
    requireSphTree(sphTreeShouldRebuild(1, 10, 0.1) && sphTreeShouldRebuild(7, 100, 0.07) &&
            !sphTreeShouldRebuild(1, 10, std::nextafter(0.1, 1.0)),
        "decimal fraction did not preserve the configured threshold");
    auto expected = getOptions();
    expected.compute.sphTreeRebuildFraction = 0.25;
    expected.run.rungMergeFraction = 0.75;
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << expected;
    }
    Options restored;
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> restored;
    }
    requireSphTree(
        restored.compute.sphTreeRebuildFraction == 0.25 && restored.run.rungMergeFraction == 0.75,
        "locality option serialization lost the independently configured fractions");
}

void verifySphMidpointSchedule() {
    requireSphTree(
        sphMidpointRung(0) == -1, "a synchronized coarse boundary scheduled an extra hydro kick");
    for (int rung = 0; rung < kMaxRung; ++rung) {
        TimeType const halfStep = TimeType{1} << (63 - rung);
        // Independently enumerate the odd half-step centers of each rung.
        // Unsigned wrap at a full interval must return to the no-H boundary.
        for (TimeType ordinal : {TimeType{1}, TimeType{3}, TimeType{5}, TimeType{7}}) {
            TimeType const center = ordinal * halfStep;
            requireSphTree(sphMidpointRung(center) == rung,
                "dyadic midpoint selected the wrong exact hydro rung");
        }
        if (rung > 0) {
            requireSphTree(sphMidpointRung(2 * halfStep) == rung - 1,
                "a fine boundary repeated fine hydro instead of selecting the coarse midpoint");
        }
    }
    std::cout << "SPH midpoint schedule regression passed\n" << std::flush;
}

void verifySphMidpointUpdates() {
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.hydro.enabled = true;
    options.hydro.method = "sph";
    options.hydro.viscosityEvolution = "morrisAndMonaghan";
    options.hydro.damping = 0.0;
    options.hydro.yAcceleration = 0.0;
    options.hydro.zPlaneAcceleration = 0.0;
    options.hydro.centralAcceleration = 0.0;
    options.chemistry.enabled = false;
    options.starFormation.enabled = false;
    options.gravity.enabled = false;
    options.initial.gasMass = 0.5;
    setOptions(options);
    bool const savedEntropyInitialized = sphParticlesEntropyInitialized;
    double const savedThermalScale = sphParticlesThermalScaleFactor;
    sphTreeDestroy(true);
    sphParticlesEntropyInitialized = false;
    initializeSphTreeParticles(2, 0);
    for (ParticleInteger gas = 0; gas < 2; ++gas) {
        sphParticlesDensities[gas] = 1.0f;
        sphParticlesGradHCorrections[gas] = 0.0f;
        sphParticlesSmoothingLength(gas) = 0.2f;
        sphParticlesRung(gas) = gas == 0 ? 0 : 2;
        sphParticlesSetProposedRung(gas, sphParticlesRung(gas));
        sphParticlesBulkViscosityCoefficients[gas] = 0.5f;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            sphParticlesPosition(dimension, gas) = dimension == 0 ? 0.45 + 0.1 * gas : 0.5;
            sphParticlesVelocity(dimension, gas) = 0.0f;
        }
    }
    sphParticlesPrepareThermodynamics(1.0, true);
    for (ParticleInteger gas = 0; gas < 2; ++gas) {
        sphParticlesVelocity(0, gas) = gas == 0 ? 0.25f : -0.25f;
        sphParticlesEntropy(gas) = 2.0f;
    }
    SphTreeCreateParameters treeParameters{};
    treeParameters.minimumRung = 2;
    treeParameters.smoothingLengthMultiplier = 1.0f;
    auto const tree = sphTreeCreate(treeParameters);
    SphTreeNeighborParameters neighbors{};
    neighbors.minimumRung = 2;
    neighbors.operation = SphNeighborOperation::RefreshBounds;
    neighbors.innerParticleSelection = neighbors.outerParticleSelection = SphParticleSelection::All;
    neighbors.smoothingLengthMultiplier = 1.0f;
    sphTreeNeighbor(neighbors, tree.id, Vector<TreeId>{}).get();
    neighbors.operation = SphNeighborOperation::FindNeighbors;
    neighbors.interactionDirection = SphInteractionDirection::EitherSupport;
    sphTreeNeighbor(neighbors, tree.id, Vector<TreeId>{tree.id}).get();
    SphRunParameters parameters;
    parameters.minimumRung = parameters.maximumRung = 2;
    parameters.baseTimestep = 1.0e-3f;
    parameters.scaleFactor = 1.0f;
    parameters.expansionRate = 0.0f;
    parameters.hydroKickTimestep = 1.0e-5;
    requireSphTree(sphMidpointEnabled() && !sphMidpointEventActive(),
        "midpoint fixture did not begin outside a hydro event");
    sphMidpointBeginEvent();
    sphMidpointForces(parameters, false);
    requireSphTree(sphParticlesVelocity(0, 0) != 0.25f && sphParticlesVelocity(0, 1) != -0.25f &&
            std::abs(sphParticlesVelocity(0, 0) + sphParticlesVelocity(0, 1)) < 1.0e-7 &&
            sphParticlesEntropy(0) > 2.0f && sphParticlesEntropy(1) > 2.0f &&
            sphParticlesRung(0) == 0 && sphParticlesRung(1) == 2,
        "fine-owned pair did not kick and heat both endpoints conservatively");
    std::array<float, 2> velocities{sphParticlesVelocity(0, 0), sphParticlesVelocity(0, 1)};
    std::array<float, 2> entropies{sphParticlesEntropy(0), sphParticlesEntropy(1)};
    parameters.hydroKickTimestep = 0.0;
    sphMidpointForces(parameters, false);
    sphMidpointEndEvent();
    for (int gas = 0; gas < 2; ++gas) {
        requireSphTree(sphParticlesVelocity(0, gas) == velocities[gas] &&
                sphParticlesEntropy(gas) == entropies[gas],
            "zero-duration estimate applied a physical pair impulse");
    }
    // Event bookkeeping and source reads must not replay a previous impulse
    // after an independent signed gravity correction to live momentum.
    sphParticlesVelocity(0, 1) -= 0.125f;
    velocities[1] = sphParticlesVelocity(0, 1);
    sphMidpointBeginEvent();
    sphMidpointEndEvent();
    ParticleGlobalRange range;
    range.proc = hpxRank();
    range.range = Pair<ParticleInteger>(0, 2);
    float velocityX[2], velocityY[2], velocityZ[2], internalEnergy[2];
    sphParticlesGlobalReadVelocities(range, velocityX, velocityY, velocityZ, 0);
    sphParticlesGlobalReadInternalEnergyAndSmoothingLength(range, internalEnergy, nullptr, 0);
    // The midpoint predictor refreshes the active particle's density (for the Balsara
    // factor), so the expected energy uses the live density with the recorded entropy.
    for (int gas = 0; gas < 2; ++gas) {
        double const expectedEnergy = sph_entropy::storedEnergy(
            entropies[gas], sphParticlesDensities[gas], options.hydro.gamma, 1.0);
        requireSphTree(velocityX[gas] == velocities[gas] &&
                std::abs(internalEnergy[gas] - expectedEnergy) < 1.0e-6 * expectedEnergy,
            "source staging replayed a rate or used stale physical state");
    }
    parameters.minimumRung = parameters.maximumRung = 0;
    parameters.hydroKickTimestep = 4.0e-5;
    sphMidpointBeginEvent();
    sphMidpointForces(parameters, false);
    sphMidpointEndEvent();
    for (int gas = 0; gas < 2; ++gas) {
        requireSphTree(sphParticlesVelocity(0, gas) == velocities[gas] &&
                sphParticlesEntropy(gas) == entropies[gas],
            "coarse midpoint repeated the finer-owned pair");
    }
    parameters.maximumRung = kMaxRung - 1;
    parameters.globalTimestep = true;
    sphMidpointBeginEvent();
    sphMidpointForces(parameters, false);
    sphMidpointEndEvent();
    requireSphTree(sphParticlesVelocity(0, 0) != velocities[0] &&
            std::abs(sphParticlesVelocity(0, 0) + sphParticlesVelocity(0, 1) -
                (velocities[0] + velocities[1])) < 1.0e-7,
        "global pair kick missed an endpoint or changed total momentum");
    for (ParticleInteger gas = 0; gas < 2; ++gas) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            requireSphTree(sphParticlesPosition(dimension, gas) ==
                    Position(dimension == 0 ? 0.45 + 0.1 * gas : 0.5),
                "fixed-position midpoint pair kick moved a particle");
        }
    }
    sphTreeDestroy(true);
    sphParticlesResize(0, false);
    particlesResize(0);
    setOptions(savedOptions);
    sphParticlesEntropyInitialized = savedEntropyInitialized;
    sphParticlesThermalScaleFactor = savedThermalScale;
    std::cout << "SPH midpoint pair, source and positive-kick regression passed\n" << std::flush;
}

using SphTemporalState = std::array<double, 4>;

SphTemporalState sphTemporalState() {
    return {sphParticlesVelocity(0, 0), sphParticlesVelocity(0, 1), sphParticlesEntropy(0),
        sphParticlesEntropy(1)};
}

void verifySphTemporalOrder() {
    auto const savedOptions = getOptions();
    bool const savedEntropyInitialized = sphParticlesEntropyInitialized;
    double const savedThermalScale = sphParticlesThermalScaleFactor;
    auto options = savedOptions;
    options.hydro.enabled = true;
    options.hydro.method = "sph";
    options.hydro.viscosityEvolution = "none";
    options.hydro.gamma = 5.0 / 3.0;
    options.hydro.bulkViscosityCoefficient = 0.7;
    options.hydro.vonNeumannRichtmyerCoefficient = 1.0;
    options.hydro.viscosityEpsilon = 0.01;
    options.hydro.yAcceleration = 0.0;
    options.hydro.zPlaneAcceleration = 0.0;
    options.hydro.centralAcceleration = 0.0;
    options.chemistry.enabled = false;
    options.starFormation.enabled = false;
    options.gravity.enabled = false;
    options.initial.gasMass = 1.0e-4;
    std::array<int, 3> const stepCounts{4, 8, 16};
    SphTemporalState const initialState{0.5, -0.3, 0.8, 1.2};

    // These are production fixed-position hydro events, including source staging,
    // reciprocal updates and commits. Hold supports and viscosity coefficients fixed
    // to isolate the temporal order of the hydro operator; this does not test drift
    // splitting or the endpoint viscosity switches. The midpoint predictor always
    // refreshes densities for the Balsara factor, so the isolated damping pair holds
    // its supports at the upper bound instead of leaving them free and unsolved.
    for (int testCase = 0; testCase < 3; ++testCase) {
        bool const damping = testCase == 0;
        options.hydro.damping = damping ? 1.25 : 0.0;
        options.hydro.conductionCoefficient = testCase == 2 ? 0.3 : 0.0;
        options.hydro.maxSmoothingLength = damping ? 0.1 : savedOptions.hydro.maxSmoothingLength;
        setOptions(options);
        sphTreeDestroy(true);
        sphParticlesEntropyInitialized = false;
        initializeSphTreeParticles(2, 0);
        for (ParticleInteger gas = 0; gas < 2; ++gas) {
            sphParticlesDensities[gas] = 1.3f + 0.8f * gas;
            sphParticlesGradHCorrections[gas] = 0.0f;
            sphParticlesSmoothingLength(gas) = damping ? 0.1f : 0.2f + 0.05f * gas;
            sphParticlesRung(gas) = gas == 0 ? 0 : 2;
            sphParticlesSetProposedRung(gas, sphParticlesRung(gas));
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                sphParticlesPosition(dimension, gas) =
                    dimension == 0 ? (damping ? 0.25 + 0.5 * gas : 0.47 + 0.06 * gas) : 0.5;
                sphParticlesVelocity(dimension, gas) = 0.0f;
            }
        }
        sphParticlesPrepareThermodynamics(1.0, true);
        SphTreeCreateParameters treeParameters{};
        treeParameters.minimumRung = damping ? 0 : 2;
        treeParameters.smoothingLengthMultiplier = 1.0f;
        auto const tree = sphTreeCreate(treeParameters);
        SphTreeNeighborParameters neighbors{};
        neighbors.minimumRung = treeParameters.minimumRung;
        neighbors.operation = SphNeighborOperation::RefreshBounds;
        neighbors.innerParticleSelection = neighbors.outerParticleSelection =
            SphParticleSelection::All;
        neighbors.smoothingLengthMultiplier = 1.0f;
        sphTreeNeighbor(neighbors, tree.id, Vector<TreeId>{}).get();
        neighbors.operation = SphNeighborOperation::FindNeighbors;
        neighbors.interactionDirection = SphInteractionDirection::EitherSupport;
        sphTreeNeighbor(neighbors, tree.id, Vector<TreeId>{tree.id}).get();

        auto const run = [&](int steps, double T) {
            for (ParticleInteger gas = 0; gas < 2; ++gas) {
                sphParticlesVelocity(0, gas) = float(initialState[gas]);
                sphParticlesEntropy(gas) = float(initialState[gas + 2]);
                sphParticlesSetProposedRung(gas, sphParticlesRung(gas));
            }
            SphRunParameters parameters;
            parameters.minimumRung = damping ? 0 : 2;
            parameters.maximumRung = damping ? kMaxRung - 1 : 2;
            parameters.globalTimestep = damping;
            parameters.baseTimestep = float(T);
            parameters.scaleFactor = 1.0f;
            parameters.expansionRate = 0.0f;
            parameters.hydroKickTimestep = T / steps;
            double const initialMomentum =
                double(float(initialState[0])) + double(float(initialState[1]));
            for (int step = 0; step < steps; ++step) {
                sphMidpointBeginEvent();
                sphMidpointForces(parameters, false);
                sphMidpointEndEvent();
                for (ParticleInteger gas = 0; gas < 2; ++gas) {
                    requireSphTree(sphParticlesEntropy(gas) > 0.0f &&
                            sphParticlesRung(gas) == (gas == 0 ? 0 : 2),
                        "temporal refinement changed rungs or lost positive entropy");
                }
                if (!damping) {
                    double const momentum =
                        double(sphParticlesVelocity(0, 0)) + double(sphParticlesVelocity(0, 1));
                    requireSphTree(std::abs(momentum - initialMomentum) < 2.0e-6,
                        "temporal refinement lost the inactive neighbor's recoil");
                }
            }
            return sphTemporalState();
        };

        double T = 1.0;
        if (!damping) {
            // Derived conditioning: choose a fixed final time from the initial
            // complete production RHS, so the pair loses an appreciable fraction
            // of its approach speed before the viscosity's separating branch.
            // T = 0.35 (v₀-v₁)/|a₀-a₁|; a is estimated by one short RK2 probe δ.
            // Symbol directory: T → duration; δ → probe duration;
            // v₀,v₁ → initial velocities; a₀-a₁ → relative acceleration.
            // Code names: a_0_minus_a_1 = a₀-a₁.
            constexpr double δ = 1.0e-4;
            auto const probe = run(1, δ);
            double const initialRelativeVelocity =
                double(float(initialState[0])) - double(float(initialState[1]));
            double const a_0_minus_a_1 = (probe[0] - probe[1] - initialRelativeVelocity) / δ;
            requireSphTree(
                a_0_minus_a_1 < 0.0, "temporal pair fixture does not oppose the approaching flow");
            T = 0.35 * initialRelativeVelocity / -a_0_minus_a_1;
        }
        auto const reference = run(damping ? 1 : 256, T);
        requireSphTree(damping ||
                (reference[0] < initialState[0] && reference[1] > initialState[1] &&
                    reference[0] > reference[1]),
            "temporal pair fixture did not remain viscous and approaching");
        if (testCase == 1) {
            requireSphTree(reference[2] > initialState[2] && reference[3] > initialState[3],
                "temporal pair fixture did not exercise viscosity heating");
        }
        {
            std::array<double, 3> velocityErrors{};
            std::array<double, 3> entropyErrors{};
            for (size_t refinement = 0; refinement < stepCounts.size(); ++refinement) {
                auto const state = run(stepCounts[refinement], T);
                for (size_t gas = 0; gas < 2; ++gas) {
                    // Derived exact solution of dv/dt = -λv:
                    // v(T) = v(0) exp(-λT). λ → damping rate;
                    // T → duration; v(0) → initial state.
                    double const expectedVelocity = damping ?
                        double(float(initialState[gas])) * std::exp(-options.hydro.damping * T) :
                        reference[gas];
                    velocityErrors[refinement] = std::max(
                        velocityErrors[refinement], std::abs(state[gas] - expectedVelocity));
                    entropyErrors[refinement] = std::max(
                        entropyErrors[refinement], std::abs(state[gas + 2] - reference[gas + 2]));
                    if (damping) {
                        requireSphTree(state[gas + 2] == double(float(initialState[gas + 2])),
                            "isolated damping changed the entropy");
                    }
                }
            }
            for (size_t refinement = 1; refinement < stepCounts.size(); ++refinement) {
                double const minimumRatio = 3.2;
                double const maximumRatio = 5.5;
                double const velocityRatio =
                    velocityErrors[refinement - 1] / velocityErrors[refinement];
                double const entropyRatio =
                    damping ? 0.0 : entropyErrors[refinement - 1] / entropyErrors[refinement];
                std::cout << "SPH production temporal order: case=" << testCase
                          << " integrator=rk2 steps=" << stepCounts[refinement]
                          << " velocity_error=" << velocityErrors[refinement]
                          << " velocity_ratio=" << velocityRatio
                          << " entropy_error=" << entropyErrors[refinement]
                          << " entropy_ratio=" << entropyRatio << '\n'
                          << std::flush;
                requireSphTree(velocityRatio > minimumRatio && velocityRatio < maximumRatio,
                    "production hydro velocity has the wrong temporal order");
                requireSphTree(
                    damping || (entropyRatio > minimumRatio && entropyRatio < maximumRatio),
                    "production hydro entropy has the wrong temporal order");
            }
        }
    }
    {
        // Derived Strang composition at fixed positions: over a coarse interval H,
        // K_f(H/2) K𞁞(H) K_f(H/2) approximates F_f + F𞁞 through second order.
        // The three-particle chain has one coarse/coarse pair and one coarse/fine
        // pair sharing the middle particle, so these velocity/entropy-dependent
        // operators do not commute. Production RK2 implements every K call.
        // Symbol directory: H -> coarse timestep; K -> advance rung;
        // F𞁞,F_f -> the production due-pair operators on rungs 0 and 1.
        constexpr ParticleInteger kParticleCount = 3;
        using SplitState = std::array<double, 2 * kParticleCount>;
        SplitState const initialSplitState{0.6, 0.1, -0.35, 0.08, 0.1, 0.12};
        std::array<int, 3> const splitStepCounts{4, 8, 16};
        options.hydro.damping = 0.0;
        // Lower entropy and stronger conduction keep the thermal O(h²) signal
        // above binary32 accumulation noise without crossing the approach branch.
        options.hydro.conductionCoefficient = 3.0;
        setOptions(options);
        sphTreeDestroy(true);
        sphParticlesEntropyInitialized = false;
        initializeSphTreeParticles(kParticleCount, 0);
        for (ParticleInteger gas = 0; gas < kParticleCount; ++gas) {
            sphParticlesDensities[gas] = 1.6f;
            sphParticlesGradHCorrections[gas] = 0.0f;
            sphParticlesSmoothingLength(gas) = 0.1f;
            sphParticlesRung(gas) = gas == 2 ? 1 : 0;
            sphParticlesSetProposedRung(gas, sphParticlesRung(gas));
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                sphParticlesPosition(dimension, gas) = dimension == 0 ? 0.44 + 0.06 * gas : 0.5;
                sphParticlesVelocity(dimension, gas) = 0.0f;
            }
        }
        sphParticlesPrepareThermodynamics(1.0, true);
        SphTreeCreateParameters treeParameters{};
        treeParameters.minimumRung = 0;
        treeParameters.smoothingLengthMultiplier = 1.0f;
        auto const tree = sphTreeCreate(treeParameters);
        SphTreeNeighborParameters neighbors{};
        neighbors.minimumRung = 0;
        neighbors.operation = SphNeighborOperation::RefreshBounds;
        neighbors.innerParticleSelection = neighbors.outerParticleSelection =
            SphParticleSelection::All;
        neighbors.smoothingLengthMultiplier = 1.0f;
        sphTreeNeighbor(neighbors, tree.id, Vector<TreeId>{}).get();
        neighbors.operation = SphNeighborOperation::FindNeighbors;
        neighbors.interactionDirection = SphInteractionDirection::EitherSupport;
        sphTreeNeighbor(neighbors, tree.id, Vector<TreeId>{tree.id}).get();

        auto const resetState = [&]() {
            for (ParticleInteger gas = 0; gas < kParticleCount; ++gas) {
                sphParticlesVelocity(0, gas) = float(initialSplitState[gas]);
                sphParticlesEntropy(gas) = float(initialSplitState[kParticleCount + gas]);
                sphParticlesSetProposedRung(gas, sphParticlesRung(gas));
            }
        };
        auto const captureState = [&]() {
            SplitState state{};
            for (ParticleInteger gas = 0; gas < kParticleCount; ++gas) {
                state[gas] = sphParticlesVelocity(0, gas);
                state[kParticleCount + gas] = sphParticlesEntropy(gas);
            }
            return state;
        };
        auto const K = [&](int rung, double timestep) {
            SphRunParameters parameters;
            parameters.minimumRung = parameters.maximumRung = rung;
            parameters.baseTimestep = std::ldexp(timestep, rung);
            parameters.scaleFactor = 1.0f;
            parameters.expansionRate = 0.0f;
            parameters.hydroKickTimestep = timestep;
            sphMidpointBeginEvent();
            sphMidpointForces(parameters, false);
            sphMidpointEndEvent();
        };
        constexpr double kProbeDuration = 1.0e-5;
        resetState();
        K(0, kProbeDuration);
        auto const coarseProbe = captureState();
        requireSphTree(coarseProbe[0] < float(initialSplitState[0]) &&
                coarseProbe[1] > float(initialSplitState[1]) &&
                coarseProbe[2] == float(initialSplitState[2]),
            "split refinement fixture did not isolate its coarse-owned pair");
        resetState();
        K(1, kProbeDuration);
        auto const fineProbe = captureState();
        requireSphTree(fineProbe[0] == float(initialSplitState[0]) &&
                fineProbe[1] < float(initialSplitState[1]) &&
                fineProbe[2] > float(initialSplitState[2]),
            "split refinement fixture did not isolate its fine-owned pair");

        auto const runSplit = [&](int steps, double duration) {
            resetState();
            double const H = duration / steps;
            double initialMomentum = 0.0;
            for (ParticleInteger gas = 0; gas < kParticleCount; ++gas) {
                initialMomentum += float(initialSplitState[gas]);
            }
            for (int step = 0; step < steps; ++step) {
                K(1, 0.5 * H);
                K(0, H);
                K(1, 0.5 * H);
                double momentum = 0.0;
                for (ParticleInteger gas = 0; gas < kParticleCount; ++gas) {
                    momentum += sphParticlesVelocity(0, gas);
                    requireSphTree(sphParticlesEntropy(gas) > 0.0f &&
                            sphParticlesRung(gas) == (gas == 2 ? 1 : 0),
                        "split refinement changed rungs or lost positive entropy");
                }
                requireSphTree(std::abs(momentum - initialMomentum) < 8.0e-6,
                    "split refinement lost reciprocal coarse/fine momentum");
            }
            return captureState();
        };
        auto const probe = runSplit(1, kProbeDuration);
        double duration = std::numeric_limits<double>::infinity();
        for (ParticleInteger gas = 0; gas + 1 < kParticleCount; ++gas) {
            double const relativeVelocity =
                double(float(initialSplitState[gas])) - float(initialSplitState[gas + 1]);
            double const relativeAcceleration =
                (probe[gas] - probe[gas + 1] - relativeVelocity) / kProbeDuration;
            requireSphTree(relativeAcceleration < 0.0,
                "split refinement fixture does not oppose each approaching pair");
            duration = std::min(duration, 0.4 * relativeVelocity / -relativeAcceleration);
        }
        auto const reference = runSplit(64, duration);
        auto const refinedReference = runSplit(128, duration);
        requireSphTree(reference[0] > reference[1] && reference[1] > reference[2],
            "split refinement crossed the viscosity approach branch");
        std::array<double, 3> velocityErrors{};
        std::array<double, 3> entropyErrors{};
        for (size_t refinement = 0; refinement < splitStepCounts.size(); ++refinement) {
            auto const state = runSplit(splitStepCounts[refinement], duration);
            for (ParticleInteger gas = 0; gas < kParticleCount; ++gas) {
                velocityErrors[refinement] =
                    std::max(velocityErrors[refinement], std::abs(state[gas] - reference[gas]));
                entropyErrors[refinement] = std::max(entropyErrors[refinement],
                    std::abs(state[kParticleCount + gas] - reference[kParticleCount + gas]));
            }
        }
        for (ParticleInteger gas = 0; gas < kParticleCount; ++gas) {
            requireSphTree(
                std::abs(reference[gas] - refinedReference[gas]) < 0.1 * velocityErrors.back(),
                "split velocity reference uncertainty masks the refinement signal");
            requireSphTree(std::abs(reference[kParticleCount + gas] -
                               refinedReference[kParticleCount + gas]) < 0.1 * entropyErrors.back(),
                "split entropy reference uncertainty masks the refinement signal");
        }
        for (size_t refinement = 1; refinement < splitStepCounts.size(); ++refinement) {
            double const velocityRatio =
                velocityErrors[refinement - 1] / velocityErrors[refinement];
            double const entropyRatio = entropyErrors[refinement - 1] / entropyErrors[refinement];
            std::cout << "SPH production temporal order: case=mixed_rung_split steps="
                      << splitStepCounts[refinement]
                      << " velocity_error=" << velocityErrors[refinement]
                      << " velocity_ratio=" << velocityRatio
                      << " entropy_error=" << entropyErrors[refinement]
                      << " entropy_ratio=" << entropyRatio << '\n'
                      << std::flush;
            requireSphTree(velocityRatio > 3.2 && velocityRatio < 5.5,
                "production fine/coarse composition has the wrong velocity order");
            requireSphTree(entropyRatio > 3.2 && entropyRatio < 5.5,
                "production fine/coarse composition has the wrong entropy order");
        }
    }
    sphTreeDestroy(true);
    sphParticlesResize(0, false);
    particlesResize(0);
    setOptions(savedOptions);
    sphParticlesEntropyInitialized = savedEntropyInitialized;
    sphParticlesThermalScaleFactor = savedThermalScale;
    std::cout << "SPH production RK2 temporal refinement passed (fixed coefficients, CPU)\n"
              << std::flush;
}

void verifyRetainedSphOctree(SphTreeCreateParameters const& params) {
    sphTreeDestroy();
    initializeSphTreeParticles(8, 0);
    for (ParticleInteger particle = 0; particle < 8; ++particle) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, particle) = ((particle >> dim) & 1) != 0 ? 0.99 : 0.49;
            particlesVel(dim, particle) = ((particle + dim) & 1) != 0 ? 0.4f : -0.4f;
        }
        particlesRung(particle) = particle < 4 ? 2 : 0;
    }
    auto const built = sphTreeCreate(params);
    requireSphTree(sphTreeIsValid(), "new SPH octree is invalid");
    std::vector<Vector3<Position>> originalPositions(sphParticlesSize());
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            originalPositions[particle][dim] = sphParticlesPosition(dim, particle);
        }
    }
    // A gravity rebuild permutes its own array but leaves the retained SPH
    // memberships intact; the inverse mapping is the only link that changes.
    particlesSwap(0, 7);
    particlesSwap(1, 6);
    particlesResolveWithSphParticles();
    requireSphTree(sphTreeIsValid(), "gravity permutation invalidated SPH membership");
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        requireSphTree(particlesCatIndex(sphParticlesParticleIndex(particle)) == particle,
            "gravity permutation failed to repair the gas-to-particle mapping");
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireSphTree(sphParticlesPosition(dim, particle) == originalPositions[particle][dim],
                "gravity permutation moved a gas particle to a different SPH leaf");
        }
    }
    std::vector<TreeId> leaves;
    std::vector<SphTreeNode> originalNodes;
    std::vector<ParticleInteger> originalOrder;
    for (int index = 0; index < sphTreeLeaflistSize(); ++index) {
        leaves.push_back(sphTreeGetLeaf(index));
        originalNodes.push_back(*sphTreeGetNode(leaves.back()));
    }
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        originalOrder.push_back(sphParticlesParticleIndex(particle));
    }
    double const driftFactor = 0.25;
    sphTreeDriftBounds(driftFactor, 2);
    bool crossedPartition = false;
    bool crossedPeriodicBoundary = false;
    for (size_t index = 0; index < leaves.size(); ++index) {
        auto const& leaf = *sphTreeGetNode(leaves[index]);
        if (originalNodes[index].numberOfActiveParticles == 0) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                requireSphTree(
                    leaf.driftBox.begin[dim] == originalNodes[index].driftBox.begin[dim] &&
                        leaf.driftBox.end[dim] == originalNodes[index].driftBox.end[dim],
                    "fine-rung drift visited an entirely inactive subtree");
            }
        }
        for (ParticleInteger particle = leaf.particleRange.first;
            particle < leaf.particleRange.second; ++particle) {
            Vector3<Position> position;
            for (int dim = 0; dim < kDimensions; ++dim) {
                position[dim] = sphParticlesPosition(dim, particle);
            }
            requireSphTree(leaf.driftBox.contains(position), "swept box lost its starting point");
            if (sphParticlesRung(particle) == 2) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    double const endpoint = position[dim].toDouble() +
                        driftFactor * sphParticlesVelocity(dim, particle);
                    crossedPeriodicBoundary = crossedPeriodicBoundary || endpoint >= 1.0;
                    position[dim] = endpoint;
                    sphParticlesPosition(dim, particle) = position[dim];
                }
            }
            crossedPartition = crossedPartition || !originalNodes[index].box.contains(position);
            requireSphTree(leaf.driftBox.contains(position),
                "velocity-expanded cell missed a periodic drift endpoint");
            sphParticlesRung(particle) = particle < 3 ? 2 : 0;
            sphParticlesSmoothingLength(particle) = 0.125f;
        }
    }
    requireSphTree(crossedPartition && crossedPeriodicBoundary,
        "drift fixture did not cross both spatial and periodic boundaries");
    auto const refreshed = sphTreeRefresh(params);
    verifySphTreeVolumeRatios(refreshed);
    requireSphTree(refreshed.id == built.id && refreshed.nodeCount == built.nodeCount &&
            refreshed.numberOfActiveParticles == 3,
        "retained tree changed topology or retained stale active counts");
    std::vector<char> returnBytes;
    {
        hpx::serialization::output_archive archive(returnBytes);
        archive << refreshed;
    }
    SphTreeCreateReturn received{};
    {
        hpx::serialization::input_archive archive(returnBytes);
        archive >> received;
    }
    requireSphTree(received.minVolumeRatio == refreshed.minVolumeRatio &&
            received.maxVolumeRatio == refreshed.maxVolumeRatio,
        "volume extrema were lost in the distributed refresh result");
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireSphTree(received.driftBox.begin[dim] == refreshed.driftBox.begin[dim] &&
                received.driftBox.end[dim] == refreshed.driftBox.end[dim],
            "drift bounds were lost in the distributed refresh result");
    }
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        requireSphTree(originalOrder[particle] == sphParticlesParticleIndex(particle),
            "refresh reordered SPH particles");
    }
    for (size_t index = 0; index < leaves.size(); ++index) {
        auto const& leaf = *sphTreeGetNode(leaves[index]);
        auto const& original = originalNodes[index];
        requireSphTree(sphTreeGetLeaf(index) == leaves[index] &&
                leaf.particleRange == original.particleRange &&
                leaf.neighborRange == Pair<int>(-1, -1),
            "retained tree changed leaf membership or kept stale neighbors");
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireSphTree(leaf.box.begin[dim] == original.box.begin[dim] &&
                    leaf.box.end[dim] == original.box.end[dim],
                "refresh changed the original octree partition");
            requireSphTree(leaf.driftBox.begin[dim] <= original.driftBox.begin[dim] &&
                    leaf.driftBox.end[dim] >= original.driftBox.end[dim],
                "refresh discarded the previous swept envelope");
        }
        size_t active = 0;
        for (ParticleInteger particle = leaf.particleRange.first;
            particle < leaf.particleRange.second; ++particle) {
            active += sphParticlesRung(particle) >= params.minimumRung;
            Vector3<Position> position;
            for (int dim = 0; dim < kDimensions; ++dim) {
                position[dim] = sphParticlesPosition(dim, particle);
                if (sphParticlesRung(particle) >= params.minimumRung) {
                    double const support =
                        params.smoothingLengthMultiplier * sphParticlesSmoothingLength(particle);
                    requireSphTree(
                        leaf.outerBox.begin[dim].toDouble() <= position[dim].toDouble() - support &&
                            leaf.outerBox.end[dim].toDouble() >= position[dim].toDouble() + support,
                        "refresh missed a growing kernel support");
                }
            }
            requireSphTree(leaf.innerBox.contains(position) && leaf.driftBox.contains(position),
                "refreshed cell does not contain a current particle");
        }
        requireSphTree(leaf.numberOfActiveParticles == active, "leaf active count is stale");
    }
    verifyOctreeNeighbors(refreshed.id);
    // Chronological midpoint integration drifts coarse sources too, even when
    // only fine targets are active. Change the velocity between the two halves
    // so the second envelope must use the post-kick momentum, not a saved rate.
    for (int half = 0; half < 2; ++half) {
        for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                sphParticlesVelocity(dimension, particle) =
                    (half == 0 ? -0.5f : 0.75f) * (dimension + 1);
            }
        }
        sphTreeDriftBounds(0.125, -1);
        for (auto const leafId : leaves) {
            auto const& leaf = *sphTreeGetNode(leafId);
            for (ParticleInteger particle = leaf.particleRange.first;
                particle < leaf.particleRange.second; ++particle) {
                Vector3<Position> endpoint;
                for (int dimension = 0; dimension < kDimensions; ++dimension) {
                    endpoint[dimension] = sphParticlesPosition(dimension, particle).toDouble() +
                        0.125 * sphParticlesVelocity(dimension, particle);
                    sphParticlesPosition(dimension, particle) = endpoint[dimension];
                }
                requireSphTree(leaf.driftBox.contains(endpoint),
                    "chronological half-drift missed an inactive source or the new velocity");
            }
        }
        auto const midpointRefit = sphTreeRefresh(params);
        verifySphTreeVolumeRatios(midpointRefit);
        requireSphTree(midpointRefit.id == built.id && midpointRefit.nodeCount == built.nodeCount &&
                midpointRefit.numberOfActiveParticles == 3,
            "chronological midpoint refit changed retained topology or active targets");
        verifyOctreeNeighbors(midpointRefit.id);
    }
    sphTreeDriftBounds(-0.5, 2);
    for (auto const leafId : leaves) {
        auto const& leaf = *sphTreeGetNode(leafId);
        for (ParticleInteger particle = leaf.particleRange.first;
            particle < leaf.particleRange.second; ++particle) {
            if (sphParticlesRung(particle) == 2) {
                Vector3<Position> endpoint;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    endpoint[dim] = sphParticlesPosition(dim, particle).toDouble() -
                        0.5 * sphParticlesVelocity(dim, particle);
                }
                requireSphTree(leaf.driftBox.contains(endpoint),
                    "signed drift missed its negative-time endpoint");
            }
        }
    }
    // A many-box displacement must saturate rather than overflow Q3.29 bounds.
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            sphParticlesVelocity(dim, particle) = 16.0f;
        }
    }
    sphTreeDriftBounds(1.0);
    for (auto const leafId : leaves) {
        auto const& leaf = *sphTreeGetNode(leafId);
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireSphTree(leaf.driftBox.begin[dim].toDouble() <= 0.0 &&
                    leaf.driftBox.end[dim].toDouble() >= 1.0,
                "long periodic drift did not cover the full axis");
        }
    }
    sphTreeInvalidate();
    requireSphTree(!sphTreeIsValid(), "particle mutation did not invalidate retained topology");
    bool rejected = false;
    try {
        sphTreeRefresh(params);
    } catch (std::runtime_error const&) {
        rejected = true;
    }
    requireSphTree(rejected, "invalid topology was refreshed instead of rejected");
}

}    // namespace detail

void initializeDistributedSphOctree() {
    detail::initializeSphTreeParticles(8, 0);
    for (ParticleInteger particle = 0; particle < 8; ++particle) {
        particlesPos(0, particle) = particle == 0 ? (hpxRank() == 0 ? 0.49 : 0.51) :
                                                    (hpxRank() + (particle + 0.5) / 8.0) / 2.0;
        particlesPos(1, particle) = 0.25 + 0.25 * (particle & 1);
        particlesPos(2, particle) = 0.25 + 0.25 * ((particle >> 1) & 1);
        particlesVel(0, particle) = particle == 0 ? (hpxRank() == 0 ? 0.08f : -0.08f) : 0.0f;
        particlesRung(particle) = particle == 0 ? 2 : 0;
        sphParticlesSmoothingLength(particle) = 0.09f;
    }
}
HPX_PLAIN_ACTION(initializeDistributedSphOctree, InitializeDistributedSphOctreeAction);

size_t driftDistributedSphOctree() {
    sphTreeDriftBounds(0.5, 2);
    size_t crossed = 0;
    auto const domain = domainsFindMyBox();
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        if (sphParticlesRung(particle) != 2) {
            continue;
        }
        sphParticlesPosition(0, particle) += Position(0.5 * sphParticlesVelocity(0, particle));
        Vector3<double> position;
        for (int dim = 0; dim < kDimensions; ++dim) {
            position[dim] = sphParticlesPosition(dim, particle).toDouble();
        }
        crossed += !domain.contains(position);
    }
    return crossed;
}
HPX_PLAIN_ACTION(driftDistributedSphOctree, DriftDistributedSphOctreeAction);

void verifyDistributedSphOctree(Vector<TreeId> const& allLeaves) {
    for (int leafIndex = 0; leafIndex < sphTreeLeaflistSize(); ++leafIndex) {
        auto const& target = *sphTreeGetNode(sphTreeGetLeaf(leafIndex));
        std::set<TreeId> neighbors;
        for (int index = target.neighborRange.first; index < target.neighborRange.second; ++index) {
            neighbors.insert(sphTreeGetNeighbor(index));
        }
        for (ParticleInteger particle = target.particleRange.first;
            particle < target.particleRange.second; ++particle) {
            if (sphParticlesRung(particle) != 2) {
                continue;
            }
            bool remotePair = false;
            for (auto const sourceId : allLeaves) {
                auto const& source = *sphTreeGetNode(sourceId);
                size_t const count = source.numberOfParticles();
                Vector<Position> positions[3];
                for (auto& coordinate : positions) {
                    coordinate.resize(count);
                }
                sphParticlesGlobalReadPositions(source.globalParticleRange(), positions[0].data(),
                    positions[1].data(), positions[2].data(), 0);
                for (size_t neighbor = 0; neighbor < count; ++neighbor) {
                    double squaredDistance = 0.0;
                    Vector3<Position> position;
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        position[dim] = positions[dim][neighbor];
                        double const separation =
                            std::remainder(sphParticlesPosition(dim, particle).toDouble() -
                                    position[dim].toDouble(),
                                1.0);
                        squaredDistance += separation * separation;
                    }
                    detail::requireSphTree(source.driftBox.contains(position),
                        "remote swept bounds missed a particle after crossing domains");
                    if (squaredDistance < 0.09 * 0.09) {
                        detail::requireSphTree(neighbors.contains(sourceId),
                            "retained distributed tree missed a brute-force neighbor");
                        remotePair = remotePair || sourceId.proc != hpxRank();
                    }
                }
            }
            detail::requireSphTree(remotePair, "distributed fixture lacks a cross-rank pair");
        }
    }
}
HPX_PLAIN_ACTION(verifyDistributedSphOctree, VerifyDistributedSphOctreeAction);

#if COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL && COSMICTIGER_ENABLE_FMM
void verifyDistributedSphHierarchyMigration(bool afterMigration) {
    static std::vector<Vector3<Position>> prefixPositions;
    auto const range = particlesCurrentRange();
    detail::requireSphTree(range.first == 7 && range.second == 8,
        "hierarchical migration changed the inactive prefix or active suffix size");
    if (!afterMigration) {
        prefixPositions.resize(range.first);
    }
    for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
        ParticleInteger const gas = particlesCatIndex(particle);
        if (!afterMigration) {
            sphParticlesEntropies[gas] = 100 * hpxRank() + particle + 1;
        }
        float const identity = sphParticlesEntropies[gas];
        if (!afterMigration) {
            sphParticlesDensities[gas] = identity + 0.5f;
            if (sphParticlesBulkViscosityCoefficients) {
                sphParticlesBulkViscosityCoefficients[gas] = identity / 256.0f;
            }
            // The last midpoint's hydro timestep bound must survive migration
            // until the next gravity boundary consumes it.
            sphParticlesSetProposedRung(gas, particlesRung(particle) + 1);
            if (particle < range.first) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    prefixPositions[particle][dim] = particlesPos(dim, particle);
                }
            }
            continue;
        }
        detail::requireSphTree(sphParticlesDensities[gas] == identity + 0.5f &&
                (!sphParticlesBulkViscosityCoefficients ||
                    sphParticlesBulkViscosityCoefficients[gas] == identity / 256.0f) &&
                sphParticlesProposedRung(gas) == particlesRung(particle) + 1,
            "migration changed midpoint gas scalars or the hydro rung bound");
        if (particle < range.first) {
            detail::requireSphTree(
                identity == 100 * hpxRank() + particle + 1 && particlesRung(particle) == 0,
                "active migration moved or replaced an inactive prefix particle");
            for (int dim = 0; dim < kDimensions; ++dim) {
                detail::requireSphTree(
                    particlesPos(dim, particle) == prefixPositions[particle][dim],
                    "migration reordered positions in a saved rung prefix");
            }
        } else {
            detail::requireSphTree(
                int(identity) / 100 == 1 - hpxRank() && particlesRung(particle) == 2,
                "crossed active particle did not arrive from the opposite rank");
        }
    }
}
HPX_PLAIN_ACTION(
    verifyDistributedSphHierarchyMigration, VerifyDistributedSphHierarchyMigrationAction);

size_t initializeOutsideDomainSphHierarchy() {
    size_t outside = 0;
    auto const domain = domainsFindMyBox();
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        bool const active = gas >= 5;
        sphParticlesRung(gas) = active ? 2 : 0;
        sphParticlesPosition(0, gas) = active ? (hpxRank() == 0 ? 0.21 : 0.71) + 0.015 * (gas - 5) :
                                                (hpxRank() == 0 ? 0.75 : 0.25) + 0.01 * gas;
        Vector3<double> position;
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (dim != 0) {
                sphParticlesPosition(dim, gas) = 0.25;
            }
            sphParticlesVelocity(dim, gas) = 0.0f;
            position[dim] = sphParticlesPosition(dim, gas).toDouble();
        }
        outside += !active && !domain.contains(position);
    }
    return outside;
}
HPX_PLAIN_ACTION(initializeOutsideDomainSphHierarchy, InitializeOutsideDomainSphHierarchyAction);
#endif

void initializeSphRungChain(bool emptyRemoteRank) {
    ParticleInteger const first = hpxSize() == 1 || hpxRank() == 0 ? 0 : 4;
    ParticleInteger const count = emptyRemoteRank ? (hpxRank() == 0 ? 7 : 0) :
        hpxSize() == 1                            ? 7 :
        hpxRank() == 0                            ? 4 :
                                                    3;
    detail::initializeSphTreeParticles(count, 0);
    for (ParticleInteger gas = 0; gas < count; ++gas) {
        int const identity = first + gas;
        sphParticlesPosition(0, gas) = std::fmod(0.76 + 0.08 * identity, 1.0);
        sphParticlesPosition(1, gas) = sphParticlesPosition(2, gas) = 0.5;
        sphParticlesSmoothingLength(gas) = identity % 2 == 0 ? 0.04f : 0.09f;
        sphParticlesRung(gas) = identity == 0 ? 7 : 0;
        sphParticlesSetProposedRung(gas, sphParticlesRung(gas));
        sphParticlesDensities[gas] = 1.0f;
        sphParticlesEntropies[gas] = 1.0f;
    }
}
HPX_PLAIN_ACTION(initializeSphRungChain, InitializeSphRungChainAction);

void verifySphRungChain(bool coarsened) {
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        int const identity = int(std::lround(
            std::remainder(sphParticlesPosition(0, gas).toDouble() - 0.76, 1.0) / 0.08));
        detail::requireSphTree(
            identity >= 0 && identity < 7, "rung closure moved a source from the chain");
        detail::requireSphTree(sphParticlesRung(gas) == (identity == 0 ? 7 : 0),
            "rung closure changed a live rung before the common-boundary commit");
        detail::requireSphTree(sphParticlesProposedRung(gas) == (coarsened ? 0 : 7 - identity),
            "rung closure missed an inactive touching-kernel chain endpoint");
        // The next call must consume the exact caller seeds, even when they are
        // below the live rungs. Gravity/hydro coarsening policy belongs upstream.
        sphParticlesSetProposedRung(gas, 0);
    }
    detail::requireSphTree(!sphRungBalanceActive() && sphRungCandidatesData() == nullptr,
        "rung closure retained its transient candidate epoch");
}
HPX_PLAIN_ACTION(verifySphRungChain, VerifySphRungChainAction);

void verifySphLiveRung(int expectedRung) {
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        detail::requireSphTree(
            sphParticlesRung(gas) == expectedRung, "live rung mutation missed a gas particle");
    }
}
HPX_PLAIN_ACTION(verifySphLiveRung, VerifySphLiveRungAction);

void sphRungChainTest() {
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.compute.sphBucketSize = 1;
    options.hydro.enabled = true;
    options.hydro.viscosityEvolution = "morrisAndMonaghan";
    // Exercise dyadic live-rung commits when that integrator is available, even
    // when the enclosing application selected global stepping. A global-only
    // build still checks the neighbor closure, which does not change live rungs.
    if (options.run.timeIntegrator == "global") {
        if (integration::kBuildHasTraditional) {
            options.run.timeIntegrator = "traditional";
        } else if (integration::kBuildHasHierarchical && options.gravity.enabled &&
            options.gravity.method == "fmm") {
            options.run.timeIntegrator = "hierarchical";
        }
    }
    options.run.hierarchicalGravity = usesHierarchicalIntegration(options.run);
    options.run.minRung = 0;
    setOptions(options);
    for (int distribution = 0; distribution < (hpxSize() == 1 ? 1 : 2); ++distribution) {
        sphTreeDestroy(true);
        Vector<hpx::future<void>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(
                hpx::async<InitializeSphRungChainAction>(locality, distribution == 1));
        }
        for (auto& future : futures) {
            future.get();
        }
        SphTreeCreateParameters parameters{};
        parameters.minimumRung = 7;
        parameters.smoothingLengthMultiplier = 1.0f;
        auto const tree = sphTreeCreate(parameters);
        detail::requireSphTree(
            tree.numberOfActiveParticles == 1, "rung chain did not contain inactive endpoints");
        for (bool coarsened : {false, true}) {
            auto const result = sphBalanceRungs(tree.id, 1.0f, false);
            detail::requireSphTree(result.maximumRung == (coarsened ? 0 : 7) &&
                    result.passes == (coarsened ? 1 : 7) &&
                    result.changedParticles == (coarsened ? 0 : 6),
                "rung chain did not reach its expected multi-pass least fixed point");
            futures.clear();
            for (auto const& locality : hpxLocalities()) {
                futures.push_back(hpx::async<VerifySphRungChainAction>(locality, coarsened));
            }
            for (auto& future : futures) {
                future.get();
            }
        }
        auto const verifyLiveRung = [&](int rung) {
            futures.clear();
            for (auto const& locality : hpxLocalities()) {
                futures.push_back(hpx::async<VerifySphLiveRungAction>(locality, rung));
            }
            for (auto& future : futures) {
                future.get();
            }
        };
        if (options.run.timeIntegrator != "global") {
            sphCommitBalancedRungs();
            verifyLiveRung(0);
            particlesSetMinrung(2);
            verifyLiveRung(2);
            sphParticlesSelectMidpointRungs(0);
            verifyLiveRung(1);
        }
    }
    sphTreeDestroy(true);
    setOptions(savedOptions);
    std::cout << "SPH common-boundary rung closure regression passed: ranks=" << hpxSize()
              << " passes=7 inactive=6\n"
              << std::flush;
}

void distributedSphOctreeTest() {
    detail::requireSphTree(hpxSize() == 2, "distributed retained-tree fixture needs two ranks");
    auto options = getOptions();
    options.hydro.enabled = true;
    options.hydro.viscosityEvolution = "morrisAndMonaghan";
    options.chemistry.enabled = false;
    options.starFormation.enabled = false;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.compute.sphBucketSize = 2;
    options.initial.numberOfParticles = 16;
    setOptions(options);
    std::vector<hpx::future<void>> initialization;
    for (auto const& locality : hpxLocalities()) {
        initialization.push_back(hpx::async<InitializeDistributedSphOctreeAction>(locality));
    }
    for (auto& future : initialization) {
        future.get();
    }
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    SphTreeCreateParameters parameters{};
    parameters.minimumRung = 2;
    parameters.smoothingLengthMultiplier = 1.0;
    auto const built = sphTreeCreate(parameters);
    detail::verifySphTreeVolumeRatios(built);
    detail::requireSphTree(built.numberOfActiveParticles == 2 && !sphTreeShouldRebuild(2, 16),
        "distributed fixture is not below the rebuild threshold");
    Vector<TreeId> leaves;
    Vector<TreeId> pending{built.id};
    while (!pending.empty()) {
        auto const id = pending.back();
        pending.pop_back();
        auto const& node = *sphTreeGetNode(id);
        if (node.leaf) {
            leaves.push_back(id);
        } else {
            for (auto const child : node.children) {
                if (child.index >= 0) {
                    pending.push_back(child);
                }
            }
        }
    }
    // The traversal above intentionally populated remote cache lines before drift.
    std::vector<hpx::future<size_t>> drifts;
    for (auto const& locality : hpxLocalities()) {
        drifts.push_back(hpx::async<DriftDistributedSphOctreeAction>(locality));
    }
    size_t crossed = 0;
    for (auto& future : drifts) {
        crossed += future.get();
    }
    detail::requireSphTree(crossed == 2, "active particles did not cross both rank boundaries");
    sphParticlesClearCaches();
    auto const refreshed = sphTreeRefresh(parameters);
    detail::verifySphTreeVolumeRatios(refreshed);
    detail::requireSphTree(refreshed.id == built.id && refreshed.nodeCount == built.nodeCount &&
            refreshed.numberOfActiveParticles == built.numberOfActiveParticles,
        "distributed refresh changed topology or active counts");
    SphTreeNeighborParameters neighbors{};
    neighbors.operation = SphNeighborOperation::FindNeighbors;
    neighbors.interactionDirection = SphInteractionDirection::EitherSupport;
    neighbors.minimumRung = 2;
    sphTreeNeighbor(neighbors, refreshed.id, Vector<TreeId>{refreshed.id}).get();
    std::vector<hpx::future<void>> validation;
    for (auto const& locality : hpxLocalities()) {
        validation.push_back(hpx::async<VerifyDistributedSphOctreeAction>(locality, leaves));
    }
    for (auto& future : validation) {
        future.get();
    }
    sphTreeDestroy(true);
#if COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL && COSMICTIGER_ENABLE_FMM
    if (options.run.hierarchicalGravity) {
        particlesPushRungs();
        particlesSortByRung(2);
        for (bool afterMigration : {false, true}) {
            if (afterMigration) {
                domainsBegin(2);
                domainsEnd();
            }
            std::vector<hpx::future<void>> stateChecks;
            for (auto const& locality : hpxLocalities()) {
                stateChecks.push_back(hpx::async<VerifyDistributedSphHierarchyMigrationAction>(
                    locality, afterMigration));
            }
            for (auto& future : stateChecks) {
                future.get();
            }
        }
        particlesPopRungs();
        detail::requireSphTree(particlesCurrentRange().first == 0,
            "hierarchical migration corrupted the saved rung stack");
        std::cout << "SPH hierarchical migration regression passed: crossed=2 prefix=7\n"
                  << std::flush;

        // A threshold rebuild can occur before slow particles reach their own
        // migration event. More than one bucket of those particles now lies
        // outside each rank's domain; a domain-limited local octree could never
        // separate their x coordinates and would exceed its maximum depth.
        std::vector<hpx::future<size_t>> outsideCounts;
        for (auto const& locality : hpxLocalities()) {
            outsideCounts.push_back(
                hpx::async<InitializeOutsideDomainSphHierarchyAction>(locality));
        }
        size_t outside = 0;
        for (auto& future : outsideCounts) {
            outside += future.get();
        }
        detail::requireSphTree(outside == 10 && sphTreeShouldRebuild(6, 16),
            "out-of-domain fixture did not reach its active rebuild threshold");
        domainsBegin(2);
        domainsEnd();
        auto const rebuilt = sphTreeCreate(parameters);
        detail::requireSphTree(sphTreeAllValid() && rebuilt.numberOfActiveParticles == 6,
            "threshold rebuild failed with inactive gas outside its owner domain");
        // The earlier brute-force check cached remote positions. Match the
        // production SPH staging boundary after manual mutation and resorting.
        sphParticlesClearCaches();
        leaves.clear();
        pending = {rebuilt.id};
        while (!pending.empty()) {
            auto const id = pending.back();
            pending.pop_back();
            auto const& node = *sphTreeGetNode(id);
            if (node.leaf) {
                leaves.push_back(id);
            } else {
                for (auto const child : node.children) {
                    if (child.index >= 0) {
                        pending.push_back(child);
                    }
                }
            }
        }
        sphTreeNeighbor(neighbors, rebuilt.id, Vector<TreeId>{rebuilt.id}).get();
        validation.clear();
        for (auto const& locality : hpxLocalities()) {
            validation.push_back(hpx::async<VerifyDistributedSphOctreeAction>(locality, leaves));
        }
        for (auto& future : validation) {
            future.get();
        }
        sphTreeDestroy(true);
        std::cout << "SPH hierarchical out-of-domain rebuild regression passed: "
                     "active=6 total=16 inactive_outside=10\n"
                  << std::flush;
    }
#endif
    std::cout << "SPH octree distributed retain regression passed: active=2 total=16 crossed=2\n"
              << std::flush;
    sphRungChainTest();
}

void sphTreeTest() {
    if (hpxSize() > 1) {
        distributedSphOctreeTest();
        return;
    }
    detail::verifySphTreePackedMetadata();
    detail::verifyEmptySphBounds();
    detail::verifySphRebuildThreshold();
    detail::verifySphMidpointSchedule();
    detail::verifySphNeighborParameterArchive();
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
    params.smoothingLengthMultiplier = detail::kSphTreeSupportMultiplier;
    detail::verifyHybridSphTree(params);
    detail::verifySphTreeVolumeEvolution(params);

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
    detail::verifyRetainedSphOctree(params);

    sphTreeDestroy();
    detail::initializeSphTreeParticles(2, 0);
    for (int dim = 0; dim < kDimensions; ++dim) {
        particlesPos(dim, 0) = particlesPos(dim, 1) = 0.0;
    }
    // Two-particle nodes bisect x/y/z in turn; the fifth z split is depth 15.
    particlesPos(2, 1) = std::ldexp(1.0, -SphTreeNode::kMaximumDepth / kDimensions);
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
    detail::verifySphMidpointUpdates();
    detail::verifySphTemporalOrder();
    sphRungChainTest();
    std::cout << "SPH tree arena regression passed (CPU, one locality)\n" << std::flush;
}
