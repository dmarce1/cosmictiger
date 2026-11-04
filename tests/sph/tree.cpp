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
#include <cosmictiger/core/sph_midpoint.hpp>
#include <cosmictiger/gravity/kick.hpp>
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
    options.chemistry.enabled = false;
    options.starFormation.enabled = false;
    options.gravity.enabled = false;
    options.initial.gasMass = 0.5;
    setOptions(options);
    bool const savedEntropyInitialized = sphParticlesEntropyInitialized;
    double const savedThermalScale = sphParticlesThermalScaleFactor;
    sphParticlesEntropyInitialized = false;
    initializeSphTreeParticles(2, 0);
    for (ParticleInteger gas = 0; gas < 2; ++gas) {
        sphParticlesDensities[gas] = 1.0f;
        sphParticlesRung(gas) = gas == 0 ? 0 : 2;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            sphParticlesPosition(dimension, gas) = gas == 0 ? 0.875 : 0.25;
            sphParticlesVelocity(dimension, gas) = 0.0f;
        }
    }
    sphParticlesPrepareThermodynamics(1.0, true);
    for (ParticleInteger gas = 0; gas < 2; ++gas) {
        sphParticlesVelocity(0, gas) = gas == 0 ? 0.25f : 0.5f;
        sphParticlesEntropy(gas) = 2.0f;
        sphParticlesEntropyRate(gas) = gas == 0 ? 0.25f : 0.5f;
        sphParticlesProposedRungs[gas] = sphParticlesRung(gas) + 1;
        sphParticlesAccelerationState(gas) = Vector3<float>(gas == 0 ? 0.5f : 1.0f, 0.0f, 0.0f);
    }
    requireSphTree(sphMidpointEnabled() && !sphMidpointEventActive(),
        "midpoint fixture did not begin outside a hydro event");
    sphMidpointBeginEvent();
    sphParticlesApplyMidpointUpdates(2, 0.25);
    requireSphTree(sphParticlesVelocity(0, 0) == 0.25f && sphParticlesVelocity(0, 1) == 0.75f &&
            sphParticlesEntropy(0) == 2.0f && sphParticlesEntropy(1) == 2.125f &&
            sphParticlesProposedRungs[1] == 3,
        "fine midpoint updated the slow target or discarded its next hydro bound");
    sphParticlesApplyMidpointUpdates(2, 0.0);
    sphMidpointEndEvent();
    // A signed gravity correction changes only live momentum. Legacy opening,
    // closing and diagnostic hydro phases must not add another hydro interval.
    sphParticlesVelocity(0, 1) -= 0.125f;
    for (int phase = 0; phase < 4; ++phase) {
        sphParticlesApplyUpdates(0, phase, 1.0f, 0.25f);
    }
    requireSphTree(!sphMidpointEventActive() && sphParticlesVelocity(0, 0) == 0.25f &&
            sphParticlesVelocity(0, 1) == 0.625f && sphParticlesEntropy(0) == 2.0f &&
            sphParticlesEntropy(1) == 2.125f,
        "a gravity-only interval applied a duplicate hydro kick");
    ParticleGlobalRange range;
    range.proc = hpxRank();
    range.range = Pair<ParticleInteger>(0, 2);
    float velocityX[2], velocityY[2], velocityZ[2], internalEnergy[2];
    sphParticlesGlobalReadVelocities(range, velocityX, velocityY, velocityZ, 0);
    sphParticlesGlobalReadInternalEnergyAndSupportRadius(range, internalEnergy, nullptr, 0);
    requireSphTree(velocityX[0] == 0.25f && velocityX[1] == 0.625f &&
            std::abs(internalEnergy[0] -
                sph_entropy::storedEnergy(2.0, 1.0, options.hydro.gamma, 1.0)) < 1e-6 &&
            std::abs(internalEnergy[1] -
                sph_entropy::storedEnergy(2.125, 1.0, options.hydro.gamma, 1.0)) < 1e-6,
        "midpoint source staging used stale velocity or entropy predictors");
    sphMidpointBeginEvent();
    sphParticlesApplyMidpointUpdates(0, 1.0);
    sphMidpointEndEvent();
    requireSphTree(sphParticlesVelocity(0, 0) == 0.75f && sphParticlesVelocity(0, 1) == 0.625f &&
            sphParticlesEntropy(0) == 2.25f && sphParticlesEntropy(1) == 2.125f,
        "coarse midpoint also kicked fine targets");
    sphMidpointBeginEvent();
    sphParticlesApplyMidpointUpdates(-1, 0.125);
    sphMidpointEndEvent();
    requireSphTree(sphParticlesVelocity(0, 0) == 0.8125f && sphParticlesVelocity(0, 1) == 0.75f &&
            sphParticlesEntropy(0) == 2.28125f && sphParticlesEntropy(1) == 2.1875f,
        "global midpoint failed to kick all gas once");
    for (ParticleInteger gas = 0; gas < 2; ++gas) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            requireSphTree(
                sphParticlesPosition(dimension, gas).toDouble() == (gas == 0 ? 0.875 : 0.25),
                "fixed-position midpoint hydro moved a particle");
        }
    }
    sphParticlesResize(0, false);
    particlesResize(0);
    setOptions(savedOptions);
    sphParticlesEntropyInitialized = savedEntropyInitialized;
    sphParticlesThermalScaleFactor = savedThermalScale;
    std::cout << "SPH midpoint target, source and positive-kick regression passed\n" << std::flush;
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
            sphParticlesSupportRadius(particle) = 0.125f;
        }
    }
    requireSphTree(crossedPartition && crossedPeriodicBoundary,
        "drift fixture did not cross both spatial and periodic boundaries");
    auto const refreshed = sphTreeRefresh(params);
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
                        params.supportRadiusMultiplier * sphParticlesSupportRadius(particle);
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
        sphParticlesSupportRadius(particle) = 0.09f;
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
            for (int dim = 0; dim < kDimensions; ++dim) {
                sphParticlesAccelerationState(gas)[dim] = identity + 0.25f * (dim + 1);
            }
            sphParticlesEntropyRate(gas) = identity / 64.0f;
            sphParticlesDensities[gas] = identity + 0.5f;
            if (sphParticlesBulkViscosityCoefficients) {
                sphParticlesBulkViscosityCoefficients[gas] = identity / 256.0f;
            }
            // The last midpoint's hydro timestep bound must survive migration
            // until the next gravity boundary consumes it.
            sphParticlesProposedRungs[gas] = particlesRung(particle) + 1;
            if (particle < range.first) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    prefixPositions[particle][dim] = particlesPos(dim, particle);
                }
            }
            continue;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            detail::requireSphTree(
                sphParticlesAccelerationState(gas)[dim] == identity + 0.25f * (dim + 1),
                "migration changed the midpoint hydro acceleration");
        }
        detail::requireSphTree(sphParticlesEntropyRate(gas) == identity / 64.0f &&
                sphParticlesDensities[gas] == identity + 0.5f &&
                (!sphParticlesBulkViscosityCoefficients ||
                    sphParticlesBulkViscosityCoefficients[gas] == identity / 256.0f) &&
                sphParticlesProposedRungs[gas] == particlesRung(particle) + 1,
            "migration changed midpoint gas rates or the hydro rung bound");
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

void distributedSphOctreeTest() {
    detail::requireSphTree(hpxSize() == 2, "distributed retained-tree fixture needs two ranks");
    auto options = getOptions();
    options.hydro.enabled = true;
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
    parameters.supportRadiusMultiplier = 1.0;
    auto const built = sphTreeCreate(parameters);
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
    detail::requireSphTree(refreshed.id == built.id && refreshed.nodeCount == built.nodeCount &&
            refreshed.numberOfActiveParticles == built.numberOfActiveParticles,
        "distributed refresh changed topology or active counts");
    SphTreeNeighborParameters neighbors{};
    neighbors.runType = kSphTreeNeighborNeighbors;
    neighbors.innerSelection = kSphInteractionsIJ;
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
    detail::verifyRetainedSphOctree(params);

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
    detail::verifySphMidpointUpdates();
    std::cout << "SPH tree arena regression passed (CPU, one locality)\n" << std::flush;
}
