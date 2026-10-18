/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fmm_acceptance.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/tree.hpp>

namespace detail {

constexpr float kFmmWalkTheta = 0.8f;

void fmmWalkRequireClose(
    std::string const& label, double actual, double expected, double tolerance = 2.0e-4) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(
            label + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}

Array<Fixed32, NDIM> fmmWalkPosition(double x, double y, double z) {
    Array<Fixed32, NDIM> position;
    position[XDIM] = x;
    position[YDIM] = y;
    position[ZDIM] = z;
    return position;
}

Array<Fixed32, NDIM> fmmWalkDistributedPosition(size_t rank) {
    // Dyadic coordinates avoid fixed-point rounding, and unequal spacing keeps
    // the middle particle's force away from cancellation at float precision.
    double const offset = static_cast<double>(rank);
    return fmmWalkPosition(0.25 + offset / 32.0 + offset * offset / 256.0,
        0.375 + 3.0 * offset / 128.0, 0.4375 + offset / 64.0);
}

KickParameters fmmWalkParameters(bool useCuda, float theta = kFmmWalkTheta) {
    KickParameters params;
    params.htime = true;
    params.sign = 1.0f;
    params.minLevel = 0;
    params.minRung = 0;
    params.a = 1.0f;
    params.t0 = 1.0f;
    params.theta = theta;
    params.gravitySupportRadius = getOptions().gravitySupportRadius;
    params.eta = getOptions().eta;
    params.gm = 1.0f;
    params.velInit = false;
    params.saveForce = true;
    params.firstCall = true;
    params.gpu = useCuda;
    params.nodeLoad = 1.0f;
    params.ascending = false;
    params.descending = false;
    params.top = true;
    params.doPhi = true;
    return params;
}

void fmmWalkResetKickState() {
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        particlesRung(i) = 1;
        for (int dim = 0; dim < NDIM; dim++) {
            particlesVel(dim, i) = 0.0f;
            particlesGravityAcceleration(dim, i) = std::numeric_limits<float>::quiet_NaN();
        }
        particlesPot(i) = std::numeric_limits<float>::quiet_NaN();
    }
}

void fmmWalkSetParticlePosition(ParticleInteger index, Array<Fixed32, NDIM> const& position) {
    for (int dim = 0; dim < NDIM; dim++) {
        particlesPos(dim, index) = position[dim];
    }
}

void fmmWalkPrepareArena() {
    particlesResize(6);
    auto const center = fmmWalkPosition(0.3, 0.35, 0.4);
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        fmmWalkSetParticlePosition(i, center);
    }
    fmmWalkResetKickState();
    particlesSortByRung(0);
    treeCreate(TreeCreateParameters(0, kFmmWalkTheta, getOptions().gravitySupportRadius));
}

TreeId fmmWalkAddNode(Array<Fixed32, NDIM> const& position, float radius,
    Pair<ParticleInteger> particleRange, int depth, bool localRoot = false,
    TreeId left = TreeId{-1, -1}, TreeId right = TreeId{-1, -1}, bool spansTwoRanks = false,
    float maxSupportRadius = 0.0f, bool tightParticleBox = false) {
    TreeNode node{};
    node.pos = position;
    node.radius = radius;
    node.particleRange = particleRange;
    node.processRange = Pair<int>(hpxRank(), hpxRank() + (spansTwoRanks ? 2 : 1));
    node.depth = depth;
    node.localRoot = localRoot;
    node.leaf = left.index == -1;
    node.children[LEFT] = left;
    node.children[RIGHT] = right;
    node.maxGravitySupportRadius = particleRange.first == particleRange.second ?
        0.0f :
        std::max(float(getOptions().gravitySupportRadius), maxSupportRadius);
    if (!node.leaf) {
        for (int child = 0; child < NCHILD; child++) {
            node.maxGravitySupportRadius = std::max(node.maxGravitySupportRadius,
                treeGetNode(node.children[child])->maxGravitySupportRadius);
        }
    }
    MultiPos multipole;
    multipole.pos = position;
    multipole.multi = 0.0f;
    for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
        Array<float, NDIM> displacement;
        for (int dim = 0; dim < NDIM; dim++) {
            displacement[dim] = distance(particlesPos(dim, i), position[dim]);
        }
        auto const particleMultipole = p2M(displacement);
        for (int coefficient = 0; coefficient < MULTIPOLE_SIZE; coefficient++) {
            multipole.multi[coefficient] += particlesMass(i) * particleMultipole[coefficient];
        }
    }
    node.mpos = treeAddMultipole(multipole);
    for (int dim = 0; dim < NDIM; dim++) {
        node.box.begin[dim] = std::max(0.0, position[dim].toDouble() - radius);
        node.box.end[dim] = std::min(1.0, position[dim].toDouble() + radius);
        if (tightParticleBox && particleRange.first != particleRange.second) {
            node.box.begin[dim] = node.box.end[dim] =
                particlesPos(dim, particleRange.first).toDouble();
            for (ParticleInteger i = particleRange.first + 1; i < particleRange.second; i++) {
                node.box.begin[dim] =
                    std::min(node.box.begin[dim].toDouble(), particlesPos(dim, i).toDouble());
                node.box.end[dim] =
                    std::max(node.box.end[dim].toDouble(), particlesPos(dim, i).toDouble());
            }
        }
    }
    return TreeId{hpxRank(), int(treeAddRemote(node))};
}

void fmmWalkVerifyEwald(std::string const& label, TreeId sink, Vector<TreeId> const& checklist,
    Vector<TreeId> const& sourceLeaves, bool useCuda) {
    Expansion<float> reference;
    reference = 0.0f;
    cpuGravityCc(kGravityEwald, reference, sourceLeaves, sink, true);
    Array<float, NDIM> displacement;
    for (int dim = 0; dim < NDIM; dim++) {
        displacement[dim] = 0.0f;
    }
    auto const value = l2P(reference, displacement, true);
    std::array<double, NDIM> const expectedAcceleration = {-SCALE_FACTOR2 * value(1, 0, 0),
        -SCALE_FACTOR2 * value(0, 1, 0), -SCALE_FACTOR2 * value(0, 0, 1)};
    double const expectedEwaldPotential = SCALE_FACTOR1 * value(0, 0, 0);
    // The call's support may differ from the defaults used to build the tree.
    float const support = 2.0f * getOptions().gravitySupportRadius;
    double const selfPotential = selfPhi() * (1.0f / support);
    double const expectedPotential = selfPotential + expectedEwaldPotential;
    auto const* sinkNode = treeGetNode(sink);
    for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
        for (int potentialMode = 0; potentialMode < 2; potentialMode++) {
            std::string const path = label + (backend == 0 ? " CPU" : " CUDA") +
                (potentialMode == 0 ? " without potential" : " with potential");
            fmmWalkResetKickState();
            Expansion<float> local;
            local = 0.0f;
            auto params = fmmWalkParameters(backend != 0);
            params.gravitySupportRadius = support;
            params.doPhi = potentialMode != 0;
            auto const result =
                kick(params, local, sinkNode->pos, sink, Vector<TreeId>{}, checklist, nullptr);
            for (ParticleInteger i = sinkNode->particleRange.first;
                 i < sinkNode->particleRange.second; i++) {
                for (int dim = 0; dim < NDIM; dim++) {
                    fmmWalkRequireClose(path + " acceleration " + std::to_string(dim),
                        particlesGravityAcceleration(dim, i), expectedAcceleration[dim]);
                }
                if (params.doPhi) {
                    fmmWalkRequireClose(path + " potential", particlesPot(i), expectedPotential);
                    fmmWalkRequireClose(path + " Ewald potential", particlesPot(i) - selfPotential,
                        expectedEwaldPotential);
                }
            }
            if (params.doPhi) {
                fmmWalkRequireClose(path + " total potential", result.pot,
                    0.5 * sinkNode->numberOfParticles() * expectedPotential);
            }
        }
    }
}

void fmmWalkRejectedSourceLeaf(bool useCuda) {
    fmmWalkPrepareArena();
    auto const sinkPosition = fmmWalkPosition(0.3, 0.35, 0.4);
    auto const sourcePosition = fmmWalkPosition(0.38, 0.42, 0.45);
    fmmWalkSetParticlePosition(2, sourcePosition);
    auto const left = fmmWalkAddNode(sinkPosition, 0.0f, Pair<ParticleInteger>(0, 1), 5);
    auto const right = fmmWalkAddNode(sinkPosition, 0.0f, Pair<ParticleInteger>(1, 2), 5);
    // Conservative radii model unequal cell sizes without relying on a particular domain split.
    // Coincident sink centers isolate the walk from translations of nonzero local expansions.
    auto const sink =
        fmmWalkAddNode(sinkPosition, 0.3f, Pair<ParticleInteger>(0, 2), 4, true, left, right);
    auto const source = fmmWalkAddNode(sourcePosition, 0.04f, Pair<ParticleInteger>(2, 3), 4);
    fmmWalkVerifyEwald(
        "retained source leaf", sink, Vector<TreeId>{source}, Vector<TreeId>{source}, useCuda);
    treeDestroy();
}

void fmmWalkRepeatedSourceOpening(bool useCuda) {
    fmmWalkPrepareArena();
    auto const sinkPosition = fmmWalkPosition(0.3, 0.35, 0.4);
    auto const sourcePosition = fmmWalkPosition(0.4, 0.43, 0.46);
    Vector<TreeId> leaves;
    for (int i = 0; i < 4; i++) {
        auto const position = fmmWalkPosition(0.385 + 0.01 * i, 0.425 + 0.003 * i, 0.455);
        fmmWalkSetParticlePosition(i + 1, position);
        leaves.push_back(fmmWalkAddNode(position, 0.0f, Pair<ParticleInteger>(i + 1, i + 2), 6));
    }
    auto const left = fmmWalkAddNode(
        sourcePosition, 0.3f, Pair<ParticleInteger>(1, 3), 5, false, leaves[0], leaves[1]);
    auto const right = fmmWalkAddNode(
        sourcePosition, 0.3f, Pair<ParticleInteger>(3, 5), 5, false, leaves[2], leaves[3]);
    auto const source =
        fmmWalkAddNode(sourcePosition, 0.35f, Pair<ParticleInteger>(1, 5), 4, false, left, right);
    auto const sink = fmmWalkAddNode(sinkPosition, 0.0f, Pair<ParticleInteger>(0, 1), 4, true);
    fmmWalkVerifyEwald("repeated source opening", sink, Vector<TreeId>{source}, leaves, useCuda);
    treeDestroy();
}

void fmmWalkEmptyLocalRoot(bool useCuda) {
    particlesResize(0);
    auto const origin = fmmWalkPosition(0.3, 0.35, 0.4);
    auto const root = fmmWalkAddNode(origin, 0.0f, Pair<ParticleInteger>(0, 0), 0, true);
    for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
        Expansion<float> local;
        local = 0.0f;
        auto const result = kick(fmmWalkParameters(backend != 0), local, origin, root,
            Vector<TreeId>{root}, Vector<TreeId>{root}, nullptr);
        fmmWalkRequireClose("empty local root potential", result.pot, 0.0);
        fmmWalkRequireClose("empty local root kinetic energy", result.kin, 0.0);
        fmmWalkRequireClose("empty local root maximum rung", result.maxRung, 0.0);
    }
    treeDestroy();
}

void fmmWalkDeferredSourceHandoff(bool useCuda) {
    fmmWalkPrepareArena();
    auto const sinkPosition = fmmWalkPosition(0.2, 0.35, 0.4);
    auto const sourcePosition = fmmWalkPosition(0.65, 0.35, 0.4);
    auto const leftSourcePosition = fmmWalkPosition(0.6, 0.35, 0.4);
    auto const rightSourcePosition = fmmWalkPosition(0.7, 0.35, 0.4);
    fmmWalkSetParticlePosition(0, sinkPosition);
    fmmWalkSetParticlePosition(1, sinkPosition);
    fmmWalkSetParticlePosition(2, leftSourcePosition);
    fmmWalkSetParticlePosition(3, rightSourcePosition);
    auto const leftSink = fmmWalkAddNode(sinkPosition, 0.0f, Pair<ParticleInteger>(0, 1), 5);
    auto const rightSink = fmmWalkAddNode(sinkPosition, 0.0f, Pair<ParticleInteger>(1, 2), 5);
    auto const sink = fmmWalkAddNode(
        sinkPosition, 0.0f, Pair<ParticleInteger>(0, 2), 4, true, leftSink, rightSink);
    auto const leafSink = fmmWalkAddNode(sinkPosition, 0.0f, Pair<ParticleInteger>(0, 1), 4, true);
    auto const leftSource =
        fmmWalkAddNode(leftSourcePosition, 0.0f, Pair<ParticleInteger>(2, 3), 5);
    auto const rightSource =
        fmmWalkAddNode(rightSourcePosition, 0.0f, Pair<ParticleInteger>(3, 4), 5);
    // This synthetic parent has a two-rank range; its children are individually single-rank.
    auto const source = fmmWalkAddNode(sourcePosition, 0.05f, Pair<ParticleInteger>(2, 4), 4, false,
        leftSource, rightSource, true);
    if (treeGetNode(source)->isLocal()) {
        throw std::runtime_error("Deferred handoff source must span two ranks");
    }
    for (int mode = 0; mode < 2; mode++) {
        // Direct mode reaches CUDA at the sink children. Ewald mode exercises the
        // CPU-only leaf fallback and its empty GPU batch completion.
        bool const ewald = mode != 0;
        TreeId const target = ewald ? leafSink : sink;
        auto const* sinkNode = treeGetNode(target);
        std::array<std::array<float, NDIM>, 2> expectedAcceleration{};
        std::array<float, 2> expectedPotential{};
        double expectedTotalPotential = 0.0;
        for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
            fmmWalkResetKickState();
            Expansion<float> local;
            local = 0.0f;
            auto const params = fmmWalkParameters(backend != 0);
            Vector<TreeId> directSources = ewald ? Vector<TreeId>{} : Vector<TreeId>{source};
            Vector<TreeId> ewaldSources = ewald ? Vector<TreeId>{source} : Vector<TreeId>{};
            auto const result = kick(params, local, sinkPosition, target, std::move(directSources),
                std::move(ewaldSources), nullptr);
            if (backend == 0) {
                expectedTotalPotential = result.pot;
            } else {
                fmmWalkRequireClose(
                    "deferred handoff total potential", result.pot, expectedTotalPotential);
            }
            for (ParticleInteger i = sinkNode->particleRange.first;
                 i < sinkNode->particleRange.second; i++) {
                for (int dim = 0; dim < NDIM; dim++) {
                    if (backend == 0) {
                        expectedAcceleration[i][dim] = particlesGravityAcceleration(dim, i);
                    } else {
                        fmmWalkRequireClose("deferred handoff acceleration",
                            particlesGravityAcceleration(dim, i), expectedAcceleration[i][dim]);
                    }
                }
                if (backend == 0) {
                    expectedPotential[i] = particlesPot(i);
                } else {
                    fmmWalkRequireClose(
                        "deferred handoff potential", particlesPot(i), expectedPotential[i]);
                }
            }
        }
    }
    treeDestroy();
}

void fmmWalkSetHybridParticles(ParticleInteger secondParticleCount) {
    // Two symmetric grids have nonzero higher moments while remaining well
    // inside the conservative radii used by the synthetic leaves below.
    for (ParticleInteger i = 0; i < 40; i++) {
        fmmWalkSetParticlePosition(i,
            fmmWalkPosition(0.25 + (i % 5 - 2) / 128.0, 0.375 + ((i / 5) % 4 - 1.5) / 128.0,
                0.4375 + (i / 20 - 0.5) / 128.0));
    }
    int const columns = secondParticleCount == 40 ? 5 : 4;
    int const layers = secondParticleCount / (columns * 4);
    for (ParticleInteger i = 0; i < secondParticleCount; i++) {
        fmmWalkSetParticlePosition(40 + i,
            fmmWalkPosition(0.4375 + (i % columns - (columns - 1) * 0.5) / 64.0,
                0.375 + ((i / columns) % 4 - 1.5) / 64.0,
                0.4375 + (i / (columns * 4) - (layers - 1) * 0.5) / 64.0));
    }
}

void fmmWalkRequireInteraction(
    TreeId sink, TreeId source, float support, FmmInteraction expected, std::string const& label) {
    auto const* sinkNode = treeGetNode(sink);
    auto const* sourceNode = treeGetNode(source);
    Array<float, NDIM> displacement;
    for (int dim = 0; dim < NDIM; dim++) {
        displacement[dim] = distance(sinkNode->pos[dim], sourceNode->pos[dim]);
    }
    float const distanceSquared = std::fma(displacement[XDIM], displacement[XDIM],
        std::fma(displacement[YDIM], displacement[YDIM], sqr(displacement[ZDIM])));
    float const pairSupport = fmmPairSupportRadius(
        support, sinkNode->maxGravitySupportRadius, sourceNode->maxGravitySupportRadius);
    auto const acceptance = computeFmmAcceptance(distanceSquared, sinkNode->radius,
        sourceNode->radius, boxDistanceSquared(sinkNode->box, sourceNode->pos),
        boxDistanceSquared(sourceNode->box, sinkNode->pos), pairSupport,
        fmmInverseOpeningAngle(kFmmWalkTheta));
    auto const actual = selectFmmInteraction(acceptance.cellCell != 0.0f,
        acceptance.particleCell != 0.0f, acceptance.cellParticle != 0.0f,
        sinkNode->leaf && sourceNode->leaf, sinkNode->numberOfParticles(),
        sourceNode->numberOfParticles(), sinkNode->radius, sourceNode->radius);
    if (actual != expected) {
        throw std::runtime_error(label + ": synthetic geometry selected the wrong interaction");
    }
}

ForceVectors fmmWalkInteractionReference(
    TreeId sink, TreeId source, float support, FmmInteraction interaction) {
    auto const* sinkNode = treeGetNode(sink);
    auto const range = sinkNode->particleRange;
    ForceVectors reference(sinkNode->numberOfParticles());
    for (ParticleInteger i = range.first; i < range.second; i++) {
        auto const j = i - range.first;
        reference.gx[j] = reference.gy[j] = reference.gz[j] = 0.0f;
        reference.phi[j] = particlesMass(i) * selfPhi() / support;
    }
    Vector<TreeId> const sources{source};
    if (interaction == FmmInteraction::ParticleCell) {
        cpuGravityPc(reference, true, sink, sources);
    } else if (interaction == FmmInteraction::CellParticle ||
        interaction == FmmInteraction::CellCell) {
        Expansion<float> local;
        local = 0.0f;
        if (interaction == FmmInteraction::CellCell) {
            cpuGravityCc(kGravityDirect, local, sources, sink, true);
        } else {
            cpuGravityCp(local, sources, sink, true);
        }
        for (ParticleInteger i = range.first; i < range.second; i++) {
            Array<float, NDIM> displacement;
            for (int dim = 0; dim < NDIM; dim++) {
                displacement[dim] = distance(particlesPos(dim, i), sinkNode->pos[dim]);
            }
            auto const value = l2P(local, displacement, true);
            auto const j = i - range.first;
            reference.gx[j] -= SCALE_FACTOR2 * value(1, 0, 0);
            reference.gy[j] -= SCALE_FACTOR2 * value(0, 1, 0);
            reference.gz[j] -= SCALE_FACTOR2 * value(0, 0, 1);
            reference.phi[j] += SCALE_FACTOR1 * value(0, 0, 0);
        }
    } else if (interaction == FmmInteraction::ParticleParticle) {
        cpuGravityPp(reference, true, sink, sources, support);
    }
    return reference;
}

ForceVectors fmmWalkApplyInteraction(TreeId sink, TreeId source, float support, bool useCuda) {
    auto const range = treeGetNode(sink)->particleRange;
    auto const position = treeGetNode(sink)->pos;
    fmmWalkResetKickState();
    Expansion<float> local;
    local = 0.0f;
    auto params = fmmWalkParameters(useCuda);
    params.gravitySupportRadius = support;
    kick(params, local, position, sink, Vector<TreeId>{source}, Vector<TreeId>{}, nullptr);
    ForceVectors result(range.second - range.first);
    for (ParticleInteger i = range.first; i < range.second; i++) {
        auto const j = i - range.first;
        result.gx[j] = particlesGravityAcceleration(XDIM, i);
        result.gy[j] = particlesGravityAcceleration(YDIM, i);
        result.gz[j] = particlesGravityAcceleration(ZDIM, i);
        result.phi[j] = particlesPot(i);
    }
    return result;
}

void fmmWalkCompareForces(
    std::string const& label, ForceVectors const& actual, ForceVectors const& expected) {
    for (size_t i = 0; i < actual.gx.size(); i++) {
        fmmWalkRequireClose(label + " x acceleration", actual.gx[i], expected.gx[i]);
        fmmWalkRequireClose(label + " y acceleration", actual.gy[i], expected.gy[i]);
        fmmWalkRequireClose(label + " z acceleration", actual.gz[i], expected.gz[i]);
        fmmWalkRequireClose(label + " potential", actual.phi[i], expected.phi[i]);
    }
}

void fmmWalkRequireForceReciprocity(std::string const& label, std::array<TreeId, 2> const& sinks,
    std::array<ForceVectors const*, 2> const& forces) {
    std::array<double, NDIM> totalForce{};
    std::array<double, NDIM> absoluteForce{};
    double forceMagnitudeSum = 0.0;
    for (int direction = 0; direction < 2; direction++) {
        auto const range = treeGetNode(sinks[direction])->particleRange;
        auto const& values = *forces[direction];
        for (ParticleInteger i = range.first; i < range.second; i++) {
            auto const j = i - range.first;
            std::array<double, NDIM> const acceleration{values.gx[j], values.gy[j], values.gz[j]};
            double forceSquared = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                double const force = particlesMass(i) * acceleration[dim];
                totalForce[dim] += force;
                absoluteForce[dim] += std::abs(force);
                forceSquared += force * force;
            }
            forceMagnitudeSum += std::sqrt(forceSquared);
        }
    }
    double totalForceSquared = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        totalForceSquared += totalForce[dim] * totalForce[dim];
        fmmWalkRequireClose(label + " total-force reciprocity",
            totalForce[dim] / std::max(1.0, absoluteForce[dim]), 0.0, 2.0e-5);
    }
    // Normalize the norm of the net force by the sum of individual force norms.
    double const residual = std::sqrt(totalForceSquared) / std::max(1.0, forceMagnitudeSum);
    fmmWalkRequireClose(label + " vector total-force reciprocity", residual, 0.0, 2.0e-5);
    PRINT("FMM %s: normalized total-force residual = %.9e\n", label.c_str(), residual);
}

void fmmWalkReciprocalHybrid(bool useCuda) {
    struct HybridCase {
        char const* label;
        ParticleInteger secondParticleCount;
        float firstRadius;
        float secondRadius;
        std::array<FmmInteraction, 2> interactions;
    };
    std::array<HybridCase, 3> const cases{{
        {"unequal counts PC/CP", 64, 0.0625f, 0.09375f,
            {FmmInteraction::ParticleCell, FmmInteraction::CellParticle}},
        {"equal counts, unequal radii CP/PC", 40, 0.0625f, 0.09375f,
            {FmmInteraction::CellParticle, FmmInteraction::ParticleCell}},
        {"equal counts and radii PP", 40, 0.078125f, 0.078125f,
            {FmmInteraction::ParticleParticle, FmmInteraction::ParticleParticle}},
    }};
    for (auto const& testCase : cases) {
        ParticleInteger const particleCount = 40 + testCase.secondParticleCount;
        particlesResize(particleCount);
        fmmWalkSetHybridParticles(testCase.secondParticleCount);
        fmmWalkResetKickState();
        particlesSortByRung(0);
        treeCreate(TreeCreateParameters(0, kFmmWalkTheta, getOptions().gravitySupportRadius));
        // The ordinary build initializes the arenas and may reorder particles.
        // Restore the prescribed ranges before constructing the synthetic moments.
        fmmWalkSetHybridParticles(testCase.secondParticleCount);
        auto const first = fmmWalkAddNode(fmmWalkPosition(0.25, 0.375, 0.4375),
            testCase.firstRadius, Pair<ParticleInteger>(0, 40), 4, true);
        auto const second = fmmWalkAddNode(fmmWalkPosition(0.4375, 0.375, 0.4375),
            testCase.secondRadius, Pair<ParticleInteger>(40, particleCount), 4, true);
        std::array<TreeId, 2> const sinks{first, second};
        int const supportModes = testCase.secondParticleCount == 64 ? 2 : 1;
        for (int supportMode = 0; supportMode < supportModes; supportMode++) {
            // Small support permits the prescribed hybrid/tie policy. Large
            // support contains every cross pair and forces softened PP.
            float const support = supportMode == 0 ? 0.015625f : 0.375f;
            std::array<FmmInteraction, 2> const interactions = supportMode == 0 ?
                testCase.interactions :
                std::array<FmmInteraction, 2>{
                    FmmInteraction::ParticleParticle, FmmInteraction::ParticleParticle};
            std::array<ForceVectors, 2> references;
            std::array<ForceVectors, 2> cpuForces;
            std::string const mode = supportMode == 0 ? testCase.label : "softened PP fallback";
            for (int direction = 0; direction < 2; direction++) {
                fmmWalkRequireInteraction(
                    sinks[direction], sinks[1 - direction], support, interactions[direction], mode);
                references[direction] = fmmWalkInteractionReference(
                    sinks[direction], sinks[1 - direction], support, interactions[direction]);
            }
            for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
                std::string const label = mode + (backend == 0 ? " CPU" : " CUDA");
                std::array<ForceVectors, 2> backendForces;
                for (int direction = 0; direction < 2; direction++) {
                    auto& forces = backendForces[direction];
                    forces = fmmWalkApplyInteraction(
                        sinks[direction], sinks[1 - direction], support, backend != 0);
                    fmmWalkCompareForces(
                        label + " operator reference", forces, references[direction]);
                    if (backend == 0) {
                        cpuForces[direction] = forces;
                    } else {
                        fmmWalkCompareForces(
                            label + " CPU agreement", forces, cpuForces[direction]);
                    }
                }
                fmmWalkRequireForceReciprocity(
                    label, sinks, {&backendForces[0], &backendForces[1]});
                if (backend != 0) {
                    fmmWalkRequireForceReciprocity(
                        mode + " mixed CPU/CUDA", sinks, {&cpuForces[0], &backendForces[1]});
                }
            }
        }
        treeDestroy();
    }
}

void fmmWalkRequireDistinctForces(
    std::string const& label, ForceVectors const& direct, ForceVectors const& formerExpansion) {
    double largestDifference = 0.0;
    for (size_t i = 0; i < direct.gx.size(); i++) {
        largestDifference = std::max(largestDifference,
            std::abs(double(direct.gx[i]) - formerExpansion.gx[i]) /
                std::max(1.0, std::abs(double(direct.gx[i]))));
    }
    // A missing metadata check must fail the actual force comparison, not just
    // the helper-level interaction expectation above it.
    if (largestDifference < 0.01) {
        throw std::runtime_error(label + ": expansion is too close to PP to test metadata routing");
    }
}

void fmmWalkCellSupportBounds(bool useCuda) {
    struct SupportCase {
        char const* label;
        float firstRadius;
        float secondRadius;
        bool tightParticleBoxes;
        bool refine;
        std::array<FmmInteraction, 2> formerInteractions;
    };
    std::array<SupportCase, 4> const cases{{
        {"cell support CC fallback", 0.0625f, 0.078125f, false, false,
            {FmmInteraction::CellCell, FmmInteraction::CellCell}},
        {"cell support radial hybrid fallback", 0.0625f, 0.09375f, false, false,
            {FmmInteraction::ParticleCell, FmmInteraction::CellParticle}},
        {"cell support box-only hybrid fallback", 0.125f, 0.078125f, true, false,
            {FmmInteraction::ParticleCell, FmmInteraction::CellParticle}},
        {"cell support CC refinement", 0.0625f, 0.078125f, false, true,
            {FmmInteraction::CellCell, FmmInteraction::CellCell}},
    }};
    float const fixedSupport = 0.015625f;
    float const largeCellSupport = 0.375f;
    std::array<Array<Fixed32, NDIM>, 2> const centers{
        fmmWalkPosition(0.25, 0.375, 0.4375), fmmWalkPosition(0.4375, 0.375, 0.4375)};
    std::array<Array<Fixed32, NDIM>, 2> const particlePositions{
        fmmWalkPosition(0.3125, 0.375, 0.4375), fmmWalkPosition(0.359375, 0.375, 0.4375)};
    std::array<Pair<ParticleInteger>, 2> const ranges{
        Pair<ParticleInteger>(0, 40), Pair<ParticleInteger>(40, 104)};
    for (auto const& testCase : cases) {
        particlesResize(104);
        for (ParticleInteger i = 0; i < particlesSize(); i++) {
            fmmWalkSetParticlePosition(i, particlePositions[i < 40 ? 0 : 1]);
        }
        fmmWalkResetKickState();
        particlesSortByRung(0);
        treeCreate(TreeCreateParameters(0, kFmmWalkTheta, fixedSupport));
        // Concentrate each group near its facing cell edge so the omitted
        // expansion terms remain measurable. The enclosing bounds stay valid.
        for (ParticleInteger i = 0; i < particlesSize(); i++) {
            fmmWalkSetParticlePosition(i, particlePositions[i < 40 ? 0 : 1]);
        }
        std::array<float, 2> const radii{testCase.firstRadius, testCase.secondRadius};
        std::array<TreeId, 2> ordinary;
        for (int side = 0; side < 2; side++) {
            ordinary[side] = fmmWalkAddNode(centers[side], radii[side], ranges[side], 4, true,
                TreeId{-1, -1}, TreeId{-1, -1}, false, fixedSupport, testCase.tightParticleBoxes);
            if (testCase.tightParticleBoxes) {
                float const centerThreshold = std::fma(fmmInverseOpeningAngle(kFmmWalkTheta),
                    testCase.secondRadius, testCase.firstRadius);
                if (0.1875f > centerThreshold) {
                    throw std::runtime_error("Support fixture must require the box shortcut");
                }
            }
        }
        std::array<ForceVectors, 2> references;
        for (int direction = 0; direction < 2; direction++) {
            fmmWalkRequireInteraction(ordinary[direction], ordinary[1 - direction], fixedSupport,
                testCase.formerInteractions[direction], testCase.label);
            references[direction] = fmmWalkInteractionReference(ordinary[direction],
                ordinary[1 - direction], fixedSupport, FmmInteraction::ParticleParticle);
            auto const former = fmmWalkInteractionReference(ordinary[direction],
                ordinary[1 - direction], fixedSupport, testCase.formerInteractions[direction]);
            fmmWalkRequireDistinctForces(testCase.label, references[direction], former);
        }
        for (int largeSide = 0; largeSide < 2; largeSide++) {
            std::array<TreeId, 2> sinks;
            for (int side = 0; side < 2; side++) {
                float const cellSupport = side == largeSide ? largeCellSupport : fixedSupport;
                TreeId left{-1, -1};
                TreeId right{-1, -1};
                if (testCase.refine) {
                    auto const midpoint = (ranges[side].first + ranges[side].second) / 2;
                    left = fmmWalkAddNode(centers[side], radii[side],
                        Pair<ParticleInteger>(ranges[side].first, midpoint), 5, false,
                        TreeId{-1, -1}, TreeId{-1, -1}, false, cellSupport);
                    right = fmmWalkAddNode(centers[side], radii[side],
                        Pair<ParticleInteger>(midpoint, ranges[side].second), 5, false,
                        TreeId{-1, -1}, TreeId{-1, -1}, false, cellSupport);
                }
                sinks[side] =
                    fmmWalkAddNode(centers[side], radii[side], ranges[side], 4, true, left, right,
                        false, testCase.refine ? 0.0f : cellSupport, testCase.tightParticleBoxes);
                fmmWalkRequireClose("synthetic cell support maximum",
                    treeGetNode(sinks[side])->maxGravitySupportRadius, cellSupport, 0.0);
            }
            // Only opening metadata differs. PP still uses fixedSupport; this
            // tests conservative routing, not variable-softening PP physics.
            std::string const mode =
                std::string(testCase.label) + (largeSide == 0 ? " first large" : " second large");
            std::array<ForceVectors, 2> cpuForces;
            for (int direction = 0; direction < 2; direction++) {
                fmmWalkRequireInteraction(sinks[direction], sinks[1 - direction], fixedSupport,
                    FmmInteraction::ParticleParticle, mode);
            }
            for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
                std::string const label = mode + (backend == 0 ? " CPU" : " CUDA");
                std::array<ForceVectors, 2> backendForces;
                for (int direction = 0; direction < 2; direction++) {
                    auto& forces = backendForces[direction];
                    forces = fmmWalkApplyInteraction(
                        sinks[direction], sinks[1 - direction], fixedSupport, backend != 0);
                    fmmWalkCompareForces(
                        label + " fixed-support PP reference", forces, references[direction]);
                    if (backend == 0) {
                        cpuForces[direction] = forces;
                    } else {
                        fmmWalkCompareForces(
                            label + " CPU agreement", forces, cpuForces[direction]);
                    }
                }
                fmmWalkRequireForceReciprocity(
                    label, sinks, {&backendForces[0], &backendForces[1]});
                if (backend != 0) {
                    fmmWalkRequireForceReciprocity(
                        mode + " mixed CPU/CUDA", sinks, {&cpuForces[0], &backendForces[1]});
                }
            }
        }
        treeDestroy();
    }
}

void fmmWalkCollectLeaves(TreeId id, Vector<TreeId>& leaves) {
    auto const* node = treeGetNode(id);
    float expectedSupportRadius = 0.0f;
    if (node->leaf) {
        if (node->numberOfParticles()) {
            leaves.push_back(id);
            expectedSupportRadius = getOptions().gravitySupportRadius;
        }
    } else {
        for (int child = 0; child < NCHILD; child++) {
            fmmWalkCollectLeaves(node->children[child], leaves);
            expectedSupportRadius = std::max(
                expectedSupportRadius, treeGetNode(node->children[child])->maxGravitySupportRadius);
        }
    }
    fmmWalkRequireClose("distributed cell support maximum", node->maxGravitySupportRadius,
        expectedSupportRadius, 0.0);
}

struct FmmWalkPeriodicField {
    double potential = 0.0;
    std::array<double, NDIM> acceleration{};
};

FmmWalkPeriodicField fmmWalkDirectPeriodicReference(
    size_t sink, Vector<Array<Fixed32, NDIM>> const& positions, Vector<double> const& masses) {
    // Independently sum the zero-mean periodic Newtonian Green function.
    // These larger real/reciprocal cubes converge beyond the production
    // alpha=2 truncations without using any generated FMM coefficient.
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kAlpha = 2.0;
    constexpr int kCutoff = 4;
    double const inverseSqrtPi = 1.0 / std::sqrt(kPi);
    FmmWalkPeriodicField result;
    for (size_t source = 0; source < positions.size(); source++) {
        double const mass = masses[source];
        std::array<double, NDIM> separation;
        for (int dim = 0; dim < NDIM; dim++) {
            separation[dim] = positions[sink][dim].toDouble() - positions[source][dim].toDouble();
            separation[dim] -= std::nearbyint(separation[dim]);
        }
        result.potential += mass * kPi / (kAlpha * kAlpha);
        if (source == sink) {
            // The direct softened self term is canceled by the kick's self
            // subtraction. Retain the finite periodic self-image potential.
            result.potential += mass * 2.0 * kAlpha * inverseSqrtPi;
        }
        for (int x = -kCutoff; x <= kCutoff; x++) {
            for (int y = -kCutoff; y <= kCutoff; y++) {
                for (int z = -kCutoff; z <= kCutoff; z++) {
                    bool const origin = x == 0 && y == 0 && z == 0;
                    std::array<double, NDIM> const displacement{
                        separation[0] + x, separation[1] + y, separation[2] + z};
                    double const radiusSquared = displacement[0] * displacement[0] +
                        displacement[1] * displacement[1] + displacement[2] * displacement[2];
                    if (!(source == sink && origin)) {
                        double const radius = std::sqrt(radiusSquared);
                        if (origin && radius <= getOptions().gravitySupportRadius) {
                            throw std::runtime_error(
                                "Periodic FMM reference requires separated particle pairs");
                        }
                        double const screened = std::erfc(kAlpha * radius) / radius;
                        double const radial = (screened +
                                                  2.0 * kAlpha * inverseSqrtPi *
                                                      std::exp(-kAlpha * kAlpha * radiusSquared)) /
                            radiusSquared;
                        result.potential -= mass * screened;
                        for (int dim = 0; dim < NDIM; dim++) {
                            result.acceleration[dim] -= mass * radial * displacement[dim];
                        }
                    }
                    if (!origin) {
                        std::array<double, NDIM> const wave{
                            2.0 * kPi * x, 2.0 * kPi * y, 2.0 * kPi * z};
                        double const waveSquared = x * x + y * y + z * z;
                        double const coefficient =
                            std::exp(-kPi * kPi * waveSquared / (kAlpha * kAlpha)) /
                            (kPi * waveSquared);
                        double const phase = wave[0] * separation[0] + wave[1] * separation[1] +
                            wave[2] * separation[2];
                        result.potential -= mass * coefficient * std::cos(phase);
                        for (int dim = 0; dim < NDIM; dim++) {
                            result.acceleration[dim] -=
                                mass * coefficient * std::sin(phase) * wave[dim];
                        }
                    }
                }
            }
        }
    }
    return result;
}

FmmWalkPeriodicField fmmWalkDirectPeriodicReference(ParticleInteger sink) {
    auto const range = particlesCurrentRange();
    Vector<Array<Fixed32, NDIM>> positions(range.second - range.first);
    Vector<double> masses(positions.size());
    for (ParticleInteger source = range.first; source < range.second; source++) {
        for (int dim = 0; dim < NDIM; dim++) {
            positions[source - range.first][dim] = particlesPos(dim, source);
        }
        masses[source - range.first] = particlesMass(source);
    }
    return fmmWalkDirectPeriodicReference(sink - range.first, positions, masses);
}

void fmmWalkPeriodicBackground(bool useCuda) {
    constexpr float kTheta = 0.1f;
    constexpr float kUntouchedPotential = 12345.0f;
    constexpr float kUntouchedAcceleration = 6789.0f;
    constexpr std::array<std::array<double, NDIM>, 6> kPositions{
        {{0.96, 0.72, 0.82}, {0.04, 0.76, 0.88}, {0.11, 0.68, 0.91}, {0.89, 0.91, 0.73},
            {0.72, 0.62, 0.69}, {0.84, 0.84, 0.95}}};
    std::array<double, NDIM> const translation{0.20, 0.31, 0.17};
    for (int scenario = 0; scenario < 3; scenario++) {
        bool const translated = scenario == 1;
        bool const selectedSubset = scenario == 2;
        particlesResize(kPositions.size() + (selectedSubset ? 2 : 0));
        for (size_t i = 0; i < kPositions.size(); i++) {
            Array<Fixed32, NDIM> position;
            for (int dim = 0; dim < NDIM; dim++) {
                double const coordinate = kPositions[i][dim] + translated * translation[dim];
                position[dim] = coordinate - std::floor(coordinate);
            }
            fmmWalkSetParticlePosition(i, position);
        }
        if (selectedSubset) {
            // HOLD trees contain only their current rung range. These two
            // excluded sources must not enter the root potential constant.
            fmmWalkSetParticlePosition(6, fmmWalkPosition(0.21, 0.13, 0.25));
            fmmWalkSetParticlePosition(7, fmmWalkPosition(0.44, 0.27, 0.16));
        }
        fmmWalkResetKickState();
        particlesSortByRung(0);
        if (selectedSubset) {
            particlesRung(6) = particlesRung(7) = 0;
            particlesSortByRung(1);
        }
        auto const range = particlesCurrentRange();
        if (range.second - range.first != kPositions.size()) {
            throw std::runtime_error("Periodic background regression selected the wrong sources");
        }
        auto const built =
            treeCreate(TreeCreateParameters(0, kTheta, getOptions().gravitySupportRadius));
        if (built.id.proc != 0 || built.id.index != 0) {
            throw std::runtime_error("Periodic background regression requires the global root");
        }
        double dipoleSquared = 0.0;
        for (int dim = 0; dim < NDIM; dim++) {
            double dipole = 0.0;
            for (ParticleInteger i = range.first; i < range.second; i++) {
                dipole += particlesMass(i) *
                    (particlesPos(dim, i).toDouble() - built.pos[dim].toDouble());
            }
            dipoleSquared += dipole * dipole;
        }
        if (dipoleSquared < 1.0e-4 || (translated == 0 && built.pos[YDIM].toDouble() <= 0.5)) {
            throw std::runtime_error("Periodic background regression lost its off-center dipole");
        }
        Vector<FmmWalkPeriodicField> reference(particlesSize());
        for (ParticleInteger i = range.first; i < range.second; i++) {
            reference[i] = fmmWalkDirectPeriodicReference(i);
        }
        for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
            for (int potentialMode = 0; potentialMode < 2; potentialMode++) {
                std::string const label = std::string("periodic root background ") +
                    (translated ? "translated " : (selectedSubset ? "rung subset " : "original ")) +
                    (backend ? "CUDA " : "CPU ") + (potentialMode ? "potential" : "force only");
                fmmWalkResetKickState();
                // Finite sentinels remain observable under production fast-math;
                // isnan checks may otherwise be folded away by the compiler.
                for (ParticleInteger i = 0; i < particlesSize(); i++) {
                    particlesPot(i) = kUntouchedPotential;
                    for (int dim = 0; dim < NDIM; dim++) {
                        particlesGravityAcceleration(dim, i) = kUntouchedAcceleration;
                    }
                }
                Expansion<float> local;
                local = 0.0f;
                auto params = fmmWalkParameters(backend != 0, kTheta);
                params.doPhi = potentialMode != 0;
                auto const origin = fmmWalkPosition(0.0, 0.0, 0.0);
                auto const result = kick(params, local, origin, built.id, Vector<TreeId>{built.id},
                    Vector<TreeId>{built.id}, nullptr);
                double totalPotential = 0.0;
                for (ParticleInteger i = range.first; i < range.second; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        fmmWalkRequireClose(label + " acceleration " + std::to_string(i) + ":" +
                                std::to_string(dim),
                            particlesGravityAcceleration(dim, i), reference[i].acceleration[dim]);
                    }
                    if (params.doPhi) {
                        fmmWalkRequireClose(label + " potential " + std::to_string(i),
                            particlesPot(i), reference[i].potential);
                    }
                    totalPotential += 0.5 * particlesMass(i) * reference[i].potential;
                }
                for (ParticleInteger i = 0; i < range.first; i++) {
                    if (particlesPot(i) != kUntouchedPotential) {
                        throw std::runtime_error(label + " changed an excluded particle");
                    }
                    for (int dim = 0; dim < NDIM; dim++) {
                        if (particlesGravityAcceleration(dim, i) != kUntouchedAcceleration) {
                            throw std::runtime_error(label + " changed an excluded particle");
                        }
                    }
                }
                if (params.doPhi) {
                    fmmWalkRequireClose(label + " total potential", result.pot, totalPotential);
                }
            }
        }
        treeDestroy();
        particlesSortByRung(0);
    }
}

}    // namespace detail

void fmmWalkInitializeDistributedParticles() {
    particlesResize(1);
    detail::fmmWalkSetParticlePosition(0, detail::fmmWalkDistributedPosition(hpxRank()));
    detail::fmmWalkResetKickState();
}
HPX_PLAIN_ACTION(
    fmmWalkInitializeDistributedParticles, FmmWalkInitializeDistributedParticlesAction);

void fmmWalkResetDistributedKick() {
    detail::fmmWalkResetKickState();
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        particlesPot(i) = 12345.0f;
        for (int dim = 0; dim < NDIM; dim++) {
            particlesGravityAcceleration(dim, i) = 6789.0f;
        }
    }
}
HPX_PLAIN_ACTION(fmmWalkResetDistributedKick, FmmWalkResetDistributedKickAction);

double fmmWalkVerifyDistributedForces(TreeId root) {
    if (particlesSize() != 1) {
        throw std::runtime_error("Shallow FMM handoff regression requires one particle per rank");
    }
    Vector<TreeId> leaves;
    detail::fmmWalkCollectLeaves(root, leaves);
    TreeId sink{-1, -1};
    Vector<TreeId> sources;
    for (auto const id : leaves) {
        if (treeGetNode(id)->processRange.first == hpxRank()) {
            sink = id;
        } else {
            sources.push_back(id);
        }
    }
    if (sink.index == -1 || sources.size() != hpxSize() - 1) {
        throw std::runtime_error("Shallow FMM handoff regression lost a source or sink leaf");
    }
    ForceVectors reference(1);
    reference.gx[0] = reference.gy[0] = reference.gz[0] = reference.phi[0] = 0.0f;
    cpuGravityPp(reference, true, sink, sources, getOptions().gravitySupportRadius);
    detail::fmmWalkRequireClose("distributed handoff x acceleration",
        particlesGravityAcceleration(XDIM, 0), reference.gx[0]);
    detail::fmmWalkRequireClose("distributed handoff y acceleration",
        particlesGravityAcceleration(YDIM, 0), reference.gy[0]);
    detail::fmmWalkRequireClose("distributed handoff z acceleration",
        particlesGravityAcceleration(ZDIM, 0), reference.gz[0]);
    detail::fmmWalkRequireClose("distributed handoff potential", particlesPot(0), reference.phi[0]);
    return reference.phi[0];
}
HPX_PLAIN_ACTION(fmmWalkVerifyDistributedForces, FmmWalkVerifyDistributedForcesAction);

double fmmWalkVerifyDistributedPeriodicForces() {
    if (particlesSize() != 1) {
        throw std::runtime_error("Periodic FMM regression requires one particle per rank");
    }
    Vector<Array<Fixed32, NDIM>> positions(hpxSize());
    Vector<double> masses(hpxSize(), 1.0);
    size_t sink = positions.size();
    for (size_t rank = 0; rank < positions.size(); rank++) {
        positions[rank] = detail::fmmWalkDistributedPosition(rank);
        bool matches = true;
        for (int dim = 0; dim < NDIM; dim++) {
            matches = matches && positions[rank][dim].raw() == particlesPos(dim, 0).raw();
        }
        if (matches) {
            sink = rank;
        }
    }
    if (sink == positions.size()) {
        throw std::runtime_error("Distributed periodic fixture lost a particle");
    }
    auto const reference = detail::fmmWalkDirectPeriodicReference(sink, positions, masses);
    for (int dim = 0; dim < NDIM; dim++) {
        detail::fmmWalkRequireClose("distributed periodic acceleration " + std::to_string(dim),
            particlesGravityAcceleration(dim, 0), reference.acceleration[dim]);
    }
    // This close-spaced fixture has a small but nonzero source trace. Check
    // potential tightly enough to detect omission of another rank's trace.
    detail::fmmWalkRequireClose(
        "distributed periodic potential", particlesPot(0), reference.potential, 2.0e-5);
    return reference.potential;
}
HPX_PLAIN_ACTION(
    fmmWalkVerifyDistributedPeriodicForces, FmmWalkVerifyDistributedPeriodicForcesAction);

void fmmWalkTest() {
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    bool const useCuda = options.cuda;
    options.sph = false;
    options.vsoft = false;
    options.gravity = true;
    options.saveForce = true;
    options.stars = false;
    options.doGroups = false;
    options.doTracers = false;
    options.bucketSize = 64;
    options.gravitySupportRadius = 0.005;
    options.numberOfParticles = hpxSize() == 1 ? 6 : hpxSize();
    setOptions(options);
    if (hpxSize() == 1) {
        detail::fmmWalkRejectedSourceLeaf(useCuda);
        detail::fmmWalkRepeatedSourceOpening(useCuda);
        detail::fmmWalkEmptyLocalRoot(useCuda);
        detail::fmmWalkDeferredSourceHandoff(useCuda);
        detail::fmmWalkReciprocalHybrid(useCuda);
        detail::fmmWalkCellSupportBounds(useCuda);
        detail::fmmWalkPeriodicBackground(useCuda);
    } else {
        Vector<hpx::future<void>> pending;
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<FmmWalkInitializeDistributedParticlesAction>(locality));
        }
        for (auto& future : pending) {
            future.get();
        }
        domainsRebound();
        domainsBegin(0);
        domainsEnd();
        particlesSortByRung(0);
        auto const built = treeCreate(TreeCreateParameters(0, 0.1, options.gravitySupportRadius));
        // Three ranks give a one-rank left child and a global right child at the first handoff.
        // First force nearby sources open against the direct reference, then
        // include Ewald to exercise the distributed root-background reduction.
        auto const* root = treeGetNode(built.id);
        if (!treeGetNode(root->children[LEFT])->isLocal() ||
            treeGetNode(root->children[RIGHT])->isLocal()) {
            throw std::runtime_error("Shallow FMM test requires a 1-rank/2-rank root split");
        }
        for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
            for (int periodic = 0; periodic < 2; periodic++) {
                pending.clear();
                for (auto const& locality : hpxLocalities()) {
                    pending.push_back(hpx::async<FmmWalkResetDistributedKickAction>(locality));
                }
                for (auto& future : pending) {
                    future.get();
                }
                Expansion<float> local;
                local = 0.0f;
                auto const origin = detail::fmmWalkPosition(0.0, 0.0, 0.0);
                auto const ewaldSources = periodic ? Vector<TreeId>{built.id} : Vector<TreeId>{};
                auto const result = kick(detail::fmmWalkParameters(backend != 0, 0.1f), local,
                    origin, built.id, Vector<TreeId>{built.id}, ewaldSources, nullptr);
                Vector<hpx::future<double>> references;
                for (auto const& locality : hpxLocalities()) {
                    references.push_back(periodic ?
                            hpx::async<FmmWalkVerifyDistributedPeriodicForcesAction>(locality) :
                            hpx::async<FmmWalkVerifyDistributedForcesAction>(locality, built.id));
                }
                double expectedPotential = 0.0;
                for (auto& future : references) {
                    expectedPotential += 0.5 * future.get();
                }
                detail::fmmWalkRequireClose(periodic ? "distributed periodic total potential" :
                                                       "distributed handoff total potential",
                    result.pot, expectedPotential, periodic ? 2.0e-5 : 2.0e-4);
            }
        }
        treeDestroy();
    }
    setOptions(originalOptions);
    PRINT("FMM walk regression passed (%s, %i rank(s))\n", useCuda ? "CPU and CUDA" : "CPU",
        hpxSize());
}
