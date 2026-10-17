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
    TreeId left = TreeId{-1, -1}, TreeId right = TreeId{-1, -1}, bool spansTwoRanks = false) {
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

void fmmWalkCollectLeaves(TreeId id, Vector<TreeId>& leaves) {
    auto const* node = treeGetNode(id);
    if (node->leaf) {
        if (node->numberOfParticles()) {
            leaves.push_back(id);
        }
    } else {
        for (int child = 0; child < NCHILD; child++) {
            fmmWalkCollectLeaves(node->children[child], leaves);
        }
    }
}

}    // namespace detail

void fmmWalkInitializeDistributedParticles() {
    particlesResize(1);
    detail::fmmWalkSetParticlePosition(0,
        detail::fmmWalkPosition(
            0.2 + 0.02 * hpxRank(), 0.3 + 0.025 * hpxRank(), 0.4 + 0.015 * hpxRank()));
    detail::fmmWalkResetKickState();
}
HPX_PLAIN_ACTION(
    fmmWalkInitializeDistributedParticles, FmmWalkInitializeDistributedParticlesAction);

void fmmWalkResetDistributedKick() {
    detail::fmmWalkResetKickState();
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
        // Omit Ewald here and force the nearby multiparticle sources open to a direct reference.
        auto const* root = treeGetNode(built.id);
        if (!treeGetNode(root->children[LEFT])->isLocal() ||
            treeGetNode(root->children[RIGHT])->isLocal()) {
            throw std::runtime_error("Shallow FMM test requires a 1-rank/2-rank root split");
        }
        for (int backend = 0; backend < (useCuda ? 2 : 1); backend++) {
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
            auto const result = kick(detail::fmmWalkParameters(backend != 0, 0.1f), local, origin,
                built.id, Vector<TreeId>{built.id}, Vector<TreeId>{}, nullptr);
            Vector<hpx::future<double>> references;
            for (auto const& locality : hpxLocalities()) {
                references.push_back(
                    hpx::async<FmmWalkVerifyDistributedForcesAction>(locality, built.id));
            }
            double expectedPotential = 0.0;
            for (auto& future : references) {
                expectedPotential += 0.5 * future.get();
            }
            detail::fmmWalkRequireClose(
                "distributed handoff total potential", result.pot, expectedPotential);
        }
        treeDestroy();
    }
    setOptions(originalOptions);
    PRINT("FMM walk regression passed (%s, %i rank(s))\n", useCuda ? "CPU and CUDA" : "CPU",
        hpxSize());
}
