/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cmath>
#include <memory>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_context.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/position.hpp>

// Shared order-templated traversal and direct interactions. KickContext supplies
// solver-specific task scheduling and completion handling.
// Per-order instance translation units compile the arithmetic once per order.

struct KickWorkspace;

namespace detail {

// Per-thread interaction lists reused across node visits.
struct KickLists {
    Vector<TreeId> nextlist;
    Vector<TreeId> cplist;
    Vector<TreeId> pclist;
    Vector<TreeId> cclist;
    Vector<TreeId> leaflist;
    KickLists() = default;
    KickLists(KickLists const&) = delete;
    KickLists& operator=(KickLists const&) = delete;
    KickLists(KickLists&&) = default;
    KickLists& operator=(KickLists&&) = default;
};

KickLists kickAcquireLists();
void kickReleaseLists(KickLists&& lists);

bool kickSourcesAreSingleRank(Vector<TreeId> const& checklist);

}    // namespace detail

template <int order>
KickReturn kickWalk(KickParameters params, ExpansionBuffer l, Vector3<Position> pos, TreeId self,
    Vector<TreeId> dchecklist, std::shared_ptr<KickWorkspace> cudaWorkspace,
    KickContext const& context);

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definition pulls in the generated kernels, so only the per-order instance
// translation units see it; kick.cpp compiles against the declaration above.

#include <algorithm>
#include <array>
#include <cmath>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/kick_interaction_counts.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#endif

namespace detail {

// Adds an opened child to a list unless it is known to hold no particles. The
// leaf criterion can split sparse boxes down to single particles beside
// empty siblings, and an empty cell contributes nothing: keeping it out of
// every list spares its acceptance tests, its own opening and its zero-mass
// M2L, as the CUDA walk does by skipping empty sources at the test. Only a
// single-rank node knows its particle count: a node spanning several ranks has
// no rank-local particle range (treeCreate leaves it at (-1, -1), so
// numberOfParticles() reads zero) and its particles live in its single-rank
// descendants, so it is always kept. The walk's lists start from the root and
// grow only here, so no list ever holds an empty single-rank cell.
inline void kickPushOccupied(Vector<TreeId>& list, TreeId id) {
    TreeNode const* node = treeGetNode(id);
    if (!node->isLocal() || node->numberOfParticles() != 0) {
        list.push_back(id);
    }
}

// Resolves a rejected source cell that is not a leaf. The lockstep rule opens
// the source: a leaf sink retests the children here and an internal sink hands
// them to its children, so cell pairs meet at equal depth. The open-larger rule
// depends only on the unordered pair, which keeps the accepted pair set
// reciprocal from both sides: a larger source is opened and retested here, a
// smaller source is handed down unopened for the sink's children, and equal
// radii open both, as the lockstep rule does. A leaf sink always opens.
inline void kickOpenOrDefer(bool openLargerCell, TreeNode const* sink, TreeNode const* source,
    TreeId sourceId, Vector<TreeId>& retry, Vector<TreeId>& deferred,
    KickInteractionCounts& counts) {
    auto const& children = source->children;
    if (!openLargerCell || sink->leaf || source->radius > sink->radius) {
        kickPushOccupied(retry, children[kLeftChild]);
        kickPushOccupied(retry, children[kRightChild]);
        counts.sourceOpenings++;
    } else if (source->radius < sink->radius) {
        deferred.push_back(sourceId);
        counts.deferredSources++;
    } else {
        kickPushOccupied(deferred, children[kLeftChild]);
        kickPushOccupied(deferred, children[kRightChild]);
        counts.sourceOpenings++;
    }
}

}    // namespace detail

template <int order>
KickReturn kickWalk(KickParameters params, ExpansionBuffer l, Vector3<Position> pos, TreeId self,
    Vector<TreeId> dchecklist, std::shared_ptr<KickWorkspace> cudaWorkspace,
    KickContext const& context) {
    using Expansion = cosmictiger::gravity::cartesian::tracefree::Expansion<float, order>;
    ALWAYS_ASSERT(!params.deferHydroOpening || (!params.forceOnly && gravityGasOpeningDeferred()));
    auto* deferredGasOpening = params.deferHydroOpening ? gravityDeferredGasOpeningData() : nullptr;
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    constexpr int kLocalDegree = FmmOrder<order>::kLocalDegree_;
    // These intervals belong to this invocation, not to a system-thread stack:
    // other HPX workers may visit nodes at the same time. Stop before descending
    // into children or handing off a packet that can synchronously run the GPU.
#if COSMICTIGER_PROFILE
    bool const profileWork = hpxRank() == 0;
#else
    bool const profileWork = false;
#endif
    profile::ScopedWork work(profile::WorkPhase::KickCpuPrepare, profileWork);
    // The buffer travels with the work item; the view aliases its leading coefficients.
    Expansion& expansion = l.template expansion<order>();
    FlopCounter<FlopCount> flops = 0;    // Float force and timestep arithmetic.
    FlopCounter<FlopCount> doubleFlops = 0;
    FlopCounter<FlopCounts> policyFlops;
    FlopCounts geometryFlops{};
    TreeNode const* selfPtr = treeGetNode(self);
    if (selfPtr->localRoot && !params.forceOnly) {
        context.beginReturn(selfPtr->numberOfParticles());
    }
    ASSERT(self.proc == hpxRank());
    if (selfPtr->isLocal() && selfPtr->numberOfParticles() == 0) {
        if (selfPtr->localRoot && !params.forceOnly) {
            // Complete the promise while HPX is alive, including on empty ranks.
            context.publishReturn(KickReturn());
        }
        return KickReturn();
    }
    bool threadLeft = true;
#ifdef USE_CUDA
    if (getOptions().gravity.cuda && params.gpu) {
        if (params.gpu && cudaWorkspace == nullptr && selfPtr->isLocal()) {
            cudaWorkspace = std::make_shared<KickWorkspace>(
                params, selfPtr->numberOfParticles(), context.gpuExecutor);
        }
        size_t maximumGpuParticles = kCudaKickPartsMax;
        auto const rng = params.forceOnly ? Pair<ParticleInteger>(0, particlesOwnedSize()) :
                                            particlesCurrentRange();
        ParticleInteger const rangeParticles = rng.second - rng.first;
        int const blockCount = kickFineBlockCount(rangeParticles);
        maximumGpuParticles = std::min((size_t) maximumGpuParticles,
            (size_t) std::max(
                rangeParticles / blockCount, (ParticleInteger) getOptions().compute.bucketSize));
        if (params.gpu && selfPtr->numberOfParticles() <= maximumGpuParticles &&
            selfPtr->isLocal() && detail::kickSourcesAreSingleRank(dchecklist)) {
            work.stop();
            if (selfPtr->numberOfParticles()) {
                if (!cudaWorkspace->isBatch()) {
                    context.coverParticles(selfPtr->numberOfParticles());
                }
                cudaWorkspace->addWork(cudaWorkspace, l, pos, self, std::move(dchecklist),
                    params.geometry, params.outputOffset);
            }
            addCpuFlops(flops, FlopPrecision::Single);
            addCpuFlops(doubleFlops, FlopPrecision::Double);
            addCpuFlops(static_cast<FlopCounts>(policyFlops));
            addCpuFlops(geometryFlops);
            if (selfPtr->localRoot && !params.forceOnly) {
                profile::ScopedWork wait(profile::WorkPhase::KickResultWait, profileWork);
                return context.waitReturn();
            } else {
                return KickReturn();
            }
        }
        threadLeft = cudaWorkspace != nullptr;
    }
#endif
    KickReturn kr;
    float const smoothingLength = params.gravitySmoothingLength;
    float const Gm = params.gm;
    float const eta = params.eta;
    bool const saveAcceleration =
        params.saveForce || saveParticleAcceleration(getOptions().analysis);
    bool const savePotential = params.saveForce || saveParticlePotential(getOptions().analysis);
    auto workspace = detail::kickAcquireLists();
    Vector<TreeId>& nextlist = workspace.nextlist;
    Vector<TreeId>& cclist = workspace.cclist;
    Vector<TreeId>& cplist = workspace.cplist;
    Vector<TreeId>& pclist = workspace.pclist;
    Vector<TreeId>& leaflist = workspace.leaflist;
    SimdFloat const thetainv = math::broadcast<SimdFloat>(fmmInverseOpeningAngle(params.theta));
    // Every node shares the fixed gravity support, so it bounds every pair.
    SimdFloat const smoothingLengthPack = math::broadcast<SimdFloat>(smoothingLength);
    // Internal sinks retest opened sources at their own level under the
    // open-larger rule, and hand down only what the rule defers.
    bool const openLargerCell = params.openLargerCell;
    KickInteractionCounts counts{};
    Array<TreeNode const*, kSimdFloatSize> otherPtrs;
    Vector3<float> ldx;
    SimdFloat selfRadius = selfPtr->radius;
    Vector3<SimdPosition> selfPos{};
    Range<SimdDouble8> selfBox{};
    Vector3<SimdPosition> otherPos{};
    Vector3<SimdFloat> dx{};
    SimdFloat otherRadius{};
    SimdFloat otherLeaf{};
    Range<SimdDouble8> otherBox{};
    for (int dim = 0; dim < kDimensions; dim++) {
        selfPos[dim] = math::broadcast<SimdPosition>(selfPtr->pos[dim].raw());
        selfBox.begin[dim] = (double) selfPtr->box.begin[dim].raw();
        selfBox.end[dim] = (double) selfPtr->box.end[dim].raw();
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        selfBox.begin[dim] *= RangeFixed::kToFloatFactor_;
        selfBox.end[dim] *= RangeFixed::kToFloatFactor_;
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        ldx[dim] = unwrappedDoubleDistance(selfPtr->pos[dim], pos[dim]);
    }
    cosmictiger::gravity::cartesian::tracefree::Expansion<float, order> translated{};
    if (!params.doPhi) {
        translated[0] = expansion[0];
    }
    cosmictiger::gravity::cartesian::tracefree::L2L(
        translated, expansion, fmmScaleDisplacement(ldx), params.doPhi);
    expansion = translated;
    flops += cosmictiger::gravity::cartesian::tracefree::detail::kL2LFlops<kLocalDegree> + 5;
    // Box scaling stays double; each local displacement scales in double and
    // narrows before the float L2L coordinate scaling. Two reciprocals are float.
    doubleFlops += 6 + 6 * kSimdFloatSize;
    float const inverseSmoothingLength = 1.0f / params.gravitySmoothingLength;
    ForceVectors forces;
    ParticleInteger const numberOfSelfParticles = selfPtr->numberOfParticles();
    if (selfPtr->leaf) {
        forces = ForceVectors(numberOfSelfParticles);
        flops += FlopCount(numberOfSelfParticles) * 2;
        for (ParticleInteger i = 0; i < numberOfSelfParticles; i++) {
            forces.gx[i] = forces.gy[i] = forces.gz[i] = 0.0f;
            ParticleInteger const particleIndex = selfPtr->particleRange.first + i;
            forces.phi[i] = particlesMass(particleIndex) * selfPhi() * inverseSmoothingLength;
        }
    }
    {
        work.next(profile::WorkPhase::KickCpuTraversal);
        nextlist.resize(0);
        cclist.resize(0);
        pclist.resize(0);
        cplist.resize(0);
        leaflist.resize(0);
        auto& checklist = dchecklist;
        do {
            for (int ci = 0; ci < checklist.size(); ci += kSimdFloatSize) {
                int const maxci = std::min((int) checklist.size(), ci + kSimdFloatSize);
                int const maxi = maxci - ci;
                for (int i = ci; i < maxci; i++) {
                    otherPtrs[i - ci] = treeGetNode(checklist[i]);
                }
                for (int i = 0; i < maxi; i++) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        simdSetLane(otherPos[dim], i, otherPtrs[i]->pos[dim].raw());
                    }
                    simdSetLane(otherRadius, i, otherPtrs[i]->radius);
                    simdSetLane(otherLeaf, i, otherPtrs[i]->leaf);
                }
                for (int i = maxi; i < kSimdFloatSize; i++) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        simdSetLane(otherPos[dim], i, 0.f);
                    }
                    simdSetLane(otherRadius, i, 0.f);
                    simdSetLane(otherLeaf, i, 0);
                }
                for (int dim = 0; dim < kDimensions; dim++) {
                    dx[dim] = fmmPositionDifference(
                        selfPos[dim], otherPos[dim], dim, params.geometry, &geometryFlops);
                }
                SimdFloat const r2 =
                    math::fma(dx[kXi], dx[kXi], math::fma(dx[kYi], dx[kYi], dx[kZi] * dx[kZi]));
                auto acceptance =
                    computeFmmAcceptance(r2, selfRadius, otherRadius, math::broadcast<SimdFloat>(0),
                        math::broadcast<SimdFloat>(0), smoothingLengthPack, thetainv);
                flops += kSimdFloatSize *
                    24;    // Float norm and acceptance; geometry counted separately.
                if (selfPtr->leaf || math::anyOf(otherLeaf != math::broadcast<SimdFloat>(0.0f))) {
                    for (int i = 0; i < maxi; i++) {
                        for (int dim = 0; dim < kDimensions; dim++) {
                            simdSetLane(otherBox.begin[dim], i, otherPtrs[i]->box.begin[dim].raw());
                            simdSetLane(otherBox.end[dim], i, otherPtrs[i]->box.end[dim].raw());
                        }
                    }
                    for (int dim = 0; dim < kDimensions; dim++) {
                        otherBox.begin[dim] *= RangeFixed::kToFloatFactor_;
                        otherBox.end[dim] *= RangeFixed::kToFloatFactor_;
                    }
                    for (int i = maxi; i < kSimdFloatSize; i++) {
                        for (int dim = 0; dim < kDimensions; dim++) {
                            simdSetLane(otherBox.begin[dim], i, 0.0);
                            simdSetLane(otherBox.end[dim], i, 0.0);
                        }
                    }
                    acceptance = computeFmmAcceptance(r2, selfRadius, otherRadius,
                        fmmSinkBoxDistanceSquared(
                            selfBox, otherPos, params.geometry, &geometryFlops),
                        fmmSourceBoxDistanceSquared(
                            otherBox, selfPos, otherPos, params.geometry, &geometryFlops),
                        smoothingLengthPack, thetainv);
                    flops += kSimdFloatSize * 19;         // Acceptance only.
                    doubleFlops += 6 * kSimdFloatSize;    // Source box scaling.
                }
                for (int i = 0; i < maxi; i++) {
                    auto const interaction = selectFmmInteraction(bool(acceptance.cellCell[i]),
                        bool(acceptance.particleCell[i]), bool(acceptance.cellParticle[i]),
                        selfPtr->leaf, bool(otherLeaf[i]), selfPtr->radius, otherPtrs[i]->radius,
                        int(numberOfSelfParticles), int(otherPtrs[i]->numberOfParticles()),
                        FmmHybridCost<order>::kHybridPerParticle_,
                        FmmHybridCost<order>::kParticlePair_,
                        params.geometry.meshSize > 0 ? FmmHybridCost<order>::kCellCell_ : 0.0f);
                    if (acceptance.cellCell[i] && selfPtr->leaf && otherLeaf[i] &&
                        params.geometry.meshSize > 0) {
                        flops += 2;
                    } else if (!acceptance.cellCell[i] && selfPtr->leaf && otherLeaf[i]) {
                        flops += 2 + int(bool(acceptance.particleCell[i])) +
                            int(bool(acceptance.cellParticle[i]));
                    }
                    if (interaction == FmmInteraction::CellCell) {
                        cclist.push_back(checklist[ci + i]);
                    } else if (interaction == FmmInteraction::CellParticle) {
                        cplist.push_back(checklist[ci + i]);
                    } else if (interaction == FmmInteraction::ParticleCell) {
                        pclist.push_back(checklist[ci + i]);
                    } else if (otherLeaf[i]) {
                        leaflist.push_back(checklist[ci + i]);
                    } else {
                        detail::kickOpenOrDefer(openLargerCell, selfPtr, otherPtrs[i],
                            checklist[ci + i], nextlist, leaflist, counts);
                    }
                }
                counts.acceptanceTests += maxi;
            }
            std::swap(checklist, nextlist);
            nextlist.resize(0);
        } while (checklist.size() && (selfPtr->leaf || openLargerCell));
        counts.cellCellDirect += cclist.size();
        counts.cellParticle += cplist.size();
        counts.particleCell += pclist.size();
        if (selfPtr->leaf) {
            counts.cellParticleAtLeafSinks += cplist.size();
            for (auto const& id : pclist) {
                counts.particleCellFromLeafSources += treeGetNode(id)->leaf;
            }
            counts.particleParticleLeafPairs += leaflist.size();
            for (auto const& id : leaflist) {
                counts.particleParticlePairs +=
                    (long long) numberOfSelfParticles * treeGetNode(id)->numberOfParticles();
            }
        }
        work.next(profile::WorkPhase::KickCpuM2L);
        cpuGravityCc<order>(expansion, cclist, self, params.doPhi, params.geometry);
        // Internal sinks pass these source-leaf contributions to their children via L2L.
        work.next(profile::WorkPhase::KickCpuP2L);
        cpuGravityCp<order>(expansion, cplist, self, params.doPhi, params.geometry);
        if (selfPtr->leaf) {
            if (selfPtr->numberOfParticles() > 0) {
                work.next(profile::WorkPhase::KickCpuM2P);
                cpuGravityPc<order>(forces, params.doPhi, self, pclist, params.geometry);
                work.next(profile::WorkPhase::KickCpuP2P);
                cpuGravityPp(forces, params.doPhi, self, leaflist, params.gravitySmoothingLength,
                    params.geometry);
            }
        } else {
            work.next(profile::WorkPhase::KickCpuTraversal);
            ALWAYS_ASSERT(pclist.size() == 0);
            checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        }
    }
    work.next(profile::WorkPhase::KickCpuPrepare);

    if (selfPtr->leaf) {
        work.next(profile::WorkPhase::KickCpuLeafKick);
        detail::kickReleaseLists(std::move(workspace));
        auto const rng = selfPtr->particleRange;
#if COSMICTIGER_WITH_SPH
        bool const midpointHydro = sphMidpointEnabled();
#else
        bool const midpointHydro = false;
#endif
        for (ParticleInteger i = rng.first; i < rng.second; i++) {
            if (params.forceOnly || particlesRung(i) >= params.minRung) {
                ParticleInteger const j = i - rng.first;
                Vector3<float> dx;
                for (ParticleInteger dim = 0; dim < kDimensions; dim++) {
                    dx[dim] = unwrappedDoubleDistance(particlesPos(dim, i), selfPtr->pos[dim]);
                }
                cosmictiger::gravity::Field<float> l2{};
                cosmictiger::gravity::cartesian::tracefree::L2P(
                    l2, expansion, fmmScaleDisplacement(dx), params.doPhi);
                float const gravityEffectiveRadius =
                    gravitySofteningEffectiveRadius(params.gravitySmoothingLength);
                forces.phi[j] += kScaleFactor1 * l2.potential();
                forces.gx[j] += kScaleFactor2 * l2.accelerationX();
                forces.gy[j] += kScaleFactor2 * l2.accelerationY();
                forces.gz[j] += kScaleFactor2 * l2.accelerationZ();
                // Three double displacements each narrow to float; the effective
                // radius divides a widened float by a double and narrows back.
                doubleFlops += 9;
                if (params.forceOnly) {
                    ALWAYS_ASSERT(cudaWorkspace != nullptr && cudaWorkspace->isBatch());
                    auto const output = cudaWorkspace->output();
                    ParticleInteger const outputIndex = i + params.outputOffset;
                    output.potential[outputIndex] = params.doPhi ? forces.phi[j] : 0.0f;
                    output.acceleration[kXi][outputIndex] = forces.gx[j];
                    output.acceleration[kYi][outputIndex] = forces.gy[j];
                    output.acceleration[kZi][outputIndex] = forces.gz[j];
                    flops += cosmictiger::gravity::cartesian::tracefree::detail::kL2PFlops<
                                 kLocalDegree> +
                        11;
                    continue;
                }
                forces.gx[j] *= Gm;
                forces.gy[j] *= Gm;
                forces.gz[j] *= Gm;
                forces.phi[j] *= Gm;
                particlesCaptureGravitySample(i, forces.gx[j], forces.gy[j], forces.gz[j]);
                if (saveAcceleration) {
                    particlesGravityAcceleration(kXi, i) = forces.gx[j];
                    particlesGravityAcceleration(kYi, i) = forces.gy[j];
                    particlesGravityAcceleration(kZi, i) = forces.gz[j];
                }
                if (savePotential && params.doPhi) {
                    particlesPot(i) = forces.phi[j];
                }
                auto& vx = particlesVel(kXi, i);
                auto& vy = particlesVel(kYi, i);
                auto& vz = particlesVel(kZi, i);
                auto& rung = particlesRung(i);
                if (params.htime) {
                    // Signed gravity-only ladder, including gas on correction levels.
                    float const mass = particlesMass(i);
                    float g2;
                    float kin0 = 0.5 * mass * sqr(vx, vy, vz);
                    float const sgn = params.sign * (params.top ? 1.f : -1.f);
                    flops += 1;
                    auto const applyKick = [&](float halfStep) {
                        vx = fmaf(forces.gx[j], halfStep, vx);
                        vy = fmaf(forces.gy[j], halfStep, vy);
                        vz = fmaf(forces.gz[j], halfStep, vz);
                        flops += 7;
                    };
                    if (params.ascending) {
                        flops += 2;
                        float const dt = 0.5f * rungDt[params.minRung] * params.t0;
                        if (!params.firstCall) {
                            applyKick(sgn * dt);
                        }
                    }
                    kr.kin += 0.5 * mass * sqr(vx, vy, vz);
                    kr.xmom += mass * vx;
                    kr.ymom += mass * vy;
                    kr.zmom += mass * vz;
                    kr.nmom += mass * sqrt(sqr(vx, vy, vz));
                    if (params.descending) {
                        g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]) + 1e-35f;
                        float const factor = eta * sqrtf(params.a);
                        float dt = std::min(
                            factor * sqrtf(gravityEffectiveRadius / sqrtf(g2)), (float) params.t0);
                        dt = std::min(params.maxDt, dt);
                        int nextRung = std::max(params.minRung +
                                int((int) ceilf(log2f(params.t0 / dt)) > params.minRung),
                            int(rung) - 1);
#if COSMICTIGER_WITH_SPH
                        if (midpointHydro && particlesType(i) == kSphType) {
                            if (params.freezeGasRungs) {
                                nextRung = int(rung);
                            } else {
                                if (params.fullGasRungSelection) {
                                    ALWAYS_ASSERT(params.top);
                                    // Derived quantization of the full-force bound:
                                    // r = max(ceil(log₂(Δτ₀/Δτ_G)), r_old−1, r_min).
                                    // Δτ₀ -> base timestep; Δτ_G -> gravity timestep; r -> next
                                    // rung.
                                    nextRung = std::max(int(ceilf(log2f(params.t0 / dt))),
                                        std::max(int(rung) - 1, params.minRung));
                                }
                                nextRung = std::max(nextRung,
                                    sphParticlesHydroRung(particlesCatIndex(i)) +
                                        hydroRungIntervalShift(params));
                            }
                        }
#endif
                        kr.maxRung = std::max(nextRung, kr.maxRung);
                        ALWAYS_ASSERT(nextRung >= 0 && nextRung < kMaxRung);
                        flops += 15;
                        rung = nextRung;
                        dt = 0.5f * rungDt[params.minRung] * params.t0;
#if COSMICTIGER_WITH_SPH
                        if (params.deferHydroOpening && midpointHydro &&
                            particlesType(i) == kSphType) {
                            auto& increment = deferredGasOpening[particlesCatIndex(i)];
                            float const signedHalfStep = sgn * dt;
                            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                                float const acceleration = dimension == kXi ?
                                    forces.gx[j] :
                                    (dimension == kYi ? forces.gy[j] : forces.gz[j]);
                                increment[dimension] =
                                    std::fma(acceleration, signedHalfStep, increment[dimension]);
                            }
                            flops += 7;
                        } else
#endif
                        {
                            applyKick(sgn * dt);
                        }
                        flops += 2;
                    }
                    float kin1 = 0.5 * mass * sqr(vx, vy, vz);
                    kr.pot += 0.5 * mass * forces.phi[j];
                    kr.dkin += kin1 - kin0;
                } else {
                    float const mass = particlesMass(i);
#if COSMICTIGER_WITH_SPH
                    bool const gas = getOptions().hydro.enabled && particlesType(i) == kSphType;
#else
                    bool const gas = false;
#endif
                    float const kin0 = 0.5f * mass * sqr(vx, vy, vz);
                    if ((!gas || midpointHydro) && !params.firstCall) {
                        flops += 9;
                        float const dt = params.sign * 0.5f * rungDt[rung] * params.t0;
                        vx = fmaf(forces.gx[j], dt, vx);
                        vy = fmaf(forces.gy[j], dt, vy);
                        vz = fmaf(forces.gz[j], dt, vz);
                    }
                    if (!gas || midpointHydro) {
                        // Legacy hydro owns gas momentum; midpoint hydro leaves gravity here.
                        kr.kin += 0.5f * mass * sqr(vx, vy, vz);
                        kr.xmom += mass * vx;
                        kr.ymom += mass * vy;
                        kr.zmom += mass * vz;
                        kr.nmom += mass * sqrt(sqr(vx, vy, vz));
                    }
                    if (!gas || midpointHydro) {
                        flops += 24;    // Timestep/rung selection and signed velocity kick.
                        float const syncVx = vx, syncVy = vy, syncVz = vz;
                        int const previousRung = int(rung);
                        float const g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]) + 1e-35f;
                        float dt = std::min(
                            eta * sqrtf(params.a) * sqrtf(gravityEffectiveRadius / sqrtf(g2)),
                            params.t0);
                        dt = std::min(dt, params.maxDt);
                        rung = std::max(std::max((int) ceilf(log2f(params.t0 / dt)),
                                            std::max(previousRung - 1, params.minRung)),
                            1);
#if COSMICTIGER_WITH_SPH
                        if (gas && midpointHydro) {
                            rung = params.freezeGasRungs ?
                                previousRung :
                                std::max(int(rung), sphParticlesHydroRung(particlesCatIndex(i)));
                        }
#endif
                        if (params.fixedRung != 0) {
                            // Preserve the required bound in the result; the driver rejects
                            // an unsafe fixed step before drifting any particle.
                            kr.maxRung = std::max(kr.maxRung, int(rung));
                            rung = params.fixedRung;
                        }
                        ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
                        dt = params.sign * 0.5f * rungDt[rung] * params.t0;
                        if (gas && params.deferHydroOpening) {
                            auto& increment = deferredGasOpening[particlesCatIndex(i)];
                            increment[kXi] = std::fma(forces.gx[j], dt, increment[kXi]);
                            increment[kYi] = std::fma(forces.gy[j], dt, increment[kYi]);
                            increment[kZi] = std::fma(forces.gz[j], dt, increment[kZi]);
                        } else {
                            vx = fmaf(forces.gx[j], dt, vx);
                            vy = fmaf(forces.gy[j], dt, vy);
                            vz = fmaf(forces.gz[j], dt, vz);
                        }
                        kr.darkMatterKineticCorrection +=
                            0.5f * mass * (sqr(syncVx, syncVy, syncVz) - sqr(vx, vy, vz));
                        kr.darkMatterXMomentumCorrection += mass * (syncVx - vx);
                        kr.darkMatterYMomentumCorrection += mass * (syncVy - vy);
                        kr.darkMatterZMomentumCorrection += mass * (syncVz - vz);
                        kr.darkMatterMomentumNormCorrection +=
                            mass * (sqrtf(sqr(syncVx, syncVy, syncVz)) - sqrtf(sqr(vx, vy, vz)));
                    }
                    kr.maxRung = std::max(kr.maxRung, (int) rung);
                    kr.pot += 0.5f * mass * forces.phi[j];
                    kr.dkin += 0.5f * mass * sqr(vx, vy, vz) - kin0;
                }
                flops +=
                    cosmictiger::gravity::cartesian::tracefree::detail::kL2PFlops<kLocalDegree> +
                    15;
            }
        }
        if (cudaWorkspace != nullptr && cudaWorkspace->isBatch()) {
            cudaWorkspace->accumulateBatchReturn(kr);
            cudaWorkspace->addInteractionCounts(counts);
        } else {
            context.accumulateReturn(kr, selfPtr->numberOfParticles(), cudaWorkspace == nullptr);
        }
        addCpuFlops(flops, FlopPrecision::Single);
        addCpuFlops(doubleFlops, FlopPrecision::Double);
        addCpuFlops(static_cast<FlopCounts>(policyFlops));
        addCpuFlops(geometryFlops);
        work.stop();
        if (cudaWorkspace != nullptr) {
            cudaWorkspace->recordProcessedParticles(cudaWorkspace, selfPtr->numberOfParticles());
        }
        if (selfPtr->localRoot && !params.forceOnly) {
            return kr;
        } else {
            return KickReturn();
        }
    } else {
        detail::kickReleaseLists(std::move(workspace));
        // Fetch both children (and their cache lines) before forking the walks.
        TreeNode const* cl = treeGetNode(selfPtr->children[kLeftChild]);
        TreeNode const* cr = treeGetNode(selfPtr->children[kRightChild]);
        (void) cl;
        (void) cr;
        work.stop();
        std::array<hpx::future<KickReturn>, kChildCount> futs;
        futs[kRightChild] = context.fork(params, l, selfPtr->pos, selfPtr->children[kRightChild],
            dchecklist, cudaWorkspace, threadLeft);
        try {
            futs[kLeftChild] = context.fork(params, l, selfPtr->pos, selfPtr->children[kLeftChild],
                std::move(dchecklist), cudaWorkspace, false);
        } catch (...) {
            // Keep shared tree data alive until the asynchronous sibling exits.
            profile::ScopedWork wait(profile::WorkPhase::KickChildJoin, profileWork);
            futs[kRightChild].wait();
            throw;
        }
        {
            profile::ScopedWork wait(profile::WorkPhase::KickChildJoin, profileWork);
            hpx::wait_all(futs);
        }
        // Surface launch/completion failures before waiting for a result that
        // only a successful GPU batch can publish.
        auto const rcl = futs[kLeftChild].get();
        auto const rcr = futs[kRightChild].get();
        if (selfPtr->processRange.second - selfPtr->processRange.first > 1) {
            kr += rcl;
            kr += rcr;
        } else {
            if (selfPtr->localRoot && !params.forceOnly) {
                profile::ScopedWork wait(profile::WorkPhase::KickResultWait, profileWork);
                kr = context.waitReturn();
            }
        }

        if (cudaWorkspace != nullptr && cudaWorkspace->isBatch()) {
            cudaWorkspace->addInteractionCounts(counts);
        }
        addCpuFlops(flops, FlopPrecision::Single);
        addCpuFlops(doubleFlops, FlopPrecision::Double);
        addCpuFlops(static_cast<FlopCounts>(policyFlops));
        addCpuFlops(geometryFlops);
        return kr;
    }
}

#endif
