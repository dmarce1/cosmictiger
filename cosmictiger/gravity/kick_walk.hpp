/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <memory>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_context.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/position.hpp>

// Shared order-templated traversal and direct interactions. KickContext supplies
// solver-specific task scheduling, periodic node work, and completion handling.
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
KickReturn kickWalk(KickParameters params, ExpansionBuffer l, Array<Position, NDIM> pos,
    TreeId self, Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace, KickContext const& context);

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definition pulls in the generated kernels, so only the per-order instance
// translation units see it; kick.cpp compiles against the declaration above.

#include <algorithm>
#include <array>
#include <cmath>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/fmm_kernels.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick_interaction_counts.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/math/simd.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

// Adds an opened child to a list unless it is known to hold no particles. The
// Ewald leaf criterion splits sparse boxes down to single particles beside
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
        kickPushOccupied(retry, children[LEFT]);
        kickPushOccupied(retry, children[RIGHT]);
        counts.sourceOpenings++;
    } else if (source->radius < sink->radius) {
        deferred.push_back(sourceId);
        counts.deferredSources++;
    } else {
        kickPushOccupied(deferred, children[LEFT]);
        kickPushOccupied(deferred, children[RIGHT]);
        counts.sourceOpenings++;
    }
}

}    // namespace detail

template <int order>
KickReturn kickWalk(KickParameters params, ExpansionBuffer l, Array<Position, NDIM> pos,
    TreeId self, Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace, KickContext const& context) {
    using Expansion = FmmExpansion<float, order>;
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    constexpr int kLocalDegree = FmmOrder<order>::kLocalDegree_;
    // The buffer travels with the work item; the view aliases its leading coefficients.
    Expansion& expansion = l.template expansion<order>();
    FlopCounter<FlopCount> flops = 0;
    stackTraceActivate();
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
    if (context.background != nullptr) {
        flops += context.background(order, params, l, pos, self, echecklist);
    }
    bool threadLeft = true;
#ifdef USE_CUDA
    if (getOptions().compute.cuda && params.gpu) {
        if (params.gpu && cudaWorkspace == nullptr && selfPtr->isLocal()) {
            cudaWorkspace = std::make_shared<KickWorkspace>(
                params, selfPtr->numberOfParticles(), context.gpuExecutor);
        }
        size_t maximumGpuParticles = CUDA_KICK_PARTS_MAX;
        auto const rng = params.forceOnly ? Pair<ParticleInteger>(0, particlesOwnedSize()) :
                                            particlesCurrentRange();
        ParticleInteger const rangeParticles = rng.second - rng.first;
        // A node that still carries Ewald checks goes over at the coarse
        // threshold, so that the kernel rather than this walk evaluates them.
        int const blockCount = echecklist.empty() ? kickFineBlockCount(rangeParticles) :
                                                    kickBlockCount(rangeParticles);
        maximumGpuParticles = std::min((size_t) maximumGpuParticles,
            (size_t) std::max(
                rangeParticles / blockCount, (ParticleInteger) getOptions().compute.bucketSize));
        if (params.gpu && selfPtr->numberOfParticles() <= maximumGpuParticles &&
            selfPtr->isLocal() && detail::kickSourcesAreSingleRank(dchecklist) &&
            detail::kickSourcesAreSingleRank(echecklist)) {
            if (selfPtr->numberOfParticles()) {
                if (!cudaWorkspace->isBatch()) {
                    context.coverParticles(selfPtr->numberOfParticles());
                }
                cudaWorkspace->addWork(cudaWorkspace, l, pos, self, std::move(dchecklist),
                    std::move(echecklist), params.geometry, params.outputOffset);
            }
            addCpuFlops(flops);
            if (selfPtr->localRoot && !params.forceOnly) {
                return context.waitReturn();
            } else {
                return KickReturn();
            }
        }
        threadLeft = cudaWorkspace != nullptr;
    }
#endif
    KickReturn kr;
    float const supportRadius = params.gravitySupportRadius;
    float const gm = params.gm;
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
    SimdFloat const thetainv = simdBroadcast<SimdFloat>(fmmInverseOpeningAngle(params.theta));
    // Internal sinks retest opened sources at their own level under the
    // open-larger rule, and hand down only what the rule defers.
    bool const openLargerCell = params.openLargerCell;
    KickInteractionCounts counts{};
    Array<TreeNode const*, SIMD_FLOAT_SIZE> otherPtrs;
    Array<float, NDIM> ldx;
    SimdFloat selfRadius = selfPtr->radius;
    Array<SimdPosition, NDIM> selfPos{};
    Range<SimdDouble8> selfBox{};
    Array<SimdPosition, NDIM> otherPos{};
    Array<SimdFloat, NDIM> dx{};
    SimdFloat otherRadius{};
    SimdFloat otherLeaf{};
    SimdFloat otherSupportRadius{};
    Range<SimdDouble8> otherBox{};
    for (int dim = 0; dim < NDIM; dim++) {
        selfPos[dim] = simdBroadcast<SimdPosition>(selfPtr->pos[dim].raw());
        selfBox.begin[dim] = (double) selfPtr->box.begin[dim].raw();
        selfBox.end[dim] = (double) selfPtr->box.end[dim].raw();
    }
    for (int dim = 0; dim < NDIM; dim++) {
        selfBox.begin[dim] *= RangeFixed::kToFloatFactor_;
        selfBox.end[dim] *= RangeFixed::kToFloatFactor_;
    }
    for (int dim = 0; dim < NDIM; dim++) {
        ldx[dim] = unwrappedDoubleDistance(selfPtr->pos[dim], pos[dim]);
    }
    expansion = l2L(expansion, ldx, params.doPhi);
    flops += detail::cartesian::L2lKernel<kLocalDegree, true>::kFlops_ + 14 + 6 * SIMD_FLOAT_SIZE;
    float const inverseSupportRadius = 1.0f / params.gravitySupportRadius;
    ForceVectors forces;
    ParticleInteger const numberOfSelfParticles = selfPtr->numberOfParticles();
    if (selfPtr->leaf) {
        forces = ForceVectors(numberOfSelfParticles);
        flops += FlopCount(numberOfSelfParticles) * 2;
        for (ParticleInteger i = 0; i < numberOfSelfParticles; i++) {
            forces.gx[i] = forces.gy[i] = forces.gz[i] = 0.0f;
            ParticleInteger const particleIndex = selfPtr->particleRange.first + i;
            forces.phi[i] = particlesMass(particleIndex) * selfPhi() * inverseSupportRadius;
        }
    }
    if (context.periodic != nullptr) {
        flops += context.periodic(order, params, l, self, echecklist, counts);
    }

    {
        nextlist.resize(0);
        cclist.resize(0);
        pclist.resize(0);
        cplist.resize(0);
        leaflist.resize(0);
        auto& checklist = dchecklist;
        do {
            for (int ci = 0; ci < checklist.size(); ci += SIMD_FLOAT_SIZE) {
                int const maxci = std::min((int) checklist.size(), ci + SIMD_FLOAT_SIZE);
                int const maxi = maxci - ci;
                for (int i = ci; i < maxci; i++) {
                    otherPtrs[i - ci] = treeGetNode(checklist[i]);
                }
                for (int i = 0; i < maxi; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(otherPos[dim], i, otherPtrs[i]->pos[dim].raw());
                    }
                    simdSetLane(otherRadius, i, otherPtrs[i]->radius);
                    simdSetLane(otherLeaf, i, otherPtrs[i]->leaf);
                    simdSetLane(otherSupportRadius, i, otherPtrs[i]->maxGravitySupportRadius);
                }
                for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(otherPos[dim], i, 0.f);
                    }
                    simdSetLane(otherRadius, i, 0.f);
                    simdSetLane(otherLeaf, i, 0);
                    simdSetLane(otherSupportRadius, i, 0.f);
                }
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] =
                        fmmPositionDifference(selfPos[dim], otherPos[dim], dim, params.geometry);
                }
                SimdFloat const r2 =
                    simdFma(dx[XDIM], dx[XDIM], simdFma(dx[YDIM], dx[YDIM], dx[ZDIM] * dx[ZDIM]));
                SimdFloat const pairSupportRadius = fmmPairSupportRadius(
                    simdBroadcast<SimdFloat>(supportRadius),
                    simdBroadcast<SimdFloat>(selfPtr->maxGravitySupportRadius), otherSupportRadius);
                auto acceptance =
                    computeFmmAcceptance(r2, selfRadius, otherRadius, simdBroadcast<SimdFloat>(0),
                        simdBroadcast<SimdFloat>(0), pairSupportRadius, thetainv);
                flops += SIMD_FLOAT_SIZE * 27;
                if (selfPtr->leaf || simdAnyOf(otherLeaf != simdBroadcast<SimdFloat>(0.0f))) {
                    for (int i = 0; i < maxi; i++) {
                        for (int dim = 0; dim < NDIM; dim++) {
                            simdSetLane(otherBox.begin[dim], i, otherPtrs[i]->box.begin[dim].raw());
                            simdSetLane(otherBox.end[dim], i, otherPtrs[i]->box.end[dim].raw());
                        }
                    }
                    for (int dim = 0; dim < NDIM; dim++) {
                        otherBox.begin[dim] *= RangeFixed::kToFloatFactor_;
                        otherBox.end[dim] *= RangeFixed::kToFloatFactor_;
                    }
                    for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
                        for (int dim = 0; dim < NDIM; dim++) {
                            simdSetLane(otherBox.begin[dim], i, 0.0);
                            simdSetLane(otherBox.end[dim], i, 0.0);
                        }
                    }
                    acceptance = computeFmmAcceptance(r2, selfRadius, otherRadius,
                        fmmSinkBoxDistanceSquared(selfBox, otherPos, params.geometry),
                        fmmSourceBoxDistanceSquared(otherBox, selfPos, otherPos, params.geometry),
                        pairSupportRadius, thetainv);
                    flops += SIMD_FLOAT_SIZE * 85;
                }
                for (int i = 0; i < maxi; i++) {
                    auto const interaction = selectFmmInteraction(bool(acceptance.cellCell[i]),
                        bool(acceptance.particleCell[i]), bool(acceptance.cellParticle[i]),
                        selfPtr->leaf, bool(otherLeaf[i]), selfPtr->radius, otherPtrs[i]->radius,
                        int(numberOfSelfParticles), int(otherPtrs[i]->numberOfParticles()),
                        FmmHybridCost<order>::kHybridPerParticle_,
                        FmmHybridCost<order>::kParticlePair_);
                    if (!acceptance.cellCell[i] && selfPtr->leaf && otherLeaf[i]) {
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
        cpuGravityCc<order>(expansion, cclist, self, params.doPhi, params.geometry);
        // Internal sinks pass these source-leaf contributions to their children via L2L.
        cpuGravityCp<order>(expansion, cplist, self, params.doPhi, params.geometry);
        if (selfPtr->leaf) {
            if (selfPtr->numberOfParticles() > 0) {
                cpuGravityPc<order>(forces, params.doPhi, self, pclist, params.geometry);
                cpuGravityPp(forces, params.doPhi, self, leaflist, params.gravitySupportRadius,
                    params.geometry);
            }
        } else {
            ALWAYS_ASSERT(pclist.size() == 0);
            checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        }
    }

    if (selfPtr->leaf) {
        detail::kickReleaseLists(std::move(workspace));
        auto const rng = selfPtr->particleRange;
        for (ParticleInteger i = rng.first; i < rng.second; i++) {
            if (params.forceOnly || particlesRung(i) >= params.minRung) {
                ParticleInteger const j = i - rng.first;
                Array<float, NDIM> dx;
                for (ParticleInteger dim = 0; dim < NDIM; dim++) {
                    dx[dim] = unwrappedDoubleDistance(particlesPos(dim, i), selfPtr->pos[dim]);
                }
                auto const l2 = l2P(expansion, dx, params.doPhi);
                float const gravityEffectiveRadius = params.gravitySupportRadius /
                    selectedGravityKernel().supportRadiusPerEffectiveRadius;
                forces.phi[j] += SCALE_FACTOR1 * l2(0, 0, 0);
                forces.gx[j] -= SCALE_FACTOR2 * l2(1, 0, 0);
                forces.gy[j] -= SCALE_FACTOR2 * l2(0, 1, 0);
                forces.gz[j] -= SCALE_FACTOR2 * l2(0, 0, 1);
                if (params.forceOnly) {
                    ALWAYS_ASSERT(cudaWorkspace != nullptr && cudaWorkspace->isBatch());
                    auto const output = cudaWorkspace->output();
                    ParticleInteger const outputIndex = i + params.outputOffset;
                    output.potential[outputIndex] = params.doPhi ? forces.phi[j] : 0.0f;
                    output.acceleration[XDIM][outputIndex] = forces.gx[j];
                    output.acceleration[YDIM][outputIndex] = forces.gy[j];
                    output.acceleration[ZDIM][outputIndex] = forces.gz[j];
                    flops += detail::cartesian::L2pKernel<kLocalDegree, true>::kFlops_ + 14;
                    continue;
                }
                forces.gx[j] *= gm;
                forces.gy[j] *= gm;
                forces.gz[j] *= gm;
                forces.phi[j] *= gm;
                if (saveAcceleration) {
                    particlesGravityAcceleration(XDIM, i) = forces.gx[j];
                    particlesGravityAcceleration(YDIM, i) = forces.gy[j];
                    particlesGravityAcceleration(ZDIM, i) = forces.gz[j];
                }
                if (savePotential && params.doPhi) {
                    particlesPot(i) = forces.phi[j];
                }
                auto& vx = particlesVel(XDIM, i);
                auto& vy = particlesVel(YDIM, i);
                auto& vz = particlesVel(ZDIM, i);
                auto& rung = particlesRung(i);
                if (params.htime) {
                    float g2;
                    float kin0 = 0.5 * sqr(vx, vy, vz);
                    float const sgn = params.top ? 1 : -1;
                    if (params.ascending) {
                        flops += 2;
                        float const dt = 0.5f * rungDt[params.minRung] * params.t0;
                        if (!params.firstCall) {
                            vx = fmaf(sgn * forces.gx[j], dt, vx);
                            vy = fmaf(sgn * forces.gy[j], dt, vy);
                            vz = fmaf(sgn * forces.gz[j], dt, vz);
                            flops += 9;
                        }
                    }
                    kr.kin += 0.5 * sqr(vx, vy, vz);
                    kr.xmom += vx;
                    kr.ymom += vy;
                    kr.zmom += vz;
                    kr.nmom += sqrt(sqr(vx, vy, vz));
                    if (params.descending) {
                        g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]) + 1e-35f;
                        float const factor = eta * sqrtf(params.a);
                        float dt = std::min(
                            factor * sqrtf(gravityEffectiveRadius / sqrtf(g2)), (float) params.t0);
                        dt = std::min(params.maxDt, dt);
                        rung = std::max(params.minRung +
                                int((int) ceilf(log2f(params.t0 / dt)) > params.minRung),
                            (int) (rung - 1));
                        kr.maxRung = std::max((int) rung, kr.maxRung);
                        ALWAYS_ASSERT(rung >= 0);
                        ALWAYS_ASSERT(rung < MAX_RUNG);
                        dt = 0.5f * rungDt[params.minRung] * params.t0;
                        vx = fmaf(sgn * forces.gx[j], dt, vx);
                        vy = fmaf(sgn * forces.gy[j], dt, vy);
                        vz = fmaf(sgn * forces.gz[j], dt, vz);
                        flops += 26;
                    }
                    float kin1 = 0.5 * sqr(vx, vy, vz);
                    kr.pot += 0.5 * forces.phi[j];
                    kr.dkin += kin1 - kin0;
                } else {
                    float const mass = particlesMass(i);
                    bool const gas = getOptions().hydro.enabled && particlesType(i) == SPH_TYPE;
                    float const kin0 = 0.5f * mass * sqr(vx, vy, vz);
                    if (gas) {
                        ParticleInteger const k = particlesCatIndex(i);
                        sphParticlesGravityAcceleration(XDIM, k) = forces.gx[j];
                        sphParticlesGravityAcceleration(YDIM, k) = forces.gy[j];
                        sphParticlesGravityAcceleration(ZDIM, k) = forces.gz[j];
                    } else if (!params.firstCall) {
                        flops += 8;
                        float const dt = 0.5f * rungDt[rung] * params.t0;
                        vx = fmaf(forces.gx[j], dt, vx);
                        vy = fmaf(forces.gy[j], dt, vy);
                        vz = fmaf(forces.gz[j], dt, vz);
                    }
                    kr.kin += 0.5f * mass * sqr(vx, vy, vz);
                    kr.xmom += mass * vx;
                    kr.ymom += mass * vy;
                    kr.zmom += mass * vz;
                    kr.nmom += mass * sqrt(sqr(vx, vy, vz));
                    if (!gas) {
                        flops += 23;    // Timestep/rung selection and velocity kick only.
                        float const syncVx = vx, syncVy = vy, syncVz = vz;
                        float const g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]) + 1e-35f;
                        float dt = std::min(
                            eta * sqrtf(params.a) * sqrtf(gravityEffectiveRadius / sqrtf(g2)),
                            params.t0);
                        dt = std::min(dt, params.maxDt);
                        rung = std::max(std::max((int) ceilf(log2f(params.t0 / dt)),
                                            std::max((int) rung - 1, params.minRung)),
                            1);
                        if (params.fixedRung != 0) {
                            // Preserve the required bound in the result; the driver rejects
                            // an unsafe fixed step before drifting any particle.
                            kr.maxRung = std::max(kr.maxRung, int(rung));
                            rung = params.fixedRung;
                        }
                        ALWAYS_ASSERT(rung >= 0 && rung < MAX_RUNG);
                        dt = 0.5f * rungDt[rung] * params.t0;
                        vx = fmaf(forces.gx[j], dt, vx);
                        vy = fmaf(forces.gy[j], dt, vy);
                        vz = fmaf(forces.gz[j], dt, vz);
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
                flops += detail::cartesian::L2pKernel<kLocalDegree, true>::kFlops_ + 25;
            }
        }
        if (cudaWorkspace != nullptr && cudaWorkspace->isBatch()) {
            cudaWorkspace->accumulateBatchReturn(kr);
            cudaWorkspace->addInteractionCounts(counts);
        } else {
            context.accumulateReturn(kr, selfPtr->numberOfParticles(), cudaWorkspace == nullptr);
            kickAddInteractionCounts(counts);
        }
        addCpuFlops(flops);
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
        TreeNode const* cl = treeGetNode(selfPtr->children[LEFT]);
        TreeNode const* cr = treeGetNode(selfPtr->children[RIGHT]);
        (void) cl;
        (void) cr;
        std::array<hpx::future<KickReturn>, NCHILD> futs;
        futs[RIGHT] = context.fork(params, l, selfPtr->pos, selfPtr->children[RIGHT], dchecklist,
            echecklist, cudaWorkspace, threadLeft);
        try {
            futs[LEFT] = context.fork(params, l, selfPtr->pos, selfPtr->children[LEFT],
                std::move(dchecklist), std::move(echecklist), cudaWorkspace, false);
        } catch (...) {
            // Keep shared tree data alive until the asynchronous sibling exits.
            futs[RIGHT].wait();
            throw;
        }
        hpx::wait_all(futs);
        // Surface launch/completion failures before waiting for a result that
        // only a successful GPU batch can publish.
        auto const rcl = futs[LEFT].get();
        auto const rcr = futs[RIGHT].get();
        if (selfPtr->processRange.second - selfPtr->processRange.first > 1) {
            kr += rcl;
            kr += rcr;
        } else {
            if (selfPtr->localRoot && !params.forceOnly) {
                kr = context.waitReturn();
            }
        }

        if (cudaWorkspace != nullptr && cudaWorkspace->isBatch()) {
            cudaWorkspace->addInteractionCounts(counts);
        } else {
            kickAddInteractionCounts(counts);
        }
        addCpuFlops(flops);
        return kr;
    }
}

#endif
