/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <memory>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fmm_types.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/tree.hpp>

// The CPU tree walk for one FMM order. kick() in kick.cpp is the order-agnostic
// dispatcher: it selects kickWalk<fmmOrder()>, which views the expansion buffer
// as FmmExpansion<float, order>, applies the templated operators and hands
// ExpansionBuffer copies back to kickFork() and the CUDA workspace. The HPX
// action, the fork policy, the per-thread interaction lists and the local-return
// bookkeeping below stay in kick.cpp.

struct KickWorkspace;

// Spawns the walk of a child: synchronously, on a new HPX thread, or remotely
// through the kick action when the child lives on another rank.
hpx::future<KickReturn> kickFork(KickParameters params, ExpansionBuffer const& l,
    Array<Fixed32, NDIM> pos, TreeId self, Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace, bool threadme);

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

// Sum of mass times squared distance from `center` over every particle, all ranks.
double kickBackgroundSecondMoment(Array<Fixed32, NDIM> center);

// Local-return bookkeeping: the local root resets the counters, leaves add their
// returns and the covered particle counts, and the result is published once every
// particle of the local root is covered by a CPU leaf (or by the GPU batch).
void kickBeginLocalReturn(ParticleInteger particleGoal);
void kickCoverParticles(ParticleInteger count);
void kickAccumulateLeafReturn(KickReturn const& kr, ParticleInteger count, bool allowPublish);
KickReturn kickWaitLocalReturn();

}    // namespace detail

template <int order>
KickReturn kickWalk(KickParameters params, ExpansionBuffer l, Array<Fixed32, NDIM> pos, TreeId self,
    Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace);

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definition pulls in the generated kernels, so only the per-order instance
// translation units see it; kick.cpp compiles against the declaration above.

#include <algorithm>
#include <array>
#include <cmath>

#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_acceptance.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stack_trace.hpp>

template <int order>
KickReturn kickWalk(KickParameters params, ExpansionBuffer l, Array<Fixed32, NDIM> pos, TreeId self,
    Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace) {
    using Expansion = FmmExpansion<float, order>;
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    constexpr int kLocalDegree = FmmOrder<order>::kLocalDegree_;
    // The buffer travels with the work item; the view aliases its leading coefficients.
    Expansion& expansion = l.template expansion<order>();
    FlopCounter<int> flops = 0;
    if (self.proc == 0 && self.index == 0) {
        profilerEnter(__FUNCTION__);
    }
    stackTraceActivate();
    TreeNode const* selfPtr = treeGetNode(self);
    if (selfPtr->localRoot) {
        detail::kickBeginLocalReturn(selfPtr->numberOfParticles());
    }
    ASSERT(self.proc == hpxRank());
    if (selfPtr->isLocal() && selfPtr->numberOfParticles() == 0) {
        if (selfPtr->localRoot) {
            // Complete the promise while HPX is alive, including on empty ranks.
            kickSetRc(KickReturn());
        }
        if (self.proc == 0 && self.index == 0) {
            profilerExit();
        }
        return KickReturn();
    }
    if (self.proc == 0 && self.index == 0 && echecklist.size() == 1 && echecklist[0] == self) {
        // The Ewald operators contain only the harmonic part. Add the uniform
        // background once, using canonical coordinates rather than periodic
        // nearest images. Its source trace is needed only for the potential
        // gauge, so compute that scalar here instead of storing it in every
        // source multipole.
        Array<float, NDIM> displacement;
        for (int dim = 0; dim < NDIM; dim++) {
            displacement[dim] = selfPtr->pos[dim].toDouble() - pos[dim].toDouble();
        }
        expansion = l2L(expansion, displacement, params.doPhi);
        pos = selfPtr->pos;
        double const secondMoment = params.doPhi ? detail::kickBackgroundSecondMoment(pos) : 0.0;
        Expansion background;
        background = 0.0f;
        initializeBackgroundExpansion<order>(
            background, fmmMultiPos<order>(selfPtr->mpos).multi, secondMoment);
        for (int i = 0; i < kLocalCoefficients; i++) {
            expansion[i] += background[i];
        }
    }
    bool threadLeft = true;
#ifdef USE_CUDA
    if (getOptions().compute.cuda && params.gpu) {
        if (params.gpu && cudaWorkspace == nullptr && selfPtr->isLocal()) {
            cudaWorkspace = std::make_shared<KickWorkspace>(params, selfPtr->numberOfParticles());
        }
        size_t maximumGpuParticles = CUDA_KICK_PARTS_MAX;
        auto const rng = particlesCurrentRange();
        maximumGpuParticles = std::min((size_t) maximumGpuParticles,
            (size_t) std::max((ParticleInteger) (rng.second - rng.first) / kickBlockCount(),
                (ParticleInteger) getOptions().compute.bucketSize));
        if (params.gpu && selfPtr->numberOfParticles() <= maximumGpuParticles &&
            selfPtr->isLocal() && detail::kickSourcesAreSingleRank(dchecklist) &&
            detail::kickSourcesAreSingleRank(echecklist)) {
            if (selfPtr->numberOfParticles()) {
                detail::kickCoverParticles(selfPtr->numberOfParticles());
                cudaWorkspace->addWork(
                    cudaWorkspace, l, pos, self, std::move(dchecklist), std::move(echecklist));
            }
            if (selfPtr->localRoot) {
                return detail::kickWaitLocalReturn();
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
    bool const saveForce = params.saveForce;
    auto workspace = detail::kickAcquireLists();
    Vector<TreeId>& nextlist = workspace.nextlist;
    Vector<TreeId>& cclist = workspace.cclist;
    Vector<TreeId>& cplist = workspace.cplist;
    Vector<TreeId>& pclist = workspace.pclist;
    Vector<TreeId>& leaflist = workspace.leaflist;
    SimdFloat const thetainv = simdBroadcast<SimdFloat>(fmmInverseOpeningAngle(params.theta));
    Array<TreeNode const*, SIMD_FLOAT_SIZE> otherPtrs;
    Array<float, NDIM> ldx;
    SimdFloat selfRadius = selfPtr->radius;
    Array<SimdInt, NDIM> selfPos{};
    Range<SimdDouble8> selfBox{};
    Array<SimdInt, NDIM> otherPos{};
    Array<SimdFloat, NDIM> dx{};
    SimdFloat otherRadius{};
    SimdFloat otherLeaf{};
    SimdFloat otherSupportRadius{};
    Range<SimdDouble8> otherBox{};
    for (int dim = 0; dim < NDIM; dim++) {
        selfPos[dim] = simdBroadcast<SimdInt>(selfPtr->pos[dim].raw());
        selfBox.begin[dim] = (double) selfPtr->box.begin[dim].raw();
        selfBox.end[dim] = (double) selfPtr->box.end[dim].raw();
    }
    for (int dim = 0; dim < NDIM; dim++) {
        selfBox.begin[dim] *= RangeFixed::kToFloatFactor_;
        selfBox.end[dim] *= RangeFixed::kToFloatFactor_;
    }
    for (int dim = 0; dim < NDIM; dim++) {
        ldx[dim] = selfPtr->pos[dim].toDouble() - pos[dim].toDouble();
    }
    expansion = l2L(expansion, ldx, params.doPhi);
    flops += detail::stf::L2lKernel<kLocalDegree, true>::kFlops_ + 3;
    float const inverseSupportRadius = 1.0f / params.gravitySupportRadius;
    ForceVectors forces;
    ParticleInteger const numberOfSelfParticles = selfPtr->numberOfParticles();
    if (selfPtr->leaf) {
        forces = ForceVectors(numberOfSelfParticles);
        for (ParticleInteger i = 0; i < numberOfSelfParticles; i++) {
            forces.gx[i] = forces.gy[i] = forces.gz[i] = 0.0f;
            ParticleInteger const particleIndex = selfPtr->particleRange.first + i;
            forces.phi[i] = particlesMass(particleIndex) * selfPhi() * inverseSupportRadius;
        }
    }
    {
        nextlist.resize(0);
        cclist.resize(0);
        leaflist.resize(0);
        auto& checklist = echecklist;
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
                }
                for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(otherPos[dim], i, 0.f);
                    }
                    simdSetLane(otherRadius, i, 0.f);
                    simdSetLane(otherLeaf, i, 0);
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(otherBox.begin[dim], i, 0.0);
                        simdSetLane(otherBox.end[dim], i, 0.0);
                    }
                }
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] =
                        simdCast<SimdFloat>(simdPositionDifference(selfPos[dim], otherPos[dim])) *
                        kFixed2float;    // 3
                }
                SimdFloat r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);    // 5
                r2 = simdMax(r2,
                    simdMax(sqr(simdBroadcast<SimdFloat>(0.5) - (selfRadius + otherRadius)),
                        simdBroadcast<SimdFloat>(0)));
                SimdFloat const dcc = (selfRadius + otherRadius) * thetainv;
                auto const cc = r2 > sqr(dcc);
                flops += maxi * 15;
                for (int i = 0; i < maxi; i++) {
                    if (cc[i]) {
                        cclist.push_back(checklist[ci + i]);
                    } else if (otherLeaf[i]) {
                        leaflist.push_back(checklist[ci + i]);
                    } else {
                        auto const childChecks = otherPtrs[i]->children;
                        nextlist.push_back(childChecks[LEFT]);
                        nextlist.push_back(childChecks[RIGHT]);
                    }
                    if (!cc[i]) {
                        ALWAYS_ASSERT(!otherLeaf[i] || !selfPtr->leaf);
                    }
                }
            }
            std::swap(checklist, nextlist);
            nextlist.resize(0);
        } while (checklist.size() && selfPtr->leaf);
        cpuGravityCc<order>(kGravityEwald, expansion, cclist, self, params.doPhi);
        if (!selfPtr->leaf) {
            checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        }
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
                        simdCast<SimdFloat>(simdPositionDifference(selfPos[dim], otherPos[dim])) *
                        kFixed2float;    // 3
                }
                SimdFloat const r2 =
                    simdFma(dx[XDIM], dx[XDIM], simdFma(dx[YDIM], dx[YDIM], dx[ZDIM] * dx[ZDIM]));
                SimdFloat const pairSupportRadius = fmmPairSupportRadius(
                    simdBroadcast<SimdFloat>(supportRadius),
                    simdBroadcast<SimdFloat>(selfPtr->maxGravitySupportRadius), otherSupportRadius);
                auto acceptance =
                    computeFmmAcceptance(r2, selfRadius, otherRadius, simdBroadcast<SimdFloat>(0),
                        simdBroadcast<SimdFloat>(0), pairSupportRadius, thetainv);
                flops += maxi * 20;
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
                        boxDistanceSquared(selfBox, otherPos),
                        boxDistanceSquared(otherBox, selfPos), pairSupportRadius, thetainv);
                    flops += maxi * 34;
                }
                for (int i = 0; i < maxi; i++) {
                    auto const interaction = selectFmmInteraction(bool(acceptance.cellCell[i]),
                        bool(acceptance.particleCell[i]), bool(acceptance.cellParticle[i]),
                        selfPtr->leaf, bool(otherLeaf[i]), selfPtr->radius, otherPtrs[i]->radius);
                    if (interaction == FmmInteraction::CellCell) {
                        cclist.push_back(checklist[ci + i]);
                    } else if (interaction == FmmInteraction::CellParticle) {
                        cplist.push_back(checklist[ci + i]);
                    } else if (interaction == FmmInteraction::ParticleCell) {
                        pclist.push_back(checklist[ci + i]);
                    } else if (otherLeaf[i]) {
                        leaflist.push_back(checklist[ci + i]);
                    } else {
                        auto const childChecks = otherPtrs[i]->children;
                        nextlist.push_back(childChecks[LEFT]);
                        nextlist.push_back(childChecks[RIGHT]);
                    }
                }
            }
            std::swap(checklist, nextlist);
            nextlist.resize(0);
        } while (checklist.size() && selfPtr->leaf);
        cpuGravityCc<order>(kGravityDirect, expansion, cclist, self, params.doPhi);
        // Internal sinks pass these source-leaf contributions to their children via L2L.
        cpuGravityCp<order>(expansion, cplist, self, params.doPhi);
        if (selfPtr->leaf) {
            if (selfPtr->numberOfParticles() > 0) {
                cpuGravityPc<order>(forces, params.doPhi, self, pclist);
                cpuGravityPp(forces, params.doPhi, self, leaflist, params.gravitySupportRadius);
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
            if (particlesRung(i) >= params.minRung) {
                ParticleInteger const j = i - rng.first;
                Array<float, NDIM> dx;
                for (ParticleInteger dim = 0; dim < NDIM; dim++) {
                    dx[dim] = particlesPos(dim, i).toDouble() - selfPtr->pos[dim].toDouble();
                }
                auto const l2 = l2P(expansion, dx, params.doPhi);
                float const gravityEffectiveRadius = params.gravitySupportRadius /
                    selectedGravityKernel().supportRadiusPerEffectiveRadius;
                forces.phi[j] += SCALE_FACTOR1 * l2(0, 0, 0);
                forces.gx[j] -= SCALE_FACTOR2 * l2(1, 0, 0);
                forces.gy[j] -= SCALE_FACTOR2 * l2(0, 1, 0);
                forces.gz[j] -= SCALE_FACTOR2 * l2(0, 0, 1);
                forces.gx[j] *= gm;
                forces.gy[j] *= gm;
                forces.gz[j] *= gm;
                forces.phi[j] *= gm;
                if (saveForce) {
                    particlesGravityAcceleration(XDIM, i) = forces.gx[j];
                    particlesGravityAcceleration(YDIM, i) = forces.gy[j];
                    particlesGravityAcceleration(ZDIM, i) = forces.gz[j];
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
                        g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]) + 1e-35f;    // 6
                        float const factor = eta * sqrtf(params.a);                     // 5
                        float dt = std::min(factor * sqrtf(gravityEffectiveRadius / sqrtf(g2)),
                            (float) params.t0);    // 14
                        dt = std::min(params.maxDt, dt);
                        rung = std::max(params.minRung +
                                int((int) ceilf(log2f(params.t0 / dt)) > params.minRung),
                            (int) (rung - 1));    // 13
                        kr.maxRung = std::max((int) rung, kr.maxRung);
                        ALWAYS_ASSERT(rung >= 0);
                        ALWAYS_ASSERT(rung < MAX_RUNG);
                        dt = 0.5f * rungDt[params.minRung] * params.t0;    // 2
                        vx = fmaf(sgn * forces.gx[j], dt, vx);             // 3
                        vy = fmaf(sgn * forces.gy[j], dt, vy);             // 3
                        vz = fmaf(sgn * forces.gz[j], dt, vz);             // 3
                        flops += 49;
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
                flops += detail::stf::L2pKernel<kLocalDegree, true>::kFlops_ + 16;
            }
        }
        detail::kickAccumulateLeafReturn(
            kr, selfPtr->numberOfParticles(), cudaWorkspace == nullptr);
        addCpuFlops(flops);
        if (cudaWorkspace != nullptr) {
            cudaWorkspace->recordProcessedParticles(cudaWorkspace, selfPtr->numberOfParticles());
        }
        if (selfPtr->localRoot) {
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
        futs[RIGHT] = kickFork(params, l, selfPtr->pos, selfPtr->children[RIGHT], dchecklist,
            echecklist, cudaWorkspace, threadLeft);
        try {
            futs[LEFT] = kickFork(params, l, selfPtr->pos, selfPtr->children[LEFT],
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
            flops += 12;
        } else {
            if (selfPtr->localRoot) {
                kr = detail::kickWaitLocalReturn();
            }
        }
        if (self.proc == 0 && self.index == 0) {
            profilerExit();
        }
        flops += 12;

        addCpuFlops(flops);
        return kr;
    }
}

#endif
