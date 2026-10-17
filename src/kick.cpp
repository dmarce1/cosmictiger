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

#include <stack>

#include <unistd.h>

#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_acceptance.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>

constexpr bool kVerbose = true;

HPX_PLAIN_ACTION(kick, KickAction);

#define MAX_ACTIVE_WORKSPACES 1

struct Workspace {
    Vector<TreeId> nextlist;
    Vector<TreeId> cplist;
    Vector<TreeId> pclist;
    Vector<TreeId> cclist;
    Vector<TreeId> leaflist;
    Workspace() = default;
    Workspace(Workspace const&) = delete;
    Workspace& operator=(Workspace const&) = delete;
    Workspace(Workspace&&) = default;
    Workspace& operator=(Workspace&&) = default;
};

static thread_local std::stack<Workspace> localWorkspaces;
static std::atomic<size_t> particlesCovered;
static hpx::promise<KickReturn> localReturnPromise;
static hpx::future<KickReturn> localReturnFuture;
static KickReturn localReturn;
static MutexType localReturnMutex;
static size_t localParticleGoal;
static bool localReturnSent;

static Workspace getWorkspace() {
    if (localWorkspaces.empty()) {
        localWorkspaces.push(Workspace());
    }
    Workspace workspace = std::move(localWorkspaces.top());
    localWorkspaces.pop();
    return std::move(workspace);
}

static void cleanupWorkspace(Workspace&& workspace) {
    workspace.cclist.resize(0);
    workspace.cplist.resize(0);
    workspace.pclist.resize(0);
    workspace.nextlist.resize(0);
    workspace.leaflist.resize(0);
    localWorkspaces.push(std::move(workspace));
}

static bool sourcesAreSingleRank(Vector<TreeId> const& checklist) {
    for (auto const& source : checklist) {
        if (!treeGetNode(source)->isLocal()) {
            return false;
        }
    }
    return true;
}

hpx::future<KickReturn> kickFork(KickParameters params, Expansion<float> l,
    Array<Fixed32, NDIM> pos, TreeId self, Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace, bool threadme) {
    static std::atomic<int> numberOfThreads(1);
    hpx::future<KickReturn> rc;
    TreeNode const* selfPtr = treeGetNode(self);
    bool remote = false;
    bool allLocal = true;
    for (auto const& i : dchecklist) {
        if (i.proc != hpxRank()) {
            allLocal = false;
            break;
        }
    }
    if (self.proc != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme =
            selfPtr->particleRange.second - selfPtr->particleRange.first > MIN_KICK_THREAD_PARTS;
        int const maximumThreads = std::max(1,
            hpxSize() == 1 ? (int) hpxHardwareConcurrency() * 2 :
                             (int) hpxHardwareConcurrency() / 2);
        if (threadme) {
            if (numberOfThreads++ < maximumThreads || !selfPtr->isLocal()) {
                threadme = true;
            } else {
                threadme = false;
                numberOfThreads--;
            }
        }
    }
    if (!threadme) {
        if (allLocal) {
            hpxYield();
        }
        rc = hpx::make_ready_future(kick(params, l, pos, self, std::move(dchecklist),
            std::move(echecklist), std::move(cudaWorkspace)));
    } else if (remote) {
        rc = hpx::async<KickAction>(hpxLocalities()[selfPtr->processRange.first], params, l, pos,
            self, std::move(dchecklist), std::move(echecklist), nullptr);
    } else {
        if (allLocal) {
            rc = hpx::async(
                [params, self, l, pos, cudaWorkspace](
                    Vector<TreeId> dchecklist, Vector<TreeId> echecklist) {
                    auto rc = kick(params, l, pos, self, std::move(dchecklist),
                        std::move(echecklist), cudaWorkspace);
                    numberOfThreads--;
                    return rc;
                },
                std::move(dchecklist), std::move(echecklist));
        } else {
            rc = hpx::async(
                HPX_PRIORITY_HI,
                [params, self, l, pos, cudaWorkspace](
                    Vector<TreeId> dchecklist, Vector<TreeId> echecklist) {
                    auto rc = kick(params, l, pos, self, std::move(dchecklist),
                        std::move(echecklist), cudaWorkspace);
                    numberOfThreads--;
                    return rc;
                },
                std::move(dchecklist), std::move(echecklist));
        }
    }
    return rc;
}

KickReturn kick(KickParameters params, Expansion<float> l, Array<Fixed32, NDIM> pos, TreeId self,
    Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace) {
    FlopCounter<int> flops = 0;
    if (self.proc == 0 && self.index == 0) {
        profilerEnter(__FUNCTION__);
    }
    stackTraceActivate();
    TreeNode const* selfPtr = treeGetNode(self);
    Timer tm;
    if (selfPtr->localRoot) {
        tm.start();
        particlesCovered = 0;
        localParticleGoal = selfPtr->numberOfParticles();
        localReturnSent = false;
        localReturn = KickReturn();
        localReturnPromise = decltype(localReturnPromise)();
        localReturnFuture = localReturnPromise.get_future();
    }
    ASSERT(self.proc == hpxRank());
    if (selfPtr->isLocal() && selfPtr->numberOfParticles() == 0) {
        if (self.proc == 0 && self.index == 0) {
            profilerExit();
        }
        return KickReturn();
    }
    bool threadLeft = true;
    size_t cudaMemUsage;
    if (getOptions().cuda && params.gpu) {
        if (params.gpu && cudaWorkspace == nullptr && selfPtr->isLocal()) {
            cudaWorkspace = std::make_shared<KickWorkspace>(params, selfPtr->numberOfParticles());
        }
        size_t maximumGpuParticles = CUDA_KICK_PARTS_MAX;
        auto const rng = particlesCurrentRange();
        maximumGpuParticles = std::min((size_t) maximumGpuParticles,
            (size_t) std::max((ParticleInteger) (rng.second - rng.first) / kickBlockCount(),
                (ParticleInteger) getOptions().bucketSize));
        if (params.gpu && selfPtr->numberOfParticles() <= maximumGpuParticles &&
            selfPtr->isLocal() && sourcesAreSingleRank(dchecklist) &&
            sourcesAreSingleRank(echecklist)) {
            if (selfPtr->numberOfParticles()) {
                particlesCovered += selfPtr->numberOfParticles();
                cudaWorkspace->addWork(
                    cudaWorkspace, l, pos, self, std::move(dchecklist), std::move(echecklist));
            }
            if (selfPtr->localRoot) {
                auto kr = localReturnFuture.get();
                return kr;
            } else {
                return KickReturn();
            }
        }
        threadLeft = cudaWorkspace != nullptr;
    }
    KickReturn kr;
    float const supportRadius = params.gravitySupportRadius;
    float const gm = params.gm;
    float const eta = params.eta;
    bool const saveForce = params.saveForce;
    auto workspace = getWorkspace();
    Vector<TreeId>& nextlist = workspace.nextlist;
    Vector<TreeId>& cclist = workspace.cclist;
    Vector<TreeId>& cplist = workspace.cplist;
    Vector<TreeId>& pclist = workspace.pclist;
    Vector<TreeId>& leaflist = workspace.leaflist;
    SimdFloat const thetainv(fmmInverseOpeningAngle(params.theta));
    Array<TreeNode const*, SIMD_FLOAT_SIZE> otherPtrs;
    Array<float, NDIM> ldx;
    SimdFloat selfRadius = selfPtr->radius;
    Array<SimdInt, NDIM> selfPos;
    Range<SimdDouble8> selfBox;
    Array<SimdInt, NDIM> otherPos;
    Array<SimdFloat, NDIM> dx;
    SimdFloat otherRadius;
    SimdFloat otherLeaf;
    SimdFloat otherSupportRadius;
    Range<SimdDouble8> otherBox;
    for (int dim = 0; dim < NDIM; dim++) {
        selfPos[dim] = selfPtr->pos[dim].raw();
        selfBox.begin[dim] = (double) selfPtr->box.begin[dim].raw();
        selfBox.end[dim] = (double) selfPtr->box.end[dim].raw();
    }
    for (int dim = 0; dim < NDIM; dim++) {
        selfBox.begin[dim] *= RangeFixed::kToFloatFactor_;
        selfBox.end[dim] *= RangeFixed::kToFloatFactor_;
    }
    for (int dim = 0; dim < NDIM; dim++) {
        ldx[dim] = distance(selfPtr->pos[dim], pos[dim]);
    }
    l = l2L(l, ldx, params.doPhi);
    flops += 1511 + params.doPhi * 178;
    float const inverseSupportRadius = 1.0f / params.gravitySupportRadius;
    ForceVectors forces;
    ParticleInteger const numberOfSelfParticles = selfPtr->numberOfParticles();
    if (selfPtr->leaf) {
        forces = ForceVectors(numberOfSelfParticles);
        for (ParticleInteger i = 0; i < numberOfSelfParticles; i++) {
            forces.gx[i] = forces.gy[i] = forces.gz[i] = 0.0f;
            forces.phi[i] =
                particlesMass(selfPtr->particleRange.first + i) * selfPhi() * inverseSupportRadius;
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
                        otherPos[dim][i] = otherPtrs[i]->pos[dim].raw();
                    }
                    otherRadius[i] = otherPtrs[i]->radius;
                    otherLeaf[i] = otherPtrs[i]->leaf;
                }
                for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        otherPos[dim][i] = 0.f;
                    }
                    otherRadius[i] = 0.f;
                    otherLeaf[i] = 0;
                    for (int dim = 0; dim < NDIM; dim++) {
                        otherBox.begin[dim][i] = otherBox.end[dim][i] = 0.0;
                    }
                }
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = SimdFloat(selfPos[dim] - otherPos[dim]) * kFixed2float;    // 3
                }
                SimdFloat r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);    // 5
                r2 = max(r2, max(sqr(SimdFloat(0.5) - (selfRadius + otherRadius)), SimdFloat(0)));
                SimdFloat const dcc = (selfRadius + otherRadius) * thetainv;
                SimdFloat const cc = r2 > sqr(dcc);
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
        cpuGravityCc(kGravityEwald, l, cclist, self, params.doPhi);
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
                        otherPos[dim][i] = otherPtrs[i]->pos[dim].raw();
                    }
                    otherRadius[i] = otherPtrs[i]->radius;
                    otherLeaf[i] = otherPtrs[i]->leaf;
                    otherSupportRadius[i] = otherPtrs[i]->maxGravitySupportRadius;
                }
                for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        otherPos[dim][i] = 0.f;
                    }
                    otherRadius[i] = 0.f;
                    otherLeaf[i] = 0;
                    otherSupportRadius[i] = 0.f;
                }
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = SimdFloat(selfPos[dim] - otherPos[dim]) * kFixed2float;    // 3
                }
                SimdFloat const r2 =
                    fmaf(dx[XDIM], dx[XDIM], fmaf(dx[YDIM], dx[YDIM], dx[ZDIM] * dx[ZDIM]));
                SimdFloat const pairSupportRadius = fmmPairSupportRadius(SimdFloat(supportRadius),
                    SimdFloat(selfPtr->maxGravitySupportRadius), otherSupportRadius);
                auto acceptance = computeFmmAcceptance(r2, selfRadius, otherRadius, SimdFloat(0),
                    SimdFloat(0), pairSupportRadius, thetainv);
                flops += maxi * 20;
                if (selfPtr->leaf) {
                    for (int i = 0; i < maxi; i++) {
                        for (int dim = 0; dim < NDIM; dim++) {
                            otherBox.begin[dim][i] = otherPtrs[i]->box.begin[dim].raw();
                            otherBox.end[dim][i] = otherPtrs[i]->box.end[dim].raw();
                        }
                    }
                    for (int dim = 0; dim < NDIM; dim++) {
                        otherBox.begin[dim] *= RangeFixed::kToFloatFactor_;
                        otherBox.end[dim] *= RangeFixed::kToFloatFactor_;
                    }
                    for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
                        for (int dim = 0; dim < NDIM; dim++) {
                            otherBox.begin[dim][i] = otherBox.end[dim][i] = 0.0;
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
                        selfPtr->leaf && bool(otherLeaf[i]), numberOfSelfParticles,
                        otherPtrs[i]->numberOfParticles(), selfPtr->radius, otherPtrs[i]->radius);
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
        cpuGravityCc(kGravityDirect, l, cclist, self, params.doPhi);
        if (selfPtr->leaf) {
            if (selfPtr->numberOfParticles() > 0) {
                cpuGravityCp(l, cplist, self, params.doPhi);
                cpuGravityPc(forces, params.doPhi, self, pclist);
                cpuGravityPp(forces, params.doPhi, self, leaflist, params.gravitySupportRadius);
            }
        } else {
            ALWAYS_ASSERT(cplist.size() == 0);
            checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        }
    }

    if (selfPtr->leaf) {
        cleanupWorkspace(std::move(workspace));
        auto const rng = selfPtr->particleRange;
        for (ParticleInteger i = rng.first; i < rng.second; i++) {
            if (particlesRung(i) >= params.minRung) {
                ParticleInteger const j = i - rng.first;
                Array<float, NDIM> dx;
                for (ParticleInteger dim = 0; dim < NDIM; dim++) {
                    dx[dim] = distance(particlesPos(dim, i), selfPtr->pos[dim]);
                }
                auto const l2 = l2P(l, dx, params.doPhi);
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
                    bool const gas = particlesType(i) == SPH_TYPE;
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
                flops += 570 + params.doPhi * 178;
            }
        }
        KickReturn completed;
        bool publish = false;
        {
            std::lock_guard<MutexType> lock(localReturnMutex);
            particlesCovered += selfPtr->numberOfParticles();
            localReturn += kr;
            if (cudaWorkspace == nullptr && particlesCovered == localParticleGoal &&
                !localReturnSent) {
                completed = localReturn;
                localReturnSent = true;
                publish = true;
            }
        }
        if (publish) {
            localReturnPromise.set_value(completed);
        }
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
        cleanupWorkspace(std::move(workspace));
        TreeNode const* cl = treeGetNode(selfPtr->children[LEFT]);
        TreeNode const* cr = treeGetNode(selfPtr->children[RIGHT]);
        std::array<hpx::future<KickReturn>, NCHILD> futs;
        futs[RIGHT] = kickFork(params, l, selfPtr->pos, selfPtr->children[RIGHT], dchecklist,
            echecklist, cudaWorkspace, threadLeft);
        futs[LEFT] = kickFork(params, l, selfPtr->pos, selfPtr->children[LEFT],
            std::move(dchecklist), std::move(echecklist), cudaWorkspace, false);
        if (selfPtr->processRange.second - selfPtr->processRange.first > 1) {
            auto const rcl = futs[LEFT].get();
            auto const rcr = futs[RIGHT].get();
            kr += rcl;
            kr += rcr;
            flops += 12;
        } else {
            if (selfPtr->localRoot) {
                kr = localReturnFuture.get();
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

void kickSetRc(KickReturn kr) {
    KickReturn completed;
    {
        std::lock_guard<MutexType> lock(localReturnMutex);
        localReturn += kr;
        completed = localReturn;
        localReturnSent = true;
    }
    localReturnPromise.set_value(completed);
}
