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

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed_capacity_vector.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_acceptance.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>

struct CudaKickParameters {
    Array<Fixed32, NDIM> lpos;
    Expansion<float> lValue;
    int self;
    int* dchecks;
    int* echecks;
    int dcount;
    int ecount;
};

namespace detail {

__device__ inline void cudaCopyExpansion(float* destination, float const* source) {
    for (int i = threadIdx.x; i < kFmmLocalCoefficients; i += WARP_SIZE) {
        destination[i] = source[i];
    }
}

}    // namespace detail

__device__ void doKick(KickReturn& result, KickParameters params, CudaKickData const& data,
    Expansion<float> const& lValue2, TreeNode const& self, CudaKickForceTile const& force,
    int sinkBegin, int sinkCount) {
    int const& tid = threadIdx.x;
    auto& allPhi = data.pot;
    auto& allGx = data.gx;
    auto& allGy = data.gy;
    auto& allGz = data.gz;
    auto* __restrict__ rungs = data.rungs;
    auto* __restrict__ vels = data.vel;
    ParticleInteger const sinkFirst = self.particleRange.first + sinkBegin;
    auto const* sinkX = data.x + sinkFirst;
    auto const* sinkY = data.y + sinkFirst;
    auto const* sinkZ = data.z + sinkFirst;
    int maxRung = 0;
    float phiTot = 0.0f;
    float kinTot = 0.f;
    float xmomTot = 0.0f;
    float ymomTot = 0.0f;
    float zmomTot = 0.0f;
    float nmomTot = 0.0f;
    float dkinTot = 0.f;
    float darkMatterKineticCorrection = 0.0f;
    float darkMatterXMomentumCorrection = 0.0f;
    float darkMatterYMomentumCorrection = 0.0f;
    float darkMatterZMomentumCorrection = 0.0f;
    float darkMatterMomentumNormCorrection = 0.0f;

    float const gravityEffectiveRadius =
        params.gravitySupportRadius / selectedGravityKernel().supportRadiusPerEffectiveRadius;
    FlopCounter<int> flops = 0;
    for (int i = tid; i < sinkCount; i += WARP_SIZE) {
        Expansion2<float> l2;
        Array<float, NDIM> dx;
        ParticleInteger sinkIndex;
        float vx;
        float vy;
        float vz;
        float dt;
        float g2;
        int rung;
        sinkIndex = sinkFirst + i;
        if (!params.htime && rungs[sinkIndex] < params.minRung) {
            continue;
        }
        ASSERT(sinkIndex >= 0);
        dx[XDIM] = sinkX[i].toDouble() - self.pos[XDIM].toDouble();
        dx[YDIM] = sinkY[i].toDouble() - self.pos[YDIM].toDouble();
        dx[ZDIM] = sinkZ[i].toDouble() - self.pos[ZDIM].toDouble();
        l2 = l2P(lValue2, dx, params.doPhi);
        float gx = force.gx[i];
        float gy = force.gy[i];
        float gz = force.gz[i];
        float phi = force.phi[i];
        phi += SCALE_FACTOR1 * l2(0, 0, 0);
        gx -= SCALE_FACTOR2 * l2(1, 0, 0);
        gy -= SCALE_FACTOR2 * l2(0, 1, 0);
        gz -= SCALE_FACTOR2 * l2(0, 0, 1);
        gz *= params.gm;
        gy *= params.gm;
        gx *= params.gm;
        phi *= params.gm;
        vx = vels[sinkIndex][XDIM];
        vy = vels[sinkIndex][YDIM];
        vz = vels[sinkIndex][ZDIM];
        float const mass = data.sph ?
            (data.type[sinkIndex] == DARK_MATTER_TYPE ? data.darkMatterMass : data.sphMass) :
            1.0f;
        float const kin0 = 0.5f * mass * sqr(vx, vy, vz);
        if (params.htime) {
            float const sgn = params.sign * (params.top ? 1.f : -1.f);
            if (params.ascending) {
                dt = 0.5f * rungDt[params.minRung] * params.t0;
                flops += 2;
                if (!params.firstCall) {
                    vx = fmaf(sgn * gx, dt, vx);
                    vy = fmaf(sgn * gy, dt, vy);
                    vz = fmaf(sgn * gz, dt, vz);
                    flops += 6;
                }
            }
            kinTot += 0.5f * sqr(vx, vy, vz);
            xmomTot += vx;
            ymomTot += vy;
            zmomTot += vz;
            nmomTot += sqrtf(sqr(vx, vy, vz));
            if (params.saveForce) {
                allGx[sinkIndex] = gx;
                allGy[sinkIndex] = gy;
                allGz[sinkIndex] = gz;
                allPhi[sinkIndex] = phi;
            }

            if (params.descending) {
                g2 = sqr(gx, gy, gz);
                dt = fminf(params.eta * sqrt(params.a * gravityEffectiveRadius * rsqrtf(g2)),
                    params.t0);    // 12
                dt = fminf(params.maxDt, dt);
                rung = rungs[sinkIndex];
                rung =
                    max(params.minRung + int((int) ceilf(log2f(params.t0 / dt)) > params.minRung),
                        rung - 1);    // 13
                rungs[sinkIndex] = rung;
                maxRung = max(rung, maxRung);
                ALWAYS_ASSERT(rung >= 0);
                ALWAYS_ASSERT(rung < MAX_RUNG);
                dt = 0.5f * rungDt[params.minRung] * params.t0;    // 2
                vx = fmaf(sgn * gx, dt, vx);                       // 3
                vy = fmaf(sgn * gy, dt, vy);                       // 3
                vz = fmaf(sgn * gz, dt, vz);                       // 3
                flops += 36;
            }
        } else {
            bool const gas = data.sph && data.type[sinkIndex] == SPH_TYPE;
            if (gas) {
                ParticleInteger const k = data.catIndex[sinkIndex];
                data.sphGravityAccelerationX[k] = gx;
                data.sphGravityAccelerationY[k] = gy;
                data.sphGravityAccelerationZ[k] = gz;
            } else if (!params.firstCall) {
                float const dt = 0.5f * rungDt[rungs[sinkIndex]] * params.t0;
                vx = fmaf(gx, dt, vx);
                vy = fmaf(gy, dt, vy);
                vz = fmaf(gz, dt, vz);
            }
            kinTot += 0.5f * mass * sqr(vx, vy, vz);
            xmomTot += mass * vx;
            ymomTot += mass * vy;
            zmomTot += mass * vz;
            nmomTot += mass * sqrtf(sqr(vx, vy, vz));
            if (params.saveForce) {
                allGx[sinkIndex] = gx;
                allGy[sinkIndex] = gy;
                allGz[sinkIndex] = gz;
                allPhi[sinkIndex] = phi;
            }
            if (!gas) {
                float const syncVx = vx, syncVy = vy, syncVz = vz;
                float const g2 = sqr(gx, gy, gz) + 1e-35f;
                float dt =
                    fminf(params.eta * sqrtf(params.a) * sqrtf(gravityEffectiveRadius / sqrtf(g2)),
                        params.t0);
                dt = fminf(dt, params.maxDt);
                rung = max(max((int) ceilf(log2f(params.t0 / dt)),
                               max((int) rungs[sinkIndex] - 1, params.minRung)),
                    1);
                if (params.fixedRung != 0) {
                    maxRung = max(maxRung, int(rung));
                    rung = params.fixedRung;
                }
                ALWAYS_ASSERT(rung >= 0 && rung < MAX_RUNG);
                rungs[sinkIndex] = rung;
                dt = 0.5f * rungDt[rung] * params.t0;
                vx = fmaf(gx, dt, vx);
                vy = fmaf(gy, dt, vy);
                vz = fmaf(gz, dt, vz);
                darkMatterKineticCorrection +=
                    0.5f * mass * (sqr(syncVx, syncVy, syncVz) - sqr(vx, vy, vz));
                darkMatterXMomentumCorrection += mass * (syncVx - vx);
                darkMatterYMomentumCorrection += mass * (syncVy - vy);
                darkMatterZMomentumCorrection += mass * (syncVz - vz);
                darkMatterMomentumNormCorrection +=
                    mass * (sqrtf(sqr(syncVx, syncVy, syncVz)) - sqrtf(sqr(vx, vy, vz)));
            }
            maxRung = max(maxRung, (int) rungs[sinkIndex]);
        }
        float kin1 = 0.5 * mass * sqr(vx, vy, vz);
        vels[sinkIndex][XDIM] = vx;
        vels[sinkIndex][YDIM] = vy;
        vels[sinkIndex][ZDIM] = vz;
        phiTot += 0.5f * mass * phi;
        dkinTot += kin1 - kin0;
        flops += detail::stf::L2pKernel<kFmmLocalDegree, true>::kFlops_ + 3;
    }
    sharedReduceAdd(phiTot);
    sharedReduceAdd(xmomTot);
    sharedReduceAdd(ymomTot);
    sharedReduceAdd(zmomTot);
    sharedReduceAdd(nmomTot);
    sharedReduceAdd(kinTot);
    sharedReduceAdd(dkinTot);
    sharedReduceAdd(darkMatterKineticCorrection);
    sharedReduceAdd(darkMatterXMomentumCorrection);
    sharedReduceAdd(darkMatterYMomentumCorrection);
    sharedReduceAdd(darkMatterZMomentumCorrection);
    sharedReduceAdd(darkMatterMomentumNormCorrection);

    flops += 30;
    sharedReduceMax(maxRung);
    if (tid == 0) {
        result.maxRung = max(result.maxRung, maxRung);
        result.pot += phiTot;
        result.xmom += xmomTot;
        result.ymom += ymomTot;
        result.zmom += zmomTot;
        result.nmom += nmomTot;
        result.kin += kinTot;
        result.dkin += dkinTot;
        result.darkMatterKineticCorrection += darkMatterKineticCorrection;
        result.darkMatterXMomentumCorrection += darkMatterXMomentumCorrection;
        result.darkMatterYMomentumCorrection += darkMatterYMomentumCorrection;
        result.darkMatterZMomentumCorrection += darkMatterZMomentumCorrection;
        result.darkMatterMomentumNormCorrection += darkMatterMomentumNormCorrection;

        flops += 6;
    }
    addGpuFlops(flops);
}

// One 32-thread warp per block. The dynamic shared memory follows the host
// computed CudaKickArenaLayout; see kick.hpp for the region lifetimes.
__global__ void cudaKickKernel(KickReturn* rc, KickParameters globalParams, CudaKickData data,
    CudaKickParameters* params, int itemCount, int* nextItem, CudaKickArenaLayout layout) {
    int const& tid = threadIdx.x;
    extern __shared__ __align__(16) char shmemPtr[];
    CudaKickControl& control = *reinterpret_cast<CudaKickControl*>(shmemPtr + layout.controlOffset);
    new (&control) CudaKickControl;
    char* arena = shmemPtr + layout.arenaOffset;
    bool const sharedExpansion = layout.expansionOffset >= 0;
    if (!sharedExpansion) {
        control.current.resize(1);
    }
    // Every lane ran the constructor; only lane zero fills the resolved layout.
    __syncwarp();
    if (tid == 0) {
        control.layout = layout;
        control.sourceX = reinterpret_cast<Fixed32*>(arena + layout.sourceOffset);
        control.sourceY = control.sourceX + layout.particleCapacity;
        control.sourceZ = control.sourceY + layout.particleCapacity;
        control.sourceMass = layout.massPlane ?
            reinterpret_cast<float*>(control.sourceZ + layout.particleCapacity) :
            nullptr;
        control.multipoles = arena + layout.sourceOffset;
        control.scratchExpansion = reinterpret_cast<float*>(arena + layout.scratchOffset);
        if (sharedExpansion) {
            char* expansionBytes = shmemPtr + layout.expansionOffset;
            control.expansion = reinterpret_cast<Expansion<float>*>(expansionBytes);
            control.expansionPos =
                reinterpret_cast<Array<Fixed32, NDIM>*>(expansionBytes + kFmmLocalExpansionBytes);
        } else {
            control.expansion = &control.current[0].expansion;
            control.expansionPos = &control.current[0].pos;
        }
        if (layout.forceOffset >= 0) {
            control.forces.gx = reinterpret_cast<float*>(arena + layout.forceOffset);
            control.forces.gy = control.forces.gx + layout.sinkCapacity;
            control.forces.gz = control.forces.gy + layout.sinkCapacity;
            control.forces.phi = control.forces.gz + layout.sinkCapacity;
        }
    }
    __syncwarp();
    auto& sparams = control.params;
    auto& ancestors = control.ancestors;
    auto& dchecks = control.dchecks;
    auto& echecks = control.echecks;
    auto& nextlist = control.nextlist;
    auto& cclist = control.cclist;
    auto& cplist = control.cplist;
    auto& pclist = control.pclist;
    auto& leaflist = control.leaflist;
    auto& barrier = control.barrier;
    Expansion<float>& expansion = *control.expansion;
    Array<Fixed32, NDIM>& expansionPos = *control.expansionPos;
    float* scratch = control.scratchExpansion;
    auto& treeNodes = data.treeNodes;
    bool const equalMass = !data.sph;
    bool const sharedForces = layout.forceOffset >= 0;
    float const& supportRadius = globalParams.gravitySupportRadius;
    float const inverseSupportRadius = 1.f / supportRadius;
    int index;
    auto group = cooperative_groups::this_thread_block();
    if (group.thread_rank() == 0) {
        init(&barrier, group.size());
    }
    group.sync();
    if (tid == 0) {
        index = atomicAdd(nextItem, 1);
    }
    index = __shfl_sync(0xFFFFFFFF, index, 0);
    __syncthreads();
    KickReturn kr;
    while (index < itemCount) {
        FlopCounter<int> flops = 0;
        ancestors.resize(0);
        dchecks.resize(0);
        echecks.resize(0);
        sparams.resize(0);
        detail::cudaCopyExpansion(&expansion[0], &params[index].lValue[0]);
        if (tid < NDIM) {
            expansionPos[tid] = params[index].lpos[tid];
        }
        dchecks.resize(params[index].dcount);
        echecks.resize(params[index].ecount);
        for (int i = tid; i < params[index].dcount; i += WARP_SIZE) {
            dchecks[i] = params[index].dchecks[i];
        }
        for (int i = tid; i < params[index].ecount; i += WARP_SIZE) {
            echecks[i] = params[index].echecks[i];
        }
        {
            SearchParameters sparam;
            sparam.phase = 0;
            sparam.self = params[index].self;
            sparams.pushBack(sparam);
        }
        __syncwarp();
        int depth = 0;
        while (depth >= 0) {
            auto const& self = treeNodes[sparams.back().self];
            switch (sparams.back().phase) {
            case 0: {
                Array<float, NDIM> dx;
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = self.pos[dim].toDouble() - expansionPos[dim].toDouble();
                }
                flops += 3;
                {
                    // Cooperative translation into the scratch expansion, then
                    // publish it as the active expansion at this node's center.
                    l2LCudaInto(scratch, expansion, dx, globalParams.doPhi);
                    detail::cudaCopyExpansion(&expansion[0], scratch);
                    if (tid < NDIM) {
                        expansionPos[tid] = self.pos[tid];
                    }
                    __syncwarp();
                    if (tid == 0) {
                        flops += globalParams.doPhi ?
                            detail::stf::L2lWarpKernel<kFmmLocalDegree, true>::kFlops_ :
                            detail::stf::L2lWarpKernel<kFmmLocalDegree, true>::kFlopsNoPotential_;
                        flops += 3 * WARP_SIZE;
                    }
                }
                ParticleInteger const numberOfSinks =
                    self.particleRange.second - self.particleRange.first;
                if (numberOfSinks) {
                    nextlist.resize(0);
                    cclist.resize(0);
                    leaflist.resize(0);
                    constexpr int kCc = 0;
                    constexpr int kLeaf = 1;
                    constexpr int kNext = 2;
                    constexpr int kNlists = 3;
                    Array<bool, kNlists> sw;
                    DeviceVector<int>* lists[kNlists] = {&cclist, &leaflist, &nextlist};

                    auto& checks = echecks;
                    float const thetainv = fmmInverseOpeningAngle(globalParams.theta);
                    do {
                        int const maxi = roundUp(checks.size(), WARP_SIZE);
                        for (int i = tid; i < maxi; i += WARP_SIZE) {
                            for (int n = 0; n < kNlists; n++) {
                                sw[n] = false;
                            }
                            if (i < checks.size()) {
                                TreeNode const& other = treeNodes[checks[i]];
                                if (other.numberOfParticles()) {
                                    for (int dim = 0; dim < NDIM; dim++) {
                                        dx[dim] = distance(self.pos[dim], other.pos[dim]);    // 3
                                    }
                                    float r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                                    r2 = fmaxf(
                                        r2, sqr(fmaxf(0.5f - (self.radius + other.radius), 0.f)));
                                    float const dcc = (self.radius + other.radius) * thetainv;
                                    sw[kCc] = r2 > sqr(dcc);
                                    flops += 24;
                                    if (!sw[kCc]) {
                                        sw[kLeaf] = other.leaf;
                                        sw[kNext] = !other.leaf;
                                        ALWAYS_ASSERT(!other.leaf || !self.leaf);
                                    }
                                }
                            }
                            {
                                Array<int, kNlists> l, total;
                                for (int n = 0; n < kNlists; n++) {
                                    l[n] = sw[n];
                                }
                                computeIndicesArray(l, total);
                                for (int n = 0; n < kNlists; n++) {
                                    int const start = lists[n]->size();
                                    __syncwarp();
                                    lists[n]->resize(start + total[n]);
                                    if (sw[n]) {
                                        (*lists[n])[l[n] + start] = checks[i];
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        checks.resize(NCHILD * nextlist.size());
                        for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
                            auto const& node = treeNodes[nextlist[i]];
                            auto const& children = node.children;
                            checks[NCHILD * i + LEFT] = children[LEFT].index;
                            checks[NCHILD * i + RIGHT] = children[RIGHT].index;
                        }
                        nextlist.resize(0);
                        __syncwarp();
                    } while (checks.size() && self.leaf);
                    cudaGravityCcEwald(data, expansion, self, cclist, globalParams.doPhi);
                    if (!self.leaf) {
                        int const start = checks.size();
                        checks.resize(start + leaflist.size());
                        for (int i = tid; i < leaflist.size(); i += WARP_SIZE) {
                            checks[start + i] = leaflist[i];
                        }
                        __syncwarp();
                    }
                }
                if (numberOfSinks) {
                    nextlist.resize(0);
                    leaflist.resize(0);
                    cclist.resize(0);
                    cplist.resize(0);
                    pclist.resize(0);
                    constexpr int kCc = 0;
                    constexpr int kLeaf = 1;
                    constexpr int kCp = 2;
                    constexpr int kPc = 3;
                    constexpr int kNext = 4;
                    constexpr int kNlists = 5;
                    Array<bool, kNlists> sw;
                    DeviceVector<int>* lists[kNlists] = {
                        &cclist, &leaflist, &cplist, &pclist, &nextlist};
                    auto& checks = dchecks;
                    float const thetainv = fmmInverseOpeningAngle(globalParams.theta);
                    do {
                        int const maxi = roundUp(checks.size(), WARP_SIZE);
                        for (int i = tid; i < maxi; i += WARP_SIZE) {
                            for (int n = 0; n < kNlists; n++) {
                                sw[n] = false;
                            }
                            if (i < checks.size()) {
                                TreeNode const& other = treeNodes[checks[i]];
                                if (other.numberOfParticles()) {
                                    for (int dim = 0; dim < NDIM; dim++) {
                                        dx[dim] = distance(self.pos[dim], other.pos[dim]);    // 3
                                    }
                                    float const r2 = fmaf(dx[XDIM], dx[XDIM],
                                        fmaf(dx[YDIM], dx[YDIM], dx[ZDIM] * dx[ZDIM]));
                                    float const pairSupportRadius = fmmPairSupportRadius(
                                        supportRadius, self.maxGravitySupportRadius,
                                        other.maxGravitySupportRadius);
                                    auto acceptance = computeFmmAcceptance(r2, self.radius,
                                        other.radius, 0.0f, 0.0f, pairSupportRadius, thetainv);
                                    flops += 20;
                                    if (!acceptance.cellCell && (other.leaf || self.leaf)) {
                                        acceptance = computeFmmAcceptance(r2, self.radius,
                                            other.radius, boxDistanceSquared(self.box, other.pos),
                                            boxDistanceSquared(other.box, self.pos),
                                            pairSupportRadius, thetainv);
                                        flops += 33;
                                    }
                                    auto const interaction = selectFmmInteraction(
                                        bool(acceptance.cellCell), bool(acceptance.particleCell),
                                        bool(acceptance.cellParticle), self.leaf, other.leaf,
                                        self.radius, other.radius);
                                    sw[kCc] = interaction == FmmInteraction::CellCell;
                                    sw[kPc] = interaction == FmmInteraction::ParticleCell;
                                    sw[kCp] = interaction == FmmInteraction::CellParticle;
                                    if (!sw[kCc] && !sw[kCp] && !sw[kPc]) {
                                        sw[kLeaf] = other.leaf;
                                        sw[kNext] = !sw[kLeaf];
                                    }
                                }
                            }
                            {
                                Array<int, kNlists> l, total;
                                for (int n = 0; n < kNlists; n++) {
                                    l[n] = sw[n];
                                }
                                computeIndicesArray(l, total);
                                for (int n = 0; n < kNlists; n++) {
                                    int const start = lists[n]->size();
                                    __syncwarp();
                                    lists[n]->resize(start + total[n]);
                                    if (sw[n]) {
                                        (*lists[n])[l[n] + start] = checks[i];
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        checks.resize(NCHILD * nextlist.size());
                        for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
                            auto const& node = treeNodes[nextlist[i]];
                            auto const& children = node.children;
                            checks[NCHILD * i + LEFT] = children[LEFT].index;
                            checks[NCHILD * i + RIGHT] = children[RIGHT].index;
                        }
                        nextlist.resize(0);
                        __syncwarp();

                    } while (checks.size() && self.leaf);
                    if (self.leaf) {
                        __syncwarp();
                        cudaGravityCcDirect(data, expansion, self, cclist, globalParams.doPhi);
                        if (equalMass) {
                            cudaGravityCpDirect<true>(
                                data, control, expansion, self, cplist, globalParams.doPhi);
                        } else {
                            cudaGravityCpDirect<false>(
                                data, control, expansion, self, cplist, globalParams.doPhi);
                        }
                        // The cell interactions are complete, so the arena scratch is
                        // free: initialize the sink forces and stream the sink tiles.
                        if (!sharedForces) {
                            control.forceBacking.resize(4 * numberOfSinks);
                            if (tid == 0) {
                                control.forces.gx = control.forceBacking.data();
                                control.forces.gy = control.forces.gx + numberOfSinks;
                                control.forces.gz = control.forces.gy + numberOfSinks;
                                control.forces.phi = control.forces.gz + numberOfSinks;
                            }
                            __syncwarp();
                        }
                        int const tile = sharedForces ? layout.sinkCapacity : numberOfSinks;
                        for (int sinkBegin = 0; sinkBegin < numberOfSinks; sinkBegin += tile) {
                            int const sinkCount = min(tile, int(numberOfSinks - sinkBegin));
                            auto const& force = control.forces;
                            for (int l = tid; l < sinkCount; l += WARP_SIZE) {
                                ParticleInteger const sinkIndex =
                                    self.particleRange.first + sinkBegin + l;
                                float const mass = data.sph ?
                                    (data.type[sinkIndex] == DARK_MATTER_TYPE ?
                                            data.darkMatterMass :
                                            data.sphMass) :
                                    1.0f;
                                force.gx[l] = force.gy[l] = force.gz[l] = 0.f;
                                force.phi[l] = mass * selfPhi() * inverseSupportRadius;
                            }
                            __syncwarp();
                            cudaGravityPcDirect(data, control, self, sinkBegin, sinkCount, pclist,
                                globalParams.doPhi);
                            if (equalMass) {
                                cudaGravityPpDirect<true>(data, control, self, sinkBegin, sinkCount,
                                    leaflist, supportRadius, globalParams.doPhi);
                            } else {
                                cudaGravityPpDirect<false>(data, control, self, sinkBegin,
                                    sinkCount, leaflist, supportRadius, globalParams.doPhi);
                            }
                            __syncwarp();
                            doKick(kr, globalParams, data, expansion, self, force, sinkBegin,
                                sinkCount);
                            // The next tile reinitializes the force planes.
                            __syncwarp();
                        }
                    } else {
                        int const start = checks.size();
                        checks.resize(start + leaflist.size());
                        cuda::memcpy_async(group, checks.data() + start, leaflist.data(),
                            leaflist.size() * sizeof(int), barrier);
                        cudaGravityCcDirect(data, expansion, self, cclist, globalParams.doPhi);
                        barrier.arrive_and_wait();
                        // CP reuses the barrier, so finish the checklist copy before calling it.
                        if (equalMass) {
                            cudaGravityCpDirect<true>(
                                data, control, expansion, self, cplist, globalParams.doPhi);
                        } else {
                            cudaGravityCpDirect<false>(
                                data, control, expansion, self, cplist, globalParams.doPhi);
                        }
                    }
                }
                if (self.leaf) {
                    __syncwarp();
                    sparams.popBack();
                    depth--;
                } else {
                    // Keep this node's expansion for the right child while the
                    // left child translates the active copy.
                    TreeId const child = self.children[LEFT];
                    int const top = ancestors.size();
                    ancestors.resize(top + 1);
                    detail::cudaCopyExpansion(&ancestors[top].expansion[0], &expansion[0]);
                    if (tid < NDIM) {
                        ancestors[top].pos[tid] = expansionPos[tid];
                    }
                    __syncwarp();
                    dchecks.pushTop();
                    echecks.pushTop();
                    sparams.back().phase += 1;
                    {
                        SearchParameters sparam;
                        sparam.phase = 0;
                        sparam.self = child.index;
                        sparams.pushBack(sparam);
                    }
                    depth++;
                }

            } break;
            case 1: {
                // Restore this node's expansion for the right child.
                int const top = ancestors.size() - 1;
                detail::cudaCopyExpansion(&expansion[0], &ancestors[top].expansion[0]);
                if (tid < NDIM) {
                    expansionPos[tid] = ancestors[top].pos[tid];
                }
                __syncwarp();
                ancestors.popBack();
                dchecks.popTop();
                echecks.popTop();
                sparams.back().phase += 1;
                {
                    SearchParameters sparam;
                    sparam.phase = 0;
                    sparam.self = self.children[RIGHT].index;
                    sparams.pushBack(sparam);
                }
                depth++;
            } break;
            case 2: {
                sparams.popBack();
                depth--;
            } break;
            }
        }

        if (tid == 0) {
            index = atomicAdd(nextItem, 1);
        }
        index = __shfl_sync(0xFFFFFFFF, index, 0);
        ASSERT(ancestors.size() == 0);
        addGpuFlops(flops);
    }
    if (tid == 0) {
        atomicAdd(&rc->dkin, kr.dkin);
        atomicAdd(&rc->kin, kr.kin);
        atomicAdd(&rc->pot, kr.pot);
        atomicAdd(&rc->xmom, kr.xmom);
        atomicAdd(&rc->ymom, kr.ymom);
        atomicAdd(&rc->zmom, kr.zmom);
        atomicAdd(&rc->nmom, kr.nmom);
        atomicAdd(&rc->darkMatterKineticCorrection, kr.darkMatterKineticCorrection);
        atomicAdd(&rc->darkMatterXMomentumCorrection, kr.darkMatterXMomentumCorrection);
        atomicAdd(&rc->darkMatterYMomentumCorrection, kr.darkMatterYMomentumCorrection);
        atomicAdd(&rc->darkMatterZMomentumCorrection, kr.darkMatterZMomentumCorrection);
        atomicAdd(&rc->darkMatterMomentumNormCorrection, kr.darkMatterMomentumNormCorrection);

        atomicMax(&rc->maxRung, kr.maxRung);
    }
    control.~CudaKickControl();
}

namespace detail {

constexpr int kKickSharedAllocationGranularity = 128;
constexpr int kKickArenaAlignment = 16;
constexpr int kKickDefaultSinkTile = 128;
constexpr int kKickDefaultParticleTile = 512;
constexpr int kKickDefaultMultipoleTile = 32;
constexpr int kKickMaximumMultipoleTile = 64;
constexpr int kKickMinimumParticleTile = WARP_SIZE;

struct KickDeviceResources {
    int numberOfRegisters;
    int staticSharedBytes;
    int sharedPerMultiprocessor;
    int reservedSharedPerBlock;
    int maximumSharedPerBlock;
    int registersPerMultiprocessor;
    int maximumBlocksPerMultiprocessor;
    int maximumWarpsPerMultiprocessor;
};

KickDeviceResources kickDeviceResources() {
    KickDeviceResources resources;
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (void const*) cudaKickKernel));
    resources.numberOfRegisters = attr.numRegs;
    resources.staticSharedBytes = int(attr.sharedSizeBytes);
    int const device = cudaGetDevice();
    int threads;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.sharedPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.reservedSharedPerBlock, cudaDevAttrReservedSharedMemoryPerBlock, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.maximumSharedPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.registersPerMultiprocessor, cudaDevAttrMaxRegistersPerMultiprocessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.maximumBlocksPerMultiprocessor, cudaDevAttrMaxBlocksPerMultiprocessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    resources.maximumWarpsPerMultiprocessor = threads / WARP_SIZE;
    return resources;
}

int kickRoundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

int kickRoundDown(int value, int multiple) {
    return (value / multiple) * multiple;
}

// Warps allocate registers in units of 256, with per-thread counts rounded to 8.
int kickRegisterResidencyLimit(KickDeviceResources const& resources) {
    int const registersPerWarp =
        kickRoundUp(kickRoundUp(resources.numberOfRegisters, 8) * WARP_SIZE, 256);
    int limit = resources.registersPerMultiprocessor / std::max(1, registersPerWarp);
    limit = std::min(limit, resources.maximumBlocksPerMultiprocessor);
    limit = std::min(limit, resources.maximumWarpsPerMultiprocessor);
    return std::max(1, limit);
}

// Dynamic shared bytes one block may use when `residency` blocks share a
// multiprocessor, after the driver reservation, static shared memory and the
// allocation granularity.
int kickSharedBudget(KickDeviceResources const& resources, int residency) {
    int const perBlock = resources.sharedPerMultiprocessor / residency;
    int budget = kickRoundDown(perBlock, kKickSharedAllocationGranularity);
    budget -= resources.reservedSharedPerBlock;
    budget -= kickRoundUp(resources.staticSharedBytes, kKickArenaAlignment);
    budget = kickRoundDown(budget, kKickSharedAllocationGranularity);
    return std::min(budget, resources.maximumSharedPerBlock - resources.staticSharedBytes);
}

// Fill the arena for a budget; returns false when the minimum tiles do not fit.
bool kickBuildLayout(Compute const& compute, bool massPlane, int residency, int budget,
    CudaKickArenaLayout& layout) {
    layout = CudaKickArenaLayout{};
    layout.residency = residency;
    layout.budgetBytes = budget;
    layout.massPlane = massPlane;
    int const controlBytes = kickRoundUp(int(sizeof(CudaKickControl)), kKickArenaAlignment);
    layout.controlOffset = 0;
    int offset = controlBytes;
    if (compute.cudaKickSharedExpansion) {
        layout.expansionOffset = offset;
        offset +=
            kickRoundUp(kFmmLocalExpansionBytes + NDIM * int(sizeof(Fixed32)), kKickArenaAlignment);
    } else {
        layout.expansionOffset = -1;
    }
    layout.arenaOffset = offset;
    int const scratchBytes = kickRoundUp(kFmmLocalExpansionBytes, kKickArenaAlignment);
    int const arenaBudget = budget - offset;
    if (arenaBudget < scratchBytes) {
        return false;
    }
    // Leaves never exceed the runtime bucket, which is capped by kMaxBucketSize,
    // so an automatic tile first tries to hold the whole bucket and only then
    // falls back to bounded sink tiles.
    int const bucket = std::min(compute.bucketSize, kMaxBucketSize);
    bool const sinkAuto = compute.cudaKickSinkTile == 0;
    int sinkCapacity = 0;
    if (compute.cudaKickSinkTile > 0) {
        sinkCapacity = kickRoundUp(compute.cudaKickSinkTile, WARP_SIZE);
    } else if (sinkAuto) {
        sinkCapacity = kickRoundUp(bucket, WARP_SIZE);
    }
    int const particleBytesPerSource =
        int(sizeof(Fixed32)) * NDIM + (massPlane ? int(sizeof(float)) : 0);
    layout.multipoleStride = kickRoundUp(int(sizeof(MultiPos)), kKickArenaAlignment);
    int const multipoleTarget = kKickDefaultMultipoleTile;
    bool const particleAuto = compute.cudaKickParticleTile <= 0;
    bool const multipoleAuto = compute.cudaKickMultipoleTile <= 0;
    int particleCapacity = particleAuto ? kKickDefaultParticleTile :
                                          kickRoundUp(compute.cudaKickParticleTile, WARP_SIZE);
    int multipoleCapacity = multipoleAuto ? multipoleTarget : compute.cudaKickMultipoleTile;
    while (true) {
        int const forceBytes = 4 * int(sizeof(float)) * sinkCapacity;
        int const particleBytes =
            kickRoundUp(particleBytesPerSource * particleCapacity, kKickArenaAlignment);
        int const multipoleBytes = multipoleCapacity * layout.multipoleStride;
        int const sourceBytes = std::max(particleBytes, multipoleBytes);
        int const arenaBytes = std::max(forceBytes + sourceBytes, scratchBytes);
        if (arenaBytes <= arenaBudget) {
            if (multipoleAuto) {
                // The staged multipoles share the particle planes; fill them.
                multipoleCapacity = std::min(kKickMaximumMultipoleTile,
                    std::max(multipoleCapacity, sourceBytes / layout.multipoleStride));
            }
            layout.sinkCapacity = sinkCapacity;
            layout.particleCapacity = particleCapacity;
            layout.multipoleCapacity = multipoleCapacity;
            layout.forceOffset = sinkCapacity > 0 ? 0 : -1;
            layout.sourceOffset = forceBytes;
            layout.scratchOffset = 0;
            layout.arenaBytes = kickRoundUp(arenaBytes, kKickArenaAlignment);
            layout.totalBytes = layout.arenaOffset + layout.arenaBytes;
            return layout.totalBytes <= budget;
        }
        // Trim: particles first, then multipoles, then an automatic sink tile.
        if (particleAuto && particleCapacity > kKickMinimumParticleTile &&
            particleBytes >= multipoleBytes) {
            particleCapacity -= WARP_SIZE;
        } else if (multipoleAuto && multipoleCapacity > 1 && multipoleBytes > particleBytes) {
            multipoleCapacity--;
        } else if (particleAuto && particleCapacity > kKickMinimumParticleTile) {
            particleCapacity -= WARP_SIZE;
        } else if (multipoleAuto && multipoleCapacity > 1) {
            multipoleCapacity--;
        } else if (sinkAuto && sinkCapacity > kKickDefaultSinkTile) {
            sinkCapacity = kKickDefaultSinkTile;
        } else if (sinkAuto && sinkCapacity > WARP_SIZE) {
            sinkCapacity -= WARP_SIZE;
        } else {
            return false;
        }
    }
}

CudaKickArenaLayout kickSelectLayout() {
    auto const& compute = getOptions().compute;
    bool const massPlane = getOptions().hydro.enabled;
    auto const resources = kickDeviceResources();
    int const registerLimit = kickRegisterResidencyLimit(resources);
    int residency = compute.cudaKickResidency > 0 ?
        std::min(compute.cudaKickResidency, registerLimit) :
        registerLimit;
    CudaKickArenaLayout layout{};
    while (residency >= 1) {
        int budget = kickSharedBudget(resources, residency);
        while (budget > 0 && kickBuildLayout(compute, massPlane, residency, budget, layout)) {
            if (layout.totalBytes > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute((void const*) cudaKickKernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, layout.totalBytes));
            }
            int achieved = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &achieved, (void const*) cudaKickKernel, WARP_SIZE, layout.totalBytes));
            if (achieved >= residency) {
                layout.residency = std::min(achieved, registerLimit);
                PRINT("CUDA kick arena: %i registers, %i static shared bytes, %i blocks/SM "
                      "(register limit %i), budget %i, dynamic shared %i bytes: control %i, "
                      "expansion %s, forces %s, particle tile %i (%s mass plane), multipole tile "
                      "%i x %i bytes\n",
                    resources.numberOfRegisters, resources.staticSharedBytes, layout.residency,
                    registerLimit, layout.budgetBytes, layout.totalBytes,
                    int(sizeof(CudaKickControl)), layout.expansionOffset >= 0 ? "shared" : "global",
                    layout.sinkCapacity > 0 ?
                        (std::string("shared tile ") + std::to_string(layout.sinkCapacity))
                            .c_str() :
                        "global whole bucket",
                    layout.particleCapacity, layout.massPlane ? "with" : "without",
                    layout.multipoleCapacity, layout.multipoleStride);
                return layout;
            }
            // The occupancy calculator disagrees with the arithmetic budget; shrink.
            budget -= kKickSharedAllocationGranularity;
        }
        if (compute.cudaKickResidency > 0) {
            break;
        }
        residency--;
    }
    throw std::runtime_error(
        "No CUDA kick shared-memory layout fits the requested residency and tiles");
}

}    // namespace detail

CudaKickArenaLayout const& kickArenaLayout() {
    static CudaKickArenaLayout const layout = detail::kickSelectLayout();
    return layout;
}

KickReturn cudaExecuteKicks(KickParameters kparams, Fixed32* devX, Fixed32* devY, Fixed32* devZ,
    TreeNode* devTreeNodes, Vector<KickWorkitem> workitems) {
    Timer tm;
    tm.start();
    auto const& layout = kickArenaLayout();
    int numberOfBlocks = layout.residency * cudaSmpCount();
    ALWAYS_ASSERT(workitems.size());
    numberOfBlocks = std::min(numberOfBlocks, (int) workitems.size());
    static char* dataPtr = nullptr;
    static char* devDataPtr = nullptr;
    static int dataSize = 0;
    int dcount = 0;
    int ecount = 0;
    for (int i = 0; i < workitems.size(); i++) {
        dcount += workitems[i].dchecklist.size();
        ecount += workitems[i].echecklist.size();
    }
    int const allocSize = sizeof(int) + sizeof(CudaKickParameters) * workitems.size() +
        sizeof(KickReturn) + sizeof(int) * dcount + sizeof(int) * ecount;
    if (dataSize < allocSize) {
        if (dataPtr) {
            CUDA_CHECK(cudaFree(devDataPtr));
            free(dataPtr);
        }
        dataSize = allocSize;
        dataPtr = (char*) malloc(allocSize);
        CUDA_CHECK(cudaMalloc(&devDataPtr, allocSize));
    }
    int offset = 0;
    KickReturn* result = (KickReturn*) (dataPtr + offset);
    offset += sizeof(KickReturn);
    CudaKickParameters* ikickParams = (CudaKickParameters*) (dataPtr + offset);
    offset += sizeof(CudaKickParameters) * workitems.size();
    int* dchecks = (int*) (dataPtr + offset);
    offset += sizeof(int) * dcount;
    int* echecks = (int*) (dataPtr + offset);
    offset += sizeof(int) * ecount;
    int* currentIndex = (int*) (dataPtr + offset);
    offset += sizeof(int);
    offset = 0;
    KickReturn* devReturnArg = (KickReturn*) (devDataPtr + offset);
    offset += sizeof(KickReturn);
    CudaKickParameters* devIkickParams = (CudaKickParameters*) (devDataPtr + offset);
    offset += sizeof(CudaKickParameters) * workitems.size();
    int* devDchecks = (int*) (devDataPtr + offset);
    offset += sizeof(int) * dcount;
    int* devEchecks = (int*) (devDataPtr + offset);
    offset += sizeof(int) * ecount;
    int* devCurrentIndex = (int*) (devDataPtr + offset);
    offset += sizeof(int);
    *currentIndex = 0;
    *result = KickReturn();
    Vector<int> dindices(workitems.size() + 1);
    Vector<int> eindices(workitems.size() + 1);
    dcount = 0;
    ecount = 0;
    for (int i = 0; i < workitems.size(); i++) {
        dindices[i] = dcount;
        eindices[i] = ecount;
        for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
            dchecks[dcount] = workitems[i].dchecklist[j].index;
            dcount++;
        }
        for (int j = 0; j < workitems[i].echecklist.size(); j++) {
            echecks[ecount] = workitems[i].echecklist[j].index;
            ecount++;
        }
    }

    dindices[workitems.size()] = dcount;
    eindices[workitems.size()] = ecount;
    CudaKickData data;
    data.x = devX;
    data.y = devY;
    data.z = devZ;
    data.treeNodes = devTreeNodes;
    data.vel = particlesVelData();
    data.rungs = &particlesRung(0);
    data.sph = getOptions().hydro.enabled;
    data.darkMatterMass = getOptions().initial.darkMatterMass;
    data.sphMass = getOptions().initial.gasMass;
    data.type = data.sph ? &particlesType(0) : nullptr;
    data.catIndex = data.sph ? &particlesCatIndex(0) : nullptr;
    data.sphGravityAccelerationX = data.sph ? &sphParticlesGravityAcceleration(XDIM, 0) : nullptr;
    data.sphGravityAccelerationY = data.sph ? &sphParticlesGravityAcceleration(YDIM, 0) : nullptr;
    data.sphGravityAccelerationZ = data.sph ? &sphParticlesGravityAcceleration(ZDIM, 0) : nullptr;
    if (kparams.saveForce) {
        data.gx = &particlesGravityAcceleration(XDIM, 0);
        data.gy = &particlesGravityAcceleration(YDIM, 0);
        data.gz = &particlesGravityAcceleration(ZDIM, 0);
        data.pot = &particlesPot(0);
    } else {
        data.gx = data.gy = data.gz = data.pot = nullptr;
    }
    ALWAYS_ASSERT(layout.massPlane == data.sph);

    for (int i = 0; i < workitems.size(); i++) {
        CudaKickParameters params;
        params.lpos = workitems[i].pos;
        params.lValue = workitems[i].l;
        params.self = workitems[i].self.index;
        params.dchecks = devDchecks + dindices[i];
        params.echecks = devEchecks + eindices[i];
        params.dcount = dindices[i + 1] - dindices[i];
        params.ecount = eindices[i + 1] - eindices[i];
        ikickParams[i] = std::move(params);
    }
    cudaSetDevice();
    CudaStream stream(cudaGetDevice());
    stream.copyAsync(devDataPtr, dataPtr, sizeof(char) * allocSize, cudaMemcpyHostToDevice);
    cudaKickKernel<<<numberOfBlocks, WARP_SIZE, layout.totalBytes, stream.nativeHandle()>>>(
        devReturnArg, kparams, data, devIkickParams, workitems.size(), devCurrentIndex, layout);
    stream.checkLaunch();
    stream.copyAsync(result, devReturnArg, sizeof(KickReturn), cudaMemcpyDeviceToHost);
    stream.wait();
    return *result;
}

// Blocks per launch; also the CPU/GPU handoff divisor in kick.cpp, unless
// compute.cudaKickHandoffBlocks pins that divisor for benchmarking.
int kickBlockCount() {
    int const handoff = getOptions().compute.cudaKickHandoffBlocks;
    if (handoff > 0) {
        return handoff;
    }
    return kickArenaLayout().residency * cudaSmpCount();
}
