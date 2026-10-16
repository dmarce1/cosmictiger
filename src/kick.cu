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

#include <atomic>

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed_capacity_vector.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>

#define MIN_PARTS_PCCP 38
#define MIN_PARTS2_CC 78

/*__managed__ int node_count;
 __managed__ double total_time;
 __managed__ double tree_time;
 __managed__ double gravity_time;
 static __managed__ double kick_time;*/

struct CudaKickParameters {
    Array<Fixed32, NDIM> lpos;
    Expansion<float> lValue;
    int self;
    int* dchecks;
    int* echecks;
    int dcount;
    int ecount;
};

__device__ void doKick(KickReturn& result, KickParameters params, CudaKickData const& data,
    Expansion<float> const& lValue2, TreeNode const& self) {
    //	auto tm = clock64();
    int const& tid = threadIdx.x;
    extern __shared__ int shmemPtr[];
    CudaKickSharedMemory& shmem = *(CudaKickSharedMemory*) shmemPtr;
    auto& allPhi = data.pot;
    auto& allGx = data.gx;
    auto& allGy = data.gy;
    auto& allGz = data.gz;
    auto* __restrict__ rungs = data.rungs;
    auto* __restrict__ vels = data.vel;
    auto& force = shmem.f;
    auto const* sinkX = data.x + self.partRange.first;
    auto const* sinkY = data.y + self.partRange.first;
    auto const* sinkZ = data.z + self.partRange.first;
    int const numberSinks = self.partRange.second - self.partRange.first;
    int maxRung = 0;
    float phiTot = 0.0f;
    float kinTot = 0.f;
    float xmomTot = 0.0f;
    float ymomTot = 0.0f;
    float zmomTot = 0.0f;
    float nmomTot = 0.0f;
    float dkinTot = 0.f;
    float dmKineticCorrection = 0.0f;
    float dmXMomentumCorrection = 0.0f;
    float dmYMomentumCorrection = 0.0f;
    float dmZMomentumCorrection = 0.0f;
    float dmMomentumNormCorrection = 0.0f;

    float const& hsoft = params.h;
    FlopCounter<int> flops = 0;
    for (int i = tid; i < numberSinks; i += WARP_SIZE) {
        Expansion2<float> l2;
        Array<float, NDIM> dx;
        ParticleInteger snki;
        float vx;
        float vy;
        float vz;
        float dt;
        float g2;
        int rung;
        snki = self.partRange.first + i;
        if (!params.htime && rungs[snki] < params.minRung) {
            continue;
        }
        ASSERT(snki >= 0);
        dx[XDIM] = distance(sinkX[i], self.pos[XDIM]);    // 1
        dx[YDIM] = distance(sinkY[i], self.pos[YDIM]);    // 1
        dx[ZDIM] = distance(sinkZ[i], self.pos[ZDIM]);    // 1
        l2 = l2P(lValue2, dx, params.doPhi);
        auto& fValue = force[i];
        fValue.phi += SCALE_FACTOR1 * l2(0, 0, 0);
        fValue.gx -= SCALE_FACTOR2 * l2(1, 0, 0);
        fValue.gy -= SCALE_FACTOR2 * l2(0, 1, 0);
        fValue.gz -= SCALE_FACTOR2 * l2(0, 0, 1);
        fValue.gz *= params.gm;
        fValue.gy *= params.gm;
        fValue.gx *= params.gm;
        fValue.phi *= params.gm;
        vx = vels[snki][XDIM];
        vy = vels[snki][YDIM];
        vz = vels[snki][ZDIM];
        float const mass =
            data.sph ? (data.type[snki] == DARK_MATTER_TYPE ? data.dmMass : data.sphMass) : 1.0f;
        float const kin0 = 0.5f * mass * sqr(vx, vy, vz);
        if (params.htime) {
            float const sgn = params.sign * (params.top ? 1.f : -1.f);
            if (params.ascending) {
                dt = 0.5f * rungDt[params.minRung] * params.t0;
                flops += 2;
                if (!params.firstCall) {
                    vx = fmaf(sgn * fValue.gx, dt, vx);
                    vy = fmaf(sgn * fValue.gy, dt, vy);
                    vz = fmaf(sgn * fValue.gz, dt, vz);
                    flops += 6;
                }
            }
            kinTot += 0.5f * sqr(vx, vy, vz);
            xmomTot += vx;
            ymomTot += vy;
            zmomTot += vz;
            nmomTot += sqrtf(sqr(vx, vy, vz));
            if (params.saveForce) {
                allGx[snki] = fValue.gx;
                allGy[snki] = fValue.gy;
                allGz[snki] = fValue.gz;
                allPhi[snki] = fValue.phi;
            }

            if (params.descending) {
                g2 = sqr(fValue.gx, fValue.gy, fValue.gz);
                dt = fminf(params.eta * sqrt(params.a * hsoft * rsqrtf(g2)), params.t0);    // 12
                dt = fminf(params.maxDt, dt);
                rung = rungs[snki];
                rung =
                    max(params.minRung + int((int) ceilf(log2f(params.t0 / dt)) > params.minRung),
                        rung - 1);    // 13
                rungs[snki] = rung;
                maxRung = max(rung, maxRung);
                ALWAYS_ASSERT(rung >= 0);
                ALWAYS_ASSERT(rung < MAX_RUNG);
                dt = 0.5f * rungDt[params.minRung] * params.t0;    // 2
                vx = fmaf(sgn * fValue.gx, dt, vx);                // 3
                vy = fmaf(sgn * fValue.gy, dt, vy);                // 3
                vz = fmaf(sgn * fValue.gz, dt, vz);                // 3
                flops += 36;
            }
        } else {
            bool const gas = data.sph && data.type[snki] == SPH_TYPE;
            if (gas) {
                ParticleInteger const k = data.catIndex[snki];
                data.sphGx[k] = fValue.gx;
                data.sphGy[k] = fValue.gy;
                data.sphGz[k] = fValue.gz;
            } else if (!params.firstCall) {
                float const dt = 0.5f * rungDt[rungs[snki]] * params.t0;
                vx = fmaf(fValue.gx, dt, vx);
                vy = fmaf(fValue.gy, dt, vy);
                vz = fmaf(fValue.gz, dt, vz);
            }
            kinTot += 0.5f * mass * sqr(vx, vy, vz);
            xmomTot += mass * vx;
            ymomTot += mass * vy;
            zmomTot += mass * vz;
            nmomTot += mass * sqrtf(sqr(vx, vy, vz));
            if (params.saveForce) {
                allGx[snki] = fValue.gx;
                allGy[snki] = fValue.gy;
                allGz[snki] = fValue.gz;
                allPhi[snki] = fValue.phi;
            }
            if (!gas) {
                float const syncVx = vx, syncVy = vy, syncVz = vz;
                float const g2 = sqr(fValue.gx, fValue.gy, fValue.gz) + 1e-35f;
                float dt =
                    fminf(params.eta * sqrtf(params.a) * sqrtf(hsoft / sqrtf(g2)), params.t0);
                dt = fminf(dt, params.maxDt);
                rung = max(max((int) ceilf(log2f(params.t0 / dt)),
                               max((int) rungs[snki] - 1, params.minRung)),
                    1);
                ALWAYS_ASSERT(rung >= 0 && rung < MAX_RUNG);
                rungs[snki] = rung;
                dt = 0.5f * rungDt[rung] * params.t0;
                vx = fmaf(fValue.gx, dt, vx);
                vy = fmaf(fValue.gy, dt, vy);
                vz = fmaf(fValue.gz, dt, vz);
                dmKineticCorrection +=
                    0.5f * mass * (sqr(syncVx, syncVy, syncVz) - sqr(vx, vy, vz));
                dmXMomentumCorrection += mass * (syncVx - vx);
                dmYMomentumCorrection += mass * (syncVy - vy);
                dmZMomentumCorrection += mass * (syncVz - vz);
                dmMomentumNormCorrection +=
                    mass * (sqrtf(sqr(syncVx, syncVy, syncVz)) - sqrtf(sqr(vx, vy, vz)));
            }
            maxRung = max(maxRung, (int) rungs[snki]);
        }
        float kin1 = 0.5 * mass * sqr(vx, vy, vz);
        vels[snki][XDIM] = vx;
        vels[snki][YDIM] = vy;
        vels[snki][ZDIM] = vz;
        phiTot += 0.5f * mass * force[i].phi;
        dkinTot += kin1 - kin0;
        flops += 557 + params.doPhi * 178;
    }
    sharedReduceAdd(phiTot);
    sharedReduceAdd(xmomTot);
    sharedReduceAdd(ymomTot);
    sharedReduceAdd(zmomTot);
    sharedReduceAdd(nmomTot);
    sharedReduceAdd(kinTot);
    sharedReduceAdd(dkinTot);
    sharedReduceAdd(dmKineticCorrection);
    sharedReduceAdd(dmXMomentumCorrection);
    sharedReduceAdd(dmYMomentumCorrection);
    sharedReduceAdd(dmZMomentumCorrection);
    sharedReduceAdd(dmMomentumNormCorrection);

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
        result.dmKineticCorrection += dmKineticCorrection;
        result.dmXMomentumCorrection += dmXMomentumCorrection;
        result.dmYMomentumCorrection += dmYMomentumCorrection;
        result.dmZMomentumCorrection += dmZMomentumCorrection;
        result.dmMomentumNormCorrection += dmMomentumNormCorrection;

        flops += 6;
    }
    addGpuFlops(flops);
}

__global__ void cudaKickKernel(KickReturn* rc, KickParameters globalParams, CudaKickData data,
    CudaKickParameters* params, int itemCount, int* nextItem) {
    //	auto tm1 = clock64();
    int const& tid = threadIdx.x;
    extern __shared__ int shmemPtr[];
    CudaKickSharedMemory& shmem = *(CudaKickSharedMemory*) shmemPtr;
    new (shmemPtr) CudaKickSharedMemory;
    auto& sparams = shmem.params;
    auto& lValue3 = shmem.l;
    auto& dchecks = shmem.dchecks;
    auto& echecks = shmem.echecks;
    auto& nextlist = shmem.nextlist;
    auto& cclist = shmem.cclist;
    auto& cplist = shmem.cplist;
    auto& pclist = shmem.pclist;
    auto& leaflist = shmem.leaflist;
    auto& barrier = shmem.barrier;
    auto& force = shmem.f;
    auto& treeNodes = data.treeNodes;
    float const& h = globalParams.h;
    float const hinv = 1.f / h;
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
        lValue3.resize(0);
        dchecks.resize(0);
        echecks.resize(0);
        sparams.resize(0);
        {
            ExpansionType thisL;
            thisL.pos = params[index].lpos;
            thisL.expansion = params[index].lValue;
            lValue3.pushBack(thisL);
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
            //			auto tm2 = clock64();
            //			node_count++;
            auto const& self = treeNodes[sparams.back().self];
            switch (sparams.back().phase) {
            case 0: {
                Array<float, NDIM> dx;
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = distance(self.pos[dim], lValue3.back().pos[dim]);
                }
                flops += 3;
                {
                    auto const thisL = l2LCuda(lValue3.back().expansion, dx, globalParams.doPhi);
                    if (tid == 0) {
                        lValue3.back().expansion = thisL;
                        lValue3.back().pos = self.pos;
                    }
                    flops += 2650 + globalParams.doPhi * 332;
                }
                ParticleInteger const numberSinks = self.partRange.second - self.partRange.first;
                if (self.leaf) {
                    force.resize(numberSinks);
                    for (int l = tid; l < numberSinks; l += WARP_SIZE) {
                        auto& f = force[l];
                        f.gx = f.gy = f.gz = 0.f;
                        float const mass = data.sph ?
                            (data.type[self.partRange.first + l] == DARK_MATTER_TYPE ?
                                    data.dmMass :
                                    data.sphMass) :
                            1.0f;
                        f.phi = mass * selfPhi() * hinv;
                    }
                }
                __syncwarp();
                if (numberSinks) {
                    nextlist.resize(0);
                    cclist.resize(0);
                    constexpr int kCc = 0;
                    constexpr int kNext = 1;
                    constexpr int kNlists = 2;
                    Array<bool, kNlists> sw;
                    DeviceVector<int>* lists[kNlists] = {&cclist, &nextlist};

                    auto& checks = echecks;
                    float const thetainv = 1.f / globalParams.theta;
                    int const maxi = roundUp(checks.size(), WARP_SIZE);
                    for (int i = tid; i < maxi; i += WARP_SIZE) {
                        for (int n = 0; n < kNlists; n++) {
                            sw[n] = false;
                        }
                        if (i < checks.size()) {
                            TreeNode const& other = treeNodes[checks[i]];
                            if (other.numberParticles()) {
                                for (int dim = 0; dim < NDIM; dim++) {
                                    dx[dim] = distance(self.pos[dim], other.pos[dim]);    // 3
                                }
                                float r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                                r2 =
                                    fmaxf(r2, sqr(fmaxf(0.5f - (self.radius + other.radius), 0.f)));
                                float const dcc = (self.radius + other.radius) * thetainv;
                                sw[kCc] = r2 > sqr(dcc);
                                flops += 24;
                                if (!sw[kCc]) {
                                    sw[kNext] = true;
                                    ALWAYS_ASSERT(!other.leaf);
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
                    cudaGravityCcEwald(
                        data, lValue3.back().expansion, self, cclist, globalParams.doPhi);
                }
                if (numberSinks) {
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
                    float const thetainv = 1.f / globalParams.theta;
                    do {
                        int const maxi = roundUp(checks.size(), WARP_SIZE);
                        for (int i = tid; i < maxi; i += WARP_SIZE) {
                            for (int n = 0; n < kNlists; n++) {
                                sw[n] = false;
                            }
                            if (i < checks.size()) {
                                TreeNode const& other = treeNodes[checks[i]];
                                if (other.numberParticles()) {
                                    for (int dim = 0; dim < NDIM; dim++) {
                                        dx[dim] = distance(self.pos[dim], other.pos[dim]);    // 3
                                    }
                                    float const r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                                    float const mind = self.radius + other.radius + h;
                                    float const dcc =
                                        fmaxf((self.radius + other.radius) * thetainv, mind);
                                    auto const selfParts = self.numberParticles();
                                    auto const otherParts = other.numberParticles();
                                    sw[kCc] =
                                        (r2 > sqr(dcc));    // && min(self_parts, (part_int)
                                                            // (2*MIN_PARTS2_CC)) * min(other_parts,
                                                            // (part_int) (2*MIN_PARTS2_CC)) >=
                                                            // MIN_PARTS2_CC;
                                    flops += 20;
                                    if (!sw[kCc] && other.leaf && self.leaf) {
                                        float const dcp =
                                            fmaxf((self.radius * thetainv + other.radius), mind);
                                        float const dpc =
                                            fmaxf((self.radius + other.radius * thetainv), mind);
                                        sw[kPc] = (r2 > sqr(dpc) ||
                                                      !boxIntersectsSphere(self.box, other.pos,
                                                          fmaxf(thetainv * other.radius, h))) &&
                                            selfParts >= MIN_PARTS_PCCP;
                                        sw[kCp] = (r2 > sqr(dcp) ||
                                                      !boxIntersectsSphere(other.box, self.pos,
                                                          fmaxf(thetainv * self.radius, h))) &&
                                            otherParts >= MIN_PARTS_PCCP;
                                        if (sw[kPc] && sw[kCp]) {
                                            if (selfParts < otherParts) {
                                                sw[kCp] = false;
                                            } else if (selfParts > otherParts) {
                                                sw[kPc] = false;
                                            } else {
                                                sw[kCp] = sw[kPc] = false;
                                            }
                                        }
                                        flops += 33;
                                    }
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
                        float const h = globalParams.h;
                        cudaGravityCcDirect(
                            data, lValue3.back().expansion, self, cclist, globalParams.doPhi);
                        cudaGravityCpDirect(
                            data, lValue3.back().expansion, self, cplist, globalParams.doPhi);
                        cudaGravityPcDirect(data, self, pclist, globalParams.doPhi);
                        cudaGravityPpDirect(data, self, leaflist, h, globalParams.doPhi);
                    } else {
                        int const start = checks.size();
                        checks.resize(start + leaflist.size());
                        cuda::memcpy_async(group, checks.data() + start, leaflist.data(),
                            leaflist.size() * sizeof(int), barrier);
                        cudaGravityCcDirect(
                            data, lValue3.back().expansion, self, cclist, globalParams.doPhi);
                        barrier.arrive_and_wait();
                    }
                }
                if (self.leaf) {
                    __syncwarp();
                    if (numberSinks) {
                        doKick(kr, globalParams, data, lValue3.back().expansion, self);
                    }
                    sparams.popBack();
                    depth--;
                } else {
                    TreeId const child = self.children[LEFT];
                    int const i1 = lValue3.size() - 1;
                    int const i2 = lValue3.size();
                    lValue3.resize(i2 + 1);
                    cuda::memcpy_async(group, &lValue3[i2].expansion, &lValue3[i1].expansion,
                        sizeof(float) * EXPANSION_SIZE, barrier);
                    if (tid < NDIM) {
                        lValue3[i2].pos[tid] = lValue3[i1].pos[tid];
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
                    barrier.arrive_and_wait();
                }

            } break;
            case 1: {
                lValue3.popBack();
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
        ASSERT(lValue3.size() == 1);
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
        atomicAdd(&rc->dmKineticCorrection, kr.dmKineticCorrection);
        atomicAdd(&rc->dmXMomentumCorrection, kr.dmXMomentumCorrection);
        atomicAdd(&rc->dmYMomentumCorrection, kr.dmYMomentumCorrection);
        atomicAdd(&rc->dmZMomentumCorrection, kr.dmZMomentumCorrection);
        atomicAdd(&rc->dmMomentumNormCorrection, kr.dmMomentumNormCorrection);

        atomicMax(&rc->maxRung, kr.maxRung);
    }
    ((CudaKickSharedMemory*) shmemPtr)->~CudaKickSharedMemory();

    //	atomicAdd(&total_time, ((double) (clock64() - tm1)));
}

KickReturn cudaExecuteKicks(KickParameters kparams, Fixed32* devX, Fixed32* devY, Fixed32* devZ,
    TreeNode* devTreeNodes, Vector<KickWorkitem> workitems) {
    Timer tm;
    tm.start();
    int numberBlocks = kickBlockCount();
    ALWAYS_ASSERT(workitems.size());
    numberBlocks = std::min(numberBlocks, (int) workitems.size());
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
    data.sph = getOptions().sph;
    data.vsoft = false;
    data.dmMass = getOptions().dmMass;
    data.sphMass = getOptions().sphMass;
    data.type = data.sph ? &particlesType(0) : nullptr;
    data.catIndex = data.sph ? &particlesCatIndex(0) : nullptr;
    data.sphGx = data.sph ? &sphParticlesGforce(XDIM, 0) : nullptr;
    data.sphGy = data.sph ? &sphParticlesGforce(YDIM, 0) : nullptr;
    data.sphGz = data.sph ? &sphParticlesGforce(ZDIM, 0) : nullptr;
    if (kparams.saveForce) {
        data.gx = &particlesGforce(XDIM, 0);
        data.gy = &particlesGforce(YDIM, 0);
        data.gz = &particlesGforce(ZDIM, 0);
        data.pot = &particlesPot(0);
    } else {
        data.gx = data.gy = data.gz = data.pot = nullptr;
    }

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
    CUDA_CHECK(
        cudaMemcpyAsync(devDataPtr, dataPtr, sizeof(char) * allocSize, cudaMemcpyHostToDevice, 0));
    cudaKickKernel<<<numberBlocks, WARP_SIZE, sizeof(CudaKickSharedMemory)>>>(
        devReturnArg, kparams, data, devIkickParams, workitems.size(), devCurrentIndex);
    CUDA_CHECK(
        cudaMemcpyAsync(result, devReturnArg, sizeof(KickReturn), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    return *result;
}

int kickBlockCount() {
    static int numberBlocks;
    static bool shown = false;
    if (!shown) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, (void const*) cudaKickKernel));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numberBlocks, (void const*) cudaKickKernel, WARP_SIZE, sizeof(CudaKickSharedMemory)));
        PRINT("Occupancy is %i shmem size = %li numregs = %i\n", numberBlocks,
            sizeof(CudaKickSharedMemory), attr.numRegs);
        numberBlocks *= cudaSmpCount();
        shown = true;
    }
    return numberBlocks;
}
