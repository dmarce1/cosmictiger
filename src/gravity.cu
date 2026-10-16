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

#include <cooperative_groups.h>
#include <cuda/barrier>

#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/gravity.hpp>

__managed__ double iclose;
__managed__ double idirect;
__managed__ double numberOfProcessedSinkParticles;
__managed__ bool useGravityCounters = false;

void resetGravityCounters() {
    iclose = idirect = numberOfProcessedSinkParticles = 0;
}

void setGravityCounterUse(bool code) {
    useGravityCounters = code;
}

void getGravityCounters(double& close, double& direct) {
    if (numberOfProcessedSinkParticles == 0.0) {
        close = direct = 0.0;
    } else {
        close = iclose / numberOfProcessedSinkParticles;
        direct = idirect / numberOfProcessedSinkParticles;
    }
}

__device__ void cudaGravityCcDirect(CudaKickData const& data, Expansion<float>& lacc,
    TreeNode const& self, DeviceVector<int> const& multlist, bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    FlopCounter<int> flops = 0;
    if (multlist.size()) {
        Expansion<float> l;
        Expansion<float> d;
        for (int i = 0; i < EXPANSION_SIZE; i++) {
            l[i] = 0.0f;
        }
        for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
            TreeNode const& other = treeNodes[multlist[i]];
            Multipole<float> const& m = other.mpos->multi;
            Array<float, NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
            }
            flops += 6 + greensFunction(d, dx);
            flops += m2L(l, m, d, doPhi);
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        flops += 6 * EXPANSION_SIZE;
        __syncwarp();
    }
    addGpuFlops(flops);
}

__device__ void cudaGravityCpDirect(CudaKickData const& data, Expansion<float>& lacc,
    TreeNode const& self, DeviceVector<int> const& partlist, bool doPhi) {
    __shared__ extern int shmemPtr[];
    CudaKickSharedMemory& shmem = *(CudaKickSharedMemory*) shmemPtr;
    auto const* mainSourceX = data.x;
    auto const* mainSourceY = data.y;
    auto const* mainSourceZ = data.z;
    auto& sourceX = shmem.xValue.x;
    auto& sourceY = shmem.xValue.y;
    auto& sourceZ = shmem.xValue.z;
    auto& sourceMass = shmem.xValue.mass;
    auto& barrier = shmem.barrier;
    auto const* treeNodes = data.treeNodes;
    int const& tid = threadIdx.x;
    FlopCounter<int> flops = 0;
    if (partlist.size()) {
        int particleIndex;
        Expansion<float> l;
        for (int j = 0; j < EXPANSION_SIZE; j++) {
            l[j] = 0.0;
        }
        int i = 0;
        auto particleRange = treeNodes[partlist[0]].particleRange;
        auto const partsz = partlist.size();
        while (i < partsz) {
            auto group = cooperative_groups::this_thread_block();
            if (group.thread_rank() == 0) {
                init(&barrier, group.size());
            }
            group.sync();
            particleIndex = 0;
            while (particleIndex < KICK_PP_MAX && i < partsz) {
                while (i + 1 < partsz) {
                    auto const otherTreeParticleRange = treeNodes[partlist[i + 1]].particleRange;
                    if (particleRange.second == otherTreeParticleRange.first) {
                        particleRange.second = otherTreeParticleRange.second;
                        i++;
                    } else {
                        break;
                    }
                }
                ParticleInteger const imin = particleRange.first;
                ParticleInteger const imax =
                    min(particleRange.first + (KICK_PP_MAX - particleIndex), particleRange.second);
                int const sz = imax - imin;
                cuda::memcpy_async(group, sourceX.data() + particleIndex, mainSourceX + imin,
                    sizeof(Fixed32) * sz, barrier);
                cuda::memcpy_async(group, sourceY.data() + particleIndex, mainSourceY + imin,
                    sizeof(Fixed32) * sz, barrier);
                cuda::memcpy_async(group, sourceZ.data() + particleIndex, mainSourceZ + imin,
                    sizeof(Fixed32) * sz, barrier);
                for (int j = tid; j < sz; j += WARP_SIZE) {
                    sourceMass[particleIndex + j] = data.sph ?
                        (data.type[imin + j] == DARK_MATTER_TYPE ? data.darkMatterMass :
                                                                   data.sphMass) :
                        1.0f;
                }
                __syncwarp();
                particleRange.first += sz;
                particleIndex += sz;
                if (particleRange.first == particleRange.second) {
                    i++;
                    if (i < partsz) {
                        particleRange = treeNodes[partlist[i]].particleRange;
                    }
                }
            }
            barrier.arrive_and_wait();
            __syncwarp();
            for (int j = tid; j < particleIndex; j += warpSize) {
                Array<float, NDIM> dx;
                dx[XDIM] = distance(self.mpos->pos[XDIM], sourceX[j]);
                dx[YDIM] = distance(self.mpos->pos[YDIM], sourceY[j]);
                dx[ZDIM] = distance(self.mpos->pos[ZDIM], sourceZ[j]);
                Expansion<float> d;
                flops += EXPANSION_SIZE + 6 + greensFunction(d, dx);
                for (int k = 0; k < EXPANSION_SIZE; k++) {
                    l[k] += sourceMass[j] * d[k];
                }
            }
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        flops += 6 * EXPANSION_SIZE;

        __syncwarp();
    }
    addGpuFlops(flops);
}

__device__ void cudaGravityPcDirect(
    CudaKickData const& data, TreeNode const& self, DeviceVector<int> const& multlist, bool doPhi) {
    int const& tid = threadIdx.x;
    __shared__ extern int shmemPtr[];
    CudaKickSharedMemory& shmem = *(CudaKickSharedMemory*) shmemPtr;
    auto& force = shmem.f;
    auto& multis = shmem.mpos;
    int const numberOfSinks = self.particleRange.second - self.particleRange.first;
    auto const* sinkX = data.x + self.particleRange.first;
    auto const* sinkY = data.y + self.particleRange.first;
    auto const* sinkZ = data.z + self.particleRange.first;
    auto& barrier = shmem.barrier;
    auto const* treeNodes = data.treeNodes;
    FlopCounter<int> flops = 0;
    auto group = cooperative_groups::this_thread_block();
    if (multlist.size()) {
        __syncwarp();
        int mi = 0;
        int const cnt = multlist.size();
        while (mi < cnt) {
            int mend = min(cnt, mi + KICK_C_MAX);
            for (int thisMi = mi; thisMi < mend; thisMi++) {
                cuda::memcpy_async(group, &multis[thisMi - mi], treeNodes[multlist[thisMi]].mpos,
                    sizeof(MultiPos), barrier);
            }
            mend -= mi;
            mi += mend;
            barrier.arrive_and_wait();
            Expansion2<float> l;
            Array<float, NDIM> dx;
            Expansion<float> d;
            for (int j = 0; j < mend; j++) {
                auto const& pos = multis[j].pos;
                auto const& m = multis[j].multi;
                for (int k = tid; k < numberOfSinks; k += WARP_SIZE) {
                    auto& fValue = force[k];
                    l(0, 0, 0) = l(1, 0, 0) = l(0, 1, 0) = l(0, 0, 1) = 0.0f;
                    dx[XDIM] = distance(sinkX[k], pos[XDIM]);
                    dx[YDIM] = distance(sinkY[k], pos[YDIM]);
                    dx[ZDIM] = distance(sinkZ[k], pos[ZDIM]);
                    flops += 6 + greensFunction(d, dx);
                    flops += m2L(l, m, d, doPhi);
                    fValue.gx -= SCALE_FACTOR2 * l(1, 0, 0);
                    fValue.gy -= SCALE_FACTOR2 * l(0, 1, 0);
                    fValue.gz -= SCALE_FACTOR2 * l(0, 0, 1);
                    fValue.phi += SCALE_FACTOR1 * l(0, 0, 0);
                    flops += 4;
                }
            }
            __syncwarp();
        }
    }
    __syncwarp();
    addGpuFlops(flops);
}

__device__ void cudaGravityCcEwald(CudaKickData const& data, Expansion<float>& lacc,
    TreeNode const& self, DeviceVector<int> const& multlist, bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    FlopCounter<int> flops = 0;
    if (multlist.size()) {
        Expansion<float> l;
        Expansion<float> d;
        for (int i = 0; i < EXPANSION_SIZE; i++) {
            l[i] = 0.0f;
        }
        for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
            TreeNode const& other = treeNodes[multlist[i]];
            Multipole<float> m = other.mpos->multi;
            //	flops += apply_scale_factor(M);
            Array<float, NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
            }
            flops += 3 + ewaldGreensFunction(d, dx);
            flops += m2L(l, m, d, doPhi);
        }
        // flops += apply_scale_factor_inv(L);
        sharedReduceAddArray(l);
        for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        flops += 6 * EXPANSION_SIZE;
        __syncwarp();
    }
    addGpuFlops(flops);
}

__device__ void cudaGravityPpDirect(CudaKickData const& data, TreeNode const& self,
    DeviceVector<int> const& partlist, float h, bool doPhi) {
    int const& tid = threadIdx.x;
    __shared__ extern int shmemPtr[];
    CudaKickSharedMemory& shmem = *(CudaKickSharedMemory*) shmemPtr;
    auto& force = shmem.f;
    int const numberOfSinks = self.particleRange.second - self.particleRange.first;
    auto const* sinkX = data.x + self.particleRange.first;
    auto const* sinkY = data.y + self.particleRange.first;
    auto const* sinkZ = data.z + self.particleRange.first;
    auto& barrier = shmem.barrier;
    auto const* mainSourceX = data.x;
    auto const* mainSourceY = data.y;
    auto const* mainSourceZ = data.z;
    auto& sourceX = shmem.xValue.x;
    auto& sourceY = shmem.xValue.y;
    auto& sourceZ = shmem.xValue.z;
    auto& sourceMass = shmem.xValue.mass;
    auto const* treeNodes = data.treeNodes;
    int particleIndex;
    float const h2 = sqr(h);
    float const hinv = 1.f / h;
    float const h2inv = sqr(hinv);
    float const h3inv = h2inv * hinv;
    FlopCounter<int> flops = 7;
    int close = 0;
    int direct = 0;
    if (partlist.size()) {
        int i = 0;
        auto particleRange = treeNodes[partlist[0]].particleRange;
        auto const partsz = partlist.size();
        while (i < partsz) {
            auto group = cooperative_groups::this_thread_block();
            particleIndex = 0;
            while (particleIndex < KICK_PP_MAX && i < partsz) {
                while (i + 1 < partsz) {
                    auto const otherTreeParticleRange = treeNodes[partlist[i + 1]].particleRange;
                    if (particleRange.second == otherTreeParticleRange.first) {
                        particleRange.second = otherTreeParticleRange.second;
                        i++;
                    } else {
                        break;
                    }
                }
                ParticleInteger const imin = particleRange.first;
                ParticleInteger const imax =
                    min(particleRange.first + (KICK_PP_MAX - particleIndex), particleRange.second);
                int const sz = imax - imin;
                cuda::memcpy_async(group, sourceX.data() + particleIndex, mainSourceX + imin,
                    sizeof(Fixed32) * sz, barrier);
                cuda::memcpy_async(group, sourceY.data() + particleIndex, mainSourceY + imin,
                    sizeof(Fixed32) * sz, barrier);
                cuda::memcpy_async(group, sourceZ.data() + particleIndex, mainSourceZ + imin,
                    sizeof(Fixed32) * sz, barrier);
                for (int j = tid; j < sz; j += WARP_SIZE) {
                    sourceMass[particleIndex + j] = data.sph ?
                        (data.type[imin + j] == DARK_MATTER_TYPE ? data.darkMatterMass :
                                                                   data.sphMass) :
                        1.0f;
                }
                __syncwarp();
                particleRange.first += sz;
                particleIndex += sz;
                if (particleRange.first == particleRange.second) {
                    i++;
                    if (i < partsz) {
                        particleRange = treeNodes[partlist[i]].particleRange;
                    }
                }
            }
            barrier.arrive_and_wait();
            float fx;
            float fy;
            float fz;
            float pot;
            float dx0;
            float dx1;
            float dx2;
            float r3inv;
            float r1inv;
            __syncwarp();
            int kmid = (numberOfSinks / WARP_SIZE) * WARP_SIZE;
            if (numberOfSinks - kmid >= WARP_SIZE / 2) {
                kmid = numberOfSinks;
            }
            for (int k = tid; k < kmid; k += WARP_SIZE) {
                fx = 0.f;
                fy = 0.f;
                fz = 0.f;
                pot = 0.f;
                for (int j = 0; j < particleIndex; j++) {
                    dx0 = distance(sinkX[k], sourceX[j]);    // 1
                    dx1 = distance(sinkY[k], sourceY[j]);    // 1
                    dx2 = distance(sinkZ[k], sourceZ[j]);    // 1
                    auto const r2 = sqr(dx0, dx1, dx2);    // 5
                    if (r2 > h2) {
                        r1inv = rsqrt(r2);             // 4
                        r3inv = sqr(r1inv) * r1inv;    // 2
                        direct++;
                    } else {
                        close++;
                        gravitySofteningKernel(r3inv, r1inv, r2, hinv, h2inv, h3inv, doPhi);
                    }
                    r3inv *= sourceMass[j];
                    if (doPhi) {
                        r1inv *= sourceMass[j];
                    }
                    fx = fmaf(dx0, r3inv, fx);    // 2
                    fy = fmaf(dx1, r3inv, fy);    // 2
                    fz = fmaf(dx2, r3inv, fz);    // 2
                    pot -= r1inv;                 // 1
                    flops += 21;
                }
                auto& fValue = force[k];
                fValue.gx -= fx;
                fValue.gy -= fy;
                fValue.gz -= fz;
                flops += 4;
                fValue.phi += pot;
            }
            for (int k = kmid; k < numberOfSinks; k++) {
                fx = 0.f;
                fy = 0.f;
                fz = 0.f;
                pot = 0.f;
                for (int j = tid; j < particleIndex; j += WARP_SIZE) {
                    dx0 = distance(sinkX[k], sourceX[j]);    // 2
                    dx1 = distance(sinkY[k], sourceY[j]);    // 2
                    dx2 = distance(sinkZ[k], sourceZ[j]);    // 2
                    auto const r2 = sqr(dx0, dx1, dx2);    // 5
                    if (r2 > h2) {
                        r1inv = rsqrt(r2);             // 4
                        r3inv = sqr(r1inv) * r1inv;    // 2
                        direct++;
                    } else {
                        close++;
                        gravitySofteningKernel(r3inv, r1inv, r2, hinv, h2inv, h3inv, doPhi);
                    }
                    r3inv *= sourceMass[j];
                    if (doPhi) {
                        r1inv *= sourceMass[j];
                    }
                    fx = fmaf(dx0, r3inv, fx);    // 2
                    fy = fmaf(dx1, r3inv, fy);    // 2
                    fz = fmaf(dx2, r3inv, fz);    // 2
                    pot -= r1inv;                 // 1
                    flops += 23;
                }
                sharedReduceAdd(fx);
                sharedReduceAdd(fy);
                sharedReduceAdd(fz);
                if (doPhi) {
                    sharedReduceAdd(pot);
                }
                if (tid == 0) {
                    auto& fValue = force[k];
                    fValue.gx -= fx;
                    fValue.gy -= fy;
                    fValue.gz -= fz;
                    flops += 4;
                    fValue.phi += pot;
                }
            }
        }
        __syncwarp();
    }
    sharedReduceAdd(close);
    sharedReduceAdd(direct);
    if (tid == 0 && useGravityCounters) {
        atomicAdd(&iclose, close);
        atomicAdd(&idirect, direct);
        atomicAdd(&numberOfProcessedSinkParticles, numberOfSinks);
    }

    __syncwarp();
    addGpuFlops(flops);
}
