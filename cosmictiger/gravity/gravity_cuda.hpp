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

// Order-templated CUDA gravity operators of the kick kernel. They are
// instantiated only inside the per-order kick instance translation units
// (kick_cuda_instances.hpp), which include this header through kick_cuda_impl.hpp.

#include <cooperative_groups.h>
#include <cuda/barrier>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/fmm/fmm_kernels.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick_cuda.hpp>

// Interaction counters shared by every order (defined in gravity.cu).
extern __managed__ double iclose;
extern __managed__ double idirect;
extern __managed__ double numberOfProcessedSinkParticles;
extern __managed__ bool useGravityCounters;

namespace detail {

// Stage particle sources from consecutive list entries into the shared source
// planes, merging adjacent particle ranges, until the planes are full or the
// list is exhausted. Returns the staged count; the staging copies are complete
// and visible to every lane on return.
template <int order, bool equalMass>
__device__ int cudaStageParticleSources(CudaKickData const& data, CudaKickControl<order>& control,
    DeviceVector<int> const& partlist, int& i, Pair<ParticleInteger>& particleRange) {
    int const& tid = threadIdx.x;
    auto group = cooperative_groups::this_thread_block();
    auto const* treeNodes = data.treeNodes;
    int const capacity = control.layout.particleCapacity;
    int const partsz = partlist.size();
    int particleIndex = 0;
    while (particleIndex < capacity && i < partsz) {
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
            min(particleRange.first + (capacity - particleIndex), particleRange.second);
        int const sz = imax - imin;
        cuda::memcpy_async(group, control.sourceX + particleIndex, data.x + imin,
            sizeof(Fixed32) * sz, control.barrier);
        cuda::memcpy_async(group, control.sourceY + particleIndex, data.y + imin,
            sizeof(Fixed32) * sz, control.barrier);
        cuda::memcpy_async(group, control.sourceZ + particleIndex, data.z + imin,
            sizeof(Fixed32) * sz, control.barrier);
        if constexpr (!equalMass) {
            for (int j = tid; j < sz; j += WARP_SIZE) {
                control.sourceMass[particleIndex + j] =
                    data.type[imin + j] == DARK_MATTER_TYPE ? data.darkMatterMass : data.sphMass;
            }
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
    control.barrier.arrive_and_wait();
    __syncwarp();
    return particleIndex;
}

}    // namespace detail

template <int order>
__device__ void cudaGravityCcDirect(CudaKickData const& data, FmmExpansion<float, order>& lacc,
    TreeNode const& self, DeviceVector<int> const& multlist, bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    FlopCounter<int> flops = 0;
    if (multlist.size()) {
        FmmExpansion<float, order> l;
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            l[i] = 0.0f;
        }
        for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
            TreeNode const& other = treeNodes[multlist[i]];
            FmmMultipole<float, order> const& m = fmmMultiPos<order>(other.mpos).multi;
            Array<float, NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
            }
            flops += 6 + m2LDirect(l, m, dx, doPhi);
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < FmmOrder<order>::kLocalCoefficients_; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        flops += 6 * FmmOrder<order>::kLocalCoefficients_;
        __syncwarp();
    }
    addGpuFlops(flops);
}

template <int order, bool equalMass>
__device__ void cudaGravityCpDirect(CudaKickData const& data, CudaKickControl<order>& control,
    FmmExpansion<float, order>& lacc, TreeNode const& self, DeviceVector<int> const& partlist,
    bool doPhi) {
    int const& tid = threadIdx.x;
    auto const* treeNodes = data.treeNodes;
    FlopCounter<int> flops = 0;
    if (partlist.size()) {
        FmmExpansion<float, order> l;
        for (int j = 0; j < FmmOrder<order>::kLocalCoefficients_; j++) {
            l[j] = 0.0;
        }
        int i = 0;
        auto particleRange = treeNodes[partlist[0]].particleRange;
        auto const partsz = partlist.size();
        while (i < partsz) {
            int const count = detail::cudaStageParticleSources<order, equalMass>(
                data, control, partlist, i, particleRange);
            for (int j = tid; j < count; j += WARP_SIZE) {
                Array<float, NDIM> dx;
                dx[XDIM] = distance(self.mpos->pos[XDIM], control.sourceX[j]);
                dx[YDIM] = distance(self.mpos->pos[YDIM], control.sourceY[j]);
                dx[ZDIM] = distance(self.mpos->pos[ZDIM], control.sourceZ[j]);
                FmmExpansion<float, order> d;
                flops += FmmOrder<order>::kLocalCoefficients_ + 6 + greensFunction(d, dx);
                if constexpr (equalMass) {
                    for (int k = 0; k < FmmOrder<order>::kLocalCoefficients_; k++) {
                        l[k] += d[k];
                    }
                } else {
                    float const mass = control.sourceMass[j];
                    for (int k = 0; k < FmmOrder<order>::kLocalCoefficients_; k++) {
                        l[k] += mass * d[k];
                    }
                }
            }
            // Every lane finishes reading the staged tile before it is refilled.
            __syncwarp();
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < FmmOrder<order>::kLocalCoefficients_; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        flops += 6 * FmmOrder<order>::kLocalCoefficients_;
        __syncwarp();
    }
    addGpuFlops(flops);
}

template <int order>
__device__ void cudaGravityPcDirect(CudaKickData const& data, CudaKickControl<order>& control,
    TreeNode const& self, int sinkBegin, int sinkCount, DeviceVector<int> const& multlist,
    bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& force = control.forces;
    int const sinkFirst = self.particleRange.first + sinkBegin;
    auto const* sinkX = data.x + sinkFirst;
    auto const* sinkY = data.y + sinkFirst;
    auto const* sinkZ = data.z + sinkFirst;
    auto& barrier = control.barrier;
    auto const* treeNodes = data.treeNodes;
    int const capacity = control.layout.multipoleCapacity;
    int const stride = control.layout.multipoleStride;
    char* multipoles = control.multipoles;
    FlopCounter<int> flops = 0;
    auto group = cooperative_groups::this_thread_block();
    if (multlist.size()) {
        __syncwarp();
        int mi = 0;
        int const cnt = multlist.size();
        while (mi < cnt) {
            int mend = min(cnt, mi + capacity);
            for (int thisMi = mi; thisMi < mend; thisMi++) {
                cuda::memcpy_async(group, multipoles + (thisMi - mi) * stride,
                    treeNodes[multlist[thisMi]].mpos, sizeof(FmmMultiPos<order>), barrier);
            }
            mend -= mi;
            mi += mend;
            barrier.arrive_and_wait();
            // Accumulate every staged multipole in registers; write each sink once.
            for (int k = tid; k < sinkCount; k += WARP_SIZE) {
                Expansion2<float> l;
                l(0, 0, 0) = l(1, 0, 0) = l(0, 1, 0) = l(0, 0, 1) = 0.0f;
                Array<float, NDIM> dx;
                for (int j = 0; j < mend; j++) {
                    FmmMultiPos<order> const& mp =
                        *reinterpret_cast<FmmMultiPos<order> const*>(multipoles + j * stride);
                    dx[XDIM] = distance(sinkX[k], mp.pos[XDIM]);
                    dx[YDIM] = distance(sinkY[k], mp.pos[YDIM]);
                    dx[ZDIM] = distance(sinkZ[k], mp.pos[ZDIM]);
                    flops += 6 + m2P(l, mp.multi, dx, doPhi);
                }
                force.gx[k] -= SCALE_FACTOR2 * l(1, 0, 0);
                force.gy[k] -= SCALE_FACTOR2 * l(0, 1, 0);
                force.gz[k] -= SCALE_FACTOR2 * l(0, 0, 1);
                force.phi[k] += SCALE_FACTOR1 * l(0, 0, 0);
                flops += 4;
            }
            // Every lane finishes reading the staged multipoles before refilling.
            __syncwarp();
        }
    }
    __syncwarp();
    addGpuFlops(flops);
}

template <int order>
__device__ void cudaGravityCcEwald(CudaKickData const& data, FmmExpansion<float, order>& lacc,
    TreeNode const& self, DeviceVector<int> const& multlist, bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    FlopCounter<int> flops = 0;
    if (multlist.size()) {
        FmmExpansion<float, order> l;
        FmmExpansion<float, order> d;
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            l[i] = 0.0f;
        }
        for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
            TreeNode const& other = treeNodes[multlist[i]];
            FmmMultipole<float, order> const m = fmmMultiPos<order>(other.mpos).multi;
            Array<float, NDIM> dx;
            Array<float, NDIM> unwrappedPosition;
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
                unwrappedPosition[dim] =
                    self.mpos->pos[dim].toDouble() - other.mpos->pos[dim].toDouble();
            }
            flops += 3 + ewaldGreensFunction(d, dx, unwrappedPosition);
            flops += m2L(l, m, d, doPhi);
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < FmmOrder<order>::kLocalCoefficients_; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        flops += 6 * FmmOrder<order>::kLocalCoefficients_;
        __syncwarp();
    }
    addGpuFlops(flops);
}

template <int order, bool equalMass>
__device__ void cudaGravityPpDirect(CudaKickData const& data, CudaKickControl<order>& control,
    TreeNode const& self, int sinkBegin, int sinkCount, DeviceVector<int> const& partlist, float h,
    bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& force = control.forces;
    int const sinkFirst = self.particleRange.first + sinkBegin;
    auto const* sinkX = data.x + sinkFirst;
    auto const* sinkY = data.y + sinkFirst;
    auto const* sinkZ = data.z + sinkFirst;
    auto const* sourceX = control.sourceX;
    auto const* sourceY = control.sourceY;
    auto const* sourceZ = control.sourceZ;
    auto const* sourceMass = control.sourceMass;
    auto const* treeNodes = data.treeNodes;
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
            int const particleIndex = detail::cudaStageParticleSources<order, equalMass>(
                data, control, partlist, i, particleRange);
            float fx;
            float fy;
            float fz;
            float pot;
            float dx0;
            float dx1;
            float dx2;
            float r3inv;
            float r1inv;
            int kmid = (sinkCount / WARP_SIZE) * WARP_SIZE;
            if (sinkCount - kmid >= WARP_SIZE / 2) {
                kmid = sinkCount;
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
                    auto const r2 = sqr(dx0, dx1, dx2);      // 5
                    if (r2 > h2) {
                        float const inverseRadius = rsqrt(r2);
                        r3inv = sqr(inverseRadius) * inverseRadius;
                        r1inv = doPhi ? inverseRadius : 0.0f;
                        direct++;
                    } else {
                        close++;
                        gravitySofteningKernel(r3inv, r1inv, r2, hinv, h2inv, h3inv, doPhi);
                    }
                    if constexpr (!equalMass) {
                        r3inv *= sourceMass[j];
                        r1inv *= sourceMass[j];
                    }
                    fx = fmaf(dx0, r3inv, fx);    // 2
                    fy = fmaf(dx1, r3inv, fy);    // 2
                    fz = fmaf(dx2, r3inv, fz);    // 2
                    pot -= r1inv;                 // 1
                    flops += 21;
                }
                force.gx[k] -= fx;
                force.gy[k] -= fy;
                force.gz[k] -= fz;
                force.phi[k] += pot;
                flops += 4;
            }
            for (int k = kmid; k < sinkCount; k++) {
                fx = 0.f;
                fy = 0.f;
                fz = 0.f;
                pot = 0.f;
                for (int j = tid; j < particleIndex; j += WARP_SIZE) {
                    dx0 = distance(sinkX[k], sourceX[j]);    // 2
                    dx1 = distance(sinkY[k], sourceY[j]);    // 2
                    dx2 = distance(sinkZ[k], sourceZ[j]);    // 2
                    auto const r2 = sqr(dx0, dx1, dx2);      // 5
                    if (r2 > h2) {
                        float const inverseRadius = rsqrt(r2);
                        r3inv = sqr(inverseRadius) * inverseRadius;
                        r1inv = doPhi ? inverseRadius : 0.0f;
                        direct++;
                    } else {
                        close++;
                        gravitySofteningKernel(r3inv, r1inv, r2, hinv, h2inv, h3inv, doPhi);
                    }
                    if constexpr (!equalMass) {
                        r3inv *= sourceMass[j];
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
                    force.gx[k] -= fx;
                    force.gy[k] -= fy;
                    force.gz[k] -= fz;
                    force.phi[k] += pot;
                    flops += 4;
                }
            }
            // Every lane finishes reading the staged tile before it is refilled.
            __syncwarp();
        }
        __syncwarp();
    }
    sharedReduceAdd(close);
    sharedReduceAdd(direct);
    if (tid == 0 && useGravityCounters) {
        atomicAdd(&iclose, close);
        atomicAdd(&idirect, direct);
        atomicAdd(&numberOfProcessedSinkParticles, sinkCount);
    }

    __syncwarp();
    addGpuFlops(flops);
}
