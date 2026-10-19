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

#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_cuda.hpp>

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

// The staged source planes derived from the dynamic shared-memory symbol. The
// pointers resolved in the control block are generic once they are reloaded
// from shared memory, and every staged source then costs a 64-bit address
// computation and a generic load; derived from the symbol they are shared-space
// pointers and the loads compile to LDS with immediate offsets. The planes are
// 16-byte aligned: the arena offset and the force region are multiples of
// kKickArenaAlignment and the plane stride is a multiple of WARP_SIZE entries.
struct CudaStagedSources {
    Fixed32 const* x;
    Fixed32 const* y;
    Fixed32 const* z;
    float const* mass;    // nullptr without a mass plane
};

template <int order>
__device__ inline CudaStagedSources cudaStagedSources(CudaKickControl<order> const& control) {
    extern __shared__ __align__(16) char shmemPtr[];
    CudaKickArenaLayout const& layout = control.layout;
    CudaStagedSources sources;
    sources.x =
        reinterpret_cast<Fixed32 const*>(shmemPtr + layout.arenaOffset + layout.sourceOffset);
    sources.y = sources.x + layout.particleCapacity;
    sources.z = sources.y + layout.particleCapacity;
    sources.mass = layout.massPlane ?
        reinterpret_cast<float const*>(sources.z + layout.particleCapacity) :
        nullptr;
    return sources;
}

__device__ inline Fixed32 cudaFixedFromRaw(uint32_t bits) {
    Fixed32 value;
    value.raw() = bits;
    return value;
}

// The inverse radius powers of kPairs sink-source pairs. The plain powers of
// every pair are computed in one straight-line block, so the scheduler overlaps
// their reciprocal square roots, and the softened powers replace them afterwards
// for the pairs inside the support radius; one test of the lane's smallest
// radius skips that substitution when none of its pairs is close. The direct
// pairs are the pair total minus the softened ones, which are counted here.
template <int kPairs>
__device__ inline void cudaPpInversePowers(float const (&r2)[kPairs], float h2, float hinv,
    float h2inv, float h3inv, bool doPhi, float (&r3inv)[kPairs], float (&r1inv)[kPairs],
    int& close) {
    float minR2 = r2[0];
#pragma unroll
    for (int p = 0; p < kPairs; p++) {
        float const inverseRadius = rsqrt(r2[p]);
        r3inv[p] = sqr(inverseRadius) * inverseRadius;
        r1inv[p] = inverseRadius;
        minR2 = fminf(minR2, r2[p]);
    }
    if (!(minR2 > h2)) {
#pragma unroll
        for (int p = 0; p < kPairs; p++) {
            if (!(r2[p] > h2)) {
                close++;
                gravitySofteningKernel(r3inv[p], r1inv[p], r2[p], hinv, h2inv, h3inv, doPhi);
            }
        }
    }
}

// The staged sources accumulated onto the nSinks sinks held by one lane: four
// sources per 16-byte shared load, every source applied to every sink, so the
// scheduler sees 4 * nSinks independent chains instead of one. Each sink's
// accumulation still runs in staged order, which keeps its forces identical
// to the one-sink loop's bit for bit. The potential accumulates whether or not
// it is wanted, which keeps a per-pair select out of the loop; callers store
// it only with doPhi.
template <int nSinks, bool equalMass>
__device__ inline void cudaPpSinkGroup(CudaStagedSources const& sources, int count,
    Fixed32 const (&sinkX)[nSinks], Fixed32 const (&sinkY)[nSinks], Fixed32 const (&sinkZ)[nSinks],
    float h2, float hinv, float h2inv, float h3inv, bool doPhi, float (&fx)[nSinks],
    float (&fy)[nSinks], float (&fz)[nSinks], float (&pot)[nSinks], int& close) {
    constexpr int kVector = 4;
    constexpr int kPairs = kVector * nSinks;
    int j = 0;
    for (; j + kVector <= count; j += kVector) {
        uint4 const x4 = *reinterpret_cast<uint4 const*>(sources.x + j);
        uint4 const y4 = *reinterpret_cast<uint4 const*>(sources.y + j);
        uint4 const z4 = *reinterpret_cast<uint4 const*>(sources.z + j);
        float4 m4 = make_float4(1.f, 1.f, 1.f, 1.f);
        if constexpr (!equalMass) {
            m4 = *reinterpret_cast<float4 const*>(sources.mass + j);
        }
        uint32_t const xs[kVector] = {x4.x, x4.y, x4.z, x4.w};
        uint32_t const ys[kVector] = {y4.x, y4.y, y4.z, y4.w};
        uint32_t const zs[kVector] = {z4.x, z4.y, z4.z, z4.w};
        float const ms[kVector] = {m4.x, m4.y, m4.z, m4.w};
        float dx0[kPairs];
        float dx1[kPairs];
        float dx2[kPairs];
        float r2[kPairs];
        float r3inv[kPairs];
        float r1inv[kPairs];
#pragma unroll
        for (int u = 0; u < kVector; u++) {
            Fixed32 const sourceX = cudaFixedFromRaw(xs[u]);
            Fixed32 const sourceY = cudaFixedFromRaw(ys[u]);
            Fixed32 const sourceZ = cudaFixedFromRaw(zs[u]);
#pragma unroll
            for (int s = 0; s < nSinks; s++) {
                int const p = u * nSinks + s;
                dx0[p] = distance(sinkX[s], sourceX);    // 1
                dx1[p] = distance(sinkY[s], sourceY);    // 1
                dx2[p] = distance(sinkZ[s], sourceZ);    // 1
                r2[p] = sqr(dx0[p], dx1[p], dx2[p]);     // 5
            }
        }
        cudaPpInversePowers<kPairs>(r2, h2, hinv, h2inv, h3inv, doPhi, r3inv, r1inv, close);
#pragma unroll
        for (int u = 0; u < kVector; u++) {
#pragma unroll
            for (int s = 0; s < nSinks; s++) {
                int const p = u * nSinks + s;
                if constexpr (!equalMass) {
                    r3inv[p] *= ms[u];
                    r1inv[p] *= ms[u];
                }
                fx[s] = fmaf(dx0[p], r3inv[p], fx[s]);    // 2
                fy[s] = fmaf(dx1[p], r3inv[p], fy[s]);    // 2
                fz[s] = fmaf(dx2[p], r3inv[p], fz[s]);    // 2
                pot[s] -= r1inv[p];                       // 1
            }
        }
    }
    for (; j < count; j++) {
        Fixed32 const sourceX = sources.x[j];
        Fixed32 const sourceY = sources.y[j];
        Fixed32 const sourceZ = sources.z[j];
        float mass = 1.f;
        if constexpr (!equalMass) {
            mass = sources.mass[j];
        }
        float dx0[nSinks];
        float dx1[nSinks];
        float dx2[nSinks];
        float r2[nSinks];
        float r3inv[nSinks];
        float r1inv[nSinks];
#pragma unroll
        for (int s = 0; s < nSinks; s++) {
            dx0[s] = distance(sinkX[s], sourceX);
            dx1[s] = distance(sinkY[s], sourceY);
            dx2[s] = distance(sinkZ[s], sourceZ);
            r2[s] = sqr(dx0[s], dx1[s], dx2[s]);
        }
        cudaPpInversePowers<nSinks>(r2, h2, hinv, h2inv, h3inv, doPhi, r3inv, r1inv, close);
#pragma unroll
        for (int s = 0; s < nSinks; s++) {
            if constexpr (!equalMass) {
                r3inv[s] *= mass;
                r1inv[s] *= mass;
            }
            fx[s] = fmaf(dx0[s], r3inv[s], fx[s]);
            fy[s] = fmaf(dx1[s], r3inv[s], fy[s]);
            fz[s] = fmaf(dx2[s], r3inv[s], fz[s]);
            pot[s] -= r1inv[s];
        }
    }
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
    detail::CudaStagedSources const sources = detail::cudaStagedSources<order>(control);
    auto const* treeNodes = data.treeNodes;
    float const h2 = sqr(h);
    float const hinv = 1.f / h;
    float const h2inv = sqr(hinv);
    float const h3inv = h2inv * hinv;
    FlopCounter<int> flops = 7;
    int close = 0;
    int pairs = 0;
    if (partlist.size()) {
        int i = 0;
        auto particleRange = treeNodes[partlist[0]].particleRange;
        auto const partsz = partlist.size();
        while (i < partsz) {
            int const particleIndex = detail::cudaStageParticleSources<order, equalMass>(
                data, control, partlist, i, particleRange);
            int kmid = (sinkCount / WARP_SIZE) * WARP_SIZE;
            if (sinkCount - kmid >= WARP_SIZE / 2) {
                kmid = sinkCount;
            }
            // Lanes own sinks k = tid, tid + WARP_SIZE, ...: two per lane while
            // every lane has two (complete 64-sink chunks, so lane utilization
            // matches the one-sink pass), then one per lane for the rest.
            constexpr int kPair = 2;
            int k = tid;
            for (; k - tid + kPair * WARP_SIZE <= kmid; k += kPair * WARP_SIZE) {
                Fixed32 const sx[kPair] = {sinkX[k], sinkX[k + WARP_SIZE]};
                Fixed32 const sy[kPair] = {sinkY[k], sinkY[k + WARP_SIZE]};
                Fixed32 const sz[kPair] = {sinkZ[k], sinkZ[k + WARP_SIZE]};
                float fx[kPair] = {0.f, 0.f};
                float fy[kPair] = {0.f, 0.f};
                float fz[kPair] = {0.f, 0.f};
                float pot[kPair] = {0.f, 0.f};
                detail::cudaPpSinkGroup<kPair, equalMass>(sources, particleIndex, sx, sy, sz, h2,
                    hinv, h2inv, h3inv, doPhi, fx, fy, fz, pot, close);
#pragma unroll
                for (int s = 0; s < kPair; s++) {
                    int const sink = k + s * WARP_SIZE;
                    force.gx[sink] -= fx[s];
                    force.gy[sink] -= fy[s];
                    force.gz[sink] -= fz[s];
                    if (doPhi) {
                        force.phi[sink] += pot[s];
                    }
                }
                pairs += kPair * particleIndex;
                flops += kPair * (21 * particleIndex + 4);
            }
            for (; k < kmid; k += WARP_SIZE) {
                Fixed32 const sx[1] = {sinkX[k]};
                Fixed32 const sy[1] = {sinkY[k]};
                Fixed32 const sz[1] = {sinkZ[k]};
                float fx[1] = {0.f};
                float fy[1] = {0.f};
                float fz[1] = {0.f};
                float pot[1] = {0.f};
                detail::cudaPpSinkGroup<1, equalMass>(sources, particleIndex, sx, sy, sz, h2, hinv,
                    h2inv, h3inv, doPhi, fx, fy, fz, pot, close);
                force.gx[k] -= fx[0];
                force.gy[k] -= fy[0];
                force.gz[k] -= fz[0];
                if (doPhi) {
                    force.phi[k] += pot[0];
                }
                pairs += particleIndex;
                flops += 21 * particleIndex + 4;
            }
            // A trailing partial group of fewer than WARP_SIZE / 2 sinks splits
            // the sources over the lanes instead.
            for (int sink = kmid; sink < sinkCount; sink++) {
                float fx = 0.f;
                float fy = 0.f;
                float fz = 0.f;
                float pot = 0.f;
                Fixed32 const sx = sinkX[sink];
                Fixed32 const sy = sinkY[sink];
                Fixed32 const sz = sinkZ[sink];
                for (int j = tid; j < particleIndex; j += WARP_SIZE) {
                    float const dx0 = distance(sx, sources.x[j]);    // 2
                    float const dx1 = distance(sy, sources.y[j]);    // 2
                    float const dx2 = distance(sz, sources.z[j]);    // 2
                    float const r2[1] = {sqr(dx0, dx1, dx2)};        // 5
                    float r3inv[1];
                    float r1inv[1];
                    detail::cudaPpInversePowers<1>(
                        r2, h2, hinv, h2inv, h3inv, doPhi, r3inv, r1inv, close);
                    if constexpr (!equalMass) {
                        r3inv[0] *= sources.mass[j];
                        r1inv[0] *= sources.mass[j];
                    }
                    fx = fmaf(dx0, r3inv[0], fx);    // 2
                    fy = fmaf(dx1, r3inv[0], fy);    // 2
                    fz = fmaf(dx2, r3inv[0], fz);    // 2
                    pot -= r1inv[0];                 // 1
                    pairs++;
                    flops += 23;
                }
                sharedReduceAdd(fx);
                sharedReduceAdd(fy);
                sharedReduceAdd(fz);
                if (doPhi) {
                    sharedReduceAdd(pot);
                }
                if (tid == 0) {
                    force.gx[sink] -= fx;
                    force.gy[sink] -= fy;
                    force.gz[sink] -= fz;
                    if (doPhi) {
                        force.phi[sink] += pot;
                    }
                    flops += 4;
                }
            }
            // Every lane finishes reading the staged tile before it is refilled.
            __syncwarp();
        }
        __syncwarp();
    }
    int direct = pairs - close;
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
