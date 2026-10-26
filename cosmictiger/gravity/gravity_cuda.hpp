/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Order-templated CUDA gravity operators of the kick kernel. They are
// instantiated only inside the per-order kick instance translation units
// (kick_cuda_instances.hpp), which include this header through kick_cuda_kernel.hpp.

#include <cooperative_groups.h>
#include <cuda/barrier>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick_cuda.hpp>

// Interaction counters shared by every order (defined in kick.cu).
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
        auto const store = treeNodes[partlist[i]].sourceStore;
        bool const ghost = store == FmmParticleStore::Ghost;
        auto const* sourceX = ghost ? data.ghostSources.positions[XDIM] : data.x;
        auto const* sourceY = ghost ? data.ghostSources.positions[YDIM] : data.y;
        auto const* sourceZ = ghost ? data.ghostSources.positions[ZDIM] : data.z;
        auto const* sourceTypes = ghost ? data.ghostSources.types : data.type;
        while (i + 1 < partsz) {
            auto const& otherNode = treeNodes[partlist[i + 1]];
            auto const otherTreeParticleRange = otherNode.particleRange;
            if (otherNode.sourceStore == store &&
                particleRange.second == otherTreeParticleRange.first) {
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
        cuda::memcpy_async(group, control.sourceX + particleIndex, sourceX + imin,
            sizeof(Position) * sz, control.barrier);
        cuda::memcpy_async(group, control.sourceY + particleIndex, sourceY + imin,
            sizeof(Position) * sz, control.barrier);
        cuda::memcpy_async(group, control.sourceZ + particleIndex, sourceZ + imin,
            sizeof(Position) * sz, control.barrier);
        if constexpr (!equalMass) {
            for (int j = tid; j < sz; j += WARP_SIZE) {
                control.sourceMass[particleIndex + j] =
                    sourceTypes[imin + j] == DARK_MATTER_TYPE ? data.darkMatterMass : data.sphMass;
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
    Position const* x;
    Position const* y;
    Position const* z;
    float const* mass;    // nullptr without a mass plane
};

template <int order>
__device__ inline CudaStagedSources cudaStagedSources(CudaKickControl<order> const& control) {
    extern __shared__ __align__(16) char shmemPtr[];
    CudaKickArenaLayout const& layout = control.layout;
    CudaStagedSources sources;
    sources.x =
        reinterpret_cast<Position const*>(shmemPtr + layout.arenaOffset + layout.sourceOffset);
    sources.y = sources.x + layout.particleCapacity;
    sources.z = sources.y + layout.particleCapacity;
    sources.mass = layout.massPlane ?
        reinterpret_cast<float const*>(sources.z + layout.particleCapacity) :
        nullptr;
    return sources;
}

__device__ inline Position cudaFixedFromRaw(PositionInteger bits) {
    Position value;
    value.raw() = bits;
    return value;
}

// The staged multipole tile derived from the dynamic shared-memory symbol, for
// the same reason as the particle planes above: addressed through the pointer
// reloaded from the control block the staged records are generic loads, one
// per coefficient of every sink-multipole pair; derived from the symbol they
// are shared-space loads. The tile shares the particle planes' region.
template <int order>
__device__ inline char* cudaStagedMultipoles(CudaKickControl<order> const& control) {
    extern __shared__ __align__(16) char shmemPtr[];
    CudaKickArenaLayout const& layout = control.layout;
    return shmemPtr + layout.arenaOffset + layout.sourceOffset;
}

// The sink-force planes of the arena derived from the symbol. Valid only when
// the layout keeps the forces in shared memory (forceOffset >= 0); with
// whole-bucket global backing the control block's pointers are the forces.
template <int order>
__device__ inline CudaKickForceTile cudaArenaForces(CudaKickControl<order> const& control) {
    extern __shared__ __align__(16) char shmemPtr[];
    CudaKickArenaLayout const& layout = control.layout;
    CudaKickForceTile forces;
    forces.gx = reinterpret_cast<float*>(shmemPtr + layout.arenaOffset + layout.forceOffset);
    forces.gy = forces.gx + layout.sinkCapacity;
    forces.gz = forces.gy + layout.sinkCapacity;
    forces.phi = forces.gz + layout.sinkCapacity;
    return forces;
}

// One staged record viewed in place. The pointer derives from the shared
// symbol, so the record's words load as LDS at immediate offsets from one
// 32-bit address, with the M2P operator's arithmetic exactly as it compiles
// from the control block's generic pointer. Copying the record to registers
// with 16-byte loads instead (ten LDS.128 at order 7) shortens the loop body
// further but lets the compiler pair the operator's fused multiply-adds
// differently, and the forces then differ from the head's at roundoff level
// ("PC loop reads from the shared-memory symbol" in
// docs/cuda_kick_shared_memory.md).
template <int order>
__device__ inline FmmMultiPos<order> const& cudaStagedMultipole(char const* record) {
    return *reinterpret_cast<FmmMultiPos<order> const*>(record);
}

// One sink's accumulated PC field applied to its force planes. Called with the
// arena planes or the global backing, so each call site keeps its address space.
__device__ inline void cudaApplyPcField(
    CudaKickForceTile const& force, int k, cosmictiger::gravity::Field<float> const& l) {
    force.gx[k] += SCALE_FACTOR2 * l.accelerationX();
    force.gy[k] += SCALE_FACTOR2 * l.accelerationY();
    force.gz[k] += SCALE_FACTOR2 * l.accelerationZ();
    force.phi[k] += SCALE_FACTOR1 * l.potential();
}

// One sink's accumulated PP force applied to its force planes, the potential
// only when it is computed. Called with the arena planes or the global
// backing, so each call site keeps its address space.
__device__ inline void cudaApplyPpForce(
    CudaKickForceTile const& force, int k, float fx, float fy, float fz, float pot, bool doPhi) {
    force.gx[k] -= fx;
    force.gy[k] -= fy;
    force.gz[k] -= fz;
    if (doPhi) {
        force.phi[k] += pot;
    }
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
    FlopCount& close, FlopCounter<FlopCount>& flops) {
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
                flops +=
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
    Position const (&sinkX)[nSinks], Position const (&sinkY)[nSinks],
    Position const (&sinkZ)[nSinks], float h2, float hinv, float h2inv, float h3inv, bool doPhi,
    float (&fx)[nSinks], float (&fy)[nSinks], float (&fz)[nSinks], float (&pot)[nSinks],
    FlopCount& close, FlopCounter<FlopCount>& flops, FmmGeometry const& geometry) {
    constexpr int kVector = 4;
    constexpr int kPairs = kVector * nSinks;
    int j = 0;
    for (; j + kVector <= count; j += kVector) {
        PositionInteger xs[kVector], ys[kVector], zs[kVector];
        if constexpr (kPositionBits == 32) {
            uint4 const x4 = *reinterpret_cast<uint4 const*>(sources.x + j);
            uint4 const y4 = *reinterpret_cast<uint4 const*>(sources.y + j);
            uint4 const z4 = *reinterpret_cast<uint4 const*>(sources.z + j);
            xs[0] = x4.x;
            xs[1] = x4.y;
            xs[2] = x4.z;
            xs[3] = x4.w;
            ys[0] = y4.x;
            ys[1] = y4.y;
            ys[2] = y4.z;
            ys[3] = y4.w;
            zs[0] = z4.x;
            zs[1] = z4.y;
            zs[2] = z4.z;
            zs[3] = z4.w;
        } else {
#pragma unroll
            for (int u = 0; u < kVector; ++u) {
                xs[u] = sources.x[j + u].raw();
                ys[u] = sources.y[j + u].raw();
                zs[u] = sources.z[j + u].raw();
            }
        }
        float4 m4 = make_float4(1.f, 1.f, 1.f, 1.f);
        if constexpr (!equalMass) {
            m4 = *reinterpret_cast<float4 const*>(sources.mass + j);
        }
        float const ms[kVector] = {m4.x, m4.y, m4.z, m4.w};
        float dx0[kPairs];
        float dx1[kPairs];
        float dx2[kPairs];
        float r2[kPairs];
        float r3inv[kPairs];
        float r1inv[kPairs];
#pragma unroll
        for (int u = 0; u < kVector; u++) {
            Position const sourceX = cudaFixedFromRaw(xs[u]);
            Position const sourceY = cudaFixedFromRaw(ys[u]);
            Position const sourceZ = cudaFixedFromRaw(zs[u]);
#pragma unroll
            for (int s = 0; s < nSinks; s++) {
                int const p = u * nSinks + s;
                dx0[p] = fmmPositionDifference(sinkX[s], sourceX, XDIM, geometry);    // 1
                dx1[p] = fmmPositionDifference(sinkY[s], sourceY, YDIM, geometry);    // 1
                dx2[p] = fmmPositionDifference(sinkZ[s], sourceZ, ZDIM, geometry);    // 1
                r2[p] = sqr(dx0[p], dx1[p], dx2[p]);                                  // 5
            }
        }
        cudaPpInversePowers<kPairs>(r2, h2, hinv, h2inv, h3inv, doPhi, r3inv, r1inv, close, flops);
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
        Position const sourceX = sources.x[j];
        Position const sourceY = sources.y[j];
        Position const sourceZ = sources.z[j];
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
            dx0[s] = fmmPositionDifference(sinkX[s], sourceX, XDIM, geometry);
            dx1[s] = fmmPositionDifference(sinkY[s], sourceY, YDIM, geometry);
            dx2[s] = fmmPositionDifference(sinkZ[s], sourceZ, ZDIM, geometry);
            r2[s] = sqr(dx0[s], dx1[s], dx2[s]);
        }
        cudaPpInversePowers<nSinks>(r2, h2, hinv, h2inv, h3inv, doPhi, r3inv, r1inv, close, flops);
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
__device__ FlopCount cudaGravityCcDirect(CudaKickData const& data,
    cosmictiger::gravity::cartesian::Expansion<float, order>& lacc, TreeNode const& self,
    DeviceVector<int> const& multlist, bool doPhi, FmmGeometry const& geometry) {
    int const& tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    FlopCounter<FlopCount> flops = 0;
    if (multlist.size()) {
        cosmictiger::gravity::cartesian::Expansion<float, order> l;
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            l[i] = 0.0f;
        }
        for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
            TreeNode const& other = treeNodes[multlist[i]];
            cosmictiger::gravity::cartesian::Multipole<float, order> const& m =
                fmmMultiPos<order>(other.mpos).multi;
            Array<float, NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] =
                    fmmPositionDifference(self.mpos->pos[dim], other.mpos->pos[dim], dim, geometry);
            }
            flops +=
                3 + cosmictiger::gravity::cartesian::M2L(l, m, fmmScaleDisplacement(dx), doPhi);
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < FmmOrder<order>::kLocalCoefficients_; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        // Five butterfly additions per coefficient in every lane, followed
        // by one distributed addition of each coefficient into lacc.
        flops += 5 * FmmOrder<order>::kLocalCoefficients_;
        if (tid == 0) {
            flops += FmmOrder<order>::kLocalCoefficients_;
        }
        __syncwarp();
    }
    return flops;
}

template <int order, bool equalMass>
__device__ FlopCount cudaGravityCpDirect(CudaKickData const& data, CudaKickControl<order>& control,
    cosmictiger::gravity::cartesian::Expansion<float, order>& lacc, TreeNode const& self,
    DeviceVector<int> const& partlist, bool doPhi, FmmGeometry const& geometry) {
    int const& tid = threadIdx.x;
    auto const* treeNodes = data.treeNodes;
    // The staged planes derived from the shared-memory symbol, as the PP loop
    // reads them: LDS per staged source instead of a generic load through the
    // control block's reloaded pointer.
    detail::CudaStagedSources const sources = detail::cudaStagedSources<order>(control);
    FlopCounter<FlopCount> flops = 0;
    if (partlist.size()) {
        cosmictiger::gravity::cartesian::Expansion<float, order> l;
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
                dx[XDIM] =
                    fmmPositionDifference(self.mpos->pos[XDIM], sources.x[j], XDIM, geometry);
                dx[YDIM] =
                    fmmPositionDifference(self.mpos->pos[YDIM], sources.y[j], YDIM, geometry);
                dx[ZDIM] =
                    fmmPositionDifference(self.mpos->pos[ZDIM], sources.z[j], ZDIM, geometry);
                cosmictiger::gravity::cartesian::Expansion<float, order> d{};
                flops += (equalMass ? 1 : 2) * FmmOrder<order>::kLocalCoefficients_ + 3 +
                    cosmictiger::gravity::cartesian::P2L(d, 1.0f, fmmScaleDisplacement(dx));
                if constexpr (equalMass) {
                    for (int k = 0; k < FmmOrder<order>::kLocalCoefficients_; k++) {
                        l[k] += d[k];
                    }
                } else {
                    float const mass = sources.mass[j];
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
        // Five butterfly additions per coefficient in every lane, followed
        // by one distributed addition of each coefficient into lacc.
        flops += 5 * FmmOrder<order>::kLocalCoefficients_;
        if (tid == 0) {
            flops += FmmOrder<order>::kLocalCoefficients_;
        }
        __syncwarp();
    }
    return flops;
}

template <int order>
__device__ FlopCount cudaGravityPcDirect(CudaKickData const& data, CudaKickControl<order>& control,
    TreeNode const& self, int sinkBegin, int sinkCount, DeviceVector<int> const& multlist,
    bool doPhi, FmmGeometry const& geometry) {
    int const& tid = threadIdx.x;
    // The forces: the arena planes derived from the shared-memory symbol, or
    // the control block's global backing when the layout keeps none.
    bool const sharedForces = control.layout.forceOffset >= 0;
    CudaKickForceTile const arenaForce = detail::cudaArenaForces<order>(control);
    CudaKickForceTile const& backingForce = control.forces;
    int const sinkFirst = self.particleRange.first + sinkBegin;
    auto const* sinkX = data.x + sinkFirst;
    auto const* sinkY = data.y + sinkFirst;
    auto const* sinkZ = data.z + sinkFirst;
    auto& barrier = control.barrier;
    auto const* treeNodes = data.treeNodes;
    int const capacity = control.layout.multipoleCapacity;
    // The tile's record stride is the tree arena's (kick_cuda_launch.hpp sets
    // layout.multipoleStride from the same constant).
    constexpr int kStride = FmmOrder<order>::kPositionedMultipoleStride_;
    char* const multipoles = detail::cudaStagedMultipoles<order>(control);
    FlopCounter<FlopCount> flops = 0;
    auto group = cooperative_groups::this_thread_block();
    // Each record is copied whole at the stride the tree arena and the staged
    // tile share. The aligned byte count makes libcu++ issue 16-byte cp.async
    // chunks that the barrier wait below completes; an untyped count would
    // deduce 1-byte alignment and copy synchronously through registers, and
    // the wait would then order nothing.
    constexpr cuda::aligned_size_t<kFmmPositionedMultipoleAlignment> kRecordBytes(kStride);
    if (multlist.size()) {
        __syncwarp();
        int mi = 0;
        int const cnt = multlist.size();
        while (mi < cnt) {
            int mend = min(cnt, mi + capacity);
            for (int thisMi = mi; thisMi < mend; thisMi++) {
                cuda::memcpy_async(group, multipoles + (thisMi - mi) * kStride,
                    treeNodes[multlist[thisMi]].mpos, kRecordBytes, barrier);
            }
            mend -= mi;
            mi += mend;
            barrier.arrive_and_wait();
            // Accumulate every staged multipole in registers; write each sink once.
            for (int k = tid; k < sinkCount; k += WARP_SIZE) {
                cosmictiger::gravity::Field<float> l{};
                Array<float, NDIM> dx;
                for (int j = 0; j < mend; j++) {
                    FmmMultiPos<order> const& mp =
                        detail::cudaStagedMultipole<order>(multipoles + j * kStride);
                    dx[XDIM] = fmmPositionDifference(sinkX[k], mp.pos[XDIM], XDIM, geometry);
                    dx[YDIM] = fmmPositionDifference(sinkY[k], mp.pos[YDIM], YDIM, geometry);
                    dx[ZDIM] = fmmPositionDifference(sinkZ[k], mp.pos[ZDIM], ZDIM, geometry);
                    flops += 3 +
                        cosmictiger::gravity::cartesian::M2P(
                            l, mp.multi, fmmScaleDisplacement(dx), doPhi);
                }
                if (sharedForces) {
                    detail::cudaApplyPcField(arenaForce, k, l);
                } else {
                    detail::cudaApplyPcField(backingForce, k, l);
                }
                flops += 8;
            }
            // Every lane finishes reading the staged multipoles before refilling.
            __syncwarp();
        }
    }
    __syncwarp();
    return flops;
}

template <int order, bool equalMass>
__device__ FlopCount cudaGravityPpDirect(CudaKickData const& data, CudaKickControl<order>& control,
    TreeNode const& self, int sinkBegin, int sinkCount, DeviceVector<int> const& partlist, float h,
    bool doPhi, FmmGeometry const& geometry) {
    int const& tid = threadIdx.x;
    // The forces: the arena planes derived from the shared-memory symbol, or
    // the control block's global backing when the layout keeps none.
    bool const sharedForces = control.layout.forceOffset >= 0;
    CudaKickForceTile const arenaForce = detail::cudaArenaForces<order>(control);
    CudaKickForceTile const& backingForce = control.forces;
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
    FlopCounter<FlopCount> flops = 4;
    FlopCount close = 0;
    FlopCount pairs = 0;
    constexpr FlopCount kPairFlops = equalMass ? 18 : 20;
    FlopCount const forceAddFlops = 3 + int(doPhi);
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
                Position const sx[kPair] = {sinkX[k], sinkX[k + WARP_SIZE]};
                Position const sy[kPair] = {sinkY[k], sinkY[k + WARP_SIZE]};
                Position const sz[kPair] = {sinkZ[k], sinkZ[k + WARP_SIZE]};
                float fx[kPair] = {0.f, 0.f};
                float fy[kPair] = {0.f, 0.f};
                float fz[kPair] = {0.f, 0.f};
                float pot[kPair] = {0.f, 0.f};
                detail::cudaPpSinkGroup<kPair, equalMass>(sources, particleIndex, sx, sy, sz, h2,
                    hinv, h2inv, h3inv, doPhi, fx, fy, fz, pot, close, flops, geometry);
#pragma unroll
                for (int s = 0; s < kPair; s++) {
                    int const sink = k + s * WARP_SIZE;
                    if (sharedForces) {
                        detail::cudaApplyPpForce(
                            arenaForce, sink, fx[s], fy[s], fz[s], pot[s], doPhi);
                    } else {
                        detail::cudaApplyPpForce(
                            backingForce, sink, fx[s], fy[s], fz[s], pot[s], doPhi);
                    }
                }
                pairs += FlopCount(kPair) * particleIndex;
                flops += kPair * forceAddFlops;
            }
            for (; k < kmid; k += WARP_SIZE) {
                Position const sx[1] = {sinkX[k]};
                Position const sy[1] = {sinkY[k]};
                Position const sz[1] = {sinkZ[k]};
                float fx[1] = {0.f};
                float fy[1] = {0.f};
                float fz[1] = {0.f};
                float pot[1] = {0.f};
                detail::cudaPpSinkGroup<1, equalMass>(sources, particleIndex, sx, sy, sz, h2, hinv,
                    h2inv, h3inv, doPhi, fx, fy, fz, pot, close, flops, geometry);
                if (sharedForces) {
                    detail::cudaApplyPpForce(arenaForce, k, fx[0], fy[0], fz[0], pot[0], doPhi);
                } else {
                    detail::cudaApplyPpForce(backingForce, k, fx[0], fy[0], fz[0], pot[0], doPhi);
                }
                pairs += particleIndex;
                flops += forceAddFlops;
            }
            // A trailing partial group of fewer than WARP_SIZE / 2 sinks splits
            // the sources over the lanes instead.
            for (int sink = kmid; sink < sinkCount; sink++) {
                float fx = 0.f;
                float fy = 0.f;
                float fz = 0.f;
                float pot = 0.f;
                Position const sx = sinkX[sink];
                Position const sy = sinkY[sink];
                Position const sz = sinkZ[sink];
                for (int j = tid; j < particleIndex; j += WARP_SIZE) {
                    float const dx0 =
                        fmmPositionDifference(sx, sources.x[j], XDIM, geometry);    // 1
                    float const dx1 =
                        fmmPositionDifference(sy, sources.y[j], YDIM, geometry);    // 1
                    float const dx2 =
                        fmmPositionDifference(sz, sources.z[j], ZDIM, geometry);    // 1
                    float const r2[1] = {sqr(dx0, dx1, dx2)};                       // 5
                    float r3inv[1];
                    float r1inv[1];
                    detail::cudaPpInversePowers<1>(
                        r2, h2, hinv, h2inv, h3inv, doPhi, r3inv, r1inv, close, flops);
                    if constexpr (!equalMass) {
                        r3inv[0] *= sources.mass[j];
                        r1inv[0] *= sources.mass[j];
                    }
                    fx = fmaf(dx0, r3inv[0], fx);    // 2
                    fy = fmaf(dx1, r3inv[0], fy);    // 2
                    fz = fmaf(dx2, r3inv[0], fz);    // 2
                    pot -= r1inv[0];                 // 1
                    pairs++;
                }
                sharedReduceAdd(fx);
                sharedReduceAdd(fy);
                sharedReduceAdd(fz);
                if (doPhi) {
                    sharedReduceAdd(pot);
                }
                flops += 5 * forceAddFlops;
                if (tid == 0) {
                    if (sharedForces) {
                        detail::cudaApplyPpForce(arenaForce, sink, fx, fy, fz, pot, doPhi);
                    } else {
                        detail::cudaApplyPpForce(backingForce, sink, fx, fy, fz, pot, doPhi);
                    }
                    flops += forceAddFlops;
                }
            }
            // Every lane finishes reading the staged tile before it is refilled.
            __syncwarp();
        }
        __syncwarp();
    }
    // Reuse the interaction total instead of updating a second counter for
    // every source pair. Softened pairs add only their table-evaluation cost.
    flops += kPairFlops * pairs;
    if (useGravityCounters) {
        FlopCount direct = pairs - close;
        sharedReduceAdd(close);
        sharedReduceAdd(direct);
        if (tid == 0) {
            atomicAdd(&iclose, double(close));
            atomicAdd(&idirect, double(direct));
            atomicAdd(&numberOfProcessedSinkParticles, sinkCount);
        }
    }

    __syncwarp();
    return flops;
}
