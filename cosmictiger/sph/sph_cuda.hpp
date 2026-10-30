/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_CUDA_HPP_
#define SPH_CUDA_HPP_

#include <numbers>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/cuda/device_vector.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_flops.hpp>
#include <cosmictiger/sph/sph_kernel.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

using SphRunCudaData = SphRunData;

inline constexpr int kPrehydro1BlockSize = 160;
inline constexpr int kPrehydro2BlockSize = 64;
inline constexpr int kHydroBlockSize = 96;
inline constexpr int kRungsBlockSize = 256;

struct SphReduction {
    float minimumTimestep;
    int nextNodeIndex;
    int pendingChanges;
    int timestepFailureReported;
    float minimumSupportRadius;
    float maximumSupportRadius;
    float maximumSignalSpeed;
    FlopCount floatingPointOperations;
    int maximumHydroRung;
    int maximumGravityRung;
    int maximumRung;
    float inverseSoundTimestep;
    float inverseViscosityTimestep;
    float inverseAccelerationTimestep;
    // Variational density-pass diagnostics; see SphRunReturn.
    float minimumGradHOmega;
    float maximumSupportResidual;
    int heldSupports;
    int illConditionedSupports;
};

#ifdef USE_CUDA
SphRunReturn sphRunCuda(SphRunParameters parameters, SphRunCudaData data, CudaStream& stream);
#endif

#ifdef __CUDACC__

__global__ void sphCudaPrehydro1(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
__global__ void sphCudaPrehydro2(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
__global__ void sphCudaHydro(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
__global__ void sphCudaRungs(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
#endif

struct SphNeighborPosition {
    Position positionX;
    Position positionY;
    Position positionZ;
};

#ifdef __CUDACC__
namespace detail {

// sharedReduceAdd executes a sum in each lane whose partner exists, including
// the redundant sums that broadcast a block result. Non-power-of-two blocks
// therefore have lane-dependent costs; integer reductions do not contribute.
template <int blockSize>
__device__ inline FlopCount sphReductionFlops() {
    int powerOfTwo = kWarpSize;
    while (powerOfTwo < blockSize) {
        powerOfTwo *= 2;
    }
    FlopCount count = 0;
    for (int bit = powerOfTwo / 2; bit > 0; bit /= 2) {
        count += (threadIdx.x + bit) % powerOfTwo < blockSize;
    }
    return count;
}

__device__ inline void recordSphGpuFlops(FlopCount count, SphReduction* reduction) {
    if constexpr (kCountFlops) {
        addGpuFlops(count);
        // Every SPH block is a multiple of a full warp. Keep the per-launch
        // diagnostic exact independently of the global reporting interval.
        sharedReduceAdd(count);
        if (threadIdx.x % kWarpSize == 0) {
            atomicAdd(&reduction->floatingPointOperations, count);
        }
    }
}

}    // namespace detail

// Section 6 of Monaghan (1992): the resolution length follows the local density,
// h_a ∝ ρ_a^(−1/ν) with ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2 or
// 6.7, the solve holds (4π/3) ρ_a H_a³/m = N_target, i.e. ΣW = Σ_b w(r_ab/H_a) = 3 N_target/(4π)
// for the selected kernel W(r, H) = w(r/H)/H³, by a bounded Newton iteration on the support
// radius H_a = supportRadius with dΣW/dH_a = −Σ_b q_ab w'(q_ab)/H_a, q_ab = r_ab/H_a. The
// bounds limit the support radius without modifying the density sum.
// NOLINTBEGIN(readability-identifier-naming)
template <int blockSize>
inline __device__ bool solveSphSupportRadius(float& supportRadius, float minimumSupportRadius,
    float maximumSupportRadius, float targetNeighborCount,
    DeviceVector<SphNeighborPosition> const& neighbors,
    Array<Position, kDimensions> const& position, Fixed32Range const& neighborSearchBox,
    FlopCounter<>& flops) {
    int const threadIndex = threadIdx.x;
    int const runtimeBlockSize = blockDim.x;
    ALWAYS_ASSERT(minimumSupportRadius > 0.0f && maximumSupportRadius >= minimumSupportRadius &&
        targetNeighborCount > 0.0f);
    if (threadIndex == 0) {
        supportRadius = fminf(fmaxf(supportRadius, minimumSupportRadius), maximumSupportRadius);
    }
    __syncthreads();
    float lower = minimumSupportRadius;
    float upper = maximumSupportRadius;
    float const ΣW_target = targetNeighborCount * (3.0f / (4.0f * std::numbers::pi_v<float>) );
    flops += 1;
    for (int iteration = 0; iteration < 128; iteration++) {
        int boxExceeded = false;
        for (int dim = 0; dim < kDimensions; dim++) {
            bool const upperExceeded = neighborSearchBox.end[dim] <
                RangeFixed(position[dim]) + RangeFixed(supportRadius) + RangeFixed::min();
            flops += 1;    // Floating support-to-fixed scaling, preserving short circuiting.
            if (upperExceeded) {
                boxExceeded = true;
            } else {
                flops += 1;
                if (RangeFixed(position[dim]) <
                    neighborSearchBox.begin[dim] + RangeFixed(supportRadius) + RangeFixed::min()) {
                    boxExceeded = true;
                }
            }
        }
        if (boxExceeded) {
            return false;
        }
        float const one_over_H_a = 1.0f / supportRadius;
        flops += 1;
        float ΣW = 0.0f;
        float dΣW_dH_a = 0.0f;
        for (int neighborIndex = threadIndex; neighborIndex < neighbors.size();
            neighborIndex += runtimeBlockSize) {
            auto const& neighbor = neighbors[neighborIndex];
            Vector3<float> const r_ab(distance(position[kXDimension], neighbor.positionX),
                distance(position[kYDimension], neighbor.positionY),
                distance(position[kZDimension], neighbor.positionZ));
            float const q_ab = sqrtf(r_ab.dot(r_ab)) * one_over_H_a;
            flops += 10;    // Three fixed distances, dot, root, normalized radius.
            if (q_ab < 1.0f) {
                ΣW += kernelW(q_ab);
                dΣW_dH_a -= q_ab * dkernelWDq(q_ab) * one_over_H_a;
                flops += 4 + sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab) +
                    sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab, true);
            }
        }
        sharedReduceAdd<float, blockSize>(ΣW);
        sharedReduceAdd<float, blockSize>(dΣW_dH_a);
        flops += 2 * detail::sphReductionFlops<blockSize>();
        float const residual = ΣW - ΣW_target;
        flops += 2;    // Residual and relative tolerance product (absolute value is free).
        if (fabsf(residual) <= kSphSupportRadiusTolerance * ΣW_target ||
            (supportRadius <= minimumSupportRadius && residual >= 0.0f) ||
            (supportRadius >= maximumSupportRadius && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = supportRadius;
        } else {
            lower = supportRadius;
        }
        float change = dΣW_dH_a > 0.0f ? -residual / dΣW_dH_a : 0.2f * supportRadius;
        change = fminf(fmaxf(change, -0.2f * supportRadius), 0.2f * supportRadius);
        float next =
            fminf(fmaxf(supportRadius + change, minimumSupportRadius), maximumSupportRadius);
        flops += 4;    // Newton/fallback change, two clamp products, radius update.
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == supportRadius) {
            return true;
        }
        __syncthreads();
        if (threadIndex == 0) {
            supportRadius = next;
        }
        __syncthreads();
    }
    if (threadIndex == 0) {
        PRINT("SPH smoothing-length solve did not converge: supportRadius=%e bounds=[%e,%e] "
              "target=%e\n",
            supportRadius, minimumSupportRadius, maximumSupportRadius, targetNeighborCount);
        __trap();
    }
    return false;
}
// NOLINTEND(readability-identifier-naming)
#endif

#endif /* SPH_CUDA_HPP_ */
