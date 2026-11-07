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
#include <cosmictiger/cuda/kernel_error.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/neighbor_buffer.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/run_data.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>

using SphRunCudaData = SphRunData;

inline constexpr int kSmoothingLengthsBlockSize = 160;
inline constexpr int kDensityAndGradientsBlockSize = 64;
inline constexpr int kHydroBlockSize = 96;
inline constexpr int kRungsBlockSize = 256;

struct SphTimestepFailure {
    ParticleInteger particleIndex;
    int requestedRung;
};

struct SphReduction {
    float minimumTimestep;
    int nextNodeIndex;
    int pendingChanges;
    CudaKernelError<SphTimestepFailure> timestepError;
    float minimumSmoothingLength;
    float maximumSmoothingLength;
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

__global__ void sphCudaSmoothingLengths(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
__global__ void sphCudaDensityAndGradients(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
__global__ void sphCudaHydro(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
__global__ void sphCudaRungs(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction);
#endif

struct SphNeighborPosition {
    sph_geometry::SpatialVector<Position> x;
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

__device__ inline void recordSphGpuFlops(sph_flops::Counter const& flops, SphReduction* reduction) {
    if constexpr (kCountFlops) {
        addGpuFlops(flops.counts());
        FlopCount count = FlopCount(flops);
        // Every SPH block is a multiple of a full warp. Keep the per-launch
        // diagnostic exact independently of the global reporting interval.
        sharedReduceAdd(count);
        if (threadIdx.x % kWarpSize == 0) {
            atomicAdd(&reduction->floatingPointOperations, count);
        }
    }
}

}    // namespace detail

// clang-format off
/*
 * Section 6 of Monaghan (1992): the resolution length follows the local density,
 * H_a ∝ ρ_a^(−1/ν) with ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2
 * or 6.7, the solve holds (4π/3) ρ_a H_a³/m = N_target, i.e. ΣW = Σ_b w(|r_ab|/H_a) = 3 N_target/(4π)
 * for the selected kernel W(r, H) = w(r/H)/H³, by a bounded Newton iteration on the support
 * radius H_a with dΣW/dH_a = −Σ_b q_ab w'(q_ab)/H_a, q_ab = |r_ab|/H_a. The
 * bounds limit the support radius without modifying the density sum.
 *
 * Symbol directory (in the code below):
 *   q_ab → normalized radius
 *   x_a → sink position; r_ab → separation
 *   H_a, H_a⁻¹ → smoothing length, its inverse; H_min, H_max → smoothing length bounds
 *   ρ_a → density
 *   ΣW, dΣW/dH_a → kernel sum, its support derivative
 *   N_target, ΣW_target → target neighbor count, target kernel sum
 *   paddedX_a → x_a as the three-axis record that the search-box bounds compare.
 *   Code names: H_inv_a = H_a⁻¹; dΣW_dH_a = dΣW/dH_a.
 */
// clang-format on

template <int blockSize, class Neighbors>
inline __device__ bool solveSphSmoothingLength(float& H_a, float H_min, float H_max, float N_target,
    Neighbors const& neighbors, sph_geometry::SpatialVector<Position> const& x_a,
    Vector3<Position> const& paddedX_a, Fixed32Range const& neighborSearchBox,
    sph_flops::Counter& flops) {
    int const threadIndex = threadIdx.x;
    int const runtimeBlockSize = blockDim.x;
    ALWAYS_ASSERT(H_min > 0.0f && H_max >= H_min && N_target > 0.0f);
    if (threadIndex == 0) {
        H_a = fminf(fmaxf(H_a, H_min), H_max);
    }
    __syncthreads();
    float lower = H_min;
    float upper = H_max;
    float const ΣW_target = N_target * float(1.0 / sph_geometry::kUnitBallVolume);
    flops += 1;
    for (int iteration = 0; iteration < 128; iteration++) {
        int boxExceeded = false;
        for (int dim = 0; dim < kDimensions; dim++) {
            bool const upperExceeded = neighborSearchBox.end[dim] <
                RangeFixed(paddedX_a[dim]) + RangeFixed(H_a) + RangeFixed::min();
            // Single-precision fixed-point scale, preserving short circuiting.
            flops.singlePrecision += 1;
            if (upperExceeded) {
                boxExceeded = true;
            } else {
                flops.singlePrecision += 1;
                if (RangeFixed(paddedX_a[dim]) <
                    neighborSearchBox.begin[dim] + RangeFixed(H_a) + RangeFixed::min()) {
                    boxExceeded = true;
                }
            }
        }
        if (boxExceeded) {
            return false;
        }
        float const H_inv_a = 1.0f / H_a;
        flops += 1;
        float ΣW = 0.0f;
        float dΣW_dH_a = 0.0f;
        for (int neighborIndex = threadIndex; neighborIndex < neighbors.size();
            neighborIndex += runtimeBlockSize) {
            auto const& neighbor = neighbors[neighborIndex];
            auto const r_ab = distance(x_a, neighbor.x);
            float const q_ab = r_ab.norm() * H_inv_a;
            // Fixed distances, dot, root, normalized radius.
            flops += sph_flops::kVectorOperation + sph_flops::kNorm + 1;
            if (q_ab < 1.0f) {
                ΣW += kernelW(q_ab);
                dΣW_dH_a -= q_ab * dkernelWDq(q_ab) * H_inv_a;
                flops += 4 + sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab) +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
            }
        }
        sharedReduceAdd<float, blockSize>(ΣW);
        sharedReduceAdd<float, blockSize>(dΣW_dH_a);
        flops += 2 * detail::sphReductionFlops<blockSize>();
        float const residual = ΣW - ΣW_target;
        flops += 2;    // Residual and relative tolerance product (absolute value is free).
        if (fabsf(residual) <= kSphSmoothingLengthTolerance * ΣW_target ||
            (H_a <= H_min && residual >= 0.0f) || (H_a >= H_max && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = H_a;
        } else {
            lower = H_a;
        }
        float change = dΣW_dH_a > 0.0f ? -residual / dΣW_dH_a : 0.2f * H_a;
        change = fminf(fmaxf(change, -0.2f * H_a), 0.2f * H_a);
        float next = fminf(fmaxf(H_a + change, H_min), H_max);
        flops += 4;    // Newton/fallback change, two clamp products, radius update.
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == H_a) {
            return true;
        }
        __syncthreads();
        if (threadIndex == 0) {
            H_a = next;
        }
        __syncthreads();
    }
    if (threadIndex == 0) {
        printf("SPH smoothing-length solve did not converge: smoothingLength=%e bounds=[%e,%e] "
               "target=%e\n",
            H_a, H_min, H_max, N_target);
        __trap();
    }
    return false;
}

#endif

#endif /* SPH_CUDA_HPP_ */
