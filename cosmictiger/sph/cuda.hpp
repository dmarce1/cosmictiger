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
template <bool retainHydroDiagnostics = false>
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
 *   r_ab → separation
 *   H_a → smoothing length
 *   ρ_a → density
 *   ΣW → kernel sum
 */
// clang-format on

template <int blockSize, class Neighbors>
inline __device__ bool solveSphSmoothingLength(float& H_a, float minimumSmoothingLength,
    float maximumSmoothingLength, float targetNeighborCount, Neighbors const& neighbors,
    Vector3<Position> const& position, Fixed32Range const& neighborSearchBox,
    sph_flops::Counter& flops) {
    int const threadIndex = threadIdx.x;
    int const runtimeBlockSize = blockDim.x;
    ALWAYS_ASSERT(minimumSmoothingLength > 0.0f &&
        maximumSmoothingLength >= minimumSmoothingLength && targetNeighborCount > 0.0f);
    if (threadIndex == 0) {
        H_a = fminf(fmaxf(H_a, minimumSmoothingLength), maximumSmoothingLength);
    }
    __syncthreads();
    float lower = minimumSmoothingLength;
    float upper = maximumSmoothingLength;
    float const targetKernelSum = targetNeighborCount * float(1.0 / sph_geometry::kUnitBallVolume);
    flops += 1;
    for (int iteration = 0; iteration < 128; iteration++) {
        int boxExceeded = false;
        for (int dim = 0; dim < kDimensions; dim++) {
            bool const upperExceeded = neighborSearchBox.end[dim] <
                RangeFixed(position[dim]) + RangeFixed(H_a) + RangeFixed::min();
            // Single-precision fixed-point scale, preserving short circuiting.
            flops.singlePrecision += 1;
            if (upperExceeded) {
                boxExceeded = true;
            } else {
                flops.singlePrecision += 1;
                if (RangeFixed(position[dim]) <
                    neighborSearchBox.begin[dim] + RangeFixed(H_a) + RangeFixed::min()) {
                    boxExceeded = true;
                }
            }
        }
        if (boxExceeded) {
            return false;
        }
        float const inverseSmoothingLength = 1.0f / H_a;
        flops += 1;
        float ΣW = 0.0f;
        float kernelSumSupportDerivative = 0.0f;
        for (int neighborIndex = threadIndex; neighborIndex < neighbors.size();
            neighborIndex += runtimeBlockSize) {
            auto const& neighbor = neighbors[neighborIndex];
            Vector3<float> const r_ab(distance(position[kXi], neighbor.positionX),
                distance(position[kYi], neighbor.positionY),
                distance(position[kZi], neighbor.positionZ));
            float const q_ab = sqrtf(sph_geometry::spatialDot(r_ab, r_ab)) * inverseSmoothingLength;
            flops += 10;    // Three fixed distances, dot, root, normalized radius.
            if (q_ab < 1.0f) {
                ΣW += kernelW(q_ab);
                kernelSumSupportDerivative -= q_ab * dkernelWDq(q_ab) * inverseSmoothingLength;
                flops += 4 + sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab) +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
            }
        }
        sharedReduceAdd<float, blockSize>(ΣW);
        sharedReduceAdd<float, blockSize>(kernelSumSupportDerivative);
        flops += 2 * detail::sphReductionFlops<blockSize>();
        float const residual = ΣW - targetKernelSum;
        flops += 2;    // Residual and relative tolerance product (absolute value is free).
        if (fabsf(residual) <= kSphSmoothingLengthTolerance * targetKernelSum ||
            (H_a <= minimumSmoothingLength && residual >= 0.0f) ||
            (H_a >= maximumSmoothingLength && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = H_a;
        } else {
            lower = H_a;
        }
        float change =
            kernelSumSupportDerivative > 0.0f ? -residual / kernelSumSupportDerivative : 0.2f * H_a;
        change = fminf(fmaxf(change, -0.2f * H_a), 0.2f * H_a);
        float next = fminf(fmaxf(H_a + change, minimumSmoothingLength), maximumSmoothingLength);
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
            H_a, minimumSmoothingLength, maximumSmoothingLength, targetNeighborCount);
        __trap();
    }
    return false;
}

#endif

#endif /* SPH_CUDA_HPP_ */
