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
    sph_geometry::SpatialVector<Position> position;
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
 * H𞁑 ∝ ρ𞁑^(−1/ν) with ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2
 * or 6.7 of Monaghan (1992), the solve holds (4π/3) ρ𞁑 H𞁑³/m = Nₛₑₜ, i.e.
 * ΣW = Σ𞁓 w(|r𞁑𞁓|/H𞁑) = 3 Nₛₑₜ/(4π) for the selected kernel W(r, H) = w(r/H)/H³, by a
 * bounded Newton iteration on the support radius H𞁑 with dΣW/dH𞁑 = −Σ𞁓 q𞁑𞁓 w'(q𞁑𞁓)/H𞁑,
 * q𞁑𞁓 = |r𞁑𞁓|/H𞁑. The bounds limit the support radius without modifying the density sum.
 *
 * Symbol directory (in the code below):
 *   q𞁑𞁓 → normalized radius
 *   x𞁑 → sink position; r𞁑𞁓 → separation
 *   H𞁑, H𞁑⁻¹ → smoothing length, its inverse; Hₘᵢₙ, Hₘ𞁑ₓ → smoothing length bounds
 *   ρ𞁑 → density
 *   ΣW, dΣW/dH𞁑 → kernel sum, its support derivative
 *   Nₛₑₜ, ΣWₛₑₜ → target neighbor count, target kernel sum
 *   paddedX_a → x𞁑 as the three-axis record that the search-box bounds compare.
 *   Code names: iH𞁑 = H𞁑⁻¹; dΣW_dH𞁑 = dΣW/dH𞁑.
 */
// clang-format on

template <int blockSize, class Neighbors>
inline __device__ bool solveSphSmoothingLength(float& H𞁑, float Hₘᵢₙ, float Hₘ𞁑ₓ, float Nₛₑₜ,
    Neighbors const& neighbors, sph_geometry::SpatialVector<Position> const& x𞁑,
    Vector3<Position> const& paddedX_a, Fixed32Range const& neighborSearchBox,
    sph_flops::Counter& flops) {
    int const threadIndex = threadIdx.x;
    int const runtimeBlockSize = blockDim.x;
    ALWAYS_ASSERT(Hₘᵢₙ > 0.0f && Hₘ𞁑ₓ >= Hₘᵢₙ && Nₛₑₜ > 0.0f);
    if (threadIndex == 0) {
        H𞁑 = fminf(fmaxf(H𞁑, Hₘᵢₙ), Hₘ𞁑ₓ);
    }
    __syncthreads();
    float lower = Hₘᵢₙ;
    float upper = Hₘ𞁑ₓ;
    float const ΣWₛₑₜ = Nₛₑₜ * float(1.0 / sph_geometry::kUnitBallVolume);
    flops += 1;
    for (int iteration = 0; iteration < 128; iteration++) {
        int boxExceeded = false;
        for (int dim = 0; dim < kDimensions; dim++) {
            bool const upperExceeded = neighborSearchBox.end[dim] <
                RangeFixed(paddedX_a[dim]) + RangeFixed(H𞁑) + RangeFixed::min();
            // Single-precision fixed-point scale, preserving short circuiting.
            flops.singlePrecision += 1;
            if (upperExceeded) {
                boxExceeded = true;
            } else {
                flops.singlePrecision += 1;
                if (RangeFixed(paddedX_a[dim]) <
                    neighborSearchBox.begin[dim] + RangeFixed(H𞁑) + RangeFixed::min()) {
                    boxExceeded = true;
                }
            }
        }
        if (boxExceeded) {
            return false;
        }
        float const iH𞁑 = 1.0f / H𞁑;
        flops += 1;
        float ΣW = 0.0f;
        float dΣW_dH𞁑 = 0.0f;
        for (int neighborIndex = threadIndex; neighborIndex < neighbors.size();
            neighborIndex += runtimeBlockSize) {
            auto const& neighbor = neighbors[neighborIndex];
            auto const r𞁑𞁓 = distance(x𞁑, neighbor.position);
            float const q𞁑𞁓 = r𞁑𞁓.norm() * iH𞁑;
            // Fixed distances, dot, root, normalized radius.
            flops += sph_flops::kVectorOperation + sph_flops::kNorm + 1;
            if (q𞁑𞁓 < 1.0f) {
                ΣW += kernelW(q𞁑𞁓);
                dΣW_dH𞁑 -= q𞁑𞁓 * dkernelWDq(q𞁑𞁓) * iH𞁑;
                flops += 4 + sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓) +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
            }
        }
        sharedReduceAdd<float, blockSize>(ΣW);
        sharedReduceAdd<float, blockSize>(dΣW_dH𞁑);
        flops += 2 * detail::sphReductionFlops<blockSize>();
        float const residual = ΣW - ΣWₛₑₜ;
        flops += 2;    // Residual and relative tolerance product (absolute value is free).
        if (fabsf(residual) <= kSphSmoothingLengthTolerance * ΣWₛₑₜ ||
            (H𞁑 <= Hₘᵢₙ && residual >= 0.0f) || (H𞁑 >= Hₘ𞁑ₓ && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = H𞁑;
        } else {
            lower = H𞁑;
        }
        float change = dΣW_dH𞁑 > 0.0f ? -residual / dΣW_dH𞁑 : 0.2f * H𞁑;
        change = fminf(fmaxf(change, -0.2f * H𞁑), 0.2f * H𞁑);
        float next = fminf(fmaxf(H𞁑 + change, Hₘᵢₙ), Hₘ𞁑ₓ);
        flops += 4;    // Newton/fallback change, two clamp products, radius update.
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == H𞁑) {
            return true;
        }
        __syncthreads();
        if (threadIndex == 0) {
            H𞁑 = next;
        }
        __syncthreads();
    }
    if (threadIndex == 0) {
        printf("SPH smoothing-length solve did not converge: smoothingLength=%e bounds=[%e,%e] "
               "target=%e\n",
            H𞁑, Hₘᵢₙ, Hₘ𞁑ₓ, Nₛₑₜ);
        __trap();
    }
    return false;
}

#endif

#endif /* SPH_CUDA_HPP_ */
