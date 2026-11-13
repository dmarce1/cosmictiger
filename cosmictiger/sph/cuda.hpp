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

/// Accumulate precision-separated GPU work and reduce each full warp into the per-launch
/// exact diagnostic total.
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
 * h𞁑 ∝ ρ𞁑^(−1/ν) with ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2
 * or 6.7 of Monaghan (1992), the solve holds (4π/3) ρ𞁑 h𞁑³/m = Nₛₑₜ, i.e.
 * ΣW = Σ𞁓 w(|r𞁑𞁓|/h𞁑) = 3 Nₛₑₜ/(4π) for the selected kernel W(r, h) = w(r/h)/h³, by a
 * bounded Newton iteration on the smoothing length h𞁑 with dΣW/dh𞁑 = −Σ𞁓 q𞁑𞁓 w'(q𞁑𞁓)/h𞁑,
 * q𞁑𞁓 = |r𞁑𞁓|/h𞁑. The bounds limit the smoothing length without modifying the density sum.
 * A support that reaches half the periodic box (kSphHalfBoxSmoothingLength) and still lacks
 * neighbors stops the run: the nearest-image neighbor search cannot serve a larger one.
 *
 * Symbol directory (in the code below):
 *   q𞁑𞁓 → normalized radius
 *   x𞁑 → sink position; r𞁑𞁓 → separation
 *   h𞁑, h𞁑⁻¹ → smoothing length, its inverse; hₘᵢₙ, hₘₐₓ → smoothing length bounds
 *   ρ𞁑 → density
 *   ΣW, dΣW/dh𞁑 → kernel sum, its smoothing-length derivative
 *   Nₛₑₜ, ΣWₛₑₜ → target neighbor count, target kernel sum
 *   paddedX_a → x𞁑 as the three-axis record that the search-box bounds compare.
 *   Code names: ih𞁑 = h𞁑⁻¹; dΣW_dh𞁑 = dΣW/dh𞁑.
 */
// clang-format on

template <int blockSize, class Neighbors>
inline __device__ bool solveSphSmoothingLength(float& h𞁑, float hₘᵢₙ, float hₘₐₓ, float Nₛₑₜ,
    Neighbors const& neighbors, sph_geometry::SpatialVector<Position> const& x𞁑,
    Vector3<Position> const& paddedX_a, SphBox const& neighborSearchBox,
    sph_flops::Counter& flops) {
    int const threadIndex = threadIdx.x;
    int const runtimeBlockSize = blockDim.x;
    ALWAYS_ASSERT(hₘᵢₙ > 0.0f && hₘₐₓ >= hₘᵢₙ && Nₛₑₜ > 0.0f);
    if (threadIndex == 0) {
        h𞁑 = fminf(fmaxf(h𞁑, hₘᵢₙ), hₘₐₓ);
    }
    __syncthreads();
    float lower = hₘᵢₙ;
    float upper = hₘₐₓ;
    float const ΣWₛₑₜ = sphTargetKernelSum(Nₛₑₜ);
    flops += 1;
    for (int iteration = 0; iteration < 128; iteration++) {
        int boxExceeded = false;
        for (int dim = 0; dim < kDimensions; dim++) {
            bool const upperExceeded = neighborSearchBox.end[dim] <
                RangeFixed(paddedX_a[dim]) + RangeFixed(h𞁑) + RangeFixed::min();
            // Single-precision fixed-point scale, preserving short circuiting.
            flops.singlePrecision += 1;
            if (upperExceeded) {
                boxExceeded = true;
            } else {
                flops.singlePrecision += 1;
                if (RangeFixed(paddedX_a[dim]) <
                    neighborSearchBox.begin[dim] + RangeFixed(h𞁑) + RangeFixed::min()) {
                    boxExceeded = true;
                }
            }
        }
        if (boxExceeded) {
            return false;
        }
        float const ih𞁑 = 1.0f / h𞁑;
        flops += 1;
        float ΣW = 0.0f;
        float dΣW_dh𞁑 = 0.0f;
        for (int neighborIndex = threadIndex; neighborIndex < neighbors.size();
            neighborIndex += runtimeBlockSize) {
            auto const& neighbor = neighbors[neighborIndex];
            auto const r𞁑𞁓 = distance(x𞁑, neighbor.position);
            float const q𞁑𞁓 = r𞁑𞁓.norm() * ih𞁑;
            // Fixed distances, dot, root, normalized radius.
            flops += sph_flops::kVectorOperation + sph_flops::kNorm + 1;
            if (q𞁑𞁓 < 1.0f) {
                ΣW += kernelW(q𞁑𞁓);
                dΣW_dh𞁑 -= q𞁑𞁓 * dkernelWDq(q𞁑𞁓) * ih𞁑;
                flops += 4 + sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓) +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
            }
        }
        sharedReduceAdd<float, blockSize>(ΣW);
        sharedReduceAdd<float, blockSize>(dΣW_dh𞁑);
        flops += 2 * detail::sphReductionFlops<blockSize>();
        float const residual = ΣW - ΣWₛₑₜ;
        flops += 2;    // Residual and relative tolerance product (absolute value is free).
        float const tolerance = kSphSmoothingLengthTolerance * ΣWₛₑₜ;
        if (hₘₐₓ >= kSphHalfBoxSmoothingLength && h𞁑 >= hₘₐₓ && residual < -tolerance) {
            // The support spans half the periodic box and still lacks neighbors.
            if (threadIndex == 0) {
                printf("SPH smoothing length exceeds half the periodic box: smoothingLength=%e "
                       "kernelSum=%e targetSum=%e target=%e\n",
                    h𞁑, ΣW, ΣWₛₑₜ, Nₛₑₜ);
                __trap();
            }
        }
        if (fabsf(residual) <= tolerance || (h𞁑 <= hₘᵢₙ && residual >= 0.0f) ||
            (h𞁑 >= hₘₐₓ && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = h𞁑;
        } else {
            lower = h𞁑;
        }
        float change = dΣW_dh𞁑 > 0.0f ? -residual / dΣW_dh𞁑 : 0.2f * h𞁑;
        change = fminf(fmaxf(change, -0.2f * h𞁑), 0.2f * h𞁑);
        float next = fminf(fmaxf(h𞁑 + change, hₘᵢₙ), hₘₐₓ);
        flops += 4;    // Newton/fallback change, two clamp products, radius update.
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == h𞁑) {
            return true;
        }
        __syncthreads();
        if (threadIndex == 0) {
            h𞁑 = next;
        }
        __syncthreads();
    }
    if (threadIndex == 0) {
        printf("SPH smoothing-length solve did not converge: smoothingLength=%e bounds=[%e,%e] "
               "target=%e\n",
            h𞁑, hₘᵢₙ, hₘₐₓ, Nₛₑₜ);
        __trap();
    }
    return false;
}

#endif

#endif /* SPH_CUDA_HPP_ */
