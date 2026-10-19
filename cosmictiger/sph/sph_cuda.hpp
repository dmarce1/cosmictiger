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

#ifndef SPH_CUDA_HPP_
#define SPH_CUDA_HPP_

#include <cosmictiger/core/defs.hpp>
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
#include <cosmictiger/sph/sph_kernel.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

using SphRunCudaData = SphRunData;

#define PREHYDRO1_BLOCK_SIZE 160
#define PREHYDRO2_BLOCK_SIZE 64
#define HYDRO_BLOCK_SIZE 96
#define RUNGS_BLOCK_SIZE 256

struct SphReduction {
    int nextNodeIndex;
    int pendingChanges;
    float minimumSupportRadius;
    float maximumSupportRadius;
    float maximumSignalSpeed;
    double floatingPointOperations;
    int maximumHydroRung;
    int maximumGravityRung;
    int maximumRung;
    float inverseSoundTimestep;
    float inverseViscosityTimestep;
    float inverseAccelerationTimestep;
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
    Fixed32 positionX;
    Fixed32 positionY;
    Fixed32 positionZ;
};

#ifdef __CUDACC__
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
    DeviceVector<SphNeighborPosition> const& neighbors, Array<Fixed32, NDIM> const& position,
    Fixed32Range const& neighborSearchBox) {
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
    float const ΣW_target = targetNeighborCount * float(3.0 / (4.0 * M_PI));
    for (int iteration = 0; iteration < 128; iteration++) {
        int boxExceeded = false;
        for (int dim = 0; dim < NDIM; dim++) {
            if (neighborSearchBox.end[dim] <
                    RangeFixed(position[dim]) + RangeFixed(supportRadius) + RangeFixed::min() ||
                RangeFixed(position[dim]) <
                    neighborSearchBox.begin[dim] + RangeFixed(supportRadius) + RangeFixed::min()) {
                boxExceeded = true;
            }
        }
        if (boxExceeded) {
            return false;
        }
        float const one_over_H_a = 1.0f / supportRadius;
        float ΣW = 0.0f;
        float dΣW_dH_a = 0.0f;
        for (int neighborIndex = threadIndex; neighborIndex < neighbors.size();
            neighborIndex += runtimeBlockSize) {
            auto const& neighbor = neighbors[neighborIndex];
            Vector3<float> const r_ab(distance(position[XDIM], neighbor.positionX),
                distance(position[YDIM], neighbor.positionY),
                distance(position[ZDIM], neighbor.positionZ));
            float const q_ab = sqrtf(r_ab.dot(r_ab)) * one_over_H_a;
            if (q_ab < 1.0f) {
                ΣW += kernelW(q_ab);
                dΣW_dH_a -= q_ab * dkernelWDq(q_ab) * one_over_H_a;
            }
        }
        sharedReduceAdd<float, blockSize>(ΣW);
        sharedReduceAdd<float, blockSize>(dΣW_dH_a);
        float const residual = ΣW - ΣW_target;
        if (fabsf(residual) <= SPH_SUPPORT_RADIUS_TOLERANCE * ΣW_target ||
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
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
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
