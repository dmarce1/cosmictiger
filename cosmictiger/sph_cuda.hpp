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

#include <cosmictiger/constants.hpp>
#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/timer.hpp>

struct SphRunCudaData {
    // Staged source arrays, including the sources that are also local sinks.
    Fixed32* sourcePositionX;
    Fixed32* sourcePositionY;
    Fixed32* sourcePositionZ;
    float* sourceVelocityX;
    float* sourceVelocityY;
    float* sourceVelocityZ;
    float* sourceInternalEnergy;
    float* sourceDensity;
    float* sourceSupportRadius;
    char* sourceRungs;

    // Local sink arrays indexed by the SPH particle index.
    float* sinkInternalEnergyRate;
    float* sinkDensity;
    char* sinkSemiactive;
    char* sinkSmoothingConverged;
    SphThermodynamicState* sinkThermodynamicStates;
    SphAcceleration* sinkAccelerations;
    float* sinkGravityAccelerationX;
    float* sinkGravityAccelerationY;
    float* sinkGravityAccelerationZ;
    ParticleInteger* sinkParticleIndices;
    char* sinkParticleRungs;    // Indexed through sinkParticleIndices into the full particle array.

    int numberOfTargetNodes;
    SphTreeNode* treeNodes;
    int* targetNodeIndices;
    int* neighborNodeIndices;
    float targetNeighborCount;    // Weighted neighbor target for the support-radius solve.
    float gasMass;
    float adiabaticIndex;    // Ideal-gas index shared by all gas particles.
    bool selfGravityEnabled;
    float gravityEffectiveRadius;        // Twice the gravity kernel standard deviation.
    float accelerationTimestepFactor;    // Coefficient of the acceleration timestep bound.
    // Optional external central potential, used by the orbiting-disc problem.
    float centralAcceleration;     // Zero disables this force.
    float centralSupportRadius;    // Support radius of the selected gravity kernel.
};

#define PREHYDRO1_BLOCK_SIZE 160
#define PREHYDRO2_BLOCK_SIZE 64
#define HYDRO_BLOCK_SIZE 96
#define RUNGS_BLOCK_SIZE 256

#define MAX_NEIGHBOR_RUNG_DIFFERENCE 1
#define SPH_SUPPORT_RADIUS_TOLERANCE float(1.0e-4)

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

SphRunReturn sphRunCuda(SphRunParameters parameters, SphRunCudaData data, cudaStream_t stream);

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

// A source can reach any sink in this box through a periodic image. Expanding
// the unwrapped interval before testing avoids inconsistent endpoint wrapping.
CUDA_EXPORT inline bool sphSupportIntersects(
    Fixed32Range const& sinkBox, Array<Fixed32, NDIM> const& sourcePosition, float sourceSupport) {
    Fixed32Range expanded = sinkBox;
    for (int dim = 0; dim < NDIM; dim++) {
        expanded.begin[dim] -= RangeFixed(sourceSupport);
        expanded.end[dim] += RangeFixed(sourceSupport);
    }
    return expanded.contains(sourcePosition);
}

#ifdef __CUDACC__
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
    float const target = targetNeighborCount * float(3.0 / (4.0 * M_PI));
    // rho H^3 / m = sum W(r/H) is held fixed, giving H proportional to rho^-1/3.
    // minimumSupportRadius and maximumSupportRadius bound the support radius without modifying the
    // density sum.
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
        float const inverseSupportRadius = 1.0f / supportRadius;
        float sum = 0.0f;
        float derivative = 0.0f;
        for (int neighborIndex = threadIndex; neighborIndex < neighbors.size();
            neighborIndex += runtimeBlockSize) {
            auto const& neighbor = neighbors[neighborIndex];
            float const separationX = distance(position[XDIM], neighbor.positionX);
            float const separationY = distance(position[YDIM], neighbor.positionY);
            float const separationZ = distance(position[ZDIM], neighbor.positionZ);
            float const normalizedRadius =
                sqrtf(sqr(separationX, separationY, separationZ)) * inverseSupportRadius;
            if (normalizedRadius < 1.0f) {
                sum += kernelW(normalizedRadius);
                derivative -=
                    normalizedRadius * dkernelWDq(normalizedRadius) * inverseSupportRadius;
            }
        }
        sharedReduceAdd<float, blockSize>(sum);
        sharedReduceAdd<float, blockSize>(derivative);
        float const residual = sum - target;
        if (fabsf(residual) <= SPH_SUPPORT_RADIUS_TOLERANCE * target ||
            (supportRadius <= minimumSupportRadius && residual >= 0.0f) ||
            (supportRadius >= maximumSupportRadius && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = supportRadius;
        } else {
            lower = supportRadius;
        }
        float change = derivative > 0.0f ? -residual / derivative : 0.2f * supportRadius;
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
#endif

#endif /* SPH_CUDA_HPP_ */
