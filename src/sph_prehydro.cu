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

#include <cosmictiger/sph_cuda.hpp>

namespace detail {

struct SmoothingWorkspace {
    DeviceVector<SphNeighborPosition> positions;
};

struct NeighborActivity {
    float supportRadius;
    char rung;
};

struct DensityWorkspace {
    DeviceVector<SphNeighborPosition> positions;
    DeviceVector<NeighborActivity> particles;
};

}    // namespace detail

__global__ void sphCudaPrehydro1(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::SmoothingWorkspace workspace;
    new (&workspace) detail::SmoothingWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    while (targetNodeIndex < data.numberOfTargetNodes) {
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        workspace.positions.resize(0);
        for (int neighborNodeIndex = targetNode.neighborRange.first;
            neighborNodeIndex < targetNode.neighborRange.second; neighborNodeIndex++) {
            auto const& neighborNode = data.treeNodes[data.neighborNodeIndices[neighborNodeIndex]];
            int const end = neighborNode.particleRange.first +
                roundUp(neighborNode.particleRange.second - neighborNode.particleRange.first,
                    blockSize);
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                Array<Fixed32, NDIM> position;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    position[XDIM] = data.sourcePositionX[sourceParticleIndex];
                    position[YDIM] = data.sourcePositionY[sourceParticleIndex];
                    position[ZDIM] = data.sourcePositionZ[sourceParticleIndex];
                    contains = targetNode.outerBox.contains(position);
                }
                int slot = contains;
                int total;
                computeIndices<PREHYDRO1_BLOCK_SIZE>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {
                        position[XDIM], position[YDIM], position[ZDIM]};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            bool const active = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] >=
                parameters.minimumRung;
            if (active && !data.sinkSmoothingConverged[sinkIndex]) {
                Array<Fixed32, NDIM> position;
                position[XDIM] = data.sourcePositionX[particleIndex];
                position[YDIM] = data.sourcePositionY[particleIndex];
                position[ZDIM] = data.sourcePositionZ[particleIndex];
                float& supportRadius = data.sinkThermodynamicStates[sinkIndex].supportRadius;
                bool const converged = solveSphSupportRadius<PREHYDRO1_BLOCK_SIZE>(supportRadius,
                    parameters.minimumSupportRadius, parameters.maximumSupportRadius,
                    data.targetNeighborCount, workspace.positions, position, targetNode.outerBox);
                if (threadIndex == 0) {
                    data.sinkSmoothingConverged[sinkIndex] = converged;
                    if (converged) {
                        atomicMin(&reduction->minimumSupportRadius, supportRadius);
                        atomicMax(&reduction->maximumSupportRadius, supportRadius);
                    } else {
                        atomicAdd(&reduction->pendingChanges, 1);
                    }
                }
            }
        }
        __syncthreads();
        if (threadIndex == 0) {
            targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
        }
        __syncthreads();
    }
    (&workspace)->~SmoothingWorkspace();
}

__global__ void sphCudaPrehydro2(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::DensityWorkspace workspace;
    new (&workspace) detail::DensityWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    while (targetNodeIndex < data.numberOfTargetNodes) {
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        workspace.positions.resize(0);
        workspace.particles.resize(0);
        for (int neighborNodeIndex = targetNode.neighborRange.first;
            neighborNodeIndex < targetNode.neighborRange.second; neighborNodeIndex++) {
            auto const& neighborNode = data.treeNodes[data.neighborNodeIndices[neighborNodeIndex]];
            int const end = neighborNode.particleRange.first +
                roundUp(neighborNode.particleRange.second - neighborNode.particleRange.first,
                    blockSize);
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                Array<Fixed32, NDIM> position;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    position[XDIM] = data.sourcePositionX[sourceParticleIndex];
                    position[YDIM] = data.sourcePositionY[sourceParticleIndex];
                    position[ZDIM] = data.sourcePositionZ[sourceParticleIndex];
                    contains = targetNode.outerBox.contains(position);
                    if (!contains) {
                        contains = sphSupportIntersects(targetNode.innerBox, position,
                            data.sourceSupportRadius[sourceParticleIndex]);
                    }
                }
                int slot = contains;
                int total;
                computeIndices<PREHYDRO2_BLOCK_SIZE>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                workspace.particles.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {
                        position[XDIM], position[YDIM], position[ZDIM]};
                    workspace.particles[offset + slot] = {
                        data.sourceSupportRadius[sourceParticleIndex],
                        data.sourceRungs[sourceParticleIndex]};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            bool const active = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] >=
                parameters.minimumRung;
            Array<Fixed32, NDIM> position;
            position[XDIM] = data.sourcePositionX[particleIndex];
            position[YDIM] = data.sourcePositionY[particleIndex];
            position[ZDIM] = data.sourcePositionZ[particleIndex];
            float& supportRadius = data.sinkThermodynamicStates[sinkIndex].supportRadius;
            int semiactive = active;
            if (!active) {
                for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                    neighborIndex += blockSize) {
                    auto const& neighbor = workspace.positions[neighborIndex];
                    auto const& particle = workspace.particles[neighborIndex];
                    if (particle.rung >= parameters.minimumRung) {
                        float const separationX = distance(position[XDIM], neighbor.positionX);
                        float const separationY = distance(position[YDIM], neighbor.positionY);
                        float const separationZ = distance(position[ZDIM], neighbor.positionZ);
                        if (sqr(separationX, separationY, separationZ) <
                            fmaxf(sqr(supportRadius), sqr(particle.supportRadius))) {
                            semiactive++;
                        }
                    }
                }
                sharedReduceAdd<int, PREHYDRO2_BLOCK_SIZE>(semiactive);
            }
            if (threadIndex == 0) {
                data.sinkSemiactive[sinkIndex] = semiactive != 0;
            }
            if (!semiactive) {
                continue;
            }
            bool converged = true;
            if (!active && !data.sinkSmoothingConverged[sinkIndex]) {
                converged = solveSphSupportRadius<PREHYDRO2_BLOCK_SIZE>(supportRadius,
                    parameters.minimumSupportRadius, parameters.maximumSupportRadius,
                    data.targetNeighborCount, workspace.positions, position, targetNode.outerBox);
            }
            if (!converged) {
                if (threadIndex == 0) {
                    data.sinkSmoothingConverged[sinkIndex] = false;
                    atomicAdd(&reduction->pendingChanges, 1);
                }
                continue;
            }
            float const inverseSupportRadius = 1.0f / supportRadius;
            float const normalization =
                data.gasMass * inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
            float density = 0.0f;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.positions[neighborIndex];
                float const separationX = distance(position[XDIM], neighbor.positionX);
                float const separationY = distance(position[YDIM], neighbor.positionY);
                float const separationZ = distance(position[ZDIM], neighbor.positionZ);
                float const normalizedRadius =
                    sqrtf(sqr(separationX, separationY, separationZ)) * inverseSupportRadius;
                density += normalization * kernelW(normalizedRadius);
            }
            sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(density);
            if (threadIndex == 0) {
                ALWAYS_ASSERT(supportRadius > 0.0f && density > 0.0f);
                data.sinkDensity[sinkIndex] = density;
                data.sinkSmoothingConverged[sinkIndex] = true;
                atomicMin(&reduction->minimumSupportRadius, supportRadius);
                atomicMax(&reduction->maximumSupportRadius, supportRadius);
            }
        }
        __syncthreads();
        if (threadIndex == 0) {
            targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
        }
        __syncthreads();
    }
    (&workspace)->~DensityWorkspace();
}
