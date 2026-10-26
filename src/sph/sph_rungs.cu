/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/sph/sph_cuda.hpp>

namespace detail {

struct RungsWorkspace {
    DeviceVector<Position> positionX;
    DeviceVector<Position> positionY;
    DeviceVector<Position> positionZ;
    DeviceVector<float> supportRadius;
    DeviceVector<char> rungs;
};

}    // namespace detail

__global__ void sphCudaRungs(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::RungsWorkspace workspace;
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    new (&workspace) detail::RungsWorkspace();
    FlopCounter<> flops = 0;
    Array<Position, kDimensions> position;
    int changes = 0;
    while (targetNodeIndex < data.numberOfTargetNodes) {
        __syncthreads();
        workspace.positionX.resize(0);
        workspace.positionY.resize(0);
        workspace.positionZ.resize(0);
        workspace.supportRadius.resize(0);
        workspace.rungs.resize(0);
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        for (int neighborNodeIndex = targetNode.neighborRange.first;
            neighborNodeIndex < targetNode.neighborRange.second; neighborNodeIndex++) {
            SphTreeNode const& neighborNode =
                data.treeNodes[data.neighborNodeIndices[neighborNodeIndex]];
            int const roundedSourceEnd =
                roundUp(neighborNode.particleRange.second - neighborNode.particleRange.first,
                    blockSize) +
                neighborNode.particleRange.first;
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < roundedSourceEnd; sourceParticleIndex += blockSize) {
                bool contains = false;
                int neighborIndex;
                int total;
                int rung;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    rung = data.sourceRungs[sourceParticleIndex];
                    if (rung >= parameters.minimumRung) {
                        position[kXDimension] = data.sourcePositionX[sourceParticleIndex];
                        position[kYDimension] = data.sourcePositionY[sourceParticleIndex];
                        position[kZDimension] = data.sourcePositionZ[sourceParticleIndex];
                        if (targetNode.outerBox.contains(position, &flops)) {
                            contains = true;
                        } else {
                            contains = sphSupportIntersects(targetNode.innerBox, position,
                                data.sourceSupportRadius[sourceParticleIndex], &flops);
                        }
                    }
                }
                neighborIndex = contains;
                computeIndices<kRungsBlockSize>(neighborIndex, total);
                int const offset = workspace.positionX.size();
                __syncthreads();
                int const nextSize = offset + total;
                workspace.positionX.resize(nextSize);
                workspace.positionY.resize(nextSize);
                workspace.positionZ.resize(nextSize);
                workspace.supportRadius.resize(nextSize);
                workspace.rungs.resize(nextSize);
                if (contains) {
                    ASSERT(neighborIndex < total);
                    int const k = offset + neighborIndex;
                    workspace.positionX[k] = position[kXDimension];
                    workspace.positionY[k] = position[kYDimension];
                    workspace.positionZ[k] = position[kZDimension];
                    workspace.supportRadius[k] = data.sourceSupportRadius[sourceParticleIndex];
                    workspace.rungs[k] = rung;
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            if (data.sourceRungs[particleIndex] >= parameters.minimumRung) {
                int const sinkIndex = targetNode.sinkParticleRange.first -
                    targetNode.particleRange.first + particleIndex;
                auto const sinkPositionX = data.sourcePositionX[particleIndex];
                auto const sinkPositionY = data.sourcePositionY[particleIndex];
                auto const sinkPositionZ = data.sourcePositionZ[particleIndex];
                auto const sinkSupportRadius =
                    data.sinkThermodynamicStates[sinkIndex].supportRadius;
                int const sinkRung = data.sourceRungs[particleIndex];
                auto const squaredSinkSupportRadius = sqr(sinkSupportRadius);
                flops += 1;
                int const roundedNeighborEnd = roundUp(workspace.positionX.size(), blockSize);
                int maximumNeighborRung = 0;
                for (int neighborIndex = threadIndex; neighborIndex < roundedNeighborEnd;
                    neighborIndex += blockSize) {
                    if (neighborIndex < workspace.positionX.size()) {
                        auto const neighborPositionX = workspace.positionX[neighborIndex];
                        auto const neighborPositionY = workspace.positionY[neighborIndex];
                        auto const neighborPositionZ = workspace.positionZ[neighborIndex];
                        int const neighborRung = workspace.rungs[neighborIndex];
                        float const separationX =
                            distance(sinkPositionX, neighborPositionX);    // 1
                        float const separationY =
                            distance(sinkPositionY, neighborPositionY);    // 1
                        float const separationZ =
                            distance(sinkPositionZ, neighborPositionZ);    // 1
                        float const squaredSeparation =
                            sqr(separationX, separationY, separationZ);    // 5
                        if (squaredSeparation < fmaxf(squaredSinkSupportRadius,
                                                    sqr(workspace.supportRadius[neighborIndex]))) {
                            maximumNeighborRung = max(maximumNeighborRung, neighborRung);
                        }
                        flops += 9;
                    }
                }
                sharedReduceMax<kRungsBlockSize>(maximumNeighborRung);
                if (threadIndex == 0) {
                    if (sinkRung < maximumNeighborRung - kMaxNeighborRungDifference) {
                        changes++;
                        data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] =
                            maximumNeighborRung - kMaxNeighborRungDifference;
                    }
                }
            }
        }
        if (threadIndex == 0) {
            targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
        }
        __syncthreads();
    }
    if (threadIndex == 0) {
        atomicAdd(&reduction->pendingChanges, changes);
    }
    detail::recordSphGpuFlops(flops, reduction);
    (&workspace)->~RungsWorkspace();
}
