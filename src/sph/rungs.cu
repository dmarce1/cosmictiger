/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/cuda.hpp>
#include <cosmictiger/sph/geometry.hpp>

using sph_geometry::SpatialVector;

namespace detail {

struct RungsWorkspace {
    // Fit 1024 fixed32 neighbors with little short-list cost in the RTX 3090 sweep.
    static constexpr int kSharedBytes = 20 * 1024;
    static constexpr int kSharedCapacity =
        kSharedBytes / (sizeof(SpatialVector<Position>) + sizeof(float) + sizeof(char));
    SharedNeighborBuffer<SpatialVector<Position>, kSharedCapacity> x;
    SharedNeighborBuffer<float, kSharedCapacity> smoothingLength;
    SharedNeighborBuffer<char, kSharedCapacity> rungs;
};

}    // namespace detail

// Locally derived monotone neighbor-graph closure of sphCpuRungs, including inactive endpoints:
//   L_a⁽ᵏ⁺¹⁾ = max(L_a⁽ᵏ⁾, max_{b: |r_ab| ≤ max(H_a, H_b)} L_b⁽ᵏ⁾ − 1).
// Symbol directory: L_a → sink rung; max L_b → maximum neighbor rung;
// H_a, H_b → sink, neighbor kernel smoothing length; x_a, x_b → sink, neighbor position;
// paddedX_b → neighbor position as the record of the box tests; r_ab → separation;
// r_ab² → squared separation. Code names: r2_ab = r_ab²; H2_a = H_a².
__global__ void sphCudaRungs(SphRunParameters, SphRunCudaData data, SphReduction* reduction) {
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::RungsWorkspace workspace;
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    new (&workspace) detail::RungsWorkspace();
    sph_flops::Counter flops = 0;
    int changes = 0;
    while (targetNodeIndex < data.numberOfTargetNodes) {
        __syncthreads();
        workspace.x.resize(0);
        workspace.smoothingLength.resize(0);
        workspace.rungs.resize(0);
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        for (int neighborNodeIndex = targetNode.neighborRange.first;
            neighborNodeIndex < targetNode.neighborRange.second; neighborNodeIndex++) {
            SphTreeNode const& neighborNode =
                data.treeNodes[data.neighborNodeIndices[neighborNodeIndex]];
            int const roundedSourceEnd =
                roundUp(neighborNode.particleRange.second - neighborNode.particleRange.first,
                    ParticleInteger(blockSize)) +
                neighborNode.particleRange.first;
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < roundedSourceEnd; sourceParticleIndex += blockSize) {
                bool contains = false;
                int neighborIndex;
                int total;
                int rung;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    rung = data.sourceRungs[sourceParticleIndex];
                    Vector3<Position> const paddedX_b =
                        data.sourcePaddedPosition(sourceParticleIndex);
                    if (targetNode.outerBox.contains(paddedX_b, &flops.doublePrecision)) {
                        contains = true;
                    } else {
                        contains = sphSupportIntersects(targetNode.innerBox, paddedX_b,
                            data.sourceSmoothingLength[sourceParticleIndex], &flops);
                    }
                }
                neighborIndex = contains;
                computeIndices<kRungsBlockSize>(neighborIndex, total);
                int const offset = workspace.x.size();
                __syncthreads();
                int const nextSize = offset + total;
                workspace.x.resize(nextSize);
                workspace.smoothingLength.resize(nextSize);
                workspace.rungs.resize(nextSize);
                if (contains) {
                    ASSERT(neighborIndex < total);
                    int const k = offset + neighborIndex;
                    workspace.x[k] = data.sourcePosition(sourceParticleIndex);
                    workspace.smoothingLength[k] = data.sourceSmoothingLength[sourceParticleIndex];
                    workspace.rungs[k] = rung;
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            auto const x_a = data.sourcePosition(particleIndex);
            auto const H_a = data.sinkSmoothingLength[sinkIndex];
            int const sinkRung = data.sourceRungs[particleIndex];
            auto const H2_a = sqr(H_a);
            flops += 1;
            int const roundedNeighborEnd = roundUp(workspace.x.size(), blockSize);
            int maximumNeighborRung = 0;
            for (int neighborIndex = threadIndex; neighborIndex < roundedNeighborEnd;
                neighborIndex += blockSize) {
                if (neighborIndex < workspace.x.size()) {
                    auto const x_b = workspace.x[neighborIndex];
                    int const neighborRung = workspace.rungs[neighborIndex];
                    auto const r_ab = distance(x_a, x_b);
                    float const r2_ab = r_ab.dot(r_ab);
                    float const H_b = workspace.smoothingLength[neighborIndex];
                    if (r2_ab <= fmaxf(H2_a, sqr(H_b))) {
                        maximumNeighborRung = max(maximumNeighborRung, neighborRung);
                    }
                    // Fixed distances, dot, H_b².
                    flops += sph_flops::kVectorOperation + sph_flops::kDotProduct + 1;
                }
            }
            sharedReduceMax<kRungsBlockSize>(maximumNeighborRung);
            if (threadIndex == 0) {
                // Same frozen-source graph closure as sphCpuRungs: every touching
                // pair constrains both endpoints, including inactive target rungs.
                if (sinkRung < maximumNeighborRung - kMaxNeighborRungDifference) {
                    changes++;
                    auto* const candidates =
                        data.sinkRungCandidates ? data.sinkRungCandidates : data.sinkParticleRungs;
                    candidates[data.sinkParticleIndices[sinkIndex]] =
                        maximumNeighborRung - kMaxNeighborRungDifference;
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
