/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/monaghan_sph.hpp>
#include <cosmictiger/sph/sph_cuda.hpp>

namespace detail {

struct SmoothingWorkspace {
    DeviceVector<SphNeighborPosition> positions;
};

struct NeighborActivity {
    float supportRadius;
    char rung;
    float velocityX;
    float velocityY;
    float velocityZ;
};

struct DensityWorkspace {
    DeviceVector<SphNeighborPosition> positions;
    DeviceVector<NeighborActivity> particles;
};

}    // namespace detail

__global__ void sphCudaPrehydro1(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    FlopCounter<> flops = 0;
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
                    contains = targetNode.outerBox.contains(position, &flops);
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
                    data.targetNeighborCount, workspace.positions, position, targetNode.outerBox,
                    flops);
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
    detail::recordSphGpuFlops(flops, reduction);
    (&workspace)->~SmoothingWorkspace();
}

__global__ void sphCudaPrehydro2(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    FlopCounter<> flops = 0;
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
                    contains = targetNode.outerBox.contains(position, &flops);
                    if (!contains) {
                        contains = sphSupportIntersects(targetNode.innerBox, position,
                            data.sourceSupportRadius[sourceParticleIndex], &flops);
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
                        data.sourceRungs[sourceParticleIndex],
                        parameters.balsaraSwitch ? data.sourceVelocityX[sourceParticleIndex] : 0.0f,
                        parameters.balsaraSwitch ? data.sourceVelocityY[sourceParticleIndex] : 0.0f,
                        parameters.balsaraSwitch ? data.sourceVelocityZ[sourceParticleIndex] :
                                                   0.0f};
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
                        flops += 10;    // Three fixed distances, squared norm, two supports.
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
                    data.targetNeighborCount, workspace.positions, position, targetNode.outerBox,
                    flops);
            }
            if (!converged) {
                if (threadIndex == 0) {
                    data.sinkSmoothingConverged[sinkIndex] = false;
                    atomicAdd(&reduction->pendingChanges, 1);
                }
                continue;
            }
            // Equation 3.8 of Monaghan (1992) (the interpolant 2.7 at r_a): ρ_a = Σ_b m_b W_ab,
            // with the sink's own support radius in W(r, H_a) = w(r/H_a)/H_a³, a gather form of
            // equation 6.3 (which uses h_b). Every gas particle has mass m_b = gasMass, and the
            // sum includes b = a. Each thread accumulates part of the sum.
            // NOLINTBEGIN(readability-identifier-naming)
            float const H_a = supportRadius;
            float const one_over_H_a = 1.0f / H_a;
            float const m_b_over_H_a3 = data.gasMass * one_over_H_a * one_over_H_a * one_over_H_a;
            flops += 4;    // Reciprocal support and the mass/support-cubed scale.
            float ρ_a = 0.0f;
            float divergence = 0.0f;
            Vector3<float> curl(0.0f);
            for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.positions[neighborIndex];
                Vector3<float> const r_ab(distance(position[XDIM], neighbor.positionX),
                    distance(position[YDIM], neighbor.positionY),
                    distance(position[ZDIM], neighbor.positionZ));
                float const radius = sqrtf(r_ab.dot(r_ab));
                float const q_ab = radius * one_over_H_a;    // r_ab/H_a
                ρ_a += m_b_over_H_a3 * kernelW(q_ab);
                flops += 12 + sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab);
                if (parameters.balsaraSwitch && radius > 0.0f && q_ab < 1.0f) {
                    // Density-normalized SPH divergence and curl share rho_a, which
                    // cancels from f_a. Use the same physical velocity as the viscosity.
                    float const gradientScale =
                        m_b_over_H_a3 * one_over_H_a * dkernelWDq(q_ab) / radius;
                    Vector3<float> const gradient = gradientScale * r_ab;
                    auto const& particle = workspace.particles[neighborIndex];
                    Vector3<float> const velocity =
                        Vector3<float>(data.sourceVelocityX[particleIndex],
                            data.sourceVelocityY[particleIndex],
                            data.sourceVelocityZ[particleIndex]) -
                        Vector3<float>(particle.velocityX, particle.velocityY, particle.velocityZ) +
                        parameters.expansionRate * r_ab;
                    divergence -= velocity.dot(gradient);
                    for (int dim = 0; dim < NDIM; ++dim) {
                        int const next = (dim + 1) % NDIM;
                        int const last = (dim + 2) % NDIM;
                        curl[dim] +=
                            velocity[next] * gradient[last] - velocity[last] * gradient[next];
                    }
                    flops += 33 + sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab, true);
                }
            }
            sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(ρ_a);
            flops += detail::sphReductionFlops<PREHYDRO2_BLOCK_SIZE>();
            if (parameters.balsaraSwitch) {
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(divergence);
                for (int dim = 0; dim < NDIM; ++dim) {
                    sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(curl[dim]);
                }
                flops += 4 * detail::sphReductionFlops<PREHYDRO2_BLOCK_SIZE>();
            }
            // NOLINTEND(readability-identifier-naming)
            if (threadIndex == 0) {
                ALWAYS_ASSERT(H_a > 0.0f && ρ_a > 0.0f);
                data.sinkDensity[sinkIndex] = ρ_a;
                data.sinkBalsaraFactor[sinkIndex] = parameters.balsaraSwitch ?
                    monaghan_sph::balsaraFactor(divergence, curl.norm()) :
                    1.0f;
                flops += parameters.balsaraSwitch ? 8 : 0;
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
    detail::recordSphGpuFlops(flops, reduction);
    (&workspace)->~DensityWorkspace();
}
