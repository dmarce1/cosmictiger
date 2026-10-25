/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/balsara/shear_limiter.hpp>
#include <cosmictiger/sph/cullen_and_dehnen/viscosity.hpp>
#include <cosmictiger/sph/monaghan/viscosity.hpp>
#include <cosmictiger/sph/morris_and_monaghan/viscosity.hpp>
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
    float accelerationX;
    float accelerationY;
    float accelerationZ;
    float divergence;
    float internalEnergy;
    float pressureEnergyRatio;
    float adiabaticIndex;
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
                Array<Position, NDIM> position;
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
                Array<Position, NDIM> position;
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
                Array<Position, NDIM> position;
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
                        (parameters.balsaraSwitch || parameters.variableViscosity) ?
                            data.sourceVelocityX[sourceParticleIndex] :
                            0.0f,
                        (parameters.balsaraSwitch || parameters.variableViscosity) ?
                            data.sourceVelocityY[sourceParticleIndex] :
                            0.0f,
                        (parameters.balsaraSwitch || parameters.variableViscosity) ?
                            data.sourceVelocityZ[sourceParticleIndex] :
                            0.0f,
                        parameters.cullenDehnen ? data.sourceAccelerationX[sourceParticleIndex] :
                                                  0.0f,
                        parameters.cullenDehnen ? data.sourceAccelerationY[sourceParticleIndex] :
                                                  0.0f,
                        parameters.cullenDehnen ? data.sourceAccelerationZ[sourceParticleIndex] :
                                                  0.0f,
                        parameters.cullenDehnen ?
                            data.sourceVelocityDivergence[sourceParticleIndex] :
                            0.0f,
                        parameters.cullenDehnen ? data.sourceInternalEnergy[sourceParticleIndex] :
                                                  0.0f,
                        parameters.cullenDehnen && data.sourcePressureEnergyRatio ?
                            data.sourcePressureEnergyRatio[sourceParticleIndex] :
                            data.adiabaticIndex - 1.0f,
                        parameters.cullenDehnen && data.sourceAdiabaticIndex ?
                            data.sourceAdiabaticIndex[sourceParticleIndex] :
                            data.adiabaticIndex};
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
            Array<Position, NDIM> position;
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
            float ρ_a_div_v_a = 0.0f;
            Vector3<float> ρ_a_curl_v_a(0.0f);
            cullen_and_dehnen::GradientSums<float> gradientSums{};
            float ρ_a_R_a = 0.0f;
            float v_sig_a = 0.0f;
            float c_a_CD = 0.0f;
            if (parameters.cullenDehnen) {
                float const γ_a = data.sourceAdiabaticIndex ?
                    data.sourceAdiabaticIndex[particleIndex] :
                    data.adiabaticIndex;
                float const P_over_ρu_a = data.sourcePressureEnergyRatio ?
                    data.sourcePressureEnergyRatio[particleIndex] :
                    γ_a - 1.0f;
                c_a_CD = sqrtf(γ_a * P_over_ρu_a * data.sourceInternalEnergy[particleIndex]);
            }
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
                if (parameters.cullenDehnen) {
                    // Cullen & Dehnen (2010), equation (17), including the self term.
                    float const div_v_b = workspace.particles[neighborIndex].divergence;
                    float const sign_div_v_b =
                        div_v_b > 0.0f ? 1.0f : (div_v_b < 0.0f ? -1.0f : 0.0f);
                    ρ_a_R_a += m_b_over_H_a3 * kernelW(q_ab) * sign_div_v_b;
                    flops += 3 + sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab);
                }
                if ((parameters.balsaraSwitch || parameters.variableViscosity) && radius > 0.0f &&
                    q_ab < 1.0f) {
                    // GADGET-4 equation (72): ρ_a div(v)_a and ρ_a curl(v)_a
                    // are −Σ_b m_b v_ab·gradW_ab and Σ_b m_b v_ab×gradW_ab.
                    float const m_b_F_a = m_b_over_H_a3 * one_over_H_a * dkernelWDq(q_ab) / radius;
                    Vector3<float> const m_b_gradW_ab = m_b_F_a * r_ab;
                    auto const& particle = workspace.particles[neighborIndex];
                    Vector3<float> const v_ab_physical =
                        Vector3<float>(data.sourceVelocityX[particleIndex],
                            data.sourceVelocityY[particleIndex],
                            data.sourceVelocityZ[particleIndex]) -
                        Vector3<float>(particle.velocityX, particle.velocityY, particle.velocityZ) +
                        parameters.expansionRate * r_ab;
                    ρ_a_div_v_a -= v_ab_physical.dot(m_b_gradW_ab);
                    for (int dim = 0; dim < NDIM; ++dim) {
                        int const next = (dim + 1) % NDIM;
                        int const last = (dim + 2) % NDIM;
                        ρ_a_curl_v_a[dim] += v_ab_physical[next] * m_b_gradW_ab[last] -
                            v_ab_physical[last] * m_b_gradW_ab[next];
                    }
                    if (parameters.cullenDehnen) {
                        // Cullen & Dehnen (2010), equations (B2), (B4), (B7), (B12).
                        // Equation (B7) permits mass/kernel-gradient weights, independent
                        // of densities being updated by other MPI ranks in this pass.
                        float const w_ab = -m_b_F_a;
                        Vector3<float> const a_ab =
                            Vector3<float>(data.sourceAccelerationX[particleIndex],
                                data.sourceAccelerationY[particleIndex],
                                data.sourceAccelerationZ[particleIndex]) -
                            Vector3<float>(particle.accelerationX, particle.accelerationY,
                                particle.accelerationZ);
                        for (int i = 0; i < NDIM; ++i) {
                            for (int j = 0; j < NDIM; ++j) {
                                gradientSums.moment[i][j] += w_ab * r_ab[i] * r_ab[j];
                                gradientSums.velocity[i][j] += w_ab * v_ab_physical[i] * r_ab[j];
                                gradientSums.acceleration[i][j] += w_ab * a_ab[i] * r_ab[j];
                            }
                        }
                        // Equation (15): maximum approaching-pair signal speed.
                        float const v_ab_dot_r̂_ab = v_ab_physical.dot(r_ab) / radius;
                        float const c_b = sqrtf(particle.adiabaticIndex *
                            particle.pressureEnergyRatio * particle.internalEnergy);
                        v_sig_a =
                            fmaxf(v_sig_a, 0.5f * (c_a_CD + c_b) - fminf(v_ab_dot_r̂_ab, 0.0f));
                        flops += 97;
                    }
                    flops += 33 + sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab, true);
                }
            }
            sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(ρ_a);
            flops += detail::sphReductionFlops<PREHYDRO2_BLOCK_SIZE>();
            if (parameters.balsaraSwitch || parameters.variableViscosity) {
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(ρ_a_div_v_a);
                for (int dim = 0; dim < NDIM; ++dim) {
                    sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(ρ_a_curl_v_a[dim]);
                }
                flops += 4 * detail::sphReductionFlops<PREHYDRO2_BLOCK_SIZE>();
            }
            if (parameters.cullenDehnen) {
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(ρ_a_R_a);
                sharedReduceMax<PREHYDRO2_BLOCK_SIZE>(v_sig_a);
                for (int i = 0; i < NDIM; ++i) {
                    for (int j = 0; j < NDIM; ++j) {
                        sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(gradientSums.moment[i][j]);
                        sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(gradientSums.velocity[i][j]);
                        sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(
                            gradientSums.acceleration[i][j]);
                    }
                }
                flops += 28 * detail::sphReductionFlops<PREHYDRO2_BLOCK_SIZE>();
            }
            // NOLINTEND(readability-identifier-naming)
            if (threadIndex == 0) {
                ALWAYS_ASSERT(H_a > 0.0f && ρ_a > 0.0f);
                data.sinkDensity[sinkIndex] = ρ_a;
                if (data.sinkBalsaraFactor) {
                    data.sinkBalsaraFactor[sinkIndex] = parameters.balsaraSwitch ?
                        balsara::balsaraFactor(ρ_a_div_v_a, ρ_a_curl_v_a.norm()) :
                        1.0f;
                }
                flops += parameters.balsaraSwitch ? 8 : 0;
                if (parameters.variableViscosity && !parameters.cullenDehnen && active) {
                    // Morris & Monaghan (1997), equations (5), (6), (13), in conformal time.
                    // NOLINTBEGIN(readability-identifier-naming)
                    float const div_v_a = ρ_a_div_v_a / (ρ_a * parameters.scaleFactor);
                    float const γ_a = data.sourceAdiabaticIndex ?
                        data.sourceAdiabaticIndex[particleIndex] :
                        data.adiabaticIndex;
                    float const P_over_ρu_a = data.sourcePressureEnergyRatio ?
                        data.sourcePressureEnergyRatio[particleIndex] :
                        γ_a - 1.0f;
                    float const c_a =
                        sqrtf(γ_a * P_over_ρu_a * data.sourceInternalEnergy[particleIndex]);
                    float const h_a = H_a /
                        float(selectedHydroKernel().supportRadiusPerEffectiveRadius) *
                        0.9128709291752769f;
                    float const one_over_τ_a =
                        parameters.viscosityDecay * c_a / (parameters.scaleFactor * h_a);
                    auto& viscosity = data.sinkViscosityStates[sinkIndex];
                    viscosity.rate = morris_and_monaghan::viscosityAlphaRate(
                        viscosity.alpha, parameters.viscosityAlphaMin, one_over_τ_a, div_v_a);
                    // NOLINTEND(readability-identifier-naming)
                    flops += 15;
                }
                if (parameters.cullenDehnen) {
                    auto const diagnostics = cullen_and_dehnen::diagnostics(gradientSums,
                        parameters.scaleFactor, parameters.expansionAcceleration, ρ_a_R_a / ρ_a,
                        H_a, fmaxf(c_a_CD, v_sig_a), parameters.viscosityAlphaMax,
                        parameters.viscosityDecayCD, flops);
                    auto& state = data.sinkCullenDehnenStates[sinkIndex];
                    state.pendingDivergence = diagnostics.divergence;
                    state.targetAlpha = diagnostics.targetAlpha;
                    state.inverseDecayTime = diagnostics.inverseDecayTime;
                }
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
