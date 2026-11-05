/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/cuda.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>
#include <cosmictiger/sph/viscosity/hu_etal.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/viscosity/morris_and_monaghan.hpp>
#include <cosmictiger/sph/viscosity/price_etal.hpp>

namespace detail {

struct SmoothingWorkspace {
    SharedNeighborBuffer<SphNeighborPosition, kSphNeighborSharedBytes / sizeof(SphNeighborPosition)>
        positions;
};

struct NeighborActivity {
    float smoothingLength;
    char rung;
    float velocityX;
    float velocityY;
    float velocityZ;
    float divergence;
    float internalEnergy;
    float accelerationX;
    float accelerationY;
    float accelerationZ;
};

struct DensityWorkspace {
    // The RTX 3090 budget sweep favored 12 KiB over 8 KiB for density.
    static constexpr int kSharedBytes = 12 * 1024;
    static constexpr int kSharedCapacity =
        kSharedBytes / (sizeof(SphNeighborPosition) + sizeof(NeighborActivity));
    SharedNeighborBuffer<SphNeighborPosition, kSharedCapacity> positions;
    SharedNeighborBuffer<NeighborActivity, kSharedCapacity> particles;
};

}    // namespace detail

__global__ void sphCudaSmoothingLengths(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    sph_flops::Counter flops = 0;
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
                    ParticleInteger(blockSize));
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                Vector3<Position> position;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    position[kXDimension] = data.sourcePositionX[sourceParticleIndex];
                    position[kYDimension] = data.sourcePositionY[sourceParticleIndex];
                    position[kZDimension] = data.sourcePositionZ[sourceParticleIndex];
                    contains = targetNode.outerBox.contains(position, &flops.doublePrecision);
                }
                int slot = contains;
                int total;
                computeIndices<kSmoothingLengthsBlockSize>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {
                        position[kXDimension], position[kYDimension], position[kZDimension]};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            bool const active = parameters.isActiveRung(
                data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
            if (active && !data.sinkFlags[sinkIndex].converged()) {
                Vector3<Position> position;
                position[kXDimension] = data.sourcePositionX[particleIndex];
                position[kYDimension] = data.sourcePositionY[particleIndex];
                position[kZDimension] = data.sourcePositionZ[particleIndex];
                float& smoothingLength = data.sinkSmoothingLength[sinkIndex];
                bool const converged = solveSphSmoothingLength<kSmoothingLengthsBlockSize>(
                    smoothingLength, parameters.minimumSmoothingLength,
                    parameters.maximumSmoothingLength, data.targetNeighborCount,
                    workspace.positions, position, targetNode.outerBox, flops);
                if (threadIndex == 0) {
                    data.sinkFlags[sinkIndex].setConverged(converged);
                    if (converged) {
                        atomicMin(&reduction->minimumSmoothingLength, smoothingLength);
                        atomicMax(&reduction->maximumSmoothingLength, smoothingLength);
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

// clang-format off
/*
 * Density and gradient sums: Monaghan (1992), ARA&A 30, 543, equation (3.8),
 * doi:10.1146/annurev.aa.30.090192.002551; Cullen & Dehnen (2010), equations
 * (15), (17), (B2), (B4), (B7), https://arxiv.org/abs/1006.1524.
 *   ρ_a = Σ_b m_b W_ab(H_a), W_ab = H_a⁻³ w(q_ab), q_ab = |r_ab|/H_a,
 *   ρ_a ∇·v_a = −Σ_b m_b v_ab · ∇_a W_ab,
 *   ρ_a ∇×v_a = Σ_b m_b v_ab × ∇_a W_ab,
 *   R_a = ρ_a⁻¹ Σ_b m_b W_ab sign(∇·v_b),
 *   T = Σ_b r_ab ⊗ r_ab w̄_ab, D = Σ_b v_ab ⊗ r_ab w̄_ab, V = D T⁻¹.
 * Direct acceleration reconstruction (Cullen & Dehnen, equation B12):
 *   C = Σ_b F_ab ⊗ r_ab w̄_ab, ∇_x F = C T⁻¹, F_ab = F_a − F_b.
 * Symbol directory (Pack denotes SIMD partial sums):
 *   H_a → sinkKernelSmoothingLength; q_ab → normalizedRadius; w(q_ab) → kernelWeight
 *   ρ_a → density; ρ_a ∇·v_a → densityWeightedDivergence
 *   ρ_a ∇×v_a → densityWeightedCurl; ρ_a R_a → densityWeightedSignAverage
 *   r_ab, v_ab → separation, velocityDifference (peculiar stored velocity)
 *   m_b ∇_a W_ab → massWeightedKernelGradient
 *   T, D, C → gradientSums.moment, gradientSums.velocity, gradientSums.acceleration
 *   F_ab → accelerationDifference; F → sourceAccelerationX/Y/Z (full canonical force)
 *   Σ_b w → kernelSum; Σ_b q_ab w′ → kernelDerivativeMoment
 *   v_sig,a → signalSpeed (CUDA), signalSpeedPack (CPU)
 *   w̄_ab = −m_b F_a → gradientWeight inside the corrected-gradient branch.
 * Peculiar velocities enter all switch gradients and densityDerivativeSum.
 * Physical pair velocities include Hubble flow only in the signal-speed estimate; division by a ρ_a Ω_a gives d ln ρ_a/dτ
 * (Price 2012, equation (27), https://arxiv.org/abs/1012.1885).
 */
// clang-format on
__global__ void sphCudaDensityAndGradients(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    bool const computeGradientDiagnostics =
        parameters.gradientSwitch() && data.sinkViscosityDiagnostics != nullptr;
    bool const computeAccelerationGradient =
        computeGradientDiagnostics && parameters.accelerationGradientViscosity;
    ALWAYS_ASSERT(!computeAccelerationGradient ||
        (data.sourceAccelerationX && data.sourceAccelerationY && data.sourceAccelerationZ));
    ALWAYS_ASSERT(!computeGradientDiagnostics || !parameters.cullenDehnen ||
        computeAccelerationGradient || data.sourceVelocityDivergence);
    bool const computeSphBalsara = parameters.balsaraSwitch && !parameters.gradientSwitch();
    bool const evolveMorrisMonaghan = parameters.variableViscosity &&
        !parameters.gradientSwitch() && parameters.viscosityTimestep > 0.0;
    bool const loadVelocity = computeSphBalsara || evolveMorrisMonaghan ||
        computeGradientDiagnostics || data.sinkLogDensityRate;
    sph_flops::Counter flops = 1;    // γ − 1.
    float const adiabaticIndex = data.adiabaticIndex;
    float const pressureEnergyRatio = adiabaticIndex - 1.0f;    // P/(ρu) for every particle.
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
                    ParticleInteger(blockSize));
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                Vector3<Position> position;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    position[kXDimension] = data.sourcePositionX[sourceParticleIndex];
                    position[kYDimension] = data.sourcePositionY[sourceParticleIndex];
                    position[kZDimension] = data.sourcePositionZ[sourceParticleIndex];
                    contains = targetNode.outerBox.contains(position, &flops.doublePrecision);
                    if (!contains) {
                        contains = sphSupportIntersects(targetNode.innerBox, position,
                            data.sourceSmoothingLength[sourceParticleIndex], &flops);
                    }
                }
                int slot = contains;
                int total;
                computeIndices<kDensityAndGradientsBlockSize>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                workspace.particles.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {
                        position[kXDimension], position[kYDimension], position[kZDimension]};
                    workspace.particles[offset + slot] = {
                        data.sourceSmoothingLength[sourceParticleIndex],
                        data.sourceRungs[sourceParticleIndex],
                        loadVelocity ? data.sourceVelocityX[sourceParticleIndex] : 0.0f,
                        loadVelocity ? data.sourceVelocityY[sourceParticleIndex] : 0.0f,
                        loadVelocity ? data.sourceVelocityZ[sourceParticleIndex] : 0.0f,
                        computeGradientDiagnostics && parameters.cullenDehnen &&
                                data.sourceVelocityDivergence ?
                            data.sourceVelocityDivergence[sourceParticleIndex] :
                            0.0f,
                        computeGradientDiagnostics &&
                                (parameters.cullenDehnen || parameters.huEtAl) ?
                            data.sourceInternalEnergy[sourceParticleIndex] :
                            0.0f,
                        computeAccelerationGradient ?
                            data.sourceAccelerationX[sourceParticleIndex] :
                            0.0f,
                        computeAccelerationGradient ?
                            data.sourceAccelerationY[sourceParticleIndex] :
                            0.0f,
                        computeAccelerationGradient ?
                            data.sourceAccelerationZ[sourceParticleIndex] :
                            0.0f};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            bool const active = parameters.isActiveRung(
                data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
            Vector3<Position> position;
            position[kXDimension] = data.sourcePositionX[particleIndex];
            position[kYDimension] = data.sourcePositionY[particleIndex];
            position[kZDimension] = data.sourcePositionZ[particleIndex];
            float& smoothingLength = data.sinkSmoothingLength[sinkIndex];
            int semiactive = active;
            if (!active) {
                for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                    neighborIndex += blockSize) {
                    auto const& neighbor = workspace.positions[neighborIndex];
                    auto const& particle = workspace.particles[neighborIndex];
                    if (parameters.isActiveRung(particle.rung)) {
                        float const separationX =
                            distance(position[kXDimension], neighbor.positionX);
                        float const separationY =
                            distance(position[kYDimension], neighbor.positionY);
                        float const separationZ =
                            distance(position[kZDimension], neighbor.positionZ);
                        flops += 10;    // Three fixed distances, squared norm, two supports.
                        if (sqr(separationX, separationY, separationZ) <
                            fmaxf(sqr(smoothingLength), sqr(particle.smoothingLength))) {
                            semiactive++;
                        }
                    }
                }
                sharedReduceAdd<int, kDensityAndGradientsBlockSize>(semiactive);
            }
            if (threadIndex == 0) {
                data.sinkFlags[sinkIndex].setSemiactive(semiactive != 0);
            }
            // Semiactive and convergence share a byte; finish the masked write
            // before any thread reads convergence from that byte.
            __syncthreads();
            if (!semiactive) {
                continue;
            }
            bool const updateMorrisMonaghan =
                evolveMorrisMonaghan && active && !data.sinkFlags[sinkIndex].converged();
            bool const computeSphDivergence = computeSphBalsara || updateMorrisMonaghan;
            bool converged = true;
            if (!active && !data.sinkFlags[sinkIndex].converged()) {
                converged = solveSphSmoothingLength<kDensityAndGradientsBlockSize>(smoothingLength,
                    parameters.minimumSmoothingLength, parameters.maximumSmoothingLength,
                    data.targetNeighborCount, workspace.positions, position, targetNode.outerBox,
                    flops);
            }
            if (!converged) {
                if (threadIndex == 0) {
                    data.sinkFlags[sinkIndex].setConverged(false);
                    atomicAdd(&reduction->pendingChanges, 1);
                }
                continue;
            }
            // clang-format off
            /*
             * Equation 3.8 of Monaghan (1992) (the interpolant 2.7 at r_a): ρ_a = Σ_b m_b W_ab,
             * with the sink's own support radius in W(r, H_a) = w(r/H_a)/H_a³, a gather form of
             * equation 6.3 (which uses h_b). Every gas particle has mass m_b = gasMass, and the
             * sum includes b = a. Each thread accumulates part of the sum.
             *
             * Symbol directory (in the code below):
             *   H_a → sinkKernelSmoothingLength
             *   m_b → data.gasMass
             *   ρ_a → density
             */
            // clang-format on

            float const sinkKernelSmoothingLength = smoothingLength;
            float const inverseSmoothingLength = 1.0f / sinkKernelSmoothingLength;
            float const massOverSupportVolume =
                data.gasMass * sph_geometry::inverseVolume(inverseSmoothingLength);
            flops += 4;    // Reciprocal support and the mass/support-cubed scale.
            float density = 0.0f;
            float densityDerivativeSum = 0.0f;
            float densityWeightedDivergence = 0.0f;
            Vector3<float> densityWeightedCurl(0.0f);
            cullen_and_dehnen::GradientSums<float> gradientSums{};
            float densityWeightedSignAverage = 0.0f;
            float signalSpeed = 0.0f;
            // clang-format off
            /*
             * Variational operator: ΣW = Σ_b w(q_ab) and Σ_b q_ab w'(q_ab) give Ω_a and ζ_a.
             *
             * Symbol directory (in the code below):
             *   q_ab → normalizedRadius
             *   ζ_a → gradH.correction.weightedSupportCorrection
             *   ΣW → kernelSum
             */
            // clang-format on
            float kernelSum = 0.0f;
            float kernelDerivativeMoment = 0.0f;
            float soundSpeedForSwitch =
                0.0f;    // Sound speed for the CD signal speed or Price target.
            if (computeGradientDiagnostics) {
                soundSpeedForSwitch = sqrtf(adiabaticIndex * pressureEnergyRatio *
                    data.sourceInternalEnergy[particleIndex]);
            }
            for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.positions[neighborIndex];
                Vector3<float> const separation(distance(position[kXDimension], neighbor.positionX),
                    distance(position[kYDimension], neighbor.positionY),
                    distance(position[kZDimension], neighbor.positionZ));
                float const radius = sqrtf(sph_geometry::spatialDot(separation, separation));
                float const normalizedRadius = radius * inverseSmoothingLength;    // r_ab/H_a
                float const kernelWeight = kernelW(normalizedRadius);
                density += massOverSupportVolume * kernelWeight;
                flops += 12 +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), normalizedRadius);
                if (normalizedRadius < 1.0f) {
                    kernelSum += kernelWeight;
                    kernelDerivativeMoment += normalizedRadius * dkernelWDq(normalizedRadius);
                    flops += 3 +
                        sph_flops::densityKernelOperations(
                            selectedHydroKernel(), normalizedRadius, true);
                }
                if (computeGradientDiagnostics && parameters.cullenDehnen) {
                    /*
                     * Cullen & Dehnen (2010), equation (17), including the self term.
                     */
                    float const neighborVelocityDivergence =
                        workspace.particles[neighborIndex].divergence;
                    float const neighborDivergenceSign = neighborVelocityDivergence > 0.0f ?
                        1.0f :
                        (neighborVelocityDivergence < 0.0f ? -1.0f : 0.0f);
                    densityWeightedSignAverage +=
                        massOverSupportVolume * kernelWeight * neighborDivergenceSign;
                    flops += 3;
                }
                if ((computeSphDivergence || computeGradientDiagnostics ||
                        data.sinkLogDensityRate) &&
                    radius > 0.0f && normalizedRadius < 1.0f) {
                    // clang-format off
                    /*
                     * For constant viscosity and Morris & Monaghan, GADGET-4 equation
                     * (72) uses −Σ_b m_b v_ab·gradW_ab and Σ_b m_b v_ab×gradW_ab.
                     * Gradient switches instead reuse V = D T⁻¹ below. The independent
                     * density derivative keeps its variational SPH sum.
                     *
                     * Symbol directory (in the code below):
                     *   v_ab → velocityDifference; m_b → data.gasMass; ρ_a → density.
                     */
                    // clang-format on
                    float const massWeightedGradientFactor = massOverSupportVolume *
                        inverseSmoothingLength * dkernelWDq(normalizedRadius) / radius;
                    auto const& particle = workspace.particles[neighborIndex];
                    Vector3<float> const velocityDifference =
                        Vector3<float>(data.sourceVelocityX[particleIndex],
                            data.sourceVelocityY[particleIndex],
                            data.sourceVelocityZ[particleIndex]) -
                        Vector3<float>(particle.velocityX, particle.velocityY, particle.velocityZ);
                    // Kernel-gradient coefficient (3) and relative velocity (3).
                    flops += 6 +
                        sph_flops::densityKernelOperations(
                            selectedHydroKernel(), normalizedRadius, true);
                    if (computeSphDivergence || data.sinkLogDensityRate) {
                        Vector3<float> const massWeightedKernelGradient =
                            massWeightedGradientFactor * separation;
                        flops += 3;
                        if (data.sinkLogDensityRate) {
                            densityDerivativeSum += sph_geometry::spatialDot(
                                velocityDifference, massWeightedKernelGradient);
                            flops += 6;
                        }
                        if (computeSphDivergence) {
                            densityWeightedDivergence -= sph_geometry::spatialDot(
                                velocityDifference, massWeightedKernelGradient);
                            flops += 6;
                        }
                        if (computeSphBalsara) {
                            for (int dim = 0; dim < kDimensions; ++dim) {
                                int const next = (dim + 1) % kDimensions;
                                int const last = (dim + 2) % kDimensions;
                                densityWeightedCurl[dim] +=
                                    velocityDifference[next] * massWeightedKernelGradient[last] -
                                    velocityDifference[last] * massWeightedKernelGradient[next];
                            }
                            flops += 12;
                        }
                    }
                    if (computeGradientDiagnostics) {
                        /*
                         * Cullen & Dehnen (2010), equations (B2), (B4), (B7), and
                         * Price et al. (2018), equations (51)-(52): V = D T⁻¹.
                         * Equation (B7) allows the common mass/kernel-gradient
                         * weights, independent of densities changing on other ranks.
                         */
                        float const gradientWeight = -massWeightedGradientFactor;
                        for (int i = 0; i < kSpatialDimensions; ++i) {
                            for (int j = 0; j < kSpatialDimensions; ++j) {
                                gradientSums.moment[i][j] +=
                                    gradientWeight * separation[i] * separation[j];
                                gradientSums.velocity[i][j] +=
                                    gradientWeight * velocityDifference[i] * separation[j];
                            }
                        }
                        flops += 6 * kSpatialDimensions * kSpatialDimensions;
                        if (computeAccelerationGradient) {
                            Vector3<float> const accelerationDifference =
                                Vector3<float>(data.sourceAccelerationX[particleIndex],
                                    data.sourceAccelerationY[particleIndex],
                                    data.sourceAccelerationZ[particleIndex]) -
                                Vector3<float>(particle.accelerationX, particle.accelerationY,
                                    particle.accelerationZ);
                            for (int row = 0; row < kSpatialDimensions; ++row) {
                                for (int column = 0; column < kSpatialDimensions; ++column) {
                                    gradientSums.acceleration[row][column] += gradientWeight *
                                        accelerationDifference[row] * separation[column];
                                }
                            }
                            flops += 3 + 3 * kSpatialDimensions * kSpatialDimensions;
                        }
                        if (parameters.cullenDehnen || parameters.huEtAl) {
                            // Hubble flow enters the approaching-pair signal speed only.
                            Vector3<float> const physicalVelocityDifference =
                                velocityDifference + parameters.expansionRate * separation;
                            float const radialVelocity =
                                sph_geometry::spatialDot(physicalVelocityDifference, separation) /
                                radius;
                            float const neighborSoundSpeed = sqrtf(
                                adiabaticIndex * pressureEnergyRatio * particle.internalEnergy);
                            signalSpeed = fmaxf(signalSpeed,
                                0.5f * (soundSpeedForSwitch + neighborSoundSpeed) -
                                    fminf(radialVelocity, 0.0f));
                            flops += 18;
                        }
                    }
                }
            }
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(density);
            flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(kernelSum);
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(kernelDerivativeMoment);
            flops += 2 * detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            if (data.sinkLogDensityRate) {
                sharedReduceAdd<float, kDensityAndGradientsBlockSize>(densityDerivativeSum);
                flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }
            if (computeSphDivergence) {
                sharedReduceAdd<float, kDensityAndGradientsBlockSize>(densityWeightedDivergence);
                flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }
            if (computeSphBalsara) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    sharedReduceAdd<float, kDensityAndGradientsBlockSize>(densityWeightedCurl[dim]);
                }
                flops += 3 * detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }
            if (computeGradientDiagnostics && parameters.cullenDehnen) {
                sharedReduceAdd<float, kDensityAndGradientsBlockSize>(densityWeightedSignAverage);
                flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }
            if (computeGradientDiagnostics && (parameters.cullenDehnen || parameters.huEtAl)) {
                sharedReduceMax<kDensityAndGradientsBlockSize>(signalSpeed);
            }
            if (computeGradientDiagnostics) {
                for (int i = 0; i < kSpatialDimensions; ++i) {
                    for (int j = 0; j < kSpatialDimensions; ++j) {
                        sharedReduceAdd<float, kDensityAndGradientsBlockSize>(
                            gradientSums.moment[i][j]);
                        sharedReduceAdd<float, kDensityAndGradientsBlockSize>(
                            gradientSums.velocity[i][j]);
                        if (computeAccelerationGradient) {
                            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(
                                gradientSums.acceleration[i][j]);
                        }
                    }
                }
                flops += (computeAccelerationGradient ? 27 : 18) *
                    detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }

            if (threadIndex == 0) {
                ALWAYS_ASSERT(sinkKernelSmoothingLength > 0.0f && density > 0.0f);
                data.sinkDensity[sinkIndex] = density;

                auto const gradH = sphGradHEvaluation(sinkKernelSmoothingLength,
                    parameters.minimumSmoothingLength, parameters.maximumSmoothingLength,
                    data.targetNeighborCount, data.gasMass, kernelSum, kernelDerivativeMoment);
                flops += 8;    // Residual (2), and Ω, ζ, the weighted sums (6).
                if (gradH.constraint != sph_volume::SupportConstraint::Free) {
                    atomicAdd(&reduction->heldSupports, 1);
                } else {
                    // clang-format off
                    /*
                     * A held support uses ζ_a = 0. An unusable free correction stops the
                     * run; a small positive Ω_a is counted.
                     *
                     * Symbol directory (in the code below):
                     *   ζ_a → gradH.correction.weightedSupportCorrection
                     */
                    // clang-format on
                    if (!sph_volume::isUsable(gradH.correction)) {
                        printf("SPH grad-H correction is unusable: smoothingLength=%e "
                               "density=%e ΣW=%e Σq dw/dq=%e Ω=%e ζ=%e residual=%e\n",
                            sinkKernelSmoothingLength, density, kernelSum, kernelDerivativeMoment,
                            gradH.correction.supportJacobian,
                            gradH.correction.weightedSupportCorrection, gradH.relativeResidual);
                        __trap();
                    }
                    atomicMin(&reduction->minimumGradHOmega, gradH.correction.supportJacobian);
                    atomicMax(&reduction->maximumSupportResidual, fabsf(gradH.relativeResidual));
                    if (!sph_volume::isWellConditioned(gradH.correction)) {
                        atomicAdd(&reduction->illConditionedSupports, 1);
                    }
                }
                data.sinkGradHCorrection[sinkIndex] = gradH.correction.weightedSupportCorrection;
                if (data.sinkLogDensityRate) {
                    // clang-format off
                    /*
                     * Price (2012), equation (27), with peculiar motion only:
                     * d log(rho_comoving)/d tau = sum m v_ab.grad W/(a rho Omega).
                     *
                     * Symbol directory (in the code below):
                     *   v_ab → velocityDifference
                     */
                    // clang-format on
                    data.sinkLogDensityRate[sinkIndex] = densityDerivativeSum /
                        (parameters.scaleFactor * density * gradH.correction.supportJacobian);
                    flops += 3;
                }

                if (computeSphBalsara && data.sinkBalsaraFactor) {
                    data.sinkBalsaraFactor[sinkIndex] =
                        balsara::encodeFactor(balsara::balsaraFactor(
                            densityWeightedDivergence, densityWeightedCurl.norm()));
                    flops += 8;
                } else if (!parameters.balsaraSwitch && data.sinkBalsaraFactor) {
                    data.sinkBalsaraFactor[sinkIndex] = 255;
                }
                if (updateMorrisMonaghan) {
                    /*
                     * Morris & Monaghan (1997), equations (5), (6), (13), in conformal time.
                     */

                    float const velocityDivergence =
                        densityWeightedDivergence / (density * parameters.scaleFactor);
                    float const soundSpeed = sqrtf(adiabaticIndex * pressureEnergyRatio *
                        data.sourceInternalEnergy[particleIndex]);
                    float const smoothingLength = sinkKernelSmoothingLength /
                        selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                        float(monaghan::kViscosityWidthScale);
                    float const inverseDecayTime = parameters.viscosityDecay * soundSpeed /
                        (parameters.scaleFactor * smoothingLength);
                    int const rung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
                    bool const actualInterval = parameters.globalTimestep;
                    float const timeStep = actualInterval ?
                        parameters.viscosityTimestep :
                        parameters.viscosityTimestep / (1 << rung);
                    auto& bulkViscosityCoefficient = data.sinkBulkViscosityCoefficient[sinkIndex];
                    bulkViscosityCoefficient = morris_and_monaghan::evolvedBulkViscosityCoefficient(
                        float(bulkViscosityCoefficient), parameters.minimumBulkViscosityCoefficient,
                        inverseDecayTime, velocityDivergence, timeStep);

                    flops += 19;
                    flops.doublePrecision += actualInterval ? 1 : 2;
                }
                if (computeGradientDiagnostics) {
                    // Both sensor routes leave endpoint history and alpha unchanged.
                    float velocityGradient[kDimensions][kDimensions];
                    float divergenceDerivative = 0.0f;
                    int rank;
                    if (computeAccelerationGradient) {
                        float accelerationGradient[kDimensions][kDimensions];
                        rank = cullen_and_dehnen::reconstructGradients(
                            gradientSums, velocityGradient, accelerationGradient, flops);
                        if (rank != 0) {
                            divergenceDerivative =
                                cullen_and_dehnen::peculiarDivergenceDerivative(velocityGradient,
                                    accelerationGradient, parameters.scaleFactor, flops);
                        }
                    } else {
                        rank = cullen_and_dehnen::reconstructVelocityGradient(
                            gradientSums, velocityGradient, flops);
                    }
                    float const canonicalDivergence =
                        velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2];
                    flops += 2;
                    if (!computeAccelerationGradient) {
                        ALWAYS_ASSERT(data.sinkScaledDivergence);
                        auto const& state = data.sinkScaledDivergence[sinkIndex];
                        int const currentRung =
                            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
                        double const sampleInterval = sphViscositySampleInterval(currentRung,
                            parameters.viscositySampleInterval, parameters.globalTimestep,
                            parameters.viscosityInitializeHistory);
                        double const previousSampleTimestep = sphViscosityStoredSampleTimestep(
                            currentRung, parameters.viscosityBaseTimestep,
                            parameters.viscosityPreviousTimestep, parameters.globalTimestep);
                        if (!parameters.globalTimestep && sampleInterval > 0.0) {
                            flops.doublePrecision += 2;
                        }
                        if (rank != 0) {
                            divergenceDerivative = cullen_and_dehnen::divergenceTimeDerivative(
                                canonicalDivergence, parameters.scaleFactor, state, sampleInterval,
                                previousSampleTimestep, flops);
                        }
                    }
                    if (parameters.balsaraSwitch && data.sinkBalsaraFactor) {
                        data.sinkBalsaraFactor[sinkIndex] = balsara::encodeFactor(
                            balsara::factorFromVelocityGradient(velocityGradient, flops));
                    }
                    cullen_and_dehnen::Result<float> diagnostics;
                    if (parameters.cullenDehnen) {
                        diagnostics = cullen_and_dehnen::diagnosticsFromGradient(velocityGradient,
                            parameters.scaleFactor, divergenceDerivative,
                            densityWeightedSignAverage / density, sinkKernelSmoothingLength,
                            fmaxf(soundSpeedForSwitch, signalSpeed),
                            parameters.maximumBulkViscosityCoefficient, parameters.viscosityDecayCD,
                            flops);
                        flops += 1;
                    } else if (parameters.huEtAl) {
                        diagnostics = hu_etal::diagnosticsFromGradient(velocityGradient,
                            parameters.scaleFactor, divergenceDerivative, sinkKernelSmoothingLength,
                            soundSpeedForSwitch, fmaxf(soundSpeedForSwitch, signalSpeed),
                            parameters.maximumBulkViscosityCoefficientHu,
                            parameters.viscosityDecayHu, flops);
                    } else {
                        /*
                         * Price et al. (2018), equations (46)-(48): the paper's smoothing
                         * length is the same calibrated viscosity length as Morris & Monaghan's h.
                         */
                        float const smoothingLength = sinkKernelSmoothingLength /
                            selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                            float(monaghan::kViscosityWidthScale);
                        diagnostics = price_etal::diagnosticsFromGradient(velocityGradient,
                            parameters.scaleFactor, divergenceDerivative, smoothingLength,
                            soundSpeedForSwitch, parameters.maximumBulkViscosityCoefficientPrice,
                            parameters.viscosityDecayPrice, flops);
                        flops += 2;
                    }
                    diagnostics.divergence = canonicalDivergence;
                    if (data.sinkViscosityDiagnostics != nullptr) {
                        auto& pending = data.sinkViscosityDiagnostics[sinkIndex];
                        pending.pendingDivergence = diagnostics.divergence;
                        pending.targetBulkViscosityCoefficient =
                            diagnostics.targetBulkViscosityCoefficient;
                        pending.inverseDecayTime = diagnostics.inverseDecayTime;
                        pending.limiter = diagnostics.limiter;
                    }
                }
                data.sinkFlags[sinkIndex].setConverged(true);
                atomicMin(&reduction->minimumSmoothingLength, smoothingLength);
                atomicMax(&reduction->maximumSmoothingLength, smoothingLength);
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
