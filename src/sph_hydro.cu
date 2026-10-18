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

#include <cosmictiger/monaghan_sph.hpp>
#include <cosmictiger/sph_cuda.hpp>

namespace detail {

struct HydroNeighbor {
    Fixed32 positionX;
    Fixed32 positionY;
    Fixed32 positionZ;
    float supportRadius;
    float velocityX;
    float velocityY;
    float velocityZ;
    float internalEnergy;
    float density;
};

struct HydroWorkspace {
    DeviceVector<HydroNeighbor> neighbors;
};

}    // namespace detail

// Monaghan (1992), equations (4.1)-(4.2) and (6.4)-(6.5).
// Positions and density are comoving; velocity and internal energy store a*v and a^2*u.
// The stored radius is the compact-support radius; the effective smoothing width is 2 sigma.
__global__ void sphCudaHydro(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::HydroWorkspace workspace;
    new (&workspace) detail::HydroWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    float const inverseScaleFactor = 1.0f / parameters.scaleFactor;
    float const gasMassOverScaleFactor = data.gasMass * inverseScaleFactor;
    float const adiabaticIndex = data.adiabaticIndex;
    // Keep the historical cubic viscosity scale while measuring every kernel
    // with the same effective width of 2 sigma: the paper smoothing length is
    // sqrt(5/6) times this width for the cubic spline.
    constexpr float kViscosityWidthScale = 0.9128709291752769f;
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4
    // defaults after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    int maximumHydroRung = 0;
    int maximumGravityRung = 0;
    int maximumRung = 0;
    float maximumSignalSpeed = 0.0f;
    while (targetNodeIndex < data.numberOfTargetNodes) {
        __syncthreads();
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        workspace.neighbors.resize(0);
        // The union of both supports is required by the symmetric gradient.
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
                computeIndices<HYDRO_BLOCK_SIZE>(slot, total);
                int const offset = workspace.neighbors.size();
                __syncthreads();
                workspace.neighbors.resize(offset + total);
                if (contains) {
                    workspace.neighbors[offset + slot] = {position[XDIM], position[YDIM],
                        position[ZDIM], data.sourceSupportRadius[sourceParticleIndex],
                        data.sourceVelocityX[sourceParticleIndex],
                        data.sourceVelocityY[sourceParticleIndex],
                        data.sourceVelocityZ[sourceParticleIndex],
                        data.sourceInternalEnergy[sourceParticleIndex],
                        data.sourceDensity[sourceParticleIndex]};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            int const previousRung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
            if (previousRung < parameters.minimumRung) {
                continue;
            }
            float const sinkSupportRadius = data.sourceSupportRadius[particleIndex];
            float const sinkDensity = data.sourceDensity[particleIndex];
            float const sinkInternalEnergy = data.sourceInternalEnergy[particleIndex];
            ALWAYS_ASSERT(
                sinkSupportRadius > 0.0f && sinkDensity > 0.0f && sinkInternalEnergy > 0.0f);
            float const sinkPressure = (adiabaticIndex - 1.0f) * sinkDensity * sinkInternalEnergy;
            float const sinkSoundSpeed =
                sqrtf(adiabaticIndex * (adiabaticIndex - 1.0f) * sinkInternalEnergy);
            float accelerationX = 0.0f;
            float accelerationY = 0.0f;
            float accelerationZ = 0.0f;
            float internalEnergyRate = 0.0f;
            float maximumCompression = 0.0f;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.neighbors.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.neighbors[neighborIndex];
                float const separationX =
                    distance(data.sourcePositionX[particleIndex], neighbor.positionX);
                float const separationY =
                    distance(data.sourcePositionY[particleIndex], neighbor.positionY);
                float const separationZ =
                    distance(data.sourcePositionZ[particleIndex], neighbor.positionZ);
                float const squaredSeparation = sqr(separationX, separationY, separationZ);
                if (squaredSeparation == 0.0f ||
                    squaredSeparation >=
                        fmaxf(sqr(sinkSupportRadius), sqr(neighbor.supportRadius))) {
                    continue;
                }
                float const radius = sqrtf(squaredSeparation);
                float const inverseSinkSupportRadius = 1.0f / sinkSupportRadius;
                float const inverseNeighborSupportRadius = 1.0f / neighbor.supportRadius;
                float const sinkGradient = dkernelWDq(radius * inverseSinkSupportRadius) *
                    sqr(sqr(inverseSinkSupportRadius)) / radius;
                float const neighborGradient = dkernelWDq(radius * inverseNeighborSupportRadius) *
                    sqr(sqr(inverseNeighborSupportRadius)) / radius;
                float const gradient = 0.5f * (sinkGradient + neighborGradient);
                float const relativeVelocityX =
                    data.sourceVelocityX[particleIndex] - neighbor.velocityX;
                float const relativeVelocityY =
                    data.sourceVelocityY[particleIndex] - neighbor.velocityY;
                float const relativeVelocityZ =
                    data.sourceVelocityZ[particleIndex] - neighbor.velocityZ;
                float const peculiarVelocityDotSeparation = relativeVelocityX * separationX +
                    relativeVelocityY * separationY + relativeVelocityZ * separationZ;
                // Include Hubble motion only in physical pair approach/viscous heating.
                float const physicalVelocityDotSeparation =
                    peculiarVelocityDotSeparation + parameters.expansionRate * squaredSeparation;
                float const viscosityApproachRate =
                    monaghan_sph::viscosityMu(squaredSeparation, physicalVelocityDotSeparation,
                        sinkSupportRadius, neighbor.supportRadius, parameters.viscosityEpsilon,
                        float(selectedHydroKernel().supportRadiusPerEffectiveRadius),
                        kViscosityWidthScale);
                float const neighborPressure =
                    (adiabaticIndex - 1.0f) * neighbor.density * neighbor.internalEnergy;
                float const neighborSoundSpeed =
                    sqrtf(adiabaticIndex * (adiabaticIndex - 1.0f) * neighbor.internalEnergy);
                float const viscosity = monaghan_sph::artificialViscosity(sinkDensity,
                    neighbor.density, sinkSoundSpeed, neighborSoundSpeed, viscosityApproachRate,
                    parameters.viscosityAlpha, parameters.viscosityBeta);
                float const pressurePair = monaghan_sph::pressureCoefficient(
                    sinkDensity, neighbor.density, sinkPressure, neighborPressure);
                float const force = gasMassOverScaleFactor * (pressurePair + viscosity) * gradient;
                accelerationX -= force * separationX;
                accelerationY -= force * separationY;
                accelerationZ -= force * separationZ;
                internalEnergyRate += gasMassOverScaleFactor *
                    monaghan_sph::pairEnergyRate(pressurePair, viscosity,
                        peculiarVelocityDotSeparation * gradient,
                        physicalVelocityDotSeparation * gradient);
                maximumCompression = fmaxf(maximumCompression, -viscosityApproachRate);
            }
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(accelerationX);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(accelerationY);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(accelerationZ);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(internalEnergyRate);
            sharedReduceMax<HYDRO_BLOCK_SIZE>(maximumCompression);
            if (threadIndex == 0) {
                float const gravityAccelerationX =
                    data.selfGravityEnabled ? data.sinkGravityAccelerationX[sinkIndex] : 0.0f;
                float const gravityAccelerationY =
                    data.selfGravityEnabled ? data.sinkGravityAccelerationY[sinkIndex] : 0.0f;
                float const gravityAccelerationZ =
                    data.selfGravityEnabled ? data.sinkGravityAccelerationZ[sinkIndex] : 0.0f;
                accelerationX += gravityAccelerationX;
                accelerationY += gravityAccelerationY + parameters.uniformAccelerationY;
                accelerationZ += gravityAccelerationZ;
                if (data.centralAcceleration != 0.0f) {
                    float const separationX =
                        distance(data.sourcePositionX[particleIndex], Fixed32(0.5));
                    float const separationY =
                        distance(data.sourcePositionY[particleIndex], Fixed32(0.5));
                    float const separationZ =
                        distance(data.sourcePositionZ[particleIndex], Fixed32(0.5));
                    float const radius = sqrtf(sqr(separationX, separationY, separationZ));
                    float const inverseSupportRadius = 1.0f / data.centralSupportRadius;
                    float const force = data.centralAcceleration *
                        radial_kernel::forceFactor(
                            selectedGravityKernel(), radius * inverseSupportRadius) *
                        inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
                    accelerationX -= force * separationX;
                    accelerationY -= force * separationY;
                    accelerationZ -= force * separationZ;
                }
                accelerationX -=
                    parameters.velocityDampingRate * data.sourceVelocityX[particleIndex];
                accelerationY -=
                    parameters.velocityDampingRate * data.sourceVelocityY[particleIndex];
                accelerationZ -=
                    parameters.velocityDampingRate * data.sourceVelocityZ[particleIndex];
                data.sinkAccelerations[sinkIndex].acceleration[XDIM] = accelerationX;
                data.sinkAccelerations[sinkIndex].acceleration[YDIM] = accelerationY;
                data.sinkAccelerations[sinkIndex].acceleration[ZDIM] = accelerationZ;
                // Homogeneous adiabatic expansion of a^2*u for any ideal-gas adiabatic index.
                internalEnergyRate += (5.0f - 3.0f * adiabaticIndex) * parameters.expansionRate *
                    inverseScaleFactor * sinkInternalEnergy;
                data.sinkInternalEnergyRate[sinkIndex] = internalEnergyRate;
                if (parameters.phase == 1) {
                    float const effectiveRadius =
                        sinkSupportRadius / selectedHydroKernel().supportRadiusPerEffectiveRadius;
                    // Section 10.3: the maximum compression is positive for approaching particles.
                    float const inverseSoundTimestep = sinkSoundSpeed / effectiveRadius;
                    float const inverseViscosityTimestep = 0.6f *
                        (parameters.viscosityAlpha * sinkSoundSpeed +
                            parameters.viscosityBeta * maximumCompression) /
                        effectiveRadius;
                    float const acceleration =
                        sqrtf(sqr(accelerationX, accelerationY, accelerationZ));
                    float const inverseAccelerationTimestep = parameters.courantFactor *
                        sqrtf(parameters.scaleFactor * acceleration / effectiveRadius) /
                        (data.accelerationTimestepFactor * kHydroAccelerationScale);
                    float const inverseTimestep =
                        fmaxf(inverseSoundTimestep + inverseViscosityTimestep,
                            inverseAccelerationTimestep);
                    float hydroTimestep = fminf(parameters.maximumTimestep,
                        parameters.courantFactor * parameters.scaleFactor /
                            fmaxf(inverseTimestep, 1.0e-30f));
                    // Bound fractional thermal change in the existing explicit integrator.
                    if (internalEnergyRate != 0.0f) {
                        hydroTimestep = fminf(hydroTimestep,
                            parameters.courantFactor / kViscosityWidthScale * sinkInternalEnergy /
                                fabsf(internalEnergyRate));
                    }
                    float const gravityAcceleration = sqrtf(
                        sqr(gravityAccelerationX, gravityAccelerationY, gravityAccelerationZ));
                    float const gravityTimestep = fminf(parameters.maximumTimestep,
                        data.accelerationTimestepFactor *
                            sqrtf(parameters.scaleFactor * data.gravityEffectiveRadius /
                                fmaxf(gravityAcceleration, 1.0e-30f)));
                    ALWAYS_ASSERT(isfinite(hydroTimestep) && hydroTimestep > 0.0f);
                    int const hydroRung =
                        max(1, int(ceilf(log2f(parameters.baseTimestep / hydroTimestep))));
                    int const gravityRung =
                        max(1, int(ceilf(log2f(parameters.baseTimestep / gravityTimestep))));
                    int const nextRung = max(max(parameters.minimumRung, previousRung - 1),
                        max(hydroRung, data.selfGravityEnabled ? gravityRung : 1));
                    ALWAYS_ASSERT(nextRung < MAX_RUNG);
                    data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
                    maximumHydroRung = max(maximumHydroRung, hydroRung);
                    maximumGravityRung = max(maximumGravityRung, gravityRung);
                    maximumRung = max(maximumRung, nextRung);
                    maximumSignalSpeed = fmaxf(maximumSignalSpeed,
                        (inverseSoundTimestep + inverseViscosityTimestep) * effectiveRadius);
                    atomicMax(&reduction->inverseSoundTimestep, inverseSoundTimestep);
                    atomicMax(&reduction->inverseViscosityTimestep, inverseViscosityTimestep);
                    atomicMax(&reduction->inverseAccelerationTimestep, inverseAccelerationTimestep);
                }
            }
        }
        __syncthreads();
        if (threadIndex == 0) {
            targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
        }
        __syncthreads();
    }
    if (threadIndex == 0) {
        atomicMax(&reduction->maximumSignalSpeed, maximumSignalSpeed);
        atomicMax(&reduction->maximumRung, maximumRung);
        atomicMax(&reduction->maximumHydroRung, maximumHydroRung);
        atomicMax(&reduction->maximumGravityRung, maximumGravityRung);
    }
    (&workspace)->~HydroWorkspace();
}
