/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/monaghan_sph.hpp>
#include <cosmictiger/sph_cpu.hpp>

SphRunReturn sphCpuHydro(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true);
    float const inverseScaleFactor = 1.0f / parameters.scaleFactor;
    float const gasMassOverScaleFactor = data.gasMass * inverseScaleFactor;
    float const adiabaticIndex = data.adiabaticIndex;
    constexpr float kViscosityWidthScale = 0.9128709291752769f;
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    SimdFloat const zero(0.0f);
    SimdFloat const one(1.0f);
    SimdFloat const half(0.5f);
    SimdFloat const gammaMinusOne(adiabaticIndex - 1.0f);
    SimdFloat const massOverScaleFactor(gasMassOverScaleFactor);
    SimdFloat const viscosityAlpha(parameters.viscosityAlpha);
    SimdFloat const viscosityBeta(parameters.viscosityBeta);
    SimdFloat const viscosityEpsilon(parameters.viscosityEpsilon);
    SimdFloat const supportRadiusPerEffectiveRadius(
        float(selectedHydroKernel().supportRadiusPerEffectiveRadius));
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; particleIndex++) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        int const previousRung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
        if (previousRung < parameters.minimumRung) {
            continue;
        }
        float const sinkSupportRadius = data.sourceSupportRadius[particleIndex];
        float const sinkDensity = data.sourceDensity[particleIndex];
        float const sinkInternalEnergy = data.sourceInternalEnergy[particleIndex];
        ALWAYS_ASSERT(sinkSupportRadius > 0.0f && sinkDensity > 0.0f && sinkInternalEnergy > 0.0f);
        float const sinkPressureEnergyRatio = data.sourcePressureEnergyRatio ?
            data.sourcePressureEnergyRatio[particleIndex] :
            adiabaticIndex - 1.0f;
        float const sinkAdiabaticIndex =
            data.sourceAdiabaticIndex ? data.sourceAdiabaticIndex[particleIndex] : adiabaticIndex;
        float const sinkPressure = sinkPressureEnergyRatio * sinkDensity * sinkInternalEnergy;
        float const sinkSoundSpeed =
            std::sqrt(sinkAdiabaticIndex * sinkPressureEnergyRatio * sinkInternalEnergy);
        SimdFloat const support(sinkSupportRadius);
        SimdFloat const density(sinkDensity);
        SimdFloat const pressure(sinkPressure);
        SimdFloat const soundSpeed(sinkSoundSpeed);
        SimdFloat const inverseSinkSupportRadius(1.0f / sinkSupportRadius);
        SimdFloat const sinkGradientNormalization = sqr(sqr(inverseSinkSupportRadius));
        SimdFloat accelerationPackX(0.0f);
        SimdFloat accelerationPackY(0.0f);
        SimdFloat accelerationPackZ(0.0f);
        SimdFloat internalEnergyRatePack(0.0f);
        SimdFloat maximumCompressionPack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const separation =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const squaredSeparation =
                sqr(separation[XDIM], separation[YDIM], separation[ZDIM]);
            SimdFloat const sourceSupportRadius =
                detail::sphCpuLoad(data.sourceSupportRadius, neighbors, first, 1.0f);
            auto const interacts = detail::sphCpuValidLanes(neighbors, first) &
                (squaredSeparation > zero) &
                (squaredSeparation < simdMax(sqr(support), sqr(sourceSupportRadius)));
            if (!simdAnyOf(interacts)) {
                continue;
            }
            // Excluded neighbors may have no density yet. Replace their inputs before
            // division or square roots; masking only the final force can still trap.
            SimdFloat const neighborSupportRadius = simdSelect(interacts, sourceSupportRadius, one);
            SimdFloat const neighborDensity = simdSelect(
                interacts, detail::sphCpuLoad(data.sourceDensity, neighbors, first, 1.0f), one);
            SimdFloat const neighborInternalEnergy = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceInternalEnergy, neighbors, first, 1.0f), one);
            SimdFloat const pairSquaredSeparation = simdSelect(interacts, squaredSeparation, one);
            SimdFloat const radius = simdSqrt(pairSquaredSeparation);
            SimdFloat const inverseNeighborSupportRadius = one / neighborSupportRadius;
            SimdFloat const sinkGradient =
                dkernelWDq(radius * inverseSinkSupportRadius) * sinkGradientNormalization / radius;
            SimdFloat const neighborGradient = dkernelWDq(radius * inverseNeighborSupportRadius) *
                sqr(sqr(inverseNeighborSupportRadius)) / radius;
            SimdFloat const gradient =
                simdSelect(interacts, half * (sinkGradient + neighborGradient), zero);
            SimdFloat const relativeVelocityX = simdSelect(interacts,
                SimdFloat(data.sourceVelocityX[particleIndex]) -
                    detail::sphCpuLoad(data.sourceVelocityX, neighbors, first, 0.0f),
                zero);
            SimdFloat const relativeVelocityY = simdSelect(interacts,
                SimdFloat(data.sourceVelocityY[particleIndex]) -
                    detail::sphCpuLoad(data.sourceVelocityY, neighbors, first, 0.0f),
                zero);
            SimdFloat const relativeVelocityZ = simdSelect(interacts,
                SimdFloat(data.sourceVelocityZ[particleIndex]) -
                    detail::sphCpuLoad(data.sourceVelocityZ, neighbors, first, 0.0f),
                zero);
            SimdFloat const peculiarVelocityDotSeparation = relativeVelocityX * separation[XDIM] +
                relativeVelocityY * separation[YDIM] + relativeVelocityZ * separation[ZDIM];
            SimdFloat const physicalVelocityDotSeparation = peculiarVelocityDotSeparation +
                SimdFloat(parameters.expansionRate) *
                    simdSelect(interacts, squaredSeparation, zero);
            SimdFloat const viscosityRadius = half * (support + neighborSupportRadius) /
                supportRadiusPerEffectiveRadius * SimdFloat(kViscosityWidthScale);
            SimdFloat const viscosityApproachRate =
                simdSelect(interacts & (physicalVelocityDotSeparation < zero),
                    viscosityRadius * physicalVelocityDotSeparation /
                        (pairSquaredSeparation + viscosityEpsilon * sqr(viscosityRadius)),
                    zero);
            SimdFloat const neighborPressureEnergyRatio = data.sourcePressureEnergyRatio ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourcePressureEnergyRatio, neighbors, first, 1.0f),
                    one) :
                gammaMinusOne;
            SimdFloat const neighborAdiabaticIndex = data.sourceAdiabaticIndex ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourceAdiabaticIndex, neighbors, first, 1.0f), one) :
                SimdFloat(adiabaticIndex);
            SimdFloat const neighborPressure =
                neighborPressureEnergyRatio * neighborDensity * neighborInternalEnergy;
            SimdFloat const neighborSoundSpeed = simdSqrt(
                neighborAdiabaticIndex * neighborPressureEnergyRatio * neighborInternalEnergy);
            SimdFloat const meanSoundSpeed = half * (soundSpeed + neighborSoundSpeed);
            SimdFloat const meanDensity = half * (density + neighborDensity);
            SimdFloat const viscosity = simdSelect(viscosityApproachRate < zero,
                (-viscosityAlpha * meanSoundSpeed * viscosityApproachRate +
                    viscosityBeta * viscosityApproachRate * viscosityApproachRate) /
                    meanDensity,
                zero);
            SimdFloat const pressurePair = monaghan_sph::pressureCoefficient(
                density, neighborDensity, pressure, neighborPressure);
            SimdFloat const force = massOverScaleFactor * (pressurePair + viscosity) * gradient;
            accelerationPackX -= force * separation[XDIM];
            accelerationPackY -= force * separation[YDIM];
            accelerationPackZ -= force * separation[ZDIM];
            internalEnergyRatePack += massOverScaleFactor *
                (half *
                    (pressurePair * (peculiarVelocityDotSeparation * gradient) +
                        viscosity * (physicalVelocityDotSeparation * gradient)));
            maximumCompressionPack = simdMax(maximumCompressionPack, -viscosityApproachRate);
        }
        float accelerationX = simdSum(accelerationPackX);
        float accelerationY = simdSum(accelerationPackY);
        float accelerationZ = simdSum(accelerationPackZ);
        float internalEnergyRate = simdSum(internalEnergyRatePack);
        float const maximumCompression = simdMaximum(maximumCompressionPack);
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
            float const separationX = distance(data.sourcePositionX[particleIndex], Fixed32(0.5));
            float const separationY = distance(data.sourcePositionY[particleIndex], Fixed32(0.5));
            float const separationZ = distance(data.sourcePositionZ[particleIndex], Fixed32(0.5));
            float const radius = std::sqrt(sqr(separationX, separationY, separationZ));
            float const inverseSupportRadius = 1.0f / data.centralSupportRadius;
            float const force = data.centralAcceleration *
                radial_kernel::forceFactor(selectedGravityKernel(), radius * inverseSupportRadius) *
                inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
            accelerationX -= force * separationX;
            accelerationY -= force * separationY;
            accelerationZ -= force * separationZ;
        }
        accelerationX -= parameters.velocityDampingRate * data.sourceVelocityX[particleIndex];
        accelerationY -= parameters.velocityDampingRate * data.sourceVelocityY[particleIndex];
        accelerationZ -= parameters.velocityDampingRate * data.sourceVelocityZ[particleIndex];
        data.sinkAccelerations[sinkIndex].acceleration[XDIM] = accelerationX;
        data.sinkAccelerations[sinkIndex].acceleration[YDIM] = accelerationY;
        data.sinkAccelerations[sinkIndex].acceleration[ZDIM] = accelerationZ;
        // E=a^2 u and homogeneous adiabatic work du/dt=-3 H P/rho give
        // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E for a composition-dependent EOS.
        float const expansionCoefficient = data.sourcePressureEnergyRatio ?
            2.0f - 3.0f * sinkPressureEnergyRatio :
            5.0f - 3.0f * adiabaticIndex;
        internalEnergyRate += expansionCoefficient * parameters.expansionRate * inverseScaleFactor *
            sinkInternalEnergy;
        data.sinkInternalEnergyRate[sinkIndex] = internalEnergyRate;
        if (parameters.phase == 1) {
            float const effectiveRadius =
                sinkSupportRadius / selectedHydroKernel().supportRadiusPerEffectiveRadius;
            float const inverseSoundTimestep = sinkSoundSpeed / effectiveRadius;
            float const inverseViscosityTimestep = 0.6f *
                (parameters.viscosityAlpha * sinkSoundSpeed +
                    parameters.viscosityBeta * maximumCompression) /
                effectiveRadius;
            float const acceleration = std::sqrt(sqr(accelerationX, accelerationY, accelerationZ));
            float const inverseAccelerationTimestep = parameters.courantFactor *
                std::sqrt(parameters.scaleFactor * acceleration / effectiveRadius) /
                (data.accelerationTimestepFactor * kHydroAccelerationScale);
            float const inverseTimestep = std::fmax(
                inverseSoundTimestep + inverseViscosityTimestep, inverseAccelerationTimestep);
            float hydroTimestep = std::fmin(parameters.maximumTimestep,
                parameters.courantFactor * parameters.scaleFactor /
                    std::fmax(inverseTimestep, 1.0e-30f));
            if (internalEnergyRate != 0.0f) {
                hydroTimestep = std::fmin(hydroTimestep,
                    parameters.courantFactor / kViscosityWidthScale * sinkInternalEnergy /
                        std::fabs(internalEnergyRate));
            }
            float const gravityAcceleration =
                std::sqrt(sqr(gravityAccelerationX, gravityAccelerationY, gravityAccelerationZ));
            float const gravityTimestep = std::fmin(parameters.maximumTimestep,
                data.accelerationTimestepFactor *
                    std::sqrt(parameters.scaleFactor * data.gravityEffectiveRadius /
                        std::fmax(gravityAcceleration, 1.0e-30f)));
            ALWAYS_ASSERT(std::isfinite(hydroTimestep) && hydroTimestep > 0.0f);
            int const hydroRung =
                std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / hydroTimestep))));
            int const gravityRung =
                std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / gravityTimestep))));
            int const nextRung = std::max(std::max(parameters.minimumRung, previousRung - 1),
                std::max(hydroRung, data.selfGravityEnabled ? gravityRung : 1));
            ALWAYS_ASSERT(nextRung < MAX_RUNG);
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
            result.maximumHydroRung = std::max(result.maximumHydroRung, hydroRung);
            result.maximumGravityRung = std::max(result.maximumGravityRung, gravityRung);
            result.maximumRung = std::max(result.maximumRung, nextRung);
            result.maximumSignalSpeed = std::fmax(result.maximumSignalSpeed,
                (inverseSoundTimestep + inverseViscosityTimestep) * effectiveRadius);
            result.inverseSoundTimestep =
                std::fmax(result.inverseSoundTimestep, inverseSoundTimestep);
            result.inverseViscosityTimestep =
                std::fmax(result.inverseViscosityTimestep, inverseViscosityTimestep);
            result.inverseAccelerationTimestep =
                std::fmax(result.inverseAccelerationTimestep, inverseAccelerationTimestep);
        }
    }
    return result;
}
