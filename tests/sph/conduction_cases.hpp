/* CosmicTiger - shared CPU/CUDA artificial-conduction regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/sph/run_data.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <tests/sph/pair_kick_cases.hpp>

namespace detail {

template <class Fixture>
void testConductionPair(Fixture& fixture) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    parameters.midpointHydro = true;
    parameters.minimumRung = 0;
    parameters.minimumProposedRung = 0;
    parameters.maximumRung = kMaxRung - 1;
    parameters.baseTimestep = 1.0e-5;
    parameters.bulkViscosityCoefficient = 0.0f;
    parameters.vonNeumannRichtmyerCoefficient = 0.0f;
    data.selfGravityEnabled = false;
    int const hot = fixture.sourceIndex(0);
    int const cold = fixture.sourceIndex(1);
    data.sourcePositionX[hot] = 0.47;
    data.sourcePositionX[cold] = 0.53;
    for (int index = 0; index < 2; ++index) {
        int const source = fixture.sourceIndex(index);
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceDensity[source] = 1.1f + index;
        data.sourceInternalEnergy[source] = index == 0 ? 2.0f : 0.5f;
        data.sourceSmoothingLength[source] = 0.22f + 0.07f * index;
        data.sourceGradHCorrection[source] = 0.0f;
    }
    // Exercise nonapproaching pairs, physical compression, unequal supports/densities,
    // and the stored-energy/entropy conversion at nonunit cosmological scale.
    for (float const scaleFactor : {1.0f, 0.7f}) {
        parameters.scaleFactor = scaleFactor;
        for (float const approach : {0.0f, 0.15f, -0.15f}) {
            parameters.expansionRate = 0.03f;
            data.sourceVelocityX[hot] = approach;
            data.sourceVelocityX[cold] = -approach;
            parameters.conductionCoefficient = 0.0f;
            fixture.hydro();
            std::array<Vector3<float>, 2> pressureAccelerations;
            for (int index = 0; index < 2; ++index) {
                pressureAccelerations[index] = data.sinkAccelerations[fixture.sinkIndex(index)];
                pairKickRequire(data.sinkEntropyRate[fixture.sinkIndex(index)] == 0.0f,
                    "disabled conduction changed entropy without viscosity");
            }
            parameters.conductionCoefficient = 0.3f;
            fixture.hydro();
            // Independent scalar evaluation of Monaghan (1992), (4.5)/(4.8),
            // with the dissipative sign fixed by (4.4)/(4.6); see conduction.hpp.
            double const radius =
                std::abs(double(distance(data.sourcePositionX[hot], data.sourcePositionX[cold])));
            double const hotSupport = data.sourceSmoothingLength[hot];
            double const coldSupport = data.sourceSmoothingLength[cold];
            double const pairLength = 0.5 * (hotSupport + coldSupport) *
                monaghan::kViscosityWidthScale /
                selectedHydroKernel().smoothingLengthPerEffectiveRadius;
            double const denominator =
                radius * radius + parameters.viscosityEpsilon * pairLength * pairLength;
            double const velocityDotRadius =
                -2.0 * approach * radius + parameters.expansionRate * radius * radius;
            double const compression = std::min(0.0, pairLength * velocityDotRadius / denominator);
            double const meanSoundSpeed = 0.5 *
                (std::sqrt(data.adiabaticIndex * (data.adiabaticIndex - 1.0) *
                     data.sourceInternalEnergy[hot]) +
                    std::sqrt(data.adiabaticIndex * (data.adiabaticIndex - 1.0) *
                        data.sourceInternalEnergy[cold]));
            double const radialGradient = 0.5 *
                (dkernelWDq(radius / hotSupport) / std::pow(hotSupport, 4) +
                    dkernelWDq(radius / coldSupport) / std::pow(coldSupport, 4));
            double const exchangeRate = -data.gasMass / scaleFactor *
                parameters.conductionCoefficient * pairLength * (meanSoundSpeed - 4 * compression) *
                radius * radialGradient /
                (0.5 * (data.sourceDensity[hot] + data.sourceDensity[cold]) * denominator);
            double const expectedHotRate =
                exchangeRate * (data.sourceInternalEnergy[cold] - data.sourceInternalEnergy[hot]);
            pairKickRequire(expectedHotRate < 0.0, "conduction reference has wrong sign");
            double totalEnergyRate = 0.0;
            for (int index = 0; index < 2; ++index) {
                int const source = fixture.sourceIndex(index);
                int const sink = fixture.sinkIndex(index);
                double const energyRate = sph_entropy::storedEnergy(data.sinkEntropyRate[sink],
                    data.sourceDensity[source], data.adiabaticIndex, scaleFactor);
                pairKickClose(energyRate, index == 0 ? expectedHotRate : -expectedHotRate,
                    "conductive rate differs from the paper's exchange");
                totalEnergyRate += energyRate;
                for (int dimension = 0; dimension < kDimensions; ++dimension) {
                    pairKickRequire(data.sinkAccelerations[sink][dimension] ==
                            pressureAccelerations[index][dimension],
                        "conduction directly changed the pressure force");
                }
            }
            pairKickRequire(std::abs(totalEnergyRate) < 2.0e-6 * std::abs(expectedHotRate),
                "unequal-density conduction failed to conserve thermal energy");
        }
    }
    // Same-temperature pairs have zero exchange but still require a diffusion
    // timestep bound: testing only net du/dt would miss this case.
    data.sourceInternalEnergy[cold] = data.sourceInternalEnergy[hot];
    parameters.globalTimestep = true;
    parameters.conductionCoefficient = 0.0f;
    fixture.hydro(1);
    double const ordinaryTimestep = fixture.reduction_->minimumTimestep;
    parameters.conductionCoefficient = 100.0f;
    fixture.hydro(1);
    double const conductionTimestep = fixture.reduction_->minimumTimestep;
    pairKickRequire(conductionTimestep < ordinaryTimestep / 10,
        "uniform-temperature diffusion did not limit the timestep");
    parameters.conductionCoefficient *= 2;
    fixture.hydro(1);
    pairKickClose(fixture.reduction_->minimumTimestep, 0.5 * conductionTimestep,
        "diffusion timestep does not scale inversely with g");
    for (int index = 0; index < 2; ++index) {
        pairKickRequire(data.sinkEntropyRate[fixture.sinkIndex(index)] == 0.0f,
            "uniform-temperature conduction generated a thermal rate");
    }
    // A fine particle owns the exchange and updates its inactive coarse neighbor.
    // Changing which endpoint owns it must preserve energy, positivity and total entropy.
    data.sourceInternalEnergy[cold] = 0.5f;
    parameters.globalTimestep = false;
    parameters.conductionCoefficient = 0.3f;
    parameters.hydroKickTimestep = conductionTimestep;
    auto* identities = fixture.enablePairKickOutputs();
    identities[hot] = {0, 11};
    identities[cold] = {1, 37};
    data.sinkInternalEnergyRate = nullptr;
    data.sinkEntropyRate = nullptr;
    for (int owner = 0; owner < 3; ++owner) {
        int const hotRung = owner == 1 ? 1 : 3;
        int const coldRung = owner == 0 ? 1 : 3;
        data.sourceRungs[hot] = data.sinkParticleRungs[fixture.particleIndex(0)] = hotRung;
        data.sourceRungs[cold] = data.sinkParticleRungs[fixture.particleIndex(1)] = coldRung;
        parameters.minimumRung = parameters.maximumRung = 3;
        data.sourcePairUpdates[hot] = {};
        data.sourcePairUpdates[cold] = {};
        fixture.hydro(1);
        double totalIncrement = 0.0;
        double entropyChange = 0.0;
        for (int index = 0; index < 2; ++index) {
            int const source = fixture.sourceIndex(index);
            double const increment =
                sph_entropy::storedEnergy(data.sourcePairUpdates[source].entropy,
                    data.sourceDensity[source], data.adiabaticIndex, parameters.scaleFactor);
            pairKickRequire(index == 0 ? increment < 0.0 : increment > 0.0,
                "pair ownership lost conductive exchange");
            double const initialEnergy = data.sourceInternalEnergy[source];
            pairKickRequire(initialEnergy + increment > 0.0, "conductive kick lost positivity");
            entropyChange += std::log1p(increment / initialEnergy);
            totalIncrement += increment;
        }
        double const hotIncrement = sph_entropy::storedEnergy(data.sourcePairUpdates[hot].entropy,
            data.sourceDensity[hot], data.adiabaticIndex, parameters.scaleFactor);
        pairKickRequire(std::abs(totalIncrement) < 2.0e-6 * std::abs(hotIncrement),
            "mixed-rung conductive kick failed energy conservation");
        pairKickRequire(entropyChange > 0.0, "conductive pair decreased total physical entropy");
    }
}

// Independent on/off evaluations check that the total rates superpose without
// duplicating the pair equations; the conduction case above checks their scalar reference.
template <class Fixture>
void testDissipativeRateSuperposition(Fixture& fixture, bool morrisAndMonaghan = false) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    parameters.midpointHydro = true;
    parameters.scaleFactor = 0.7f;
    parameters.expansionRate = 0.0f;
    parameters.hydroKickTimestep = 0.0;
    data.selfGravityEnabled = false;
    for (int index = 0; index < 2; ++index) {
        int const source = fixture.sourceIndex(index);
        data.sourcePositionX[source] = index == 0 ? 0.47 : 0.53;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceVelocityX[source] = index == 0 ? 0.15f : -0.15f;
        data.sourceVelocityY[source] = data.sourceVelocityZ[source] = 0.0f;
        data.sourceDensity[source] = 1.1f + index;
        data.sourceInternalEnergy[source] = index == 0 ? 2.0f : 0.5f;
        data.sourceSmoothingLength[source] = 0.22f + 0.07f * index;
        data.sourceGradHCorrection[source] = 0.0f;
    }
    if (morrisAndMonaghan) {
        fixture.enableVariableViscosity();
    }
    std::array<SphAcceleration, 2> pressureAccelerations;
    std::array<float, 2> viscosityRates{}, conductionRates{};
    for (int mode = 0; mode < 4; ++mode) {
        parameters.bulkViscosityCoefficient = (mode & 1) ? 0.7f : 0.0f;
        parameters.vonNeumannRichtmyerCoefficient = (mode & 1) ? 2.0f : 0.0f;
        parameters.conductionCoefficient = (mode & 2) ? 0.3f : 0.0f;
        if (morrisAndMonaghan) {
            for (int index = 0; index < 2; ++index) {
                data.sourceBulkViscosityCoefficient[fixture.sourceIndex(index)] =
                    (mode & 1) ? 0.4f + 0.2f * index : 0.0f;
            }
        }
        parameters.uniformAccelerationY = 0.0f;
        parameters.velocityDampingRate = 0.0f;
        fixture.hydro();
        std::array<SphAcceleration, 2> hydroAccelerations;
        std::array<float, 2> entropyRates;
        for (int index = 0; index < 2; ++index) {
            int const sink = fixture.sinkIndex(index);
            hydroAccelerations[index] = data.sinkAccelerations[sink];
            entropyRates[index] = data.sinkEntropyRate[sink];
        }
        // These forces must affect integration but never the entropy rate.
        parameters.uniformAccelerationY = 0.25f;
        parameters.velocityDampingRate = 0.1f;
        fixture.hydro();
        for (int index = 0; index < 2; ++index) {
            int const sink = fixture.sinkIndex(index);
            pairKickClose(data.sinkEntropyRate[sink], entropyRates[index],
                "external sources changed the entropy rate");
            pairKickClose(data.sinkAccelerations[sink][1], hydroAccelerations[index][1] + 0.25f,
                "external acceleration was not applied to the integrator");
            for (int dimension = 0; dimension < kSpatialDimensions; ++dimension) {
                if (mode == 0) {
                    pressureAccelerations[index][dimension] = hydroAccelerations[index][dimension];
                }
                if ((mode & 1) == 0) {
                    pairKickClose(hydroAccelerations[index][dimension],
                        pressureAccelerations[index][dimension],
                        "conduction changed the pressure force");
                }
            }
            if (mode & 1) {
                pairKickRequire(hydroAccelerations[index][0] != pressureAccelerations[index][0],
                    "approaching viscous pair has no viscous force");
            }
            if (mode == 0) {
                pairKickRequire(
                    entropyRates[index] == 0.0f, "disabled dissipation generated entropy");
            } else if (mode == 1) {
                viscosityRates[index] = entropyRates[index];
                pairKickRequire(viscosityRates[index] > 0.0f, "viscous pair generated no entropy");
            } else if (mode == 2) {
                conductionRates[index] = entropyRates[index];
            } else {
                pairKickClose(entropyRates[index], viscosityRates[index] + conductionRates[index],
                    "viscosity and conduction entropy rates do not superpose");
            }
        }
    }
}

template <class Fixture>
void testContinuityDensityRate(Fixture& fixture) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableLogDensityRates();
    parameters.minimumSmoothingLength = parameters.maximumSmoothingLength = 0.2f;
    parameters.scaleFactor = 0.7f;
    parameters.expansionRate = 0.3f;
    for (int index = 0; index < 2; ++index) {
        int const source = fixture.sourceIndex(index);
        data.sourcePositionX[source] = index == 0 ? 0.47 : 0.53;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceVelocityY[source] = data.sourceVelocityZ[source] = 0.0f;
    }
    for (float const approach : {0.0f, 0.15f, -0.15f}) {
        data.sourceVelocityX[fixture.sourceIndex(0)] = approach;
        data.sourceVelocityX[fixture.sourceIndex(1)] = -approach;
        fixture.densityPass();
        double const radius = std::abs(double(distance(data.sourcePositionX[fixture.sourceIndex(0)],
            data.sourcePositionX[fixture.sourceIndex(1)])));
        double const support = parameters.maximumSmoothingLength;
        // Fixed-support continuity equation, evaluated independently for this pair.
        double const expectedRate = data.gasMass * (-2.0 * approach) *
            dkernelWDq(radius / support) / (parameters.scaleFactor * std::pow(support, 4));
        for (int index = 0; index < 2; ++index) {
            int const sink = fixture.sinkIndex(index);
            pairKickClose(double(data.sinkLogDensityRate[sink]) * data.sinkDensity[sink],
                expectedRate, "density rate differs from the continuity equation");
        }
    }
}

}    // namespace detail
