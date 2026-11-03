/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/sph/sph.hpp>

namespace detail {

void require(bool condition, char const* label);
void requireClose(char const* label, double actual, double expected, double tolerance);

template <typename Fixture>
void testPhantomPairs(Fixture& fixture) {
    fixture.enableVariableViscosity(false, true);
    auto& data = fixture.data_;
    float const gamma = data.adiabaticIndex;
    for (int i = 0; i < 2; ++i) {
        int const source = fixture.sourceIndex(i);
        data.sourcePositionX[source] = i == 0 ? 0.35 : 0.65;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceDensity[source] = i == 0 ? 0.8f : 2.3f;
        data.sourceInternalEnergy[source] = i == 0 ? 0.7f : 1.9f;
        data.sourceGradHCorrection[source] = i == 0 ? 0.1f : -0.2f;
        data.sourceBalsaraFactor[source] = i == 0 ? 0.2f : 0.8f;
    }
    // Equation (39) must retain the two unequal kernels/densities/sound speeds and
    // equation (42) must assign directed local heating, even outside one support.
    for (float const supportA : {0.2f, 0.5f}) {
        data.sourceSupportRadius[fixture.sourceIndex(0)] = supportA;
        data.sourceSupportRadius[fixture.sourceIndex(1)] = 0.45f;
        for (float const scaleFactor : {0.4f, 1.0f, 2.0f}) {
            fixture.params_.scaleFactor = scaleFactor;
            for (float const expansionRate : {0.0f, 1.0f}) {
                fixture.params_.expansionRate = expansionRate;
                for (float const velocity : {-0.1f, 0.0f, 0.1f}) {
                    data.sourceVelocityX[fixture.sourceIndex(0)] = velocity;
                    data.sourceVelocityX[fixture.sourceIndex(1)] = -velocity;
                    for (float const bulkViscosityCoefficient : {0.0f, 0.2f}) {
                        data.sourceBulkViscosityCoefficient[fixture.sourceIndex(0)] =
                            bulkViscosityCoefficient;
                        data.sourceBulkViscosityCoefficient[fixture.sourceIndex(1)] =
                            3.0f * bulkViscosityCoefficient;
                        for (float const vonNeumannRichtmyerCoefficient : {0.0f, 2.0f}) {
                            for (bool const balsara : {false, true}) {
                                fixture.params_.balsaraSwitch = balsara;
                                // Reference removes only dissipation, leaving pressure/work intact.
                                fixture.params_.priceEtAl = false;
                                fixture.params_.variableViscosity = false;
                                fixture.params_.bulkViscosityCoefficient =
                                    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
                                fixture.hydro();
                                std::array<float, 2> energy;
                                std::array<SphAcceleration, 2> acceleration;
                                for (int i = 0; i < 2; ++i) {
                                    int const sink = fixture.sinkIndex(i);
                                    energy[i] = data.sinkInternalEnergyRate[sink];
                                    acceleration[i] = data.sinkAccelerations[sink];
                                }
                                fixture.params_.priceEtAl = true;
                                fixture.params_.variableViscosity = true;
                                fixture.params_.vonNeumannRichtmyerCoefficient =
                                    vonNeumannRichtmyerCoefficient;
                                fixture.hydro();
                                if (expansionRate == 0.0f) {
                                    fixture.checkConservation();
                                }
                                int const a = fixture.sourceIndex(0);
                                int const b = fixture.sourceIndex(1);
                                double const dx =
                                    distance(data.sourcePositionX[a], data.sourcePositionX[b]);
                                double const pairDistance = std::abs(dx);
                                double const dot = (2.0 * velocity + expansionRate * dx) * dx;
                                double const w = std::min(dot / pairDistance, 0.0);
                                double directed[2];
                                double gradient[2];
                                for (int i = 0; i < 2; ++i) {
                                    int const source = fixture.sourceIndex(i);
                                    double const rho = data.sourceDensity[source];
                                    double const sound = std::sqrt(
                                        gamma * (gamma - 1.0f) * data.sourceInternalEnergy[source]);
                                    double const smoothingNormalizedRadius = -0.5 * rho *
                                        (data.sourceBulkViscosityCoefficient[source] * sound -
                                            vonNeumannRichtmyerCoefficient * w) *
                                        w;
                                    directed[i] = smoothingNormalizedRadius *
                                        (data.gasMass - data.sourceGradHCorrection[source]) /
                                        (rho * rho);
                                    if (balsara) {
                                        directed[i] *= 0.5;
                                    }
                                    float const support = data.sourceSupportRadius[source];
                                    gradient[i] = dkernelWDq(float(pairDistance) / support) /
                                        (std::pow(double(support), 4) * pairDistance);
                                }
                                double const force =
                                    (directed[0] * gradient[0] + directed[1] * gradient[1]) /
                                    scaleFactor;
                                for (int i = 0; i < 2; ++i) {
                                    int const sink = fixture.sinkIndex(i);
                                    double const heat =
                                        directed[i] * gradient[i] * dot / scaleFactor;
                                    requireClose("Phantom equation 42 local heating",
                                        data.sinkInternalEnergyRate[sink] - energy[i], heat,
                                        4.0e-5);
                                    require(heat >= 0.0, "Phantom shock heating is nonnegative");
                                    requireClose("Phantom equation 39 directed pair force",
                                        data.sinkAccelerations[sink][0] - acceleration[i][0],
                                        (i == 0 ? -1.0 : 1.0) * force * dx, 4.0e-5);
                                    for (int dim = 1; dim < kDimensions; ++dim) {
                                        requireClose(
                                            "Phantom central force has no transverse component",
                                            data.sinkAccelerations[sink][dim] -
                                                acceleration[i][dim],
                                            0.0, 1.0e-6);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // No h/eta regularization is present in Phantom's force or approach-speed CFL bound.
    fixture.params_.expansionRate = 0.0f;
    data.sourceVelocityX[fixture.sourceIndex(0)] = 0.1f;
    data.sourceVelocityX[fixture.sourceIndex(1)] = -0.1f;
    fixture.params_.viscosityEpsilon = 0.0f;
    fixture.hydro(1);
    float const force = data.sinkAccelerations[fixture.sinkIndex(0)][0];
    double const inverseTimestep = fixture.reduction_->inverseViscosityTimestep;
    fixture.params_.viscosityEpsilon = 100.0f;
    fixture.hydro(1);
    requireClose("Phantom force ignores viscosityEpsilon",
        data.sinkAccelerations[fixture.sinkIndex(0)][0], force, 0.0);
    requireClose("Phantom CFL ignores viscosityEpsilon",
        fixture.reduction_->inverseViscosityTimestep, inverseTimestep, 0.0);
    require(inverseTimestep > 0.0, "Phantom CFL includes approaching velocity");
}

}    // namespace detail
