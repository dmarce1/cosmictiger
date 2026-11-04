/* CosmicTiger - shared CPU/CUDA conservative pair regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>

namespace detail {

inline void pairKickRequire(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void pairKickClose(double actual, double expected, char const* message) {
    pairKickRequire(
        std::abs(actual - expected) <= 2.0e-6 * std::max(std::abs(expected), 1.0e-20), message);
}

template <class Fixture>
void testConservativePairKick(Fixture& fixture, bool directedViscosity) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    parameters.midpointHydro = true;
    parameters.minimumRung = 0;
    parameters.minimumProposedRung = 0;
    parameters.baseTimestep = 1.0e-5f;
    parameters.scaleFactor = 0.7f;
    parameters.expansionRate = 0.03f;
    data.selfGravityEnabled = false;
    if (directedViscosity) {
        fixture.enableVariableViscosity(false, true);
    }
    std::array<std::array<float, 3>, 2> const velocities{
        {{0.2f, 0.07f, 0.04f}, {-0.1f, -0.03f, -0.06f}}};
    for (int index = 0; index < 2; ++index) {
        int const source = fixture.sourceIndex(index);
        data.sourcePositionX[source] = 0.47 + 0.06 * index;
        data.sourcePositionY[source] = 0.49 + 0.03 * index;
        data.sourcePositionZ[source] = 0.48 + 0.02 * index;
        data.sourceVelocityX[source] = velocities[index][0];
        data.sourceVelocityY[source] = velocities[index][1];
        data.sourceVelocityZ[source] = velocities[index][2];
        data.sourceDensity[source] = 1.3f + 0.8f * index;
        data.sourceInternalEnergy[source] = 0.8f + 0.6f * index;
        data.sourceSmoothingLength[source] = 0.2f + 0.05f * index;
        data.sourceGradHCorrection[source] = 0.0f;    // Fixed-support pair.
        if (directedViscosity) {
            data.sourceBulkViscosityCoefficient[source] = 0.4f + 0.5f * index;
        }
    }
    // The ordinary complete-target RHS is an independent reference for both
    // sides' force and heating, including unequal directed Price q terms.
    fixture.hydro(0);
    std::array<Vector3<float>, 2> acceleration;
    std::array<float, 2> entropyRate;
    for (int index = 0; index < 2; ++index) {
        acceleration[index] = data.sinkAccelerations[fixture.sinkIndex(index)];
        entropyRate[index] = data.sinkEntropyRate[fixture.sinkIndex(index)];
        pairKickRequire(entropyRate[index] > 0.0f, "pair fixture missed viscous heating");
    }
    if (directedViscosity) {
        double const firstHeating =
            sph_entropy::storedEnergy(entropyRate[0], data.sourceDensity[fixture.sourceIndex(0)],
                data.adiabaticIndex, parameters.scaleFactor);
        double const secondHeating =
            sph_entropy::storedEnergy(entropyRate[1], data.sourceDensity[fixture.sourceIndex(1)],
                data.adiabaticIndex, parameters.scaleFactor);
        pairKickRequire(std::abs(firstHeating / secondHeating - 1.0) > 0.05,
            "Price fixture did not exercise different endpoint heating");
    }
    auto* identities = fixture.enablePairKickOutputs();
    identities[fixture.sourceIndex(0)] = {1, 37};
    identities[fixture.sourceIndex(1)] = {0, 92};
    data.sourceRungs[fixture.sourceIndex(0)] = data.sinkParticleRungs[fixture.particleIndex(0)] = 3;
    data.sourceRungs[fixture.sourceIndex(1)] = data.sinkParticleRungs[fixture.particleIndex(1)] = 1;
    auto clearUpdates = [&] {
        for (int index = 0; index < 2; ++index) {
            data.sourcePairUpdates[fixture.sourceIndex(index)] = {};
        }
    };
    auto requireZero = [&] {
        for (int index = 0; index < 2; ++index) {
            auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
            pairKickRequire(update.entropy == 0.0, "inactive pair generated entropy");
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                pairKickRequire(update.velocity[dimension] == 0.0,
                    "pair was integrated on its coarser endpoint's clock");
            }
        }
    };
    parameters.hydroKickTimestep = 0.0005;
    parameters.minimumRung = parameters.maximumRung = 1;
    fixture.hydro(1);
    requireZero();
    parameters.minimumRung = parameters.maximumRung = 3;
    fixture.hydro(1);
    for (int index = 0; index < 2; ++index) {
        auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
        pairKickClose(update.entropy, parameters.hydroKickTimestep * entropyRate[index],
            "pair kick lost an endpoint's viscous heating");
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            pairKickClose(update.velocity[dimension],
                parameters.hydroKickTimestep * acceleration[index][dimension],
                "fine-owned pair impulse differs from the complete pair force");
        }
    }
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        double const firstIncrement =
            data.sourcePairUpdates[fixture.sourceIndex(0)].velocity[dimension];
        double const secondIncrement =
            data.sourcePairUpdates[fixture.sourceIndex(1)].velocity[dimension];
        pairKickRequire(firstIncrement == -secondIncrement,
            "pair endpoints did not receive the identical opposite impulse");
        float const firstVelocity = float(double(velocities[0][dimension]) + firstIncrement);
        float const secondVelocity = float(double(velocities[1][dimension]) + secondIncrement);
        double const momentumChange = double(data.gasMass) *
            (double(firstVelocity) + double(secondVelocity) - double(velocities[0][dimension]) -
                double(velocities[1][dimension]));
        double const roundingBound = 2.0 * std::numeric_limits<float>::epsilon() * data.gasMass *
            (std::abs(double(velocities[0][dimension])) +
                std::abs(double(velocities[1][dimension])) + std::abs(firstIncrement) +
                std::abs(secondIncrement));
        pairKickRequire(std::abs(momentumChange) <= roundingBound,
            "float velocity commits exceeded their momentum roundoff bound");
    }
    // Equal-rung ownership must integrate exactly once across ranks, and must
    // also use the local index to break ties within one rank.
    data.sourceRungs[fixture.sourceIndex(1)] = data.sinkParticleRungs[fixture.particleIndex(1)] = 3;
    for (int ownershipCase = 0; ownershipCase < 2; ++ownershipCase) {
        clearUpdates();
        if (ownershipCase == 1) {
            identities[fixture.sourceIndex(0)] = {0, 37};
            identities[fixture.sourceIndex(1)] = {0, 92};
        }
        fixture.hydro(1);
        for (int index = 0; index < 2; ++index) {
            auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
            pairKickClose(update.entropy, parameters.hydroKickTimestep * entropyRate[index],
                "equal-rung pair heating was counted twice or lost");
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                pairKickClose(update.velocity[dimension],
                    parameters.hydroKickTimestep * acceleration[index][dimension],
                    "equal-rung pair force was counted twice or lost");
            }
        }
    }
    // Production does not need persistent acceleration or entropy-rate output.
    data.sinkAccelerations = nullptr;
    data.sinkEntropyRate = nullptr;
    data.sinkInternalEnergyRate = nullptr;
    clearUpdates();
    parameters.hydroKickTimestep = 0.0;
    fixture.hydro(1);
    requireZero();
    clearUpdates();
    parameters.hydroKickTimestep = 0.0005;
    parameters.minimumRung = parameters.maximumRung = 1;
    data.sourceRungs[fixture.sourceIndex(1)] = data.sinkParticleRungs[fixture.particleIndex(1)] = 1;
    parameters.uniformAccelerationY = 0.4f;
    parameters.velocityDampingRate = 0.125f;
    fixture.hydro(1);
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        pairKickRequire(data.sourcePairUpdates[fixture.sourceIndex(0)].velocity[dimension] == 0.0,
            "external force touched an inactive target");
        float const externalAcceleration = (dimension == 1 ? 0.4f : 0.0f) -
            parameters.velocityDampingRate * velocities[1][dimension];
        pairKickClose(data.sourcePairUpdates[fixture.sourceIndex(1)].velocity[dimension],
            parameters.hydroKickTimestep * externalAcceleration,
            "external force was omitted or counted more than once");
    }
}

}    // namespace detail
