/* CosmicTiger - shared CPU/CUDA midpoint SPH regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cosmictiger/sph/run_data.hpp>
#include <cosmictiger/sph/thermodynamics.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <tests/sph/pair_kick_cases.hpp>

namespace detail {

inline void midpointRequire(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fixture>
void testFineNeighborsOfCoarseParticle() {
    Fixture fixture(4, 4);
    fixture.enableMidpointOutputs();
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    parameters.midpointHydro = true;
    parameters.minimumRung = parameters.maximumRung = 2;
    parameters.minimumProposedRung = 0;
    parameters.baseTimestep = 1.0e-6f;
    parameters.hydroKickTimestep = 1.0e-5;
    parameters.scaleFactor = 0.7f;
    parameters.expansionRate = 0.03f;
    parameters.balsaraSwitch = false;
    parameters.variableViscosity = false;
    parameters.uniformAccelerationY = 0.0f;
    parameters.zPlaneAcceleration = 0.0f;
    parameters.velocityDampingRate = 0.0f;
    data.selfGravityEnabled = false;
    data.centralAcceleration = 0.0f;
    auto* identities = fixture.enablePairKickOutputs();
    std::array<std::array<float, 3>, 4> const positions{{{0.50f, 0.50f, 0.50f},
        {0.46f, 0.49f, 0.50f}, {0.51f, 0.55f, 0.49f}, {0.49f, 0.50f, 0.44f}}};
    std::array<std::array<float, 3>, 4> const velocities{{{0.03f, -0.02f, 0.01f},
        {0.20f, 0.01f, 0.00f}, {0.00f, -0.10f, 0.05f}, {-0.02f, 0.00f, 0.15f}}};
    std::array<float, 4> const energies{0.8f, 1.0f, 1.3f, 0.9f};
    std::array<double, 4> entropy;
    float* velocity[] = {data.sourceVelocityX, data.sourceVelocityY, data.sourceVelocityZ};
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        data.sourcePositionX[source] = positions[index][0];
        data.sourcePositionY[source] = positions[index][1];
        data.sourcePositionZ[source] = positions[index][2];
        data.sourceDensity[source] = 1.0f + 0.2f * index;
        data.sourceGradHCorrection[source] = 0.0f;
        data.sourceSmoothingLength[source] = 0.2f;
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(index)] =
            index == 0 ? 1 : 2;
        identities[source] = {index % 3, 27 - index};
        entropy[index] = sph_entropy::fromStoredEnergy(energies[index], data.sourceDensity[source],
            data.adiabaticIndex, parameters.scaleFactor);
    }
    std::array<SphPairUpdate, 4> reference;
    std::array<int, 4> order{0, 1, 2, 3};
    bool firstOrdering = true;
    do {
        for (int node = 0; node < 4; ++node) {
            data.targetNodeIndices[node] = order[node];
            data.neighborNodeIndices[node] = order[3 - node];
        }
        for (int index = 0; index < 4; ++index) {
            int const source = fixture.sourceIndex(index);
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                velocity[dimension][source] = velocities[index][dimension];
            }
            data.sourceInternalEnergy[source] = energies[index];
            data.sourcePairUpdates[source] = {};
        }
        // All fine targets read the same initial coarse state. Only after every
        // fine-owned pair has deposited do both endpoints enter the predictor.
        fixture.hydro(1);
        bool coarseMoved = false;
        for (int index = 0; index < 4; ++index) {
            int const source = fixture.sourceIndex(index);
            auto const predictor = data.sourcePairUpdates[source];
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                coarseMoved = coarseMoved || (index == 0 && predictor.velocity[dimension] != 0.0);
                velocity[dimension][source] = float(
                    double(velocities[index][dimension]) + 0.5 * predictor.velocity[dimension]);
            }
            data.sourceInternalEnergy[source] =
                float(sph_entropy::storedEnergy(entropy[index] + 0.5 * predictor.entropy,
                    data.sourceDensity[source], data.adiabaticIndex, parameters.scaleFactor));
            data.sourcePairUpdates[source] = {};
        }
        midpointRequire(
            coarseMoved, "fine neighbors did not predict their inactive coarse endpoint");
        fixture.hydro(1);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            double total = 0.0;
            double absolute = 0.0;
            for (int index = 0; index < 4; ++index) {
                double const increment =
                    data.sourcePairUpdates[fixture.sourceIndex(index)].velocity[dimension];
                total += increment;
                absolute += std::abs(increment);
            }
            // Opposite impulses are identical; independent endpoint sums round
            // to float. Kahan bounds summation error at O(epsilon_float), not
            // epsilon_double, including cancellation among several neighbors.
            midpointRequire(
                std::abs(total) <= 4.0 * std::numeric_limits<float>::epsilon() * absolute,
                "multiple fine neighbors lost reciprocal coarse recoil");
        }
        for (int index = 0; index < 4; ++index) {
            auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
            if (firstOrdering) {
                reference[index] = update;
            } else {
                double const entropyScale =
                    std::max(1.0e-12f, std::abs(float(reference[index].entropy)));
                midpointRequire(std::abs(update.entropy - reference[index].entropy) <=
                        32.0 * std::numeric_limits<float>::epsilon() * entropyScale,
                    "fine-neighbor evaluation order changed corrected entropy");
                for (int dimension = 0; dimension < kDimensions; ++dimension) {
                    double scale = 1.0e-12;
                    for (int particle = 0; particle < 4; ++particle) {
                        scale += std::abs(reference[particle].velocity[dimension]);
                    }
                    midpointRequire(std::abs(update.velocity[dimension] -
                                        reference[index].velocity[dimension]) <=
                            32.0 * std::numeric_limits<float>::epsilon() * scale,
                        "fine-neighbor evaluation order changed the corrected pair impulse");
                }
            }
            midpointRequire(
                data.sinkParticleRungs[fixture.particleIndex(index)] == (index == 0 ? 1 : 2),
                "shared-coarse correction changed an interval's live rung");
        }
        firstOrdering = false;
    } while (std::next_permutation(order.begin(), order.end()));
}

template <typename Fixture>
void testMidpointHydroTargets(Fixture& fixture) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    fixture.enableKickObservation();
    parameters.minimumRung = 0;
    parameters.minimumProposedRung = 0;
    parameters.baseTimestep = 1.0e-6f;
    data.selfGravityEnabled = true;
    // Neither a gravity buffer nor a valid gravity timestep scale is available
    // at a hydro midpoint. The independent gravity operator owns that criterion.
    data.sinkGravityAccelerationX = data.sinkGravityAccelerationY = data.sinkGravityAccelerationZ =
        nullptr;
    data.gravityEffectiveRadius = 0.0f;
    std::array<float, 4> const velocities{0.3f, -0.2f, 0.1f, -0.4f};
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        data.sourcePositionX[source] = 0.45 + 0.03 * index;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceVelocityX[source] = velocities[index];
        data.sourceInternalEnergy[source] = 0.8f + 0.1f * index;
        data.sinkFlags[sink].setConverged(true);
    }
    fixture.densityPass();
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        midpointRequire(data.sinkFlags[sink].proposedRung() == kMaxRung - 1,
            "density updates changed the existing hydro rung proposal");
        data.sourceDensity[source] = data.sinkDensity[sink];
        data.sourceGradHCorrection[source] = data.sinkGradHCorrection[sink];
    }
    fixture.hydro(0);
    std::array<sph_geometry::SpatialVector<float>, 4> referenceAcceleration;
    std::array<float, 4> referenceEntropyRate;
    std::array<int, 4> const rungs{1, 2, 1, 2};
    double impulseScale = 0.0;
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        referenceAcceleration[index] = fixture.kickAcceleration(index);
        referenceEntropyRate[index] = data.sinkEntropyRate[sink];
        for (int dimension = 0; dimension < kSpatialDimensions; ++dimension) {
            impulseScale += std::abs(referenceAcceleration[index][dimension]);
        }
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(index)] =
            rungs[index];
        data.sinkEntropyRate[sink] = -123.0f;
    }
    parameters.minimumRung = parameters.maximumRung = 2;
    fixture.hydro(1);
    midpointRequire(fixture.reduction_->maximumGravityRung == 0,
        "midpoint hydro published a gravity timestep criterion");
    midpointRequire(referenceEntropyRate[1] > 0.0f && referenceEntropyRate[3] > 0.0f,
        "midpoint fixture did not exercise viscous entropy generation");
    // Every pair of an active target is kicked once, by the finer endpoint or the lower
    // identity, so active targets recover the complete force while inactive neighbors receive
    // only their recoil. Compensated sums deposited in another order agree to float rounding.
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        bool const active = rungs[index] == 2;
        if (active) {
            auto const acceleration = fixture.kickAcceleration(index);
            for (int dimension = 0; dimension < kSpatialDimensions; ++dimension) {
                midpointRequire(
                    std::abs(acceleration[dimension] - referenceAcceleration[index][dimension]) <=
                        32.0 * std::numeric_limits<float>::epsilon() * impulseScale,
                    "exact-rung midpoint force lost neighbors");
            }
        }
        midpointRequire(
            data.sinkEntropyRate[sink] == (active ? referenceEntropyRate[index] : -123.0f),
            "midpoint entropy rate ignored exact-rung selection or updated an inactive target");
        midpointRequire(data.sinkFlags[sink].proposedRung() == (active ? 1 : kMaxRung - 1),
            "midpoint selection prevented coarsening or proposed an inactive rung");
        midpointRequire(data.sinkFlags[sink].converged() && data.sinkFlags[sink].semiactive(),
            "hydro rung proposals changed the packed density flags");
        midpointRequire(data.sinkParticleRungs[fixture.particleIndex(index)] == rungs[index],
            "midpoint force changed a live rung before its gravity boundary");
        midpointRequire(data.sourceVelocityX[source] == velocities[index] &&
                data.sourceInternalEnergy[source] == 0.8f + 0.1f * index,
            "midpoint force changed the frozen source state");
    }
    for (int dimension = 0; dimension < kSpatialDimensions; ++dimension) {
        double total = 0.0;
        double absolute = 0.0;
        for (int index = 0; index < 4; ++index) {
            double const increment = compensatedKickSum(
                data.sourcePairUpdates[fixture.sourceIndex(index)].velocity[dimension]);
            total += increment;
            absolute += std::abs(increment);
        }
        midpointRequire(std::abs(total) <= 4.0 * std::numeric_limits<float>::epsilon() * absolute,
            "fine-owned pairs lost the inactive endpoints' recoil");
    }
    // A faster touching source constrains the proposal without becoming an H target.
    data.sourceRungs[fixture.sourceIndex(2)] = data.sinkParticleRungs[fixture.particleIndex(2)] = 5;
    fixture.hydro(1);
    for (int index : {1, 3}) {
        midpointRequire(data.sinkFlags[fixture.sinkIndex(index)].proposedRung() ==
                5 - kMaxNeighborRungDifference,
            "midpoint proposal omitted a faster touching neighbor");
        midpointRequire(data.sinkParticleRungs[fixture.particleIndex(index)] == 2,
            "neighbor timestep bound committed before the gravity boundary");
    }
    midpointRequire(data.sinkEntropyRate[fixture.sinkIndex(2)] == -123.0f &&
            data.sinkFlags[fixture.sinkIndex(2)].proposedRung() == kMaxRung - 1,
        "midpoint neighbor limiter updated a target above the selected rung");
    testFineNeighborsOfCoarseParticle<Fixture>();
}

template <typename Fixture>
void testEndpointMorrisInterval(Fixture& fixture) {
    fixture.enableVariableViscosity();
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    parameters.midpointHydro = true;
    parameters.minimumRung = 3;
    parameters.maximumRung = kMaxRung - 1;
    for (int index = 0; index < 3; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        data.sourcePositionX[source] = 0.45 + 0.05 * index;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceInternalEnergy[source] = 0.9f;
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(index)] = index + 2;
        data.sinkFlags[sink].setConverged(false);
        data.sinkFlags[sink].setProposedRung(index + 4);
    }
    fixture.smoothingLengthsPass();
    for (int index = 0; index < 3; ++index) {
        midpointRequire(data.sinkFlags[fixture.sinkIndex(index)].converged() == (index >= 1),
            "endpoint support solve ignored the closing-rung range");
        midpointRequire(data.sinkFlags[fixture.sinkIndex(index)].proposedRung() == index + 4,
            "support convergence changed the packed hydro rung");
    }
    for (bool const global : {false, true}) {
        parameters.globalTimestep = global;
        parameters.viscosityTimestep = 0.08;
        for (int index = 0; index < 3; ++index) {
            int const sink = fixture.sinkIndex(index);
            data.sinkBulkViscosityCoefficient[sink] = 0.7f;
            data.sinkFlags[sink].setConverged(false);
        }
        fixture.densityPass();
        std::array<float, 3> coefficients;
        for (int index = 0; index < 3; ++index) {
            int const sink = fixture.sinkIndex(index);
            int const source = fixture.sourceIndex(index);
            int const rung = data.sinkParticleRungs[fixture.particleIndex(index)];
            double const soundSpeed = std::sqrt(double(data.adiabaticIndex) *
                (data.adiabaticIndex - 1.0) * double(data.sourceInternalEnergy[source]));
            double const smoothingLength = data.sinkSmoothingLength[sink] /
                selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                monaghan::kViscosityWidthScale;
            double const inverseDecayTime =
                parameters.viscosityDecay * soundSpeed / (parameters.scaleFactor * smoothingLength);
            double const completedInterval = global ?
                parameters.viscosityTimestep :
                std::ldexp(parameters.viscosityTimestep, -rung);
            double const expected = index == 0 ? double(float(Float16Storage(0.7f))) :
                                                 parameters.minimumBulkViscosityCoefficient +
                    (double(float(Float16Storage(0.7f))) -
                        parameters.minimumBulkViscosityCoefficient) /
                        (1.0 + completedInterval * inverseDecayTime);
            midpointRequire(std::abs(data.sinkBulkViscosityCoefficient[sink] -
                                float(Float16Storage(float(expected)))) < 2.0e-6,
                "endpoint M&M used the wrong completed interval or advanced inactive gas");
            midpointRequire(
                data.sinkFlags[sink].semiactive(), "endpoint fixture lost a touching neighbor");
            midpointRequire(data.sinkFlags[sink].proposedRung() == index + 4,
                "endpoint density flags changed the packed hydro rung");
            midpointRequire(data.sourceBulkViscosityCoefficient[source] == 0.7f,
                "endpoint density mutated a staged viscosity source");
            coefficients[index] = data.sinkBulkViscosityCoefficient[sink];
        }
        fixture.densityPass();
        for (int index = 0; index < 3; ++index) {
            midpointRequire(
                data.sinkBulkViscosityCoefficient[fixture.sinkIndex(index)] == coefficients[index],
                "repeated endpoint density pass advanced viscosity twice");
            data.sinkFlags[fixture.sinkIndex(index)].setConverged(false);
        }
        parameters.viscosityTimestep = 0.0;
        fixture.densityPass();
        for (int index = 0; index < 3; ++index) {
            midpointRequire(
                data.sinkBulkViscosityCoefficient[fixture.sinkIndex(index)] == coefficients[index],
                "midpoint or diagnostic density pass advanced endpoint viscosity");
        }
    }
}

}    // namespace detail
