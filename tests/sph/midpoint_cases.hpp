/* CosmicTiger - shared CPU/CUDA midpoint SPH regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>

namespace detail {

inline void midpointRequire(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fixture>
void testMidpointHydroTargets(Fixture& fixture) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    parameters.midpointHydro = true;
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
        data.sinkSmoothingConverged[sink] = true;
    }
    fixture.densityPass();
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        data.sourceDensity[source] = data.sinkDensity[sink];
        data.sourceGradHCorrection[source] = data.sinkGradHCorrection[sink];
    }
    fixture.hydro(0);
    std::array<Vector3<float>, 4> referenceAcceleration;
    std::array<float, 4> referenceEntropyRate;
    std::array<int, 4> const rungs{1, 2, 1, 2};
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        referenceAcceleration[index] = data.sinkAccelerations[sink];
        referenceEntropyRate[index] = data.sinkEntropyRate[sink];
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(index)] =
            rungs[index];
        data.sinkAccelerations[sink] = Vector3<float>(-123.0f);
        data.sinkEntropyRate[sink] = -123.0f;
    }
    parameters.minimumRung = parameters.maximumRung = 2;
    fixture.hydro(1);
    midpointRequire(fixture.reduction_->maximumGravityRung == 0,
        "midpoint hydro published a gravity timestep criterion");
    midpointRequire(referenceEntropyRate[1] > 0.0f && referenceEntropyRate[3] > 0.0f,
        "midpoint fixture did not exercise viscous entropy generation");
    for (int index = 0; index < 4; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        bool const active = rungs[index] == 2;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            midpointRequire(data.sinkAccelerations[sink][dimension] ==
                    (active ? referenceAcceleration[index][dimension] : -123.0f),
                "exact-rung midpoint force lost neighbors or updated an inactive target");
        }
        midpointRequire(
            data.sinkEntropyRate[sink] == (active ? referenceEntropyRate[index] : -123.0f),
            "midpoint entropy rate ignored exact-rung selection");
        midpointRequire(data.sinkProposedRungs[sink] == (active ? 1 : -1),
            "midpoint selection prevented coarsening or proposed an inactive rung");
        midpointRequire(data.sinkParticleRungs[fixture.particleIndex(index)] == rungs[index],
            "midpoint force changed a live rung before its gravity boundary");
        midpointRequire(data.sourceVelocityX[source] == velocities[index] &&
                data.sourceInternalEnergy[source] == 0.8f + 0.1f * index,
            "midpoint force changed the frozen source state");
    }
    // A faster touching source constrains the proposal without becoming an H target.
    data.sourceRungs[fixture.sourceIndex(2)] = data.sinkParticleRungs[fixture.particleIndex(2)] = 5;
    fixture.hydro(1);
    for (int index : {1, 3}) {
        midpointRequire(
            data.sinkProposedRungs[fixture.sinkIndex(index)] == 5 - kMaxNeighborRungDifference,
            "midpoint proposal omitted a faster touching neighbor");
        midpointRequire(data.sinkParticleRungs[fixture.particleIndex(index)] == 2,
            "neighbor timestep bound committed before the gravity boundary");
    }
    midpointRequire(data.sinkAccelerations[fixture.sinkIndex(2)][0] == -123.0f &&
            data.sinkEntropyRate[fixture.sinkIndex(2)] == -123.0f &&
            data.sinkProposedRungs[fixture.sinkIndex(2)] == -1,
        "midpoint neighbor limiter updated a target above the selected rung");
}

template <typename Fixture>
void testMidpointMorrisInterval(Fixture& fixture) {
    fixture.enableVariableViscosity();
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    parameters.midpointHydro = true;
    parameters.minimumRung = parameters.maximumRung = 3;
    parameters.viscosityTimestep = 0.08;
    for (int index = 0; index < 3; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        data.sourcePositionX[source] = 0.45 + 0.05 * index;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceInternalEnergy[source] = 0.9f;
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(index)] = index + 2;
        data.sinkSmoothingConverged[sink] = false;
    }
    fixture.prehydro1Pass();
    for (int index = 0; index < 3; ++index) {
        int const sink = fixture.sinkIndex(index);
        midpointRequire(bool(data.sinkSmoothingConverged[sink]) == (index == 1),
            "midpoint active support solve included another rung");
        data.sinkSmoothingConverged[sink] = false;
    }
    fixture.densityPass();
    int const active = fixture.sinkIndex(1);
    double const soundSpeed = std::sqrt(double(data.adiabaticIndex) * (data.adiabaticIndex - 1.0) *
        double(data.sourceInternalEnergy[fixture.sourceIndex(1)]));
    double const smoothingLength = data.sinkSupportRadius[active] /
        selectedHydroKernel().supportRadiusPerEffectiveRadius * monaghan::kViscosityWidthScale;
    double const inverseDecayTime =
        parameters.viscosityDecay * soundSpeed / (parameters.scaleFactor * smoothingLength);
    double const expected = parameters.minimumBulkViscosityCoefficient +
        (double(float(Float16Storage(0.7f))) - parameters.minimumBulkViscosityCoefficient) /
            (1.0 + parameters.viscosityTimestep * inverseDecayTime);
    midpointRequire(std::abs(data.sinkBulkViscosityCoefficient[active] -
                        float(Float16Storage(float(expected)))) < 2.0e-6,
        "midpoint Morris-Monaghan interval was scaled by its rung twice");
    for (int index = 0; index < 3; ++index) {
        int const sink = fixture.sinkIndex(index);
        midpointRequire(data.sinkSemiactive[sink], "midpoint fixture lost a touching neighbor");
        if (index != 1) {
            midpointRequire(
                float(data.sinkBulkViscosityCoefficient[sink]) == float(Float16Storage(0.7f)),
                "midpoint advanced the viscosity coefficient on another rung");
        }
        midpointRequire(data.sourceBulkViscosityCoefficient[fixture.sourceIndex(index)] == 0.7f,
            "midpoint density mutated a staged viscosity source");
    }
    float const coefficient = data.sinkBulkViscosityCoefficient[active];
    fixture.densityPass();
    midpointRequire(data.sinkBulkViscosityCoefficient[active] == coefficient,
        "repeated midpoint density pass advanced viscosity twice");
    parameters.viscosityTimestep = 0.0;
    data.sinkSmoothingConverged[active] = false;
    fixture.densityPass();
    midpointRequire(data.sinkBulkViscosityCoefficient[active] == coefficient,
        "midpoint timestep estimate advanced viscosity");
}

}    // namespace detail
