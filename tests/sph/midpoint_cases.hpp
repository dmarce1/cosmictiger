/* CosmicTiger - shared CPU/CUDA midpoint SPH regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>

namespace detail {

inline void midpointRequire(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fixture>
void requireAffineBalsaraFactor(Fixture& fixture, int flow) {
    // These affine fields have (|trace V|, |curl V|) = (1,10), (0,4), or (3,0).
    // The common a² normalization cancels. Check boundary particles too, where
    // uncorrected kernel-weighted derivatives need not recover the affine field.
    float const expected = flow == 2 ? 1.0f / 11.0f : flow == 3 ? 0.0f : 1.0f;
    for (int index = 0; index < fixture.count_; ++index) {
        midpointRequire(fixture.data_.sinkBalsaraFactor[fixture.sinkIndex(index)] ==
                balsara::encodeFactor(expected),
            "gradient-switch Balsara factor did not use the reconstructed affine gradient");
    }
}

template <typename Fixture>
void testGradientBalsaraPreservation(Fixture& fixture) {
    auto& data = fixture.data_;
    auto* diagnostics = data.sinkViscosityDiagnostics;
    data.sinkViscosityDiagnostics = nullptr;
    for (bool const midpoint : {false, true}) {
        fixture.params_.midpointHydro = midpoint;
        for (int index = 0; index < fixture.count_; ++index) {
            data.sinkBalsaraFactor[fixture.sinkIndex(index)] = uint8_t(17 + index);
        }
        fixture.densityPass();
        for (int index = 0; index < fixture.count_; ++index) {
            midpointRequire(data.sinkBalsaraFactor[fixture.sinkIndex(index)] == uint8_t(17 + index),
                "midpoint or diagnostic pass replaced the endpoint Balsara byte");
        }
    }
    data.sinkViscosityDiagnostics = diagnostics;
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
    midpointRequire(data.sinkAccelerations[fixture.sinkIndex(2)][0] == -123.0f &&
            data.sinkEntropyRate[fixture.sinkIndex(2)] == -123.0f &&
            data.sinkFlags[fixture.sinkIndex(2)].proposedRung() == kMaxRung - 1,
        "midpoint neighbor limiter updated a target above the selected rung");
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
