/* CosmicTiger - shared CPU/CUDA deferred source regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>

namespace detail {

template <typename Fixture>
void testDeferredSourceCompletion() {
    Fixture fixture(2);
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    parameters.minimumSupportRadius = parameters.maximumSupportRadius = 0.35f;
    parameters.uniformAccelerationY = 0.375f;
    parameters.zPlaneAcceleration = 0.25f;
    parameters.velocityDampingRate = 0.125f;
    parameters.expansionRate = 0.2f;
    data.centralAcceleration = 0.003f;
    data.centralSupportRadius = 0.2f;
    data.selfGravityEnabled = true;
    std::array<Vector3<float>, 2> gravity;
    for (int i = 0; i < 2; ++i) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = data.sourcePositionZ[source] = 0.45 + 0.1 * i;
        data.sourcePositionY[source] = 0.5;
        data.sourceVelocityX[source] = i == 0 ? 0.2f : -0.2f;
        data.sourceVelocityY[source] = 0.1f;
        data.sourceVelocityZ[source] = -0.3f;
        data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sink] = 0.35f;
        gravity[i] = Vector3<float>(3.0f + i, -4.0f + i, 2.0f - i);
        data.sinkGravityAccelerationX[sink] = gravity[i][0];
        data.sinkGravityAccelerationY[sink] = gravity[i][1];
        data.sinkGravityAccelerationZ[sink] = gravity[i][2];
    }
    fixture.density();
    fixture.hydro(0);
    std::array<Vector3<float>, 2> expectedAcceleration;
    std::array<float, 2> expectedHeating;
    for (int i = 0; i < 2; ++i) {
        int const sink = fixture.sinkIndex(i);
        expectedAcceleration[i] = data.sinkAccelerations[sink];
        expectedHeating[i] = data.sinkInternalEnergyRate[sink];
    }
    // No gravity buffer is readable while the pair pass is in flight. A stale
    // field read must fail, even if a later correction could hide its value.
    parameters.deferAccelerationSources = true;
    data.sinkGravityAccelerationX = nullptr;
    data.sinkGravityAccelerationY = nullptr;
    data.sinkGravityAccelerationZ = nullptr;
    fixture.hydro(0);
    for (int i = 0; i < 2; ++i) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        sph_flops::Counter flops;
        auto const completed = sphCompleteAcceleration(parameters,
            Vector3<Position>(data.sourcePositionX[source], data.sourcePositionY[source],
                data.sourcePositionZ[source]),
            Vector3<float>(data.sourceVelocityX[source], data.sourceVelocityY[source],
                data.sourceVelocityZ[source]),
            data.sinkAccelerations[sink], gravity[i], data.centralAcceleration,
            data.centralSupportRadius, flops);
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (std::abs(completed[dim] - expectedAcceleration[i][dim]) >
                1.0e-6f * std::max(1.0f, std::abs(expectedAcceleration[i][dim]))) {
                throw std::runtime_error("Deferred SPH source assembly differs from serial force");
            }
        }
        if (data.sinkInternalEnergyRate[sink] != expectedHeating[i]) {
            throw std::runtime_error("Deferred SPH sources changed pressure or viscous work");
        }
    }
}

}    // namespace detail
