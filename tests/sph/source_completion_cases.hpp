/* CosmicTiger - shared CPU/CUDA deferred source regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>

namespace detail {

// deferAccelerationSources keeps the gravity buffer unread while a pair pass is in flight.
// The pair and external kicks are the force outputs production integrates; neither depends
// on the deferral, and neither carries self-gravity, which the gravity kick integrates.
// Only a closing (non-midpoint) pass reads the gravity buffer at all, so the deferral guard
// is exercised by a closing pass over null buffers, observed through the viscous heating.
template <typename Fixture>
void testDeferredSourceCompletion() {
    Fixture fixture(2);
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableKickObservation();
    parameters.minimumSmoothingLength = parameters.maximumSmoothingLength = 0.35f;
    parameters.uniformAccelerationY = 0.375f;
    parameters.zPlaneAcceleration = 0.25f;
    parameters.velocityDampingRate = 0.125f;
    parameters.expansionRate = 0.2f;
    data.centralAcceleration = 0.003f;
    data.centralSmoothingLength = 0.2f;
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
    std::array<sph_geometry::SpatialVector<float>, 2> expectedAcceleration;
    std::array<double, 2> expectedHeating;
    for (int i = 0; i < 2; ++i) {
        expectedAcceleration[i] = fixture.kickAcceleration(i);
        expectedHeating[i] = fixture.dissipativeHeating(i);
    }
    // The kicks and the viscous heating never depend on the deferral.
    parameters.deferAccelerationSources = true;
    data.sinkGravityAccelerationX = nullptr;
    data.sinkGravityAccelerationY = nullptr;
    data.sinkGravityAccelerationZ = nullptr;
    fixture.hydro(0);
    // The one pair and the external kicks deposit the same impulses in the same order.
    for (int i = 0; i < 2; ++i) {
        auto const acceleration = fixture.kickAcceleration(i);
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            if (acceleration[dim] != expectedAcceleration[i][dim]) {
                throw std::runtime_error("Deferred SPH sources changed the pair or external kicks");
            }
        }
        if (fixture.dissipativeHeating(i) != expectedHeating[i]) {
            throw std::runtime_error("Deferred SPH sources changed the viscous heating");
        }
    }
    // A midpoint pass never reads the gravity buffer; only a closing pass does, and only
    // when the sources are not deferred. No gravity buffer is readable while the pair pass
    // is in flight: a stale field read must fail, even if a later correction could hide its
    // value. The closing pass completes over the null buffers and sums the same heating.
    fixture.disableKickObservation();
    fixture.hydro(0);
    for (int i = 0; i < 2; ++i) {
        if (fixture.dissipativeHeating(i) != expectedHeating[i]) {
            throw std::runtime_error("Deferred closing SPH pass changed the viscous heating");
        }
    }
}

}    // namespace detail
