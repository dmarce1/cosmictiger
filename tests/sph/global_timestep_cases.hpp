/* CosmicTiger - shared CPU/CUDA global timestep regressions; BSL-1.0. */
#pragma once

#include <cmath>
#include <initializer_list>
#include <stdexcept>

// Exercise the production hydro kernel on an isolated particle. The prescribed
// ceiling is deliberately not a rung interval, and the old rung would allow
// only a much smaller step under the block-step growth restriction.
namespace detail {

template <typename Fixture>
void testGlobalRawTimestep() {
    Fixture fixture(1);
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    int const source = fixture.sourceIndex(0);
    int const sink = fixture.sinkIndex(0);
    int const particle = fixture.particleIndex(0);
    data.sourcePositionX[source] = data.sourcePositionY[source] = data.sourcePositionZ[source] =
        0.5;
    data.sourceInternalEnergy[source] = 1.0e-12f;
    data.sinkSmoothingConverged[sink] = true;
    parameters.minimumRung = 0;
    parameters.globalTimestep = true;
    parameters.maximumTimestep = 0.073123f;
    data.selfGravityEnabled = false;
    fixture.density();
    double const ceiling = parameters.maximumTimestep;
    for (int const oldRung : {0, 20}) {
        data.sourceRungs[source] = data.sinkParticleRungs[particle] = oldRung;
        fixture.hydro(1);
        if (std::abs(fixture.reduction_->minimumTimestep - ceiling) > 1.0e-8 ||
            data.sinkParticleRungs[particle] != 0 || fixture.reduction_->maximumRung != 0) {
            throw std::runtime_error("Global hydro ceiling was rounded or limited by an old rung");
        }
    }
#if COSMICTIGER_WITH_GRAVITY
    // Make self-gravity more restrictive than hydro acceleration or sound speed.
    data.selfGravityEnabled = true;
    data.gravityEffectiveRadius = 1.0e-4f;
    data.sinkGravityAccelerationX[sink] = 3.0f;
    fixture.hydro(1);
    double const expected = data.accelerationTimestepFactor *
        std::sqrt(parameters.scaleFactor * double(data.gravityEffectiveRadius) / 3.0);
    if (std::abs(fixture.reduction_->minimumTimestep - expected) > 5.0e-7 * expected) {
        throw std::runtime_error("Global hydro did not retain the raw self-gravity timestep");
    }
    // A disabled gravity capability must not keep the stale acceleration bound.
    data.selfGravityEnabled = false;
    fixture.hydro(1);
    if (std::abs(fixture.reduction_->minimumTimestep - ceiling) > 1.0e-8) {
        throw std::runtime_error("Disabled gravity constrained the global hydro timestep");
    }
#endif
}

}    // namespace detail
