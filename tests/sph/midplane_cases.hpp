// Dimensional imposed-midplane force regressions for shared CPU/CUDA completion.
// Copyright (C) 2026 Dominic C. Marcello; BSL-1.0 (see LICENSE_1_0.txt).
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>

namespace detail {

inline void testRayleighTaylorMidplaneAcceleration() {
    if (spatialDimensions() < 2) {
        throw std::invalid_argument("Rayleigh-Taylor force requires at least two dimensions");
    }
    int const verticalAxis = spatialDimensions() - 1;
    SphRunParameters parameters;
    parameters.uniformAccelerationY = 0.0f;
    parameters.zPlaneAcceleration = 0.5f;
    parameters.velocityDampingRate = 0.0f;
    PositionInteger const middle = Position(0.5).raw();
    std::array const coordinates = {PositionInteger(0), PositionInteger(1), Position(0.125).raw(),
        PositionInteger(middle - 1), middle, PositionInteger(middle + 1), Position(0.875).raw(),
        std::numeric_limits<PositionInteger>::max()};
    std::array const expected = {0.5f, 0.5f, 0.5f, 0.5f, 0.0f, -0.5f, -0.5f, -0.5f};
    auto const check = [](float actual, float expectedValue, char const* label) {
        if (actual != expectedValue) {
            throw std::runtime_error(label);
        }
    };
    auto const force = [&](Vector3<Position> const& position) {
        sph_flops::Counter flops;
        return sphCompleteAcceleration(parameters, position,
            sph_geometry::SpatialVector<float>(0.0f), sph_geometry::SpatialVector<float>(0.0f),
            sph_geometry::SpatialVector<float>(0.0f), 0.0f, 1.0f, flops);
    };
    for (size_t index = 0; index < coordinates.size(); ++index) {
        // Deliberately place the inactive z padding above the midplane in 2D, so selecting
        // the old z coordinate would either reverse or entirely lose the intended force.
        Vector3<Position> position(Position(0.75));
        position[verticalAxis].raw() = coordinates[index];
        auto const acceleration = force(position);
        sph_flops::Counter flops;
        auto const paddedAcceleration = sphCompleteAcceleration(parameters, position,
            Vector3<float>(0.0f), Vector3<float>(0.0f), Vector3<float>(0.0f), 0.0f, 1.0f, flops);
        for (int dim = 0; dim < kDimensions; ++dim) {
            float const expectedComponent = dim == verticalAxis ? expected[index] : 0.0f;
            check(paddedAcceleration[dim], expectedComponent,
                "Midplane force used the wrong coordinate or padded component");
            if (dim < spatialDimensions()) {
                check(acceleration[dim], expectedComponent,
                    "Spatial force completion lost the imposed midplane force");
            }
        }
        check(acceleration.norm(), std::abs(expected[index]),
            "Acceleration timestep input has the wrong imposed-force magnitude");
    }

    Vector3<Position> position(Position(0.75));
    parameters.zPlaneAcceleration = 0.0f;
    check(force(position).norm(), 0.0f, "Zero magnitude must disable the midplane force");
    parameters.zPlaneAcceleration = 0.5f;
    parameters.uniformAccelerationY = 0.25f;
    auto const combined = force(position);
    for (int dim = 0; dim < spatialDimensions(); ++dim) {
        float const expectedComponent =
            (dim == kYi ? 0.25f : 0.0f) + (dim == verticalAxis ? -0.5f : 0.0f);
        check(combined[dim], expectedComponent,
            "Uniform y acceleration and midplane acceleration must add independently");
    }
}

}    // namespace detail
