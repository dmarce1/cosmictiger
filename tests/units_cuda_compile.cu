/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <type_traits>

#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/units.hpp>

// Only the host boundary is typed; the kernel payload retains the original Position.
double unitsFixedPositionHostCompileRegression(
    Position position, units::UnitScales const& scales, units::CosmologicalContext const& context) {
    units::ComovingPosition<Position> const wrapped(position);
    static_assert(std::is_same_v<decltype(wrapped.storage()), Position const&>);
    return units::toProperLength(wrapped, scales, context).quantity().value();
}

__global__ void unitsFixedPositionCudaCompileRegression(Position const* positions,
    double lengthScale, double* coordinates, PositionInteger* rawValues) {
    coordinates[0] = positions[0].toDouble() * lengthScale;
    rawValues[0] = positions[0].raw();
}
