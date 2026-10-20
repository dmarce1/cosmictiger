/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <type_traits>

#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/units.hpp>

// Only the host boundary is typed; the kernel payload retains the original Fixed32.
double unitsFixedPositionHostCompileRegression(
    Fixed32 position, units::UnitScales const& scales, units::CosmologicalContext const& context) {
    units::ComovingPosition<Fixed32> const wrapped(position);
    static_assert(std::is_same_v<decltype(wrapped.storage()), Fixed32 const&>);
    return units::toProperLength(wrapped, scales, context).quantity().value();
}

__global__ void unitsFixedPositionCudaCompileRegression(
    Fixed32 const* positions, double lengthScale, double* coordinates, uint32_t* rawValues) {
    coordinates[0] = positions[0].toDouble() * lengthScale;
    rawValues[0] = positions[0].raw();
}
