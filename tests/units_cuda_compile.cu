/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <type_traits>

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/units.hpp>

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
