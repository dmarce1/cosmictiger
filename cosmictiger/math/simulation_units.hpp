/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE for details.
 */

#pragma once

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/units.hpp>

namespace units {

/// Construct validated physical unit scales from the configured CGS length, mass, and time
/// units.
inline UnitScales scalesFromOptions(Options const& options) {
    return UnitScales(options.units.lengthCm, options.units.massG, options.units.timeSeconds);
}

/// Express 100 km/s/Mpc in the configured inverse simulation-time units.
inline double normalizedHubble100(Options const& options) {
    return (kHubble100 * Time::from_value(options.units.timeSeconds)).value();
}

}    // namespace units
