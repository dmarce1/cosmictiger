/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the GNU General Public License, version 2 or later.
 See LICENSE for details.
 */

#pragma once

#include <cosmictiger/options.hpp>
#include <cosmictiger/units.hpp>

namespace units {

inline UnitScales scalesFromOptions(Options const& options) {
    return UnitScales(options.codeToCm, options.codeToG, options.codeToS);
}

inline double normalizedHubble100(Options const& options) {
    return (kHubble100 * Time::from_value(options.codeToS)).value();
}

}    // namespace units
