// Mirrored Rayleigh-Taylor initialization and its dimension-dependent lattice size.
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/scenarios/scenario.hpp>

namespace scenarios {

// Derived complete-cell population: N = 6 * 4^(D-1) * n^D for the mirrored
// light/heavy layers, with two heavy sites for each light site.
inline size_t rayleighTaylorLatticeMultiplicity() {
    return spatialDimensions() == 2 ? 24 : 96;
}

// The imposed field is along y in 2D and z in 3D. The public
// scenario.rayleighTaylor.zPlaneAcceleration key retains its historical spelling.
ScenarioPlan initializeRayleighTaylor();

}    // namespace scenarios
