/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/cosmology/background.hpp>
#endif
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/particles/particle_types.hpp>

using LightconeReal = Fixed<int, 31>;

// Every species crosses the lightcone: the HEALPix maps weight entries by
// species mass, while the lightcone halo finder keeps dark matter only.
struct LightconeEntry {
    LightconeReal x, y, z;
    float vx, vy, vz;
    int pix;
    char type = kDarkMatterType;
};

size_t lcAddParticles(
    std::vector<std::vector<LightconeEntry>> const& entries, double scale, double tau);

// Positions encode the remaining light-travel conformal time, not emission time.
#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
/// Sample the background at observer time minus lightcone radius, clamped to the initial
/// epoch; reject invalid inputs.
inline double lightconeEmissionScale(
    cosmology::SimulationBackgroundTable const& background, double tauMax, double radius) {
    if (!std::isfinite(radius) || radius < 0.0 || !std::isfinite(tauMax)) {
        throw std::invalid_argument("Lightcone radius and observation time must be finite");
    }
    return background.sample(std::max(0.0, tauMax - radius)).state.scaleFactor;
}
#endif
