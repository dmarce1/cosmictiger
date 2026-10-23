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

#include <cosmictiger/cosmology/background.hpp>
#include <cosmictiger/math/fixed.hpp>

using LightconeReal = Fixed<int, 31>;

struct LightconeEntry {
    LightconeReal x, y, z;
    float vx, vy, vz;
    int pix;
};

size_t lcAddParticles(
    std::vector<std::vector<LightconeEntry>> const& entries, double scale, double tau);

// Positions encode the remaining light-travel conformal time, not emission time.
inline double lightconeEmissionScale(
    cosmology::SimulationBackgroundTable const& background, double tauMax, double radius) {
    if (!std::isfinite(radius) || radius < 0.0 || !std::isfinite(tauMax)) {
        throw std::invalid_argument("Lightcone radius and observation time must be finite");
    }
    return background.sample(std::max(0.0, tauMax - radius)).state.scaleFactor;
}
