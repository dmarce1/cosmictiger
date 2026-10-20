/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cstddef>
#include <vector>

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
    double scale, double derivative, double tau, double tauMax, double radius) {
    return scale + derivative * (tauMax - radius - tau);
}
