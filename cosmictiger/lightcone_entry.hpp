/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <cosmictiger/fixed.hpp>

using LightconeReal = Fixed<int, 31>;

struct LightconeEntry {
    LightconeReal x, y, z;
    float vx, vy, vz;
    int pix;
};

size_t lcAddParticles(
    std::vector<std::vector<LightconeEntry>> const& entries, double scale, double tau);
