/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/lightcone_entry.hpp>

struct DriftParameters {
    int rung = -1;
    double scale = 1.0;
    double timestep = 0.0;
    double tau0 = 0.0;
    double tau1 = 0.0;
    double tauMax = 0.0;
    double drag = 0.0;
    bool lightcone = false;
    int mapSize = 1;
};

// Arrays describe one disjoint particle range. Lightcone entries are appended
// before wrapping positions, using the same periodic images as CUDA drift.
void cpuDriftKernel(Fixed32* x, Fixed32* y, Fixed32* z, Array<float, NDIM>* velocities,
    char const* rungs, size_t count, DriftParameters const& parameters,
    std::vector<LightconeEntry>* entries = nullptr);
