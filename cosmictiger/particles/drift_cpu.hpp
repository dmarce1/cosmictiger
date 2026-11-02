/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <cosmictiger/analysis/lightcone_entry.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/hydro/hopkins/hopkins_state.hpp>
#include <cosmictiger/particles/particles.hpp>

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
// A null types array means every particle is dark matter.
void cpuDriftKernel(Position* x, Position* y, Position* z, Array<float, kDimensions>* velocities,
    char const* rungs, char const* types, size_t count, DriftParameters const& parameters,
    std::vector<LightconeEntry>* entries = nullptr, HopkinsParticleState* meshStates = nullptr,
    ParticleInteger const* gasIndices = nullptr);
