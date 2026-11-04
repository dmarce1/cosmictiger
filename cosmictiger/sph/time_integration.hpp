/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstddef>

// The midpoint split uses live operator-stage velocities and entropies. Its
// driver drifts all particles chronologically; no per-particle predictor clocks
// or future-position restoration are needed.
bool sphMidpointEnabled();
bool sphMidpointEventActive();
void sphMidpointBeginEvent();
void sphMidpointEndEvent();

// Count exact-rung targets collectively; a negative rung selects all gas.
size_t sphMidpointTargetCount(int exactRung);

// Apply the force pass's hydro acceleration and entropy rate over one positive
// interval. A negative exactRung selects all gas for a global step. Rung bounds
// remain staged until overwritten by the next midpoint force calculation.
void sphParticlesApplyMidpointUpdates(int exactRung, double timestep);

// Gravity-disabled traditional steps choose a new rung only for particles whose
// interval ends at this boundary. Return the maximum occupied rung globally.
int sphParticlesSelectMidpointRungs(int minimumRung);
