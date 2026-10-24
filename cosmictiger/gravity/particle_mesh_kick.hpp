/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/gravity/kick.hpp>

// Both operations recurse over all localities. Preparation closes the previous
// dark-matter half-step and proposes a common rung, leaving its opening kick
// pending until hydro has contributed its timestep bound.
KickReturn particleMeshPrepareKick(
    double scale, double baseTimestep, double maximumTimestep, bool firstCall, int minimumRung);
void particleMeshFinishKick(double baseTimestep, int rung);
