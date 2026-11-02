/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/math/vector3.hpp>

// Consume one completed, gm-scaled near+far field without retaining a force
// array. Particle membership is fixed by the caller's current ladder stage;
// only this owned sink's velocity/rung and requested diagnostics are modified.
// Traditional hydro sinks receive acceleration only; hydro owns their kicks
// and rungs. Traditional sources must contain every owned particle even when
// minRung selects only a subset of the sinks.
// Accumulate source-operation counts in flopCount when supplied, so callers can
// submit a whole sink batch once; otherwise submit this particle's count here.
// precisionCounts optionally receives the same work split by arithmetic precision.
// Supplying either output defers global submission to the caller.
KickReturn particleMeshApplyField(ParticleInteger index, Vector3<float> const& acceleration,
    float potential, KickParameters const& parameters, FlopCount* flopCount = nullptr,
    FlopCounts* precisionCounts = nullptr);

// PMMM integration using the total far-field and near-field force.
// Both operations recurse over all localities. Preparation closes the previous
// dark-matter half-step and proposes a common rung, leaving its opening kick
// pending until hydro has contributed its timestep bound.
KickReturn particleMeshPrepareKick(double scale, double baseTimestep, double maximumTimestep,
    bool firstCall, int minimumRung, double previousTimestep = -1.0);
// Return the global sum of 0.5 * mass * stored canonical velocity squared after
// opening the dark-matter kick. Gas velocities are unchanged here; hydro callers
// finish their gas update separately and must not use this as synchronized energy.
double particleMeshFinishKick(double baseTimestep, int rung, double explicitTimestep = -1.0);
// Continuous global step: open using the raw reduced interval, with every rung zero.
double particleMeshFinishGlobalKick(double timestep);
