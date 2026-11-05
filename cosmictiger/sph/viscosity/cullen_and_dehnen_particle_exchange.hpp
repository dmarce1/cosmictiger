/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/particles/particles.hpp>

struct SphRunParameters;

// currentGradient selects pendingDivergence from a completed all-gas diagnostic
// pass. The caller must clear diagnostic caches between source stages. No history
// is committed by either read mode; the default preserves the legacy endpoint signs.
void sphParticlesGlobalReadViscosityDiagnostics(ParticleGlobalRange range, float* divergence,
    ParticleInteger offset, bool currentGradient = false);

// Publish the completed density pass for output, using the existing checkpoint
// normalization. This does not evolve alpha or feed the direct-gradient sensor.
void sphParticlesPublishDivergence(SphRunParameters const& parameters);

namespace cullen_and_dehnen {

// Called after the locality-wide SPH phase has completed.
void clearDiagnosticCache();

}    // namespace cullen_and_dehnen
