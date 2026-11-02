/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/particles/particles.hpp>

void sphParticlesGlobalReadViscosityDiagnostics(ParticleGlobalRange range, float* accelerationX,
    float* accelerationY, float* accelerationZ, float* divergence, ParticleInteger offset);

namespace cullen_and_dehnen {

// Called after the locality-wide SPH phase has completed.
void clearDiagnosticCache();

}    // namespace cullen_and_dehnen
