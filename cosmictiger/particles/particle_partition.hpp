/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/particles/particles.hpp>

// The existing in-place kd partition, with an exact fixed-point split plane.
ParticleInteger particlesPartition(Pair<ParticleInteger> range, Position split, int dimension);
