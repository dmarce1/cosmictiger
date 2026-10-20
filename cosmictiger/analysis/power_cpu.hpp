/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#ifndef COSMICTIGER_POWER_CPU_HPP_
#define COSMICTIGER_POWER_CPU_HPP_

#include <cstddef>
#include <cstdint>

#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/range.hpp>

// Add the particles owned by the tile interior to its padded density buffer.
// The caller initializes density; nullptr masses assigns unit mass to each particle.
// Periodic wrapping of the padded cells is performed by fft3dAccumulateReal.
void accumulateDensityCpuKernel(Fixed32 const* positionX, Fixed32 const* positionY,
    Fixed32 const* positionZ, float const* masses, size_t particleCount, int fold, int gridSize,
    Range<int64_t> const& box, float* density);

#endif
