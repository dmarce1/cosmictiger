/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#ifndef COSMICTIGER_POWER_CUDA_HPP_
#define COSMICTIGER_POWER_CUDA_HPP_

#include <cstddef>
#include <cstdint>

#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/range.hpp>

#ifdef __CUDACC__
// Separate from the particle-storage adapter so tests can run the production
// kernel with a bounded launch and ordinary test fixtures.
__global__ void accumulateDensityKernel(Position const* xValue, Position const* yValue,
    Position const* zValue, float* rho, float const* masses, int mfold, size_t numberOfParticles,
    int ndim, Range<int64_t> intbox);
#endif

#endif
