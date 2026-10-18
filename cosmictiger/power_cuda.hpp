/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef COSMICTIGER_POWER_CUDA_HPP_
#define COSMICTIGER_POWER_CUDA_HPP_

#include <cstddef>
#include <cstdint>

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/range.hpp>

#ifdef __CUDACC__
// Separate from the particle-storage adapter so tests can run the production
// kernel with a bounded launch and ordinary test fixtures.
__global__ void accumulateDensityKernel(Fixed32 const* xValue, Fixed32 const* yValue,
    Fixed32 const* zValue, float* rho, float const* masses, int mfold, size_t numberOfParticles,
    int ndim, Range<int64_t> intbox);
#endif

#endif
