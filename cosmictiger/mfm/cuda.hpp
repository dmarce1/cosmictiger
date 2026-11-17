/*
 CosmicTiger - A cosmological N-Body code
 Typed CUDA launch interfaces for MFM support, density, and inverse-moment preparation.
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/mfm/run_data.hpp>

namespace mfm {

#ifdef USE_CUDA
/// Solve active supports, wait for completion, and return the combined node diagnostics.
/// Source/sink arrays must be device-accessible; overflow caching uses the CUDA pool.
template <typename Real, int dimensions>
PrehydroResult<Real> smoothingLengthsCuda(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, CudaStream& stream);

/// Prepare semiactive density and checked inverse moments without evolving hydro state.
/// Source/sink arrays must be device-accessible; overflow caching uses the CUDA pool.
template <typename Real, int dimensions>
PrehydroResult<Real> densityAndGeometryCuda(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, CudaStream& stream);
#endif

}    // namespace mfm
