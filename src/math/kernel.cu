/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/math/kernel.hpp>

namespace detail {
__device__ __constant__ HydroKernelTable deviceHydroKernel;
__device__ __constant__ radial_kernel::KernelTable<double> deviceGravityKernel;
__device__ __constant__ radial_kernel::KernelTable<float> deviceSphGravityKernel;
}    // namespace detail

/// Copy the selected gravity and hydrodynamic kernel tables into device constant memory.
void uploadKernelSelections() {
    auto status = cudaMemcpyToSymbol(
        detail::deviceGravityKernel, &detail::gravityKernel, sizeof(detail::gravityKernel));
    if (status == cudaSuccess) {
        status = cudaMemcpyToSymbol(detail::deviceSphGravityKernel, &detail::sphGravityKernel,
            sizeof(detail::sphGravityKernel));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyToSymbol(
            detail::deviceHydroKernel, &detail::hydroKernel, sizeof(detail::hydroKernel));
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Uploading radial kernel tables: ") + cudaGetErrorString(status));
    }
}
