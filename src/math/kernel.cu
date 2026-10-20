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
__device__ __constant__ radial_kernel::KernelTable deviceHydroKernel;
__device__ __constant__ radial_kernel::KernelTable deviceGravityKernel;
}    // namespace detail

void uploadKernelSelections() {
    auto status = cudaMemcpyToSymbol(
        detail::deviceGravityKernel, &detail::gravityKernel, sizeof(detail::gravityKernel));
    if (status == cudaSuccess) {
        status = cudaMemcpyToSymbol(
            detail::deviceHydroKernel, &detail::hydroKernel, sizeof(detail::hydroKernel));
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Uploading radial kernel tables: ") + cudaGetErrorString(status));
    }
}
