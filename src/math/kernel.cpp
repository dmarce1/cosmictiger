/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <stdexcept>

#include <cosmictiger/math/kernel.hpp>

void setKernelSelections(
    radial_kernel::KernelSpec gravity, radial_kernel::KernelSpec hydro, bool useCuda) {
    if (!radial_kernel::isValid(gravity) || !radial_kernel::isValid(hydro)) {
        throw std::invalid_argument("Unsupported radial kernel family or polynomial degree");
    }
    detail::gravityKernel = radial_kernel::makeKernelTable(gravity);
    detail::hydroKernel = radial_kernel::makeKernelTable(hydro);
    if (useCuda) {
#ifdef USE_CUDA
        uploadKernelSelections();
#else
        throw std::runtime_error("CUDA kernel selection requires a CUDA build");
#endif
    }
}
