/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/radial_density_kernel.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

// Keep the established float table and arithmetic in 3D; hydro-only dimensions
// use density/gradient tables, which have no Newtonian force/potential contract.
using HydroKernelTable =
    std::conditional_t<kSpatialDimensions == 3, radial_kernel::KernelTable<float>,
        radial_kernel::DensityKernelTable<kSpatialDimensions, float>>;
/// Build the selected density-kernel table for the configured spatial dimension.
inline HydroKernelTable makeHydroKernelTable(radial_kernel::KernelSpec spec) {
#if COSMICTIGER_DIMENSION_COUNT == 3
    return radial_kernel::makeKernelTable<float>(spec);
#else
    return radial_kernel::makeDensityKernelTable<kSpatialDimensions, float>(spec);
#endif
}

namespace detail {
inline HydroKernelTable hydroKernel = makeHydroKernelTable({radial_kernel::Family::Spline, 3});
inline radial_kernel::KernelTable<double> gravityKernel =
    radial_kernel::makeKernelTable({radial_kernel::Family::Ferrers, 4});
inline radial_kernel::KernelTable<float> sphGravityKernel =
    radial_kernel::makeKernelTable<float>({radial_kernel::Family::Ferrers, 4});
#ifdef __CUDACC__
extern __device__ __constant__ HydroKernelTable deviceHydroKernel;
extern __device__ __constant__ radial_kernel::KernelTable<double> deviceGravityKernel;
extern __device__ __constant__ radial_kernel::KernelTable<float> deviceSphGravityKernel;
#endif
}    // namespace detail

/// Return the active host or device hydrodynamic kernel table.
CUDA_EXPORT inline HydroKernelTable const& selectedHydroKernel() {
#ifdef __CUDA_ARCH__
    return detail::deviceHydroKernel;
#else
    return detail::hydroKernel;
#endif
}

/// Return the active host/device gravity kernel table for the requested float or double
/// precision.
template <class Real = double>
CUDA_EXPORT inline radial_kernel::KernelTable<Real> const& selectedGravityKernel() {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<Real, float>) {
        return detail::deviceSphGravityKernel;
    } else {
        return detail::deviceGravityKernel;
    }
#else
    if constexpr (std::is_same_v<Real, float>) {
        return detail::sphGravityKernel;
    } else {
        return detail::gravityKernel;
    }
#endif
}

// Restore CUDA constant tables after device initialization/reset.
#ifdef USE_CUDA
void uploadKernelSelections();
#endif

// Call on each locality and its selected CUDA device before launching kernels.
void setKernelSelections(
    radial_kernel::KernelSpec gravity, radial_kernel::KernelSpec hydro, bool useCuda);

/// Evaluate the selected dimensionless SPH density kernel at normalized radius q.
template <typename Real>
CUDA_EXPORT inline Real kernelW(Real q) {
    return radial_kernel::value(selectedHydroKernel(), q);
}

/// Evaluate the derivative of the selected SPH density kernel with respect to q.
template <typename Real>
CUDA_EXPORT inline Real dkernelWDq(Real q) {
    return radial_kernel::derivative(selectedHydroKernel(), q);
}
