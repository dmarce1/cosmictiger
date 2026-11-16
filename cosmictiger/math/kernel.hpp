// Runtime-selected density kernels and fixed three-dimensional gravity kernels.
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <iterator>
#include <type_traits>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/radial_density_kernel.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

// A common table holds dimension-dependent density coefficients. Gravity fields are
// used only in 3D, where the original table construction and arithmetic are retained.
using HydroKernelTable = radial_kernel::KernelTable<float>;

inline bool isValidHydroKernel(radial_kernel::KernelSpec spec) {
    switch (spatialDimensions()) {
    case 1:
        return radial_kernel::isValidDensity<1>(spec);
    case 2:
        return radial_kernel::isValidDensity<2>(spec);
    default:
        return radial_kernel::isValidDensity<3>(spec);
    }
}

/// Build density coefficients in the selected runtime dimension.
template <class TableReal = float>
inline radial_kernel::KernelTable<TableReal> makeHydroKernelTable(radial_kernel::KernelSpec spec) {
    if (spatialDimensions() == kDimensions) {
        return radial_kernel::makeKernelTable<TableReal>(spec);
    }
    radial_kernel::KernelTable<TableReal> result{};
    auto const copyDensity =
        [&]<size_t dimension>(
            radial_kernel::DensityKernelTable<dimension, TableReal> const& table) {
            result.spec = table.spec;
            result.degree = table.degree;
            result.pieceCount = table.pieceCount;
            result.smoothingLengthPerEffectiveRadius = table.smoothingLengthPerEffectiveRadius;
            for (size_t piece = 0; piece < std::size(table.changePoints); ++piece) {
                result.changePoints[piece] = table.changePoints[piece];
            }
            for (size_t piece = 0; piece < std::size(table.coefficients); ++piece) {
                for (size_t i = 0; i < std::size(table.coefficients[piece]); ++i) {
                    result.coefficients[piece][i] = table.coefficients[piece][i];
                }
                for (size_t i = 0; i < std::size(table.derivativeCoefficients[piece]); ++i) {
                    result.derivativeCoefficients[piece][i] =
                        table.derivativeCoefficients[piece][i];
                }
            }
        };
    if (spatialDimensions() == 1) {
        copyDensity(radial_kernel::makeDensityKernelTable<1, TableReal>(spec));
    } else {
        copyDensity(radial_kernel::makeDensityKernelTable<2, TableReal>(spec));
    }
    return result;
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
