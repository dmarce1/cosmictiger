/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <cosmictiger/radial_kernel.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_KERNEL_HD __host__ __device__
#else
#define COSMICTIGER_KERNEL_HD
#endif

namespace detail {
inline radial_kernel::KernelTable hydroKernel =
    radial_kernel::makeKernelTable({radial_kernel::Family::Spline, 3});
inline radial_kernel::KernelTable gravityKernel =
    radial_kernel::makeKernelTable({radial_kernel::Family::Ferrers, 8});
#ifdef __CUDACC__
extern __device__ __constant__ radial_kernel::KernelTable deviceHydroKernel;
extern __device__ __constant__ radial_kernel::KernelTable deviceGravityKernel;
#endif
}    // namespace detail

COSMICTIGER_KERNEL_HD inline radial_kernel::KernelTable const& selectedHydroKernel() {
#ifdef __CUDA_ARCH__
    return detail::deviceHydroKernel;
#else
    return detail::hydroKernel;
#endif
}

COSMICTIGER_KERNEL_HD inline radial_kernel::KernelTable const& selectedGravityKernel() {
#ifdef __CUDA_ARCH__
    return detail::deviceGravityKernel;
#else
    return detail::gravityKernel;
#endif
}

// Restore CUDA constant tables after device initialization/reset.
#ifdef USE_CUDA
void uploadKernelSelections();
#endif

// Call on each locality and its selected CUDA device before launching kernels.
void setKernelSelections(
    radial_kernel::KernelSpec gravity, radial_kernel::KernelSpec hydro, bool useCuda);

template <typename Real>
COSMICTIGER_KERNEL_HD inline Real kernelW(Real q) {
    return radial_kernel::value(selectedHydroKernel(), q);
}

template <typename Real>
COSMICTIGER_KERNEL_HD inline Real dkernelWDq(Real q) {
    return radial_kernel::derivative(selectedHydroKernel(), q);
}

template <typename Real>
COSMICTIGER_KERNEL_HD inline Real kernelFqinv(Real q) {
    return radial_kernel::forceFactor(selectedHydroKernel(), q);
}

template <typename Real>
COSMICTIGER_KERNEL_HD inline Real kernelPot(Real q) {
    return radial_kernel::potential(selectedHydroKernel(), q);
}

// Existing support-bound smoothing correction used by particle prediction.
COSMICTIGER_KERNEL_HD inline void dsmoothXDh(
    float h, float hmin, float, float& value, float& derivative) {
    if (h < 2.0f * hmin) {
        value = h / hmin - 1.0f;
        derivative = 1.0f / hmin;
    } else {
        value = 1.0f;
        derivative = 0.0f;
    }
}

COSMICTIGER_KERNEL_HD inline float dlogsmoothXDlogh(float h, float hmin, float hmax) {
    float value, derivative;
    dsmoothXDh(h, hmin, hmax, value, derivative);
    return derivative * h / value;
}

COSMICTIGER_KERNEL_HD inline float smoothX(float h, float hmin, float hmax) {
    float value, derivative;
    dsmoothXDh(h, hmin, hmax, value, derivative);
    return value;
}

#undef COSMICTIGER_KERNEL_HD
