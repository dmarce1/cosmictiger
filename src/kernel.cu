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

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/kernel.hpp>

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
