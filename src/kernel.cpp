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

#include <cosmictiger/kernel.hpp>

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
