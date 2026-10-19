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

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/math/tensor.hpp>

#define NREAL 178
#define NFOUR 92

// Ewald image lists and Fourier derivative tables. The tables are filled once
// for the runtime FMM order: each Fourier row holds a
// SymmetricTracelessTensor<float, order> and d0 a SymmetricTensor<float, order>,
// both stored in buffers sized for kFmmMaxOrder. The order-templated accessors
// view the leading entries; they are only valid for the order the tables were
// built for, which the generated Ewald derivative of that order guarantees.
struct EwaldConstants {
    Array<Array<float, NDIM>, NREAL> realIndices;
    Array<Array<float, NDIM>, NFOUR> fourIndices;
    Array<Array<float, kFmmMaxTracelessTensorCount>, NFOUR> fourExpanse;
    Array<float, kFmmMaxSymmetricTensorCount> d0;
    int order;
};

extern EwaldConstants ec;
#ifdef __CUDACC__
extern __constant__ EwaldConstants ecDev;
#endif

struct EwaldConst {
    CUDA_EXPORT static int nfour() {
        return NFOUR;
    }
    CUDA_EXPORT static int nreal() {
        return NREAL;
    }
    // Build the image lists and the tables for fmmOrder() on every locality.
    static void init();
    static void initGpu();
    CUDA_EXPORT static EwaldConstants const& constants() {
#ifdef __CUDA_ARCH__
        return ecDev;
#else
        return ec;
#endif
    }
    CUDA_EXPORT static Array<float, NDIM> const& realIndex(int i) {
        return constants().realIndices[i];
    }
    CUDA_EXPORT static Array<float, NDIM> const& fourIndex(int i) {
        return constants().fourIndices[i];
    }
    template <int order>
    CUDA_EXPORT static SymmetricTracelessTensor<float, order> const& fourExpansion(int i) {
        static_assert(SymmetricTracelessTensor<float, order>::kN_ <= kFmmMaxTracelessTensorCount);
        return *reinterpret_cast<SymmetricTracelessTensor<float, order> const*>(
            constants().fourExpanse[i].data());
    }
    template <int order>
    CUDA_EXPORT static SymmetricTensor<float, order> const& d0() {
        static_assert(SymmetricTensor<float, order>::kN_ <= kFmmMaxSymmetricTensorCount);
        return *reinterpret_cast<SymmetricTensor<float, order> const*>(constants().d0.data());
    }
};
