/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
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
