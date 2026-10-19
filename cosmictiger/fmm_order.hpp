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

#include <type_traits>
#include <utility>

#include <cosmictiger/assert.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>

// FMM order conventions.
//
// The FMM order equals the multipole degree p plus two: source multipoles carry
// degree p, local expansions carry degree p + 1, and the Ewald derivative
// tensors carry ranks zero through p + 1, so their tensor "order" template
// argument equals the FMM order. Order 3 (dipoles with quadratic expansions) is
// the smallest order whose local expansion can hold the periodic uniform
// background quadratic; order 9 (degree-7 multipoles, degree-8 expansions) is
// the largest: single precision gains nothing measurable beyond it.
//
// Every order in [kFmmMinOrder, kFmmMaxOrder] is compiled into the binary. The
// run selects one order at startup (gravity.fmmOrder, defaulting to the build's
// COSMICTIGER_WITH_ORDER) and fmmDispatch() switches to the matching
// instantiation at coarse boundaries: tree construction, a CPU walk, a CUDA
// launch. Inner loops never test the order.
//
// Coefficient counts use compact rank-major STF storage: a degree-d harmonic
// tensor stores (d + 1)^2 coefficients, and local expansions append one
// constant Laplacian (the periodic background trace).

inline constexpr int kFmmMinOrder = 3;
inline constexpr int kFmmMaxOrder = 9;

#ifndef ORDER
#error "ORDER (the default FMM order, COSMICTIGER_WITH_ORDER) must be defined"
#endif
inline constexpr int kFmmDefaultOrder = ORDER;
static_assert(kFmmDefaultOrder >= kFmmMinOrder && kFmmDefaultOrder <= kFmmMaxOrder,
    "COSMICTIGER_WITH_ORDER must select an FMM order from 3 to 9");
static_assert(LORDER == ORDER, "LORDER must equal the default FMM order");

CUDA_EXPORT constexpr int fmmMultipoleDegree(int order) {
    return order - 2;
}

CUDA_EXPORT constexpr int fmmLocalDegree(int order) {
    return order - 1;
}

CUDA_EXPORT constexpr int fmmHarmonicCoefficientCount(int degree) {
    return (degree + 1) * (degree + 1);
}

CUDA_EXPORT constexpr int fmmMultipoleCoefficientCount(int order) {
    return fmmHarmonicCoefficientCount(fmmMultipoleDegree(order));
}

CUDA_EXPORT constexpr int fmmLocalHarmonicCoefficientCount(int order) {
    return fmmHarmonicCoefficientCount(fmmLocalDegree(order));
}

// The index of the constant Laplacian stored after the harmonic coefficients.
CUDA_EXPORT constexpr int fmmLocalTraceIndex(int order) {
    return fmmLocalHarmonicCoefficientCount(order);
}

CUDA_EXPORT constexpr int fmmLocalCoefficientCount(int order) {
    return fmmLocalHarmonicCoefficientCount(order) + 1;
}

// A positioned multipole stores NDIM Fixed32 positions followed by the coefficients.
CUDA_EXPORT constexpr int fmmPositionedMultipoleBytes(int order) {
    return int(sizeof(float)) * (fmmMultipoleCoefficientCount(order) + NDIM);
}

CUDA_EXPORT constexpr int fmmLocalExpansionBytes(int order) {
    return int(sizeof(float)) * fmmLocalCoefficientCount(order);
}

// Ewald derivative tensors: SymmetricTracelessTensor<T, order> holds ranks 0..order-1.
CUDA_EXPORT constexpr int fmmTracelessTensorCount(int order) {
    return order * order + 1;
}

CUDA_EXPORT constexpr int fmmSymmetricTensorCount(int order) {
    return order * (order + 1) * (order + 2) / 6;
}

inline constexpr int kFmmMaxMultipoleCoefficients = fmmMultipoleCoefficientCount(kFmmMaxOrder);
inline constexpr int kFmmMaxLocalCoefficients = fmmLocalCoefficientCount(kFmmMaxOrder);
inline constexpr int kFmmMaxTracelessTensorCount = fmmTracelessTensorCount(kFmmMaxOrder);
inline constexpr int kFmmMaxSymmetricTensorCount = fmmSymmetricTensorCount(kFmmMaxOrder);

static_assert(fmmPositionedMultipoleBytes(3) == 28 && fmmPositionedMultipoleBytes(7) == 156 &&
        fmmPositionedMultipoleBytes(9) == 268,
    "Positioned multipole byte sizes");
static_assert(fmmLocalExpansionBytes(3) == 40 && fmmLocalExpansionBytes(7) == 200 &&
        fmmLocalExpansionBytes(9) == 328,
    "Local expansion byte sizes");

// Compile-time view of one order.
template <int order>
struct FmmOrder {
    static_assert(order >= kFmmMinOrder && order <= kFmmMaxOrder, "Unsupported FMM order");
    static constexpr int kOrder_ = order;
    static constexpr int kMultipoleDegree_ = fmmMultipoleDegree(order);
    static constexpr int kLocalDegree_ = fmmLocalDegree(order);
    static constexpr int kMultipoleCoefficients_ = fmmMultipoleCoefficientCount(order);
    static constexpr int kLocalHarmonicCoefficients_ = fmmLocalHarmonicCoefficientCount(order);
    static constexpr int kLocalTraceIndex_ = fmmLocalTraceIndex(order);
    static constexpr int kLocalCoefficients_ = fmmLocalCoefficientCount(order);
    static constexpr int kPositionedMultipoleBytes_ = fmmPositionedMultipoleBytes(order);
    static constexpr int kLocalExpansionBytes_ = fmmLocalExpansionBytes(order);
    static constexpr int kEwaldTensorOrder_ = order;
};

// The order selected for this run. setFmmOrder validates the range; options
// parsing calls it before any tree, kick or Ewald table is built.
int fmmOrder();
void setFmmOrder(int order);

// Call f(std::integral_constant<int, order>{}) for the runtime order.
template <class F>
decltype(auto) fmmDispatch(int order, F&& f) {
    switch (order) {
    case 3:
        return f(std::integral_constant<int, 3>{});
    case 4:
        return f(std::integral_constant<int, 4>{});
    case 5:
        return f(std::integral_constant<int, 5>{});
    case 6:
        return f(std::integral_constant<int, 6>{});
    case 7:
        return f(std::integral_constant<int, 7>{});
    case 8:
        return f(std::integral_constant<int, 8>{});
    case 9:
        return f(std::integral_constant<int, 9>{});
    default:
        break;
    }
    ALWAYS_ASSERT(false);
    return f(std::integral_constant<int, kFmmDefaultOrder>{});
}

template <class F>
decltype(auto) fmmDispatch(F&& f) {
    return fmmDispatch(fmmOrder(), std::forward<F>(f));
}
