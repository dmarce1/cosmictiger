/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <type_traits>
#include <utility>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/position.hpp>

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
// Coefficient counts use compact rank-major Cartesian storage: a degree-d
// harmonic tensor stores (d + 1)^2 coefficients, and local expansions append one constant Laplacian
// (the periodic background trace).

#ifndef FMM_MIN_ORDER
#error "FMM_MIN_ORDER must be defined by the build configuration"
#endif
#ifndef FMM_MAX_ORDER
#error "FMM_MAX_ORDER must be defined by the build configuration"
#endif
inline constexpr int kFmmMinOrder = FMM_MIN_ORDER;
inline constexpr int kFmmMaxOrder = FMM_MAX_ORDER;

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

// A positioned multipole stores NDIM Position positions followed by the coefficients.
CUDA_EXPORT constexpr int fmmPositionedMultipoleBytes(int order) {
    int const bytes =
        int(sizeof(float)) * fmmMultipoleCoefficientCount(order) + NDIM * int(sizeof(Position));
    // Include the C++ record's tail padding for its position alignment.
    return (bytes + int(alignof(Position)) - 1) / int(alignof(Position)) * int(alignof(Position));
}

// The tree's multipole arena and the CUDA kick's staged multipole tile store
// the records at this stride: the record size rounded up to a multiple of
// kFmmPositionedMultipoleAlignment, so that every record starts on a 16-byte
// boundary and the kick stages a whole record with 16-byte asynchronous
// copies. The required padding depends on the order and position width.
inline constexpr int kFmmPositionedMultipoleAlignment = 16;

CUDA_EXPORT constexpr int fmmPositionedMultipoleStride(int order) {
    return (fmmPositionedMultipoleBytes(order) + kFmmPositionedMultipoleAlignment - 1) /
        kFmmPositionedMultipoleAlignment * kFmmPositionedMultipoleAlignment;
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

static_assert(kPositionBits != 32 ||
    (fmmPositionedMultipoleBytes(3) == 28 && fmmPositionedMultipoleBytes(7) == 156 &&
        fmmPositionedMultipoleBytes(9) == 268));
static_assert(kPositionBits != 64 ||
    (fmmPositionedMultipoleBytes(3) == 40 && fmmPositionedMultipoleBytes(7) == 168 &&
        fmmPositionedMultipoleBytes(9) == 280));
static_assert(kPositionBits != 32 ||
    (fmmPositionedMultipoleStride(3) == 32 && fmmPositionedMultipoleStride(4) == 48 &&
        fmmPositionedMultipoleStride(7) == 160 && fmmPositionedMultipoleStride(8) == 208 &&
        fmmPositionedMultipoleStride(9) == 272));
static_assert(kPositionBits != 64 ||
    (fmmPositionedMultipoleStride(3) == 48 && fmmPositionedMultipoleStride(4) == 64 &&
        fmmPositionedMultipoleStride(7) == 176 && fmmPositionedMultipoleStride(8) == 224 &&
        fmmPositionedMultipoleStride(9) == 288));
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
    static constexpr int kPositionedMultipoleStride_ = fmmPositionedMultipoleStride(order);
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
#if FMM_MIN_ORDER <= 3 && FMM_MAX_ORDER >= 3
    case 3:
        return f(std::integral_constant<int, 3>{});
#endif
#if FMM_MIN_ORDER <= 4 && FMM_MAX_ORDER >= 4
    case 4:
        return f(std::integral_constant<int, 4>{});
#endif
#if FMM_MIN_ORDER <= 5 && FMM_MAX_ORDER >= 5
    case 5:
        return f(std::integral_constant<int, 5>{});
#endif
#if FMM_MIN_ORDER <= 6 && FMM_MAX_ORDER >= 6
    case 6:
        return f(std::integral_constant<int, 6>{});
#endif
#if FMM_MIN_ORDER <= 7 && FMM_MAX_ORDER >= 7
    case 7:
        return f(std::integral_constant<int, 7>{});
#endif
#if FMM_MIN_ORDER <= 8 && FMM_MAX_ORDER >= 8
    case 8:
        return f(std::integral_constant<int, 8>{});
#endif
#if FMM_MIN_ORDER <= 9 && FMM_MAX_ORDER >= 9
    case 9:
        return f(std::integral_constant<int, 9>{});
#endif
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

// Internal normalized FMM coordinate scale. Physical lengths use the unit box.
#define SCALE_FACTOR 1.000000e+03f
#define SCALE_FACTOR0 1.000000e+00f
#define SCALE_FACTOR_INV0 1.000000e+00f
#define SCALE_FACTOR1 1.000000e+03f
#define SCALE_FACTOR_INV1 1.000000e-03f
#define SCALE_FACTOR2 1.000000e+06f
#define SCALE_FACTOR_INV2 1.000000e-06f
#define SCALE_FACTOR3 1.000000e+09f
#define SCALE_FACTOR_INV3 1.000000e-09f
#define SCALE_FACTOR4 1.000000e+12f
#define SCALE_FACTOR_INV4 1.000000e-12f
#define SCALE_FACTOR5 1.000000e+15f
#define SCALE_FACTOR_INV5 1.000000e-15f
#define SCALE_FACTOR6 1.000000e+18f
#define SCALE_FACTOR_INV6 1.000000e-18f
#define SCALE_FACTOR7 1.000000e+21f
#define SCALE_FACTOR_INV7 1.000000e-21f
