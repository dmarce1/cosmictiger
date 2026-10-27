/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/gravity/build_config.hpp>
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
// COSMICTIGER_WITH_ORDER selects the only order compiled into this binary.
// fmmDispatch() directly instantiates it at tree, CPU walk, and CUDA launch
// boundaries; no runtime order selection is performed.
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

inline constexpr int kFmmDefaultOrder = gravity::kBuildFmmOrder;
static_assert(kFmmMinOrder == kFmmDefaultOrder && kFmmMaxOrder == kFmmDefaultOrder,
    "FMM_MIN_ORDER and FMM_MAX_ORDER must equal the selected COSMICTIGER_WITH_ORDER");

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
    int const bytes = int(sizeof(float)) * fmmMultipoleCoefficientCount(order) +
        kDimensions * int(sizeof(Position));
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

// Compatibility checks for callers carrying serialized or explicit orders.
// The selected order itself is a constant for the lifetime of the executable.
inline constexpr int fmmOrder() {
    return kFmmDefaultOrder;
}
void setFmmOrder(int order);

// Explicit-order callers must request this binary's sole compiled order.
template <class F>
decltype(auto) fmmDispatch(int order, F&& f) {
    ALWAYS_ASSERT(order == kFmmDefaultOrder);
    return f(std::integral_constant<int, kFmmDefaultOrder>{});
}

template <class F>
decltype(auto) fmmDispatch(F&& f) {
    return f(std::integral_constant<int, kFmmDefaultOrder>{});
}

// Internal normalized FMM coordinate scale. Physical lengths use the unit box.
inline constexpr float kScaleFactor = 1.000000e+03f;
inline constexpr float kScaleFactor1 = 1.000000e+03f;
inline constexpr float kScaleFactorInverse1 = 1.000000e-03f;
inline constexpr float kScaleFactor2 = 1.000000e+06f;
inline constexpr float kScaleFactorInverse2 = 1.000000e-06f;
inline constexpr float kScaleFactorInverse3 = 1.000000e-09f;
