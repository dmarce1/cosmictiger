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
#include <cosmictiger/gravity/build_config.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_config.hpp>
#include <cosmictiger/math/position.hpp>

// FMM order conventions.
//
// The FMM order equals the multipole degree p plus two: source multipoles carry
// degree p and local expansions carry degree p + 1. Order 3 (dipoles with
// quadratic expansions) is the smallest order whose local expansion can hold
// the periodic uniform background quadratic; order 9 (degree-7 multipoles,
// degree-8 expansions) is the largest: single precision gains nothing
// measurable beyond it.
//
// COSMICTIGER_EXPANSION_DEGREE selects one potential degree per executable.
// Checkpoints retain the internal order (potential degree plus one).
//
// Coefficient counts use compact rank-major harmonic storage plus the independent
// quadrupole zz component when enabled by the Cartesian generator. Degrees above
// two remain trace-free.

inline constexpr int kFmmOrder = gravity::kBuildFmmOrder;

/// Return the maximum source-multipole degree for this FMM order.
CUDA_EXPORT constexpr int fmmMultipoleDegree(int order) {
    return order - 2;
}

/// Return the maximum local-expansion degree for this FMM order.
CUDA_EXPORT constexpr int fmmLocalDegree(int order) {
    return order - 1;
}

/// Count independent real harmonic coefficients through the supplied degree.
CUDA_EXPORT constexpr int fmmHarmonicCoefficientCount(int degree) {
    return (degree + 1) * (degree + 1);
}

/// Count Cartesian coefficients, including the independent quadrupole trace.
CUDA_EXPORT constexpr int fmmCartesianCoefficientCount(int degree) {
    return fmmHarmonicCoefficientCount(degree) +
        int(cosmictiger::gravity::cartesian::kQuadrupoleTrace && degree >= 2);
}

/// Count source-multipole coefficients for the requested FMM order.
CUDA_EXPORT constexpr int fmmMultipoleCoefficientCount(int order) {
    return fmmCartesianCoefficientCount(fmmMultipoleDegree(order));
}

/// Count harmonic local-expansion coefficients for the requested FMM order.
CUDA_EXPORT constexpr int fmmLocalHarmonicCoefficientCount(int order) {
    return fmmHarmonicCoefficientCount(fmmLocalDegree(order));
}

/// Return the stored local-expansion coefficient count for this FMM order.
CUDA_EXPORT constexpr int fmmLocalCoefficientCount(int order) {
    return fmmCartesianCoefficientCount(fmmLocalDegree(order));
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

/// Round a positioned multipole record size up to its required alignment.
CUDA_EXPORT constexpr int fmmPositionedMultipoleStride(int order) {
    return (fmmPositionedMultipoleBytes(order) + kFmmPositionedMultipoleAlignment - 1) /
        kFmmPositionedMultipoleAlignment * kFmmPositionedMultipoleAlignment;
}

/// Return the byte count of a float local expansion at the requested order.
CUDA_EXPORT constexpr int fmmLocalExpansionBytes(int order) {
    return int(sizeof(float)) * fmmLocalCoefficientCount(order);
}

inline constexpr int kFmmMultipoleCoefficients = fmmMultipoleCoefficientCount(kFmmOrder);
inline constexpr int kFmmLocalCoefficients = fmmLocalCoefficientCount(kFmmOrder);

static_assert(kPositionBits != 32 ||
    (fmmPositionedMultipoleBytes(3) == 28 &&
        fmmPositionedMultipoleBytes(7) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 160 : 156) &&
        fmmPositionedMultipoleBytes(9) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 272 : 268)));
static_assert(kPositionBits != 64 ||
    (fmmPositionedMultipoleBytes(3) == 40 &&
        fmmPositionedMultipoleBytes(7) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 176 : 168) &&
        fmmPositionedMultipoleBytes(9) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 288 : 280)));
static_assert(kPositionBits != 32 ||
    (fmmPositionedMultipoleStride(3) == 32 &&
        fmmPositionedMultipoleStride(4) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 64 : 48) &&
        fmmPositionedMultipoleStride(7) == 160 &&
        fmmPositionedMultipoleStride(8) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 224 : 208) &&
        fmmPositionedMultipoleStride(9) == 272));
static_assert(kPositionBits != 64 ||
    (fmmPositionedMultipoleStride(3) == 48 && fmmPositionedMultipoleStride(4) == 64 &&
        fmmPositionedMultipoleStride(7) == 176 && fmmPositionedMultipoleStride(8) == 224 &&
        fmmPositionedMultipoleStride(9) == 288));
static_assert(
    fmmLocalExpansionBytes(3) == (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 40 : 36) &&
        fmmLocalExpansionBytes(7) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 200 : 196) &&
        fmmLocalExpansionBytes(9) ==
            (cosmictiger::gravity::cartesian::kQuadrupoleTrace ? 328 : 324),
    "Local expansion byte sizes");

// Compile-time view of one order.
template <int order>
struct FmmOrder {
    static_assert(order == kFmmOrder, "Unsupported FMM order");
    static constexpr int kOrder_ = order;
    static constexpr int kMultipoleDegree_ = fmmMultipoleDegree(order);
    static constexpr int kLocalDegree_ = fmmLocalDegree(order);
    static constexpr int kMultipoleCoefficients_ = fmmMultipoleCoefficientCount(order);
    static constexpr int kLocalHarmonicCoefficients_ = fmmLocalHarmonicCoefficientCount(order);
    static constexpr int kLocalCoefficients_ = fmmLocalCoefficientCount(order);
    static constexpr int kPositionedMultipoleBytes_ = fmmPositionedMultipoleBytes(order);
    static constexpr int kPositionedMultipoleStride_ = fmmPositionedMultipoleStride(order);
    static constexpr int kLocalExpansionBytes_ = fmmLocalExpansionBytes(order);
};

// Internal normalized FMM coordinate scale. Physical lengths use the unit box.
inline constexpr float kScaleFactor = 1.000000e+03f;
inline constexpr float kScaleFactor1 = 1.000000e+03f;
inline constexpr float kScaleFactorInverse1 = 1.000000e-03f;
inline constexpr float kScaleFactor2 = 1.000000e+06f;
inline constexpr float kScaleFactorInverse2 = 1.000000e-06f;
inline constexpr float kScaleFactorInverse3 = 1.000000e-09f;
