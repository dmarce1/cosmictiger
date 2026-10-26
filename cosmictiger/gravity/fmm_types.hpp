/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_types.hpp>
#include <cosmictiger/gravity/fmm_kernels/spherical_types.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/math/position.hpp>

// Order-dependent FMM value types. This header is light: it carries no
// generated kernels, so order-agnostic code (tree nodes, work items, options)
// can include it everywhere. The operators live in fmm_kernels/*_operators.hpp.

namespace cosmictiger::gravity {
// Physical point result in normalized FMM units. Acceleration is -grad(potential).
template <class Real>
struct Field : Array<Real, kDimensions + 1> {
    CUDA_EXPORT Real& potential() {
        return (*this)[0];
    }
    CUDA_EXPORT Real const& potential() const {
        return (*this)[0];
    }
    CUDA_EXPORT Real& accelerationX() {
        return (*this)[1];
    }
    CUDA_EXPORT Real const& accelerationX() const {
        return (*this)[1];
    }
    CUDA_EXPORT Real& accelerationY() {
        return (*this)[2];
    }
    CUDA_EXPORT Real const& accelerationY() const {
        return (*this)[2];
    }
    CUDA_EXPORT Real& accelerationZ() {
        return (*this)[3];
    }
    CUDA_EXPORT Real const& accelerationZ() const {
        return (*this)[3];
    }
};
static_assert(sizeof(Field<float>) == (kDimensions + 1) * sizeof(float));
}    // namespace cosmictiger::gravity

static_assert(sizeof(cosmictiger::gravity::cartesian::Multipole<float, 7>) ==
    4 * fmmMultipoleCoefficientCount(7));
static_assert(
    sizeof(cosmictiger::gravity::cartesian::Expansion<float, 7>) == fmmLocalExpansionBytes(7));

// Positioned multipole record. The center comes first so order-agnostic tree
// code reads it through MultipolePosition; the per-order view adds the moments.
struct MultipolePosition {
    Array<Position, kDimensions> pos;
};

template <int order>
struct FmmMultiPos {
    Array<Position, kDimensions> pos;
    cosmictiger::gravity::cartesian::Multipole<float, order> multi;
};

static_assert(std::is_standard_layout_v<FmmMultiPos<kFmmDefaultOrder>>);
static_assert(sizeof(FmmMultiPos<7>) == fmmPositionedMultipoleBytes(7));
static_assert(sizeof(FmmMultiPos<9>) == fmmPositionedMultipoleBytes(9));

template <int order>
constexpr bool fmmTypedLayout() {
    using CartesianMultipole = cosmictiger::gravity::cartesian::Multipole<float, order>;
    using CartesianExpansion = cosmictiger::gravity::cartesian::Expansion<float, order>;
    using CartesianGreen = cosmictiger::gravity::cartesian::Green<float, order>;
    using SphericalMultipole = cosmictiger::gravity::spherical::Multipole<float, order>;
    using SphericalExpansion = cosmictiger::gravity::spherical::Expansion<float, order>;
    using SphericalGreen = cosmictiger::gravity::spherical::Green<float, order>;
    return sizeof(CartesianMultipole) == sizeof(float) * fmmMultipoleCoefficientCount(order) &&
        sizeof(CartesianExpansion) == sizeof(float) * fmmLocalCoefficientCount(order) &&
        sizeof(CartesianGreen) == sizeof(CartesianExpansion) &&
        sizeof(SphericalMultipole) == sizeof(CartesianMultipole) &&
        sizeof(SphericalExpansion) == sizeof(float) * fmmLocalHarmonicCoefficientCount(order) &&
        sizeof(SphericalGreen) == sizeof(SphericalExpansion) &&
        sizeof(FmmMultiPos<order>) == fmmPositionedMultipoleBytes(order) &&
        alignof(CartesianMultipole) == alignof(float) &&
        alignof(CartesianExpansion) == alignof(float) &&
        alignof(SphericalMultipole) == alignof(float) &&
        alignof(SphericalExpansion) == alignof(float) &&
        std::is_standard_layout_v<CartesianMultipole> &&
        std::is_standard_layout_v<CartesianExpansion> &&
        std::is_standard_layout_v<SphericalMultipole> &&
        std::is_standard_layout_v<SphericalExpansion> &&
        std::is_trivially_copyable_v<CartesianMultipole> &&
        std::is_trivially_copyable_v<CartesianExpansion> &&
        std::is_trivially_copyable_v<SphericalMultipole> &&
        std::is_trivially_copyable_v<SphericalExpansion>;
}
static_assert(fmmTypedLayout<3>() && fmmTypedLayout<4>() && fmmTypedLayout<5>() &&
    fmmTypedLayout<6>() && fmmTypedLayout<7>() && fmmTypedLayout<8>() && fmmTypedLayout<9>());

template <int order>
CUDA_EXPORT inline FmmMultiPos<order>& fmmMultiPos(MultipolePosition* record) {
    return *reinterpret_cast<FmmMultiPos<order>*>(record);
}

template <int order>
CUDA_EXPORT inline FmmMultiPos<order> const& fmmMultiPos(MultipolePosition const* record) {
    return *reinterpret_cast<FmmMultiPos<order> const*>(record);
}

// Transport buffers sized for the largest order. Order-agnostic interfaces
// (kick work items, HPX actions, tree returns) pass these; the per-order code
// views the leading coefficients through expansion<order>() / multipole<order>().
struct ExpansionBuffer {
    Array<float, kFmmMaxLocalCoefficients> coefficients;

    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmMaxLocalCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Expansion<float, order>& expansion() {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Expansion<float, order>*>(
            coefficients.data());
    }
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Expansion<float, order> const& expansion() const {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Expansion<float, order> const*>(
            coefficients.data());
    }
    template <int order>
    CUDA_EXPORT void assign(cosmictiger::gravity::cartesian::Expansion<float, order> const& value) {
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            coefficients[i] = value[i];
        }
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        int const count = fmmLocalCoefficientCount(fmmOrder());
        for (int i = 0; i < count; i++) {
            arc& coefficients[i];
        }
    }
};

struct MultipoleBuffer {
    Array<float, kFmmMaxMultipoleCoefficients> coefficients;

    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmMaxMultipoleCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Multipole<float, order>& multipole() {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Multipole<float, order>*>(
            coefficients.data());
    }
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Multipole<float, order> const& multipole() const {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Multipole<float, order> const*>(
            coefficients.data());
    }
    template <int order>
    CUDA_EXPORT void assign(cosmictiger::gravity::cartesian::Multipole<float, order> const& value) {
        for (int i = 0; i < FmmOrder<order>::kMultipoleCoefficients_; i++) {
            coefficients[i] = value[i];
        }
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        int const count = fmmMultipoleCoefficientCount(fmmOrder());
        for (int i = 0; i < count; i++) {
            arc& coefficients[i];
        }
    }
};
