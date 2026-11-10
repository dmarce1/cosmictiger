/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp>
#include <cosmictiger/gravity/kernels/spherical_types.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/vector3.hpp>

// Order-dependent FMM value types and their inline basis casts. Tree operator
// kernels live separately in kernels/*_operators.hpp.

namespace cosmictiger::gravity {
// Physical point result in normalized FMM units. Acceleration is -grad(potential).
template <class Real>
struct Field : Array<Real, kDimensions + 1> {
    /// Access the potential component of the particle field.
    CUDA_EXPORT Real& potential() {
        return (*this)[0];
    }
    /// Access the potential component of the particle field.
    CUDA_EXPORT Real const& potential() const {
        return (*this)[0];
    }
    /// Access the x component of particle acceleration.
    CUDA_EXPORT Real& accelerationX() {
        return (*this)[1];
    }
    /// Access the x component of particle acceleration.
    CUDA_EXPORT Real const& accelerationX() const {
        return (*this)[1];
    }
    /// Access the y component of particle acceleration.
    CUDA_EXPORT Real& accelerationY() {
        return (*this)[2];
    }
    /// Access the y component of particle acceleration.
    CUDA_EXPORT Real const& accelerationY() const {
        return (*this)[2];
    }
    /// Access the z component of particle acceleration.
    CUDA_EXPORT Real& accelerationZ() {
        return (*this)[3];
    }
    /// Access the z component of particle acceleration.
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
static_assert(sizeof(Vector3<Position>) == kDimensions * sizeof(Position));
static_assert(alignof(Vector3<Position>) == alignof(Position));
static_assert(std::is_standard_layout_v<Vector3<Position>>);
static_assert(std::is_trivially_copyable_v<Vector3<Position>>);

struct MultipolePosition {
    Vector3<Position> pos;
};

template <int order>
struct FmmMultiPos {
    Vector3<Position> pos;
    cosmictiger::gravity::cartesian::Multipole<float, order> multi;
};

static_assert(std::is_standard_layout_v<FmmMultiPos<kFmmDefaultOrder>>);
static_assert(sizeof(FmmMultiPos<7>) == fmmPositionedMultipoleBytes(7));
static_assert(sizeof(FmmMultiPos<9>) == fmmPositionedMultipoleBytes(9));

/// Check coefficient counts, transport sizes, alignment assumptions, and trivial
/// typed-storage layout at compile time.
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

/// Interpret an aligned positioned-multipole record at its actual compile-time FMM order.
template <int order>
CUDA_EXPORT inline FmmMultiPos<order>& fmmMultiPos(MultipolePosition* record) {
    return *reinterpret_cast<FmmMultiPos<order>*>(record);
}

/// Interpret an aligned positioned-multipole record at its actual compile-time FMM order.
template <int order>
CUDA_EXPORT inline FmmMultiPos<order> const& fmmMultiPos(MultipolePosition const* record) {
    return *reinterpret_cast<FmmMultiPos<order> const*>(record);
}

// Transport buffers sized for the largest order. Order-agnostic interfaces
// (kick work items, HPX actions, tree returns) pass these; the per-order code
// views the leading coefficients through expansion<order>() / multipole<order>().
struct ExpansionBuffer {
    Array<float, kFmmMaxLocalCoefficients> coefficients;

    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    /// Clear the complete maximum-capacity coefficient buffer.
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmMaxLocalCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    /// View the coefficient prefix as a Cartesian local expansion of the supplied
    /// order.
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Expansion<float, order>& expansion() {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Expansion<float, order>*>(
            coefficients.data());
    }
    /// View the coefficient prefix as a Cartesian local expansion of the supplied
    /// order.
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Expansion<float, order> const& expansion() const {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Expansion<float, order> const*>(
            coefficients.data());
    }
    /// Copy the active coefficients from a typed expansion into the transport buffer.
    template <int order>
    CUDA_EXPORT void assign(cosmictiger::gravity::cartesian::Expansion<float, order> const& value) {
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            coefficients[i] = value[i];
        }
    }
    /// Archive `coefficients` in transport order.
    template <class A>
    void serialize(A&& arc, unsigned) {
        fmmDispatch([&](auto orderTag) {
            constexpr int order = decltype(orderTag)::value;
            for (int i = 0; i < fmmLocalCoefficientCount(order); i++) {
                arc& coefficients[i];
            }
        });
    }
};

static_assert(sizeof(ExpansionBuffer) == sizeof(float) * kFmmMaxLocalCoefficients);

struct MultipoleBuffer {
    Array<float, kFmmMaxMultipoleCoefficients> coefficients;

    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    /// Clear the complete maximum-capacity coefficient buffer.
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmMaxMultipoleCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    /// View the coefficient prefix as a Cartesian source multipole of the supplied
    /// order.
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Multipole<float, order>& multipole() {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Multipole<float, order>*>(
            coefficients.data());
    }
    /// View the coefficient prefix as a Cartesian source multipole of the supplied
    /// order.
    template <int order>
    CUDA_EXPORT cosmictiger::gravity::cartesian::Multipole<float, order> const& multipole() const {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Multipole<float, order> const*>(
            coefficients.data());
    }
    /// Copy the active coefficients from a typed expansion into the transport buffer.
    template <int order>
    CUDA_EXPORT void assign(cosmictiger::gravity::cartesian::Multipole<float, order> const& value) {
        for (int i = 0; i < FmmOrder<order>::kMultipoleCoefficients_; i++) {
            coefficients[i] = value[i];
        }
    }
    /// Archive `coefficients` in transport order.
    template <class A>
    void serialize(A&& arc, unsigned) {
        fmmDispatch([&](auto orderTag) {
            constexpr int order = decltype(orderTag)::value;
            for (int i = 0; i < fmmMultipoleCoefficientCount(order); i++) {
                arc& coefficients[i];
            }
        });
    }
};
