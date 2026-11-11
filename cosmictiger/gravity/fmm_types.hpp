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

// Build-order FMM value types and their inline basis casts. Tree operator
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

// Positioned multipole record. The center comes first so generic tree
// code reads it through MultipolePosition; the typed view adds the moments.
static_assert(sizeof(Vector3<Position>) == kDimensions * sizeof(Position));
static_assert(alignof(Vector3<Position>) == alignof(Position));
static_assert(std::is_standard_layout_v<Vector3<Position>>);
static_assert(std::is_trivially_copyable_v<Vector3<Position>>);

struct MultipolePosition {
    Vector3<Position> pos;
};

struct FmmMultiPos {
    Vector3<Position> pos;
    cosmictiger::gravity::cartesian::Multipole<float> multi;
};

static_assert(std::is_standard_layout_v<FmmMultiPos>);
static_assert(sizeof(FmmMultiPos) == fmmPositionedMultipoleBytes(kFmmOrder));

/// Check coefficient counts, transport sizes, alignment assumptions, and trivial
/// typed-storage layout at compile time.
constexpr bool fmmTypedLayout() {
    using CartesianMultipole = cosmictiger::gravity::cartesian::Multipole<float>;
    using CartesianExpansion = cosmictiger::gravity::cartesian::Expansion<float>;
    using CartesianGreen = cosmictiger::gravity::cartesian::Green<float>;
    using SphericalMultipole = cosmictiger::gravity::spherical::Multipole<float>;
    using SphericalExpansion = cosmictiger::gravity::spherical::Expansion<float>;
    using SphericalGreen = cosmictiger::gravity::spherical::Green<float>;
    return sizeof(CartesianMultipole) == sizeof(float) * fmmMultipoleCoefficientCount(kFmmOrder) &&
        sizeof(CartesianExpansion) == sizeof(float) * fmmLocalCoefficientCount(kFmmOrder) &&
        sizeof(CartesianGreen) == sizeof(CartesianExpansion) &&
        sizeof(SphericalMultipole) == sizeof(CartesianMultipole) &&
        sizeof(SphericalExpansion) == sizeof(float) * fmmLocalHarmonicCoefficientCount(kFmmOrder) &&
        sizeof(SphericalGreen) == sizeof(SphericalExpansion) &&
        sizeof(FmmMultiPos) == fmmPositionedMultipoleBytes(kFmmOrder) &&
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
static_assert(fmmTypedLayout());

/// Interpret an aligned positioned-multipole record at its actual build FMM order.
CUDA_EXPORT inline FmmMultiPos& fmmMultiPos(MultipolePosition* record) {
    return *reinterpret_cast<FmmMultiPos*>(record);
}

/// Interpret an aligned positioned-multipole record at its actual build FMM order.
CUDA_EXPORT inline FmmMultiPos const& fmmMultiPos(MultipolePosition const* record) {
    return *reinterpret_cast<FmmMultiPos const*>(record);
}

// Transport buffers sized exactly for the build order. Shared interfaces
// (kick work items, HPX actions, tree returns) pass these; the typed code
// views the leading coefficients through expansion() / multipole().
struct ExpansionBuffer {
    Array<float, kFmmLocalCoefficients> coefficients;

    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    /// Clear the build-sized coefficient buffer.
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmLocalCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    /// View the coefficient prefix as a Cartesian local expansion at the build order.
    CUDA_EXPORT cosmictiger::gravity::cartesian::Expansion<float>& expansion() {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Expansion<float>*>(
            coefficients.data());
    }
    /// View the coefficient prefix as a Cartesian local expansion at the build order.
    CUDA_EXPORT cosmictiger::gravity::cartesian::Expansion<float> const& expansion() const {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Expansion<float> const*>(
            coefficients.data());
    }
    /// Copy the active coefficients from a typed expansion into the transport buffer.
    CUDA_EXPORT void assign(cosmictiger::gravity::cartesian::Expansion<float> const& value) {
        for (int i = 0; i < kFmmLocalCoefficients; i++) {
            coefficients[i] = value[i];
        }
    }
    /// Archive `coefficients` in transport order.
    template <class A>
    void serialize(A&& arc, unsigned) {
        for (int i = 0; i < kFmmLocalCoefficients; i++) {
            arc& coefficients[i];
        }
    }
};

static_assert(sizeof(ExpansionBuffer) == sizeof(float) * kFmmLocalCoefficients);

struct MultipoleBuffer {
    Array<float, kFmmMultipoleCoefficients> coefficients;

    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    /// Access a coefficient by its packed storage index.
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    /// Clear the build-sized coefficient buffer.
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmMultipoleCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    /// View the coefficient prefix as a Cartesian source multipole at the build order.
    CUDA_EXPORT cosmictiger::gravity::cartesian::Multipole<float>& multipole() {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Multipole<float>*>(
            coefficients.data());
    }
    /// View the coefficient prefix as a Cartesian source multipole at the build order.
    CUDA_EXPORT cosmictiger::gravity::cartesian::Multipole<float> const& multipole() const {
        return *reinterpret_cast<cosmictiger::gravity::cartesian::Multipole<float> const*>(
            coefficients.data());
    }
    /// Copy the active coefficients from a typed expansion into the transport buffer.
    CUDA_EXPORT void assign(cosmictiger::gravity::cartesian::Multipole<float> const& value) {
        for (int i = 0; i < kFmmMultipoleCoefficients; i++) {
            coefficients[i] = value[i];
        }
    }
    /// Archive `coefficients` in transport order.
    template <class A>
    void serialize(A&& arc, unsigned) {
        for (int i = 0; i < kFmmMultipoleCoefficients; i++) {
            arc& coefficients[i];
        }
    }
};
