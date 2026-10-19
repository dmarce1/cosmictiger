/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cosmictiger/math/simd_math.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_VECTOR3_EXPORT __host__ __device__
#else
#define COSMICTIGER_VECTOR3_EXPORT
#endif

// Three-component spatial vector, modeled on Octotiger's Vector<T, N>. The element
// type supplies the arithmetic: Vector3<float> and Vector3<double> work in host and
// device code, and Vector3<SimdFloat> operates lane-wise on a pack of neighbors.
template <class T>
class Vector3 {
    T components_[3];

public:
    Vector3() = default;
    COSMICTIGER_VECTOR3_EXPORT explicit Vector3(T const& value)
      : components_{value, value, value} {}
    COSMICTIGER_VECTOR3_EXPORT Vector3(T const& x, T const& y, T const& z)
      : components_{x, y, z} {}
    // Converts each component, for example Vector3<float> to Vector3<SimdFloat>.
    template <class U>
    COSMICTIGER_VECTOR3_EXPORT explicit Vector3(Vector3<U> const& other)
      : components_{T(other[0]), T(other[1]), T(other[2])} {}
    COSMICTIGER_VECTOR3_EXPORT static constexpr int size() {
        return 3;
    }
    COSMICTIGER_VECTOR3_EXPORT T& operator[](int dim) {
        return components_[dim];
    }
    COSMICTIGER_VECTOR3_EXPORT T const& operator[](int dim) const {
        return components_[dim];
    }
    COSMICTIGER_VECTOR3_EXPORT Vector3& operator+=(Vector3 const& other) {
        for (int dim = 0; dim < 3; dim++) {
            components_[dim] += other.components_[dim];
        }
        return *this;
    }
    COSMICTIGER_VECTOR3_EXPORT Vector3& operator-=(Vector3 const& other) {
        for (int dim = 0; dim < 3; dim++) {
            components_[dim] -= other.components_[dim];
        }
        return *this;
    }
    COSMICTIGER_VECTOR3_EXPORT Vector3& operator*=(T const& scalar) {
        // The scalar may alias one of our components, as in v *= v[0].
        T const value = scalar;
        for (int dim = 0; dim < 3; dim++) {
            components_[dim] *= value;
        }
        return *this;
    }
    COSMICTIGER_VECTOR3_EXPORT Vector3& operator/=(T const& scalar) {
        return *this *= T(1) / scalar;
    }
    COSMICTIGER_VECTOR3_EXPORT T dot(Vector3 const& other) const {
        return components_[0] * other.components_[0] + components_[1] * other.components_[1] +
            components_[2] * other.components_[2];
    }
    COSMICTIGER_VECTOR3_EXPORT T squaredNorm() const {
        return dot(*this);
    }
    COSMICTIGER_VECTOR3_EXPORT T norm() const {
        return simdSqrt(squaredNorm());
    }
    friend COSMICTIGER_VECTOR3_EXPORT Vector3 operator+(Vector3 const& value) {
        return value;
    }
    friend COSMICTIGER_VECTOR3_EXPORT Vector3 operator-(Vector3 const& value) {
        return Vector3(-value.components_[0], -value.components_[1], -value.components_[2]);
    }
    friend COSMICTIGER_VECTOR3_EXPORT Vector3 operator+(Vector3 left, Vector3 const& right) {
        left += right;
        return left;
    }
    friend COSMICTIGER_VECTOR3_EXPORT Vector3 operator-(Vector3 left, Vector3 const& right) {
        left -= right;
        return left;
    }
    friend COSMICTIGER_VECTOR3_EXPORT Vector3 operator*(Vector3 value, T const& scalar) {
        value *= scalar;
        return value;
    }
    friend COSMICTIGER_VECTOR3_EXPORT Vector3 operator*(T const& scalar, Vector3 value) {
        value *= scalar;
        return value;
    }
    friend COSMICTIGER_VECTOR3_EXPORT Vector3 operator/(Vector3 value, T const& scalar) {
        value /= scalar;
        return value;
    }
};

// Component-wise selection; Mask is bool for scalar element types and a SIMD mask for packs.
template <class Mask, class T>
COSMICTIGER_VECTOR3_EXPORT inline Vector3<T> simdSelect(
    Mask const& mask, Vector3<T> const& yes, Vector3<T> const& no) {
    return Vector3<T>(simdSelect(mask, yes[0], no[0]), simdSelect(mask, yes[1], no[1]),
        simdSelect(mask, yes[2], no[2]));
}

#undef COSMICTIGER_VECTOR3_EXPORT
