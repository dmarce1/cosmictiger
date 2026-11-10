/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <type_traits>
#include <utility>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace math {

// Fixed-dimensional arithmetic vector. Derived optionally preserves a specialized
// vector's result type while sharing its storage and all dimension-independent math.
// The existing global Vector<T, Allocator> remains the dynamic project container.
template <class T, int dimensions, class Derived = void>
class Vector {
    static_assert(dimensions > 0);
    using Result = std::conditional_t<std::is_void_v<Derived>, Vector, Derived>;

    T components_[dimensions];

    /// Broadcast the scalar into all components using the compile-time index sequence.
    template <size_t... indices>
    CUDA_EXPORT constexpr Vector(T const& value, std::index_sequence<indices...>)
      : components_{(static_cast<void>(indices), value)...} {}

    /// Convert each source component to the destination scalar type.
    template <class U, class OtherDerived, size_t... indices>
    CUDA_EXPORT constexpr Vector(
        Vector<U, dimensions, OtherDerived> const& other, std::index_sequence<indices...>)
      : components_{T(other[indices])...} {}

    /// Expose this base object as the configured vector result type.
    CUDA_EXPORT constexpr Result& result() {
        return static_cast<Result&>(*this);
    }

    /// Form the dot product with a left-associated sum over the compile-time component
    /// indices.
    template <size_t... indices>
    CUDA_EXPORT constexpr T dotComponents(
        Result const& other, std::index_sequence<indices...>) const {
        // A left fold preserves Vector3's association and its three multiplies/two adds.
        return (... + (components_[indices] * other[indices]));
    }

    /// Return a vector with each selected component negated.
    template <size_t... indices>
    CUDA_EXPORT constexpr Result negateComponents(std::index_sequence<indices...>) const {
        return Result(-components_[indices]...);
    }

    /// Combine componentwise equality tests into the scalar or SIMD equality mask.
    template <size_t... indices>
    CUDA_EXPORT static constexpr auto equalComponents(
        Result const& left, Result const& right, std::index_sequence<indices...>) {
        return (... && (left[indices] == right[indices]));
    }

public:
    CUDA_EXPORT constexpr Vector() = default;

    /// Broadcast one scalar value to every vector component.
    CUDA_EXPORT constexpr explicit Vector(T const& value)
      : Vector(value, std::make_index_sequence<dimensions>{}) {}

    // A one-component vector already has the scalar/broadcast constructor.
    template <class... Values>
        requires(dimensions > 1 && sizeof...(Values) == dimensions &&
            (std::is_convertible_v<Values const&, T> && ...))
    CUDA_EXPORT constexpr Vector(Values const&... values)
      : components_{T(values)...} {}

    /// Construct a vector by converting each component of another vector.
    template <class U, class OtherDerived>
    CUDA_EXPORT constexpr explicit Vector(Vector<U, dimensions, OtherDerived> const& other)
      : Vector(other, std::make_index_sequence<dimensions>{}) {}

    /// Return the logical number of stored elements.
    CUDA_EXPORT static constexpr int size() {
        return dimensions;
    }
    /// Access the indexed element; the index must lie within the logical bounds.
    CUDA_EXPORT constexpr T& operator[](int dim) {
        return components_[dim];
    }
    /// Access the indexed element; the index must lie within the logical bounds.
    CUDA_EXPORT constexpr T const& operator[](int dim) const {
        return components_[dim];
    }
    /// Expose the contiguous element storage without copying.
    CUDA_EXPORT constexpr T* data() {
        return components_;
    }
    /// Expose the contiguous element storage without copying.
    CUDA_EXPORT constexpr T const* data() const {
        return components_;
    }
    /// Return a pointer to the first stored element.
    CUDA_EXPORT constexpr T* begin() {
        return data();
    }
    /// Return a pointer to the first stored element.
    CUDA_EXPORT constexpr T const* begin() const {
        return data();
    }
    /// Return the one-past-end pointer for the stored elements.
    CUDA_EXPORT constexpr T* end() {
        return data() + size();
    }
    /// Return the one-past-end pointer for the stored elements.
    CUDA_EXPORT constexpr T const* end() const {
        return data() + size();
    }
    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        for (int dim = 0; dim < size(); dim++) {
            archive& components_[dim];
        }
    }
    /// Add another vector componentwise in place and return this vector.
    CUDA_EXPORT constexpr Result& operator+=(Result const& other) {
        for (int dim = 0; dim < size(); dim++) {
            components_[dim] += other[dim];
        }
        return result();
    }
    /// Subtract another vector componentwise in place and return this vector.
    CUDA_EXPORT constexpr Result& operator-=(Result const& other) {
        for (int dim = 0; dim < size(); dim++) {
            components_[dim] -= other[dim];
        }
        return result();
    }
    /// Scale every component in place, preserving a scalar argument that aliases a
    /// component.
    CUDA_EXPORT constexpr Result& operator*=(T const& scalar) {
        // The scalar may alias one of our components, as in v *= v[0].
        T const value = scalar;
        for (int dim = 0; dim < size(); dim++) {
            components_[dim] *= value;
        }
        return result();
    }
    /// Scale by the reciprocal of the supplied scalar and return this vector.
    CUDA_EXPORT constexpr Result& operator/=(T const& scalar) {
        return *this *= T(1) / scalar;
    }
    /// Return the scalar or lane-wise Euclidean dot product.
    CUDA_EXPORT constexpr T dot(Result const& other) const {
        return dotComponents(other, std::make_index_sequence<dimensions>{});
    }
    /// Return the dot product of the vector with itself.
    CUDA_EXPORT constexpr T squaredNorm() const {
        return dot(static_cast<Result const&>(*this));
    }
    /// Return the Euclidean magnitude, independently in each SIMD lane when applicable.
    CUDA_EXPORT T norm() const {
        return math::sqrt(squaredNorm());
    }
    // SIMD elements return a per-lane mask; scalar elements return bool.
    friend CUDA_EXPORT constexpr auto operator==(Result const& left, Result const& right) {
        return equalComponents(left, right, std::make_index_sequence<dimensions>{});
    }
    /// Negate the vector equality result, preserving scalar or SIMD mask semantics.
    friend CUDA_EXPORT constexpr auto operator!=(Result const& left, Result const& right) {
        return !(left == right);
    }
    /// Return an unchanged copy for unary plus.
    friend CUDA_EXPORT constexpr Result operator+(Result const& value) {
        return value;
    }
    /// Return the componentwise negation of the vector.
    friend CUDA_EXPORT constexpr Result operator-(Result const& value) {
        return static_cast<Vector const&>(value).negateComponents(
            std::make_index_sequence<dimensions>{});
    }
    /// Return the componentwise sum of two vectors.
    friend CUDA_EXPORT constexpr Result operator+(Result left, Result const& right) {
        left += right;
        return left;
    }
    /// Return the componentwise difference of two vectors.
    friend CUDA_EXPORT constexpr Result operator-(Result left, Result const& right) {
        left -= right;
        return left;
    }
    /// Return a copy with each component scaled by the scalar.
    friend CUDA_EXPORT constexpr Result operator*(Result value, T const& scalar) {
        value *= scalar;
        return value;
    }
    /// Return a copy with each component scaled by the scalar.
    friend CUDA_EXPORT constexpr Result operator*(T const& scalar, Result value) {
        value *= scalar;
        return value;
    }
    /// Return a copy divided componentwise by the scalar.
    friend CUDA_EXPORT constexpr Result operator/(Result value, T const& scalar) {
        value /= scalar;
        return value;
    }
};

namespace detail {

    /// Choose each component from yes or no using the same scalar/SIMD mask.
    template <class Result, class Mask, class Input, size_t... indices>
    CUDA_EXPORT inline Result selectVector(
        Mask const& mask, Input const& yes, Input const& no, std::index_sequence<indices...>) {
        return Result(math::select(mask, yes[indices], no[indices])...);
    }

}    // namespace detail

/// Select between vectors componentwise while preserving the specialized result type.
template <class Mask, class T, int dimensions, class Derived>
CUDA_EXPORT inline auto select(Mask const& mask, math::Vector<T, dimensions, Derived> const& yes,
    math::Vector<T, dimensions, Derived> const& no) {
    using Result =
        std::conditional_t<std::is_void_v<Derived>, math::Vector<T, dimensions, Derived>, Derived>;
    return math::detail::selectVector<Result>(
        mask, yes, no, std::make_index_sequence<dimensions>{});
}
}    // namespace math
