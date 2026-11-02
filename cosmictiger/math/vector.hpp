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

    template <size_t... indices>
    CUDA_EXPORT constexpr Vector(T const& value, std::index_sequence<indices...>)
      : components_{(static_cast<void>(indices), value)...} {}

    template <class U, class OtherDerived, size_t... indices>
    CUDA_EXPORT constexpr Vector(
        Vector<U, dimensions, OtherDerived> const& other, std::index_sequence<indices...>)
      : components_{T(other[indices])...} {}

    CUDA_EXPORT constexpr Result& result() {
        return static_cast<Result&>(*this);
    }

    template <size_t... indices>
    CUDA_EXPORT constexpr T dotComponents(
        Result const& other, std::index_sequence<indices...>) const {
        // A left fold preserves Vector3's association and its three multiplies/two adds.
        return (... + (components_[indices] * other[indices]));
    }

    template <size_t... indices>
    CUDA_EXPORT constexpr Result negateComponents(std::index_sequence<indices...>) const {
        return Result(-components_[indices]...);
    }

    template <size_t... indices>
    CUDA_EXPORT static constexpr auto equalComponents(
        Result const& left, Result const& right, std::index_sequence<indices...>) {
        return (... && (left[indices] == right[indices]));
    }

public:
    CUDA_EXPORT constexpr Vector() = default;

    CUDA_EXPORT constexpr explicit Vector(T const& value)
      : Vector(value, std::make_index_sequence<dimensions>{}) {}

    // A one-component vector already has the scalar/broadcast constructor.
    template <class... Values>
        requires(dimensions > 1 && sizeof...(Values) == dimensions &&
            (std::is_convertible_v<Values const&, T> && ...))
    CUDA_EXPORT constexpr Vector(Values const&... values)
      : components_{T(values)...} {}

    template <class U, class OtherDerived>
    CUDA_EXPORT constexpr explicit Vector(Vector<U, dimensions, OtherDerived> const& other)
      : Vector(other, std::make_index_sequence<dimensions>{}) {}

    CUDA_EXPORT static constexpr int size() {
        return dimensions;
    }
    CUDA_EXPORT constexpr T& operator[](int dim) {
        return components_[dim];
    }
    CUDA_EXPORT constexpr T const& operator[](int dim) const {
        return components_[dim];
    }
    CUDA_EXPORT constexpr T* data() {
        return components_;
    }
    CUDA_EXPORT constexpr T const* data() const {
        return components_;
    }
    CUDA_EXPORT constexpr T* begin() {
        return data();
    }
    CUDA_EXPORT constexpr T const* begin() const {
        return data();
    }
    CUDA_EXPORT constexpr T* end() {
        return data() + size();
    }
    CUDA_EXPORT constexpr T const* end() const {
        return data() + size();
    }
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        for (int dim = 0; dim < size(); dim++) {
            archive& components_[dim];
        }
    }
    CUDA_EXPORT constexpr Result& operator+=(Result const& other) {
        for (int dim = 0; dim < size(); dim++) {
            components_[dim] += other[dim];
        }
        return result();
    }
    CUDA_EXPORT constexpr Result& operator-=(Result const& other) {
        for (int dim = 0; dim < size(); dim++) {
            components_[dim] -= other[dim];
        }
        return result();
    }
    CUDA_EXPORT constexpr Result& operator*=(T const& scalar) {
        // The scalar may alias one of our components, as in v *= v[0].
        T const value = scalar;
        for (int dim = 0; dim < size(); dim++) {
            components_[dim] *= value;
        }
        return result();
    }
    CUDA_EXPORT constexpr Result& operator/=(T const& scalar) {
        return *this *= T(1) / scalar;
    }
    CUDA_EXPORT constexpr T dot(Result const& other) const {
        return dotComponents(other, std::make_index_sequence<dimensions>{});
    }
    CUDA_EXPORT constexpr T squaredNorm() const {
        return dot(static_cast<Result const&>(*this));
    }
    CUDA_EXPORT T norm() const {
        return simdSqrt(squaredNorm());
    }
    // SIMD elements return a per-lane mask; scalar elements return bool.
    friend CUDA_EXPORT constexpr auto operator==(Result const& left, Result const& right) {
        return equalComponents(left, right, std::make_index_sequence<dimensions>{});
    }
    friend CUDA_EXPORT constexpr auto operator!=(Result const& left, Result const& right) {
        return !(left == right);
    }
    friend CUDA_EXPORT constexpr Result operator+(Result const& value) {
        return value;
    }
    friend CUDA_EXPORT constexpr Result operator-(Result const& value) {
        return static_cast<Vector const&>(value).negateComponents(
            std::make_index_sequence<dimensions>{});
    }
    friend CUDA_EXPORT constexpr Result operator+(Result left, Result const& right) {
        left += right;
        return left;
    }
    friend CUDA_EXPORT constexpr Result operator-(Result left, Result const& right) {
        left -= right;
        return left;
    }
    friend CUDA_EXPORT constexpr Result operator*(Result value, T const& scalar) {
        value *= scalar;
        return value;
    }
    friend CUDA_EXPORT constexpr Result operator*(T const& scalar, Result value) {
        value *= scalar;
        return value;
    }
    friend CUDA_EXPORT constexpr Result operator/(Result value, T const& scalar) {
        value /= scalar;
        return value;
    }
};

namespace detail {

    template <class Result, class Mask, class Input, size_t... indices>
    CUDA_EXPORT inline Result selectVector(
        Mask const& mask, Input const& yes, Input const& no, std::index_sequence<indices...>) {
        return Result(simdSelect(mask, yes[indices], no[indices])...);
    }

}    // namespace detail
}    // namespace math

template <class Mask, class T, int dimensions, class Derived>
CUDA_EXPORT inline auto simdSelect(Mask const& mask,
    math::Vector<T, dimensions, Derived> const& yes,
    math::Vector<T, dimensions, Derived> const& no) {
    using Result =
        std::conditional_t<std::is_void_v<Derived>, math::Vector<T, dimensions, Derived>, Derived>;
    return math::detail::selectVector<Result>(
        mask, yes, no, std::make_index_sequence<dimensions>{});
}
