/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/math/vector.hpp>

// Preserve the spatial-vector interface while sharing fixed-dimensional storage
// and arithmetic with math::Vector. Only the cross product requires three dimensions.
template <class T>
class Vector3 : public math::Vector<T, 3, Vector3<T>> {
    using Base = math::Vector<T, 3, Vector3<T>>;

public:
    CUDA_EXPORT constexpr Vector3() = default;
    /// Broadcast one scalar value into all three components.
    CUDA_EXPORT constexpr explicit Vector3(T const& value)
      : Base(value) {}
    /// Construct a three-component vector from its x, y, and z values.
    CUDA_EXPORT constexpr Vector3(T const& x, T const& y, T const& z)
      : Base(x, y, z) {}
    // Converts each component, for example Vector3<float> to Vector3<SimdFloat>.
    template <class U>
    CUDA_EXPORT constexpr explicit Vector3(Vector3<U> const& other)
      : Base(static_cast<math::Vector<U, 3, Vector3<U>> const&>(other)) {}

    /// Return the right-handed three-dimensional cross product with the other vector.
    CUDA_EXPORT constexpr Vector3 cross(Vector3 const& other) const {
        return {(*this)[1] * other[2] - (*this)[2] * other[1],
            (*this)[2] * other[0] - (*this)[0] * other[2],
            (*this)[0] * other[1] - (*this)[1] * other[0]};
    }
};

// Keep this exact overload: the scalar/SIMD helper also accepts arbitrary Real
// types, so relying only on deduction through the math::Vector base is insufficient.
namespace math {
/// Select each of the three components from yes or no using a common scalar/SIMD mask.
template <class Mask, class T>
CUDA_EXPORT inline Vector3<T> select(
    Mask const& mask, Vector3<T> const& yes, Vector3<T> const& no) {
    return math::detail::selectVector<Vector3<T>>(mask, yes, no, std::make_index_sequence<3>{});
}
}    // namespace math
