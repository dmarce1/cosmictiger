/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef CONTAINERS_HPP_
#define CONTAINERS_HPP_

#include <array>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cuda/cuda.hpp>

#ifdef __CUDA_ARCH__
#define DO_BOUNDS_CHECK(i)                                                                   \
    if (i >= this->size()) {                                                                 \
        printf("Bounds error - %lld is not between 0 and %lld\n", static_cast<long long>(i), \
            static_cast<long long>(this->size()));                                           \
        __trap();                                                                            \
    }
#else
#define DO_BOUNDS_CHECK(i)                                                               \
    if (i >= this->size()) {                                                             \
        throw std::out_of_range(std::format("Bounds error - {} is not between 0 and {}", \
            static_cast<long long>(i), static_cast<long long>(this->size())));           \
    }
#endif

template <class T, class Alloc = std::allocator<T>>
class Vector : public std::vector<T, Alloc> {
    using BaseType = std::vector<T, Alloc>;
    using BaseType::BaseType;

public:
#ifdef CHECK_BOUNDS
    /// Access the indexed element; the index must lie within the logical bounds.
    inline T& operator[](size_t i) {
        DO_BOUNDS_CHECK(i);
        return BaseType::operator[](i);
    }
    /// Access the indexed element; the index must lie within the logical bounds.
    inline T const& operator[](size_t i) const {
        DO_BOUNDS_CHECK(i);
        return BaseType::operator[](i);
    }
#endif
};

template <class T, int n>
class Array {
    T a_[n];

public:
    /// Access the indexed element; the index must lie within the logical bounds.
    CUDA_EXPORT
    inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
        DO_BOUNDS_CHECK(i);
#endif
        return a_[i];
    }
    /// Access the indexed element; the index must lie within the logical bounds.
    CUDA_EXPORT
    inline T const& operator[](int i) const {
#ifdef CHECK_BOUNDS
        DO_BOUNDS_CHECK(i);
#endif
        return a_[i];
    }
    /// Expose the contiguous element storage without copying.
    CUDA_EXPORT
    inline T const* data() const {
        return a_;
    }
    /// Expose the contiguous element storage without copying.
    CUDA_EXPORT
    inline T* data() {
        return a_;
    }
    /// Return the logical number of stored elements.
    CUDA_EXPORT
    inline int size() const {
        return n;
    }
    /// Return a pointer to the first stored element.
    CUDA_EXPORT
    inline T const* begin() const {
        return a_;
    }
    /// Return a pointer to the first stored element.
    CUDA_EXPORT
    inline T* begin() {
        return a_;
    }
    /// Return the one-past-end pointer for the stored elements.
    CUDA_EXPORT
    inline T const* end() const {
        return a_ + n;
    }
    /// Return the one-past-end pointer for the stored elements.
    CUDA_EXPORT
    inline T* end() {
        return a_ + n;
    }
    /// Archive `a_` in transport order.
    template <class Arc>
    void serialize(Arc&& arc, unsigned) {
        for (int i = 0; i < n; i++) {
            arc& a_[i];
        }
    }
};

/// Return an array with every spatial component multiplied by the scalar.
template <class T>
inline Array<T, kDimensions> operator*(Array<T, kDimensions> const& a, T b) {
    Array<T, kDimensions> c;
    for (int dim = 0; dim < kDimensions; dim++) {
        c[dim] = a[dim] * b;
    }
    return c;
}

/// Compare corresponding stored values for equality.
template <class T>
inline bool operator==(Array<T, kDimensions> const& a, Array<T, kDimensions> const& b) {
    for (int dim = 0; dim < kDimensions; dim++) {
        if (a[dim] != b[dim]) {
            return false;
        }
    }
    return true;
}

/// Test whether any corresponding array component differs.
template <class T>
inline bool operator!=(Array<T, kDimensions> const& a, Array<T, kDimensions> const& b) {
    return !(a == b);
}

/// Return the componentwise sum of two spatial arrays.
template <class T>
inline Array<T, kDimensions> operator+(
    Array<T, kDimensions> const& a, Array<T, kDimensions> const& b) {
    Array<T, kDimensions> c;
    for (int dim = 0; dim < kDimensions; dim++) {
        c[dim] = a[dim] + b[dim];
    }
    return c;
}

/// Return the componentwise difference of two spatial arrays.
template <class T>
inline Array<T, kDimensions> operator-(
    Array<T, kDimensions> const& a, Array<T, kDimensions> const& b) {
    Array<T, kDimensions> c;
    for (int dim = 0; dim < kDimensions; dim++) {
        c[dim] = a[dim] - b[dim];
    }
    return c;
}

template <class T, class V = T>
struct Pair {
    T first;
    V second;
    Pair() = default;
    Pair(Pair const&) = default;
    Pair& operator=(Pair const&) = default;
    Pair(Pair&&) = default;
    Pair& operator=(Pair&&) = default;
    /// Initialize the first and second values from the supplied pair of arguments.
    Pair(T a, V b)
      : first(a)
      , second(b) {}
    /// Compare corresponding stored values for equality.
    inline bool operator==(Pair<T> other) const {
        return other.first == first && second == other.second;
    }
    /// Archive this record in the field order required by distributed transport.
    template <class A>
    void serialize(A&& a, unsigned) {
        a & first;
        a & second;
    }
};

#endif /* CONTAINERS_HPP_ */
