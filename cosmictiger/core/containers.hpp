/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef CONTAINERS_HPP_
#define CONTAINERS_HPP_

#include <array>
#include <vector>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cuda/cuda.hpp>

#ifdef __CUDA_ARCH__
#define DO_BOUNDS_CHECK(i)                                                           \
    if (i >= this->size()) {                                                         \
        THROW_ERROR("Bounds error - %i is not between 0 and %i\n", i, this->size()); \
        __trap();                                                                    \
    }
#else
#define DO_BOUNDS_CHECK(i)                                                           \
    if (i >= this->size()) {                                                         \
        THROW_ERROR("Bounds error - %i is not between 0 and %i\n", i, this->size()); \
    }
#endif

template <class T, class Alloc = std::allocator<T>>
class Vector : public std::vector<T, Alloc> {
    using BaseType = std::vector<T, Alloc>;
    using BaseType::BaseType;

public:
#ifdef CHECK_BOUNDS
    inline T& operator[](size_t i) {
        DO_BOUNDS_CHECK(i);
        return (*((std::vector<T>*) this))[i];
    }
    inline T const& operator[](size_t i) const {
        DO_BOUNDS_CHECK(i);
        return (*((std::vector<T>*) this))[i];
    }
#endif
};

template <class T, int n>
class Array {
    T a_[n];

public:
    CUDA_EXPORT
    inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
        DO_BOUNDS_CHECK(i);
#endif
        return a_[i];
    }
    CUDA_EXPORT
    inline T const& operator[](int i) const {
#ifdef CHECK_BOUNDS
        DO_BOUNDS_CHECK(i);
#endif
        return a_[i];
    }
    CUDA_EXPORT
    inline T const* data() const {
        return a_;
    }
    CUDA_EXPORT
    inline T* data() {
        return a_;
    }
    CUDA_EXPORT
    inline int size() const {
        return n;
    }
    CUDA_EXPORT
    inline T const* begin() const {
        return a_;
    }
    CUDA_EXPORT
    inline T* begin() {
        return a_;
    }
    CUDA_EXPORT
    inline T const* end() const {
        return a_ + n;
    }
    CUDA_EXPORT
    inline T* end() {
        return a_ + n;
    }
    template <class Arc>
    void serialize(Arc&& arc, unsigned) {
        for (int i = 0; i < n; i++) {
            arc& a_[i];
        }
    }
};

template <class T>
inline Array<T, NDIM> operator*(Array<T, NDIM> const& a, T b) {
    Array<T, NDIM> c;
    for (int dim = 0; dim < NDIM; dim++) {
        c[dim] = a[dim] * b;
    }
    return c;
}

template <class T>
inline bool operator==(Array<T, NDIM> const& a, Array<T, NDIM> const& b) {
    for (int dim = 0; dim < NDIM; dim++) {
        if (a[dim] != b[dim]) {
            return false;
        }
    }
    return true;
}

template <class T>
inline bool operator!=(Array<T, NDIM> const& a, Array<T, NDIM> const& b) {
    return !(a == b);
}

template <class T>
inline Array<T, NDIM> operator+(Array<T, NDIM> const& a, Array<T, NDIM> const& b) {
    Array<T, NDIM> c;
    for (int dim = 0; dim < NDIM; dim++) {
        c[dim] = a[dim] + b[dim];
    }
    return c;
}

template <class T>
inline Array<T, NDIM> operator-(Array<T, NDIM> const& a, Array<T, NDIM> const& b) {
    Array<T, NDIM> c;
    for (int dim = 0; dim < NDIM; dim++) {
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
    Pair(T a, V b)
      : first(a)
      , second(b) {}
    inline bool operator==(Pair<T> other) const {
        return other.first == first && second == other.second;
    }
    template <class A>
    void serialize(A&& a, unsigned) {
        a & first;
        a & second;
    }
};

#endif /* CONTAINERS_HPP_ */
