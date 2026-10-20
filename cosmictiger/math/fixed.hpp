/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cstdint>
#include <limits.h>
#include <limits>
#include <utility>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/simd.hpp>

template <class, int>
class Fixed;

using Fixed32 = Fixed<uint32_t, 32>;
using Fixed64 = Fixed<uint64_t, 32>;

static constexpr float kFixed2float = 1.f / float(size_t(1) << size_t(32));

template <class T, int sigbits = 32>
class Fixed {
    T i_;
    static constexpr float kC0_ = float(size_t(1) << size_t(sigbits));
    static constexpr float kC0dbl_ = double(size_t(1) << size_t(sigbits));
    static constexpr float kCinv_ = 1.f / kC0_;
    static constexpr double kDblecinv_ = 1.f / kC0dbl_;
    static constexpr T kWidth_ = (sigbits);

public:
    static constexpr double kToFloatFactor_ = 1.f / kC0dbl_;
    friend class SimdFixed32;

    inline void setInteger(T j) {
        i_ = j;
    }

    template <int m>
    CUDA_EXPORT inline Fixed operator=(Fixed<T, m> const& other) {
        if (m > sigbits) {
            i_ = unsigned(other.i_) >> (m - sigbits);
        } else {
            i_ = unsigned(other.i_) << (sigbits - m);
        }
        return *this;
    }

    CUDA_EXPORT
    inline T raw() const {
        return i_;
    }

    CUDA_EXPORT
    inline T& raw() {
        return i_;
    }

    CUDA_EXPORT
    inline static Fixed<T, sigbits> max() {
        Fixed<T, sigbits> num;
#ifdef __CUDA_ARCH__
        num.i_ = 0xFFFFFFFFUL;
#else
        num.i_ = std::numeric_limits<T>::max();
#endif
        return num;
    }
    CUDA_EXPORT
    inline static Fixed<T, sigbits> min() {
        Fixed<T, sigbits> num;
        num.i_ = 1;
        return num;
    }
    inline Fixed<T, sigbits>() = default;

    CUDA_EXPORT
    inline Fixed<T, sigbits>& operator=(double number) {
        i_ = (kC0dbl_ * number);
        return *this;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits>(float number)
      : i_(kC0_ * number) {}
    CUDA_EXPORT
    inline Fixed<T, sigbits>(int number)
      : i_(kC0_ * number) {}
    CUDA_EXPORT
    inline Fixed<T, sigbits>(double number)
      : i_(kC0dbl_ * number) {}

    template <class V, int m>
    CUDA_EXPORT inline constexpr Fixed<T, sigbits>(Fixed<V, m> other) {
        if (m > sigbits) {
            i_ = unsigned(other.i_) >> abs(m - sigbits);
        } else {
            i_ = unsigned(other.i_) << abs(sigbits - m);
        }
    }

    CUDA_EXPORT
    inline bool operator<(Fixed other) const {
        return i_ < other.i_;
    }

    CUDA_EXPORT
    inline bool operator>(Fixed other) const {
        return i_ > other.i_;
    }

    CUDA_EXPORT
    inline bool operator<=(Fixed other) const {
        return i_ <= other.i_;
    }

    CUDA_EXPORT
    inline bool operator>=(Fixed other) const {
        return i_ >= other.i_;
    }

    CUDA_EXPORT
    inline bool operator==(Fixed other) const {
        return i_ == other.i_;
    }

    CUDA_EXPORT
    inline bool operator!=(Fixed other) const {
        return i_ != other.i_;
    }

    CUDA_EXPORT
    inline float toFloat() const {
        return (float(i_)) * kCinv_;
    }

    CUDA_EXPORT
    inline int toInt() const {
        return i_ >> kWidth_;
    }

    CUDA_EXPORT
    inline double toDouble() const {
        return (i_) *kDblecinv_;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator*(Fixed<T, sigbits> const& other) const {
        int64_t a;
        int64_t const b = i_;
        int64_t const c = other.i_;
        a = (b * c) >> kWidth_;
        Fixed<T, sigbits> res;
        res.i_ = (T) a;
        return res;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator*=(Fixed<T, sigbits> const& other) {
        int64_t a;
        int64_t const b = i_;
        int64_t const c = other.i_;
        a = (b * c) >> kWidth_;
        i_ = (T) a;
        return *this;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator*=(int other) {
        int64_t a;
        int64_t const b = i_;
        int64_t const c = other;
        a = b * c;
        i_ = (T) a;
        return *this;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator/(Fixed<T, sigbits> const& other) const {
        int64_t a;
        int64_t const b = i_;
        int64_t const c = other.i_;
        a = b / (c >> kWidth_);
        Fixed<T, sigbits> res;
        res.i_ = (T) a;
        return res;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator/=(Fixed<T, sigbits> const& other) {
        int64_t a;
        int64_t const b = i_;
        int64_t const c = other.i_;
        a = b / (c >> kWidth_);
        i_ = (T) a;
        return *this;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator+(Fixed<T, sigbits> const& other) const {
        Fixed<T, sigbits> a;
        a.i_ = i_ + other.i_;
        return a;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator-(Fixed<T, sigbits> const& other) const {
        Fixed<T, sigbits> a;
        a.i_ = i_ - other.i_;
        return a;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits> operator-() const {
        Fixed<T, sigbits> a;
        a.i_ = -i_;
        return a;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits>& operator+=(Fixed<T, sigbits> const& other) {
        i_ += other.i_;
        return *this;
    }

    CUDA_EXPORT
    inline Fixed<T, sigbits>& operator-=(Fixed<T, sigbits> const& other) {
        i_ -= other.i_;
        return *this;
    }

    CUDA_EXPORT
    inline T getInteger() const {
        return i_;
    }

    template <class A>
    void serialize(A& arc, unsigned) {
        arc & i_;
    }

    template <class, int>
    friend class Fixed;

    template <class V>
    friend void swap(Fixed<V>& first, Fixed<V>& second);

    friend Fixed32 randFixed32();
};

template <class T, int m>
CUDA_EXPORT inline Fixed<T, m> max(Fixed<T, m> const& a, Fixed<T, m> const& b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

template <class T, int m>
CUDA_EXPORT inline Fixed<T, m> min(Fixed<T, m> const& a, Fixed<T, m> const& b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

template <class T>
inline void swap(Fixed<T>& first, Fixed<T>& second) {
    std::swap(first, second);
}

CUDA_EXPORT inline float distance(Fixed32 a, Fixed32 b) {
    return (Fixed<int32_t>(a) - Fixed<int32_t>(b)).toFloat();
}

// Subtract Q0.32 positions before narrowing, without periodic minimum-image wrapping.
// The magnitude fits uint32_t, so the one integer-to-float rounding matches
// float(a.toDouble() - b.toDouble()) without device FP64 arithmetic.
CUDA_EXPORT inline float unwrappedDistance(Fixed32 a, Fixed32 b) {
    uint32_t const aRaw = a.raw();
    uint32_t const bRaw = b.raw();
    uint32_t const magnitude = aRaw >= bRaw ? aRaw - bRaw : bRaw - aRaw;
#ifdef __CUDA_ARCH__
    float const separation = __uint2float_rn(magnitude) * kFixed2float;
#else
    float const separation = static_cast<float>(magnitude) * kFixed2float;
#endif
    return aRaw >= bRaw ? separation : -separation;
}

template <int n>
CUDA_EXPORT inline float distance(Fixed<int, n> a, Fixed<int, n> b) {
    return (a - b).toFloat();
}

CUDA_EXPORT inline float distance(double a, double b) {
    double dif = a - b;
    while (dif > 0.5) {
        dif -= 1.0;
    }
    while (dif < -0.5) {
        dif += 1.0;
    }
    return dif;
}

CUDA_EXPORT inline float distance(Fixed32 a, double b) {
    double dif = a.toDouble() - b;
    while (dif > 0.5) {
        dif -= 1.0;
    }
    while (dif < -0.5) {
        dif += 1.0;
    }
    return dif;
}

CUDA_EXPORT inline float distance(double a, Fixed32 b) {
    double dif = a - b.toDouble();
    while (dif > 0.5) {
        dif -= 1.0;
    }
    while (dif < -0.5) {
        dif += 1.0;
    }
    return dif;
}

CUDA_EXPORT inline Fixed32 sum(Fixed32 a, Fixed32 b) {
    return (Fixed<int32_t>(a) + Fixed<int32_t>(b));
}

CUDA_EXPORT inline float doubleDistance(Fixed32 a, Fixed32 b) {
    return (Fixed<int32_t>(a) - Fixed<int32_t>(b)).toDouble();
}

#endif /* COSMICTIGER_FIXED_HPP_ */
