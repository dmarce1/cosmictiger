/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/simd.hpp>

template <class, int>
class Fixed;

using Fixed32 = Fixed<uint32_t, 32>;
using Fixed64 = Fixed<uint64_t, 64>;

static constexpr float kFixed2float = 0x1p-32f;

namespace detail {

CUDA_EXPORT constexpr double fixedPowerOfTwo(int exponent) {
    double result = 1.0;
    for (int bit = 0; bit < exponent; ++bit) {
        result *= 2.0;
    }
    return result;
}

CUDA_EXPORT inline uint64_t fixedMultiplyHigh(uint64_t a, uint64_t b) {
#ifdef __CUDA_ARCH__
    return __umul64hi(a, b);
#else
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
#endif
}

// The low 64 bits of (numerator << fractionBits) / denominator. Long division
// on the device avoids depending on device support for a 128-bit integer type.
CUDA_EXPORT inline uint64_t fixedDivide(
    uint64_t numerator, uint64_t denominator, int fractionBits) {
#ifdef __CUDA_ARCH__
    uint64_t quotient = numerator / denominator;
    uint64_t remainder = numerator % denominator;
    for (int bit = 0; bit < fractionBits; ++bit) {
        bool const carry = (remainder >> 63) != 0;
        remainder <<= 1;
        bool const next = carry || remainder >= denominator;
        if (next) {
            remainder -= denominator;
        }
        quotient = (quotient << 1) | uint64_t(next);
    }
    return quotient;
#else
    return static_cast<uint64_t>(
        (static_cast<unsigned __int128>(numerator) << fractionBits) / denominator);
#endif
}

}    // namespace detail

// Arithmetic wraps modulo the storage width. Unsigned Q0.32 and Q0.64 values
// therefore represent periodic positions in [0, 1). Signed instantiations also
// support the wider coordinate ranges used by tree bounds and lightcones.
template <class IntegerType, int fractionBits = 32>
class Fixed {
public:
    using Integer = IntegerType;
    using UnsignedInteger = std::make_unsigned_t<Integer>;
    static constexpr int kStorageBits_ = sizeof(Integer) * 8;
    static constexpr int kFractionBits_ = fractionBits;
    static constexpr double kToFloatFactor_ = 1.0 / detail::fixedPowerOfTwo(fractionBits);

private:
    static_assert(std::is_integral_v<Integer> && sizeof(Integer) <= sizeof(uint64_t));
    static_assert(fractionBits > 0 && fractionBits <= kStorageBits_);
    Integer i_;

    CUDA_EXPORT static constexpr Fixed fromBits(UnsignedInteger bits) {
        Fixed result;
        result.i_ = static_cast<Integer>(bits);
        return result;
    }

    CUDA_EXPORT static Integer floatingBits(double number) {
        // Reduce before scaling: casting a negative number or exactly 2^64 to
        // an unsigned integer is undefined. Keeping the sign separate also
        // retains tiny negative displacements that would vanish in 1 + number.
        double const period = detail::fixedPowerOfTwo(kStorageBits_ - fractionBits);
        double const reduced =
            number > -period && number < period ? number : ::fmod(number, period);
        bool const negative = reduced < 0.0;
        double const magnitude = negative ? -reduced : reduced;
        double const scaled = magnitude * detail::fixedPowerOfTwo(fractionBits);
        UnsignedInteger bits;
#ifndef __CUDA_ARCH__
        if constexpr (sizeof(UnsignedInteger) == sizeof(uint64_t)) {
            // GCC can lower double-to-uint64_t to a signed conversion followed
            // by a correction. For valid values >= 2^63 that first conversion
            // raises FE_INVALID when floating-point traps are enabled. Each
            // limb stays below 2^32, so both conversions are safe even if the
            // compiler speculates instructions across a range check.
            uint32_t const high = static_cast<uint32_t>(scaled * 0x1p-32);
            uint32_t const low = static_cast<uint32_t>(scaled - double(high) * 0x1p32);
            bits = (uint64_t(high) << 32) | low;
        } else
#endif
        {
            bits = static_cast<UnsignedInteger>(scaled);
        }
        return static_cast<Integer>(negative ? UnsignedInteger(0) - bits : bits);
    }

    CUDA_EXPORT uint64_t magnitude() const {
        UnsignedInteger const bits = static_cast<UnsignedInteger>(i_);
        if constexpr (std::is_signed_v<Integer>) {
            return i_ < 0 ? UnsignedInteger(0) - bits : bits;
        } else {
            return bits;
        }
    }

public:
    CUDA_EXPORT constexpr Fixed() = default;

    CUDA_EXPORT inline void setInteger(Integer value) {
        i_ = value;
    }

    CUDA_EXPORT constexpr Integer raw() const {
        return i_;
    }

    CUDA_EXPORT constexpr Integer& raw() {
        return i_;
    }

    CUDA_EXPORT static constexpr Fixed max() {
        UnsignedInteger bits = ~UnsignedInteger(0);
        if constexpr (std::is_signed_v<Integer>) {
            bits >>= 1;
        }
        return fromBits(bits);
    }

    CUDA_EXPORT static constexpr Fixed min() {
        return fromBits(1);
    }

    CUDA_EXPORT Fixed& operator=(double number) {
        i_ = floatingBits(number);
        return *this;
    }

    CUDA_EXPORT Fixed(float number)
      : i_(floatingBits(number)) {}

    CUDA_EXPORT Fixed(int number)
      : i_(floatingBits(number)) {}

    CUDA_EXPORT Fixed(double number)
      : i_(floatingBits(number)) {}

    template <class OtherInteger, int otherFractionBits>
    CUDA_EXPORT constexpr Fixed(Fixed<OtherInteger, otherFractionBits> other) {
        if constexpr (otherFractionBits > fractionBits) {
            i_ = static_cast<Integer>(other.raw() >> (otherFractionBits - fractionBits));
        } else {
            // Widen first, including sign extension, before increasing precision.
            uint64_t const bits = static_cast<uint64_t>(other.raw());
            i_ = static_cast<Integer>(bits << (fractionBits - otherFractionBits));
        }
    }

    template <class OtherInteger, int otherFractionBits>
    CUDA_EXPORT constexpr Fixed& operator=(Fixed<OtherInteger, otherFractionBits> other) {
        i_ = Fixed(other).raw();
        return *this;
    }

    CUDA_EXPORT bool operator<(Fixed other) const {
        return i_ < other.i_;
    }

    CUDA_EXPORT bool operator>(Fixed other) const {
        return i_ > other.i_;
    }

    CUDA_EXPORT bool operator<=(Fixed other) const {
        return i_ <= other.i_;
    }

    CUDA_EXPORT bool operator>=(Fixed other) const {
        return i_ >= other.i_;
    }

    CUDA_EXPORT bool operator==(Fixed other) const {
        return i_ == other.i_;
    }

    CUDA_EXPORT bool operator!=(Fixed other) const {
        return i_ != other.i_;
    }

    CUDA_EXPORT float toFloat() const {
        return static_cast<float>(i_) * static_cast<float>(kToFloatFactor_);
    }

    CUDA_EXPORT int toInt() const {
        if constexpr (fractionBits == kStorageBits_) {
            if constexpr (std::is_signed_v<Integer>) {
                return i_ < 0 ? -1 : 0;
            } else {
                return 0;
            }
        } else {
            return static_cast<int>(i_ >> fractionBits);
        }
    }

    CUDA_EXPORT double toDouble() const {
        return static_cast<double>(i_) * kToFloatFactor_;
    }

    CUDA_EXPORT Fixed operator*(Fixed other) const {
        uint64_t const a = magnitude();
        uint64_t const b = other.magnitude();
        uint64_t const low = a * b;
        uint64_t const high = detail::fixedMultiplyHigh(a, b);
        uint64_t shifted;
        bool remainder;
        if constexpr (fractionBits == 64) {
            shifted = high;
            remainder = low != 0;
        } else {
            shifted = (low >> fractionBits) | (high << (64 - fractionBits));
            remainder = (low & ((uint64_t(1) << fractionBits) - 1)) != 0;
        }
        if constexpr (std::is_signed_v<Integer>) {
            if ((i_ < 0) != (other.i_ < 0)) {
                // Signed fixed multiplication rounds down, as an arithmetic
                // right shift of the exact full-width product would do.
                shifted = uint64_t(0) - shifted - uint64_t(remainder);
            }
        }
        return fromBits(static_cast<UnsignedInteger>(shifted));
    }

    CUDA_EXPORT Fixed& operator*=(Fixed other) {
        *this = *this * other;
        return *this;
    }

    CUDA_EXPORT Fixed& operator*=(int other) {
        i_ = static_cast<Integer>(
            static_cast<UnsignedInteger>(i_) * static_cast<UnsignedInteger>(other));
        return *this;
    }

    CUDA_EXPORT Fixed operator/(Fixed other) const {
        ALWAYS_ASSERT(other.i_ != 0);
        uint64_t quotient = detail::fixedDivide(magnitude(), other.magnitude(), fractionBits);
        if constexpr (std::is_signed_v<Integer>) {
            if ((i_ < 0) != (other.i_ < 0)) {
                quotient = uint64_t(0) - quotient;
            }
        }
        return fromBits(static_cast<UnsignedInteger>(quotient));
    }

    CUDA_EXPORT Fixed& operator/=(Fixed other) {
        *this = *this / other;
        return *this;
    }

    CUDA_EXPORT Fixed operator+(Fixed other) const {
        return fromBits(static_cast<UnsignedInteger>(i_) + static_cast<UnsignedInteger>(other.i_));
    }

    CUDA_EXPORT Fixed operator-(Fixed other) const {
        return fromBits(static_cast<UnsignedInteger>(i_) - static_cast<UnsignedInteger>(other.i_));
    }

    CUDA_EXPORT Fixed operator-() const {
        return fromBits(UnsignedInteger(0) - static_cast<UnsignedInteger>(i_));
    }

    CUDA_EXPORT Fixed& operator+=(Fixed other) {
        *this = *this + other;
        return *this;
    }

    CUDA_EXPORT Fixed& operator-=(Fixed other) {
        *this = *this - other;
        return *this;
    }

    CUDA_EXPORT Integer getInteger() const {
        return i_;
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & i_;
    }
};

template <class Integer, int fractionBits>
CUDA_EXPORT inline Fixed<Integer, fractionBits> max(
    Fixed<Integer, fractionBits> a, Fixed<Integer, fractionBits> b) {
    return a > b ? a : b;
}

template <class Integer, int fractionBits>
CUDA_EXPORT inline Fixed<Integer, fractionBits> min(
    Fixed<Integer, fractionBits> a, Fixed<Integer, fractionBits> b) {
    return a < b ? a : b;
}

template <class Integer, int fractionBits>
inline void swap(Fixed<Integer, fractionBits>& first, Fixed<Integer, fractionBits>& second) {
    std::swap(first, second);
}

template <class Integer, int fractionBits>
CUDA_EXPORT inline float distance(Fixed<Integer, fractionBits> a, Fixed<Integer, fractionBits> b) {
    using SignedFixed = Fixed<std::make_signed_t<Integer>, fractionBits>;
    return (SignedFixed(a) - SignedFixed(b)).toFloat();
}

// Subtract native fixed-point integers before conversion so nearby Q0.64
// positions retain their separation even when both round to the same double.
template <class Integer, int fractionBits>
CUDA_EXPORT inline float unwrappedDistance(
    Fixed<Integer, fractionBits> a, Fixed<Integer, fractionBits> b) {
    static_assert(std::is_unsigned_v<Integer>);
    Integer const aRaw = a.raw();
    Integer const bRaw = b.raw();
    Integer const magnitude = aRaw >= bRaw ? aRaw - bRaw : bRaw - aRaw;
#ifdef __CUDA_ARCH__
    float const converted =
        sizeof(Integer) == 8 ? __ull2float_rn(magnitude) : __uint2float_rn(magnitude);
#else
    float const converted = static_cast<float>(magnitude);
#endif
    float const separation =
        converted * static_cast<float>(Fixed<Integer, fractionBits>::kToFloatFactor_);
    return aRaw >= bRaw ? separation : -separation;
}

template <class Integer, int fractionBits>
CUDA_EXPORT inline double unwrappedDoubleDistance(
    Fixed<Integer, fractionBits> a, Fixed<Integer, fractionBits> b) {
    static_assert(std::is_unsigned_v<Integer>);
    Integer const aRaw = a.raw();
    Integer const bRaw = b.raw();
    Integer const magnitude = aRaw >= bRaw ? aRaw - bRaw : bRaw - aRaw;
    double const separation =
        static_cast<double>(magnitude) * Fixed<Integer, fractionBits>::kToFloatFactor_;
    return aRaw >= bRaw ? separation : -separation;
}

CUDA_EXPORT inline float distance(double a, double b) {
    double difference = a - b;
    while (difference > 0.5) {
        difference -= 1.0;
    }
    while (difference < -0.5) {
        difference += 1.0;
    }
    return difference;
}

template <class Integer, int fractionBits>
CUDA_EXPORT inline float distance(Fixed<Integer, fractionBits> a, double b) {
    if constexpr (std::is_unsigned_v<Integer> && fractionBits == 64) {
        return distance(a, Fixed<Integer, fractionBits>(b));
    } else {
        return distance(a.toDouble(), b);
    }
}

template <class Integer, int fractionBits>
CUDA_EXPORT inline float distance(double a, Fixed<Integer, fractionBits> b) {
    if constexpr (std::is_unsigned_v<Integer> && fractionBits == 64) {
        return distance(Fixed<Integer, fractionBits>(a), b);
    } else {
        return distance(a, b.toDouble());
    }
}

template <class Integer, int fractionBits>
CUDA_EXPORT inline Fixed<Integer, fractionBits> sum(
    Fixed<Integer, fractionBits> a, Fixed<Integer, fractionBits> b) {
    return a + b;
}

template <class Integer, int fractionBits>
CUDA_EXPORT inline double doubleDistance(
    Fixed<Integer, fractionBits> a, Fixed<Integer, fractionBits> b) {
    using SignedFixed = Fixed<std::make_signed_t<Integer>, fractionBits>;
    return (SignedFixed(a) - SignedFixed(b)).toDouble();
}

#endif /* COSMICTIGER_FIXED_HPP_ */
