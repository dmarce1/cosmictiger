/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <bit>
#include <cstdint>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace scaled_int16_storage {
namespace detail {

    CUDA_EXPORT inline uint32_t floatBits(float value) {
#ifdef __CUDA_ARCH__
        return __float_as_uint(value);
#else
        return std::bit_cast<uint32_t>(value);
#endif
    }

    CUDA_EXPORT inline float fromFloatBits(uint32_t bits) {
#ifdef __CUDA_ARCH__
        return __uint_as_float(bits);
#else
        return std::bit_cast<float>(bits);
#endif
    }

}    // namespace detail

struct ScaledValue {
    int16_t significand = 0;
    int8_t exponent = 0;
};

inline constexpr int8_t kInvalidExponent = -128;

CUDA_EXPORT inline bool isValid(int16_t significand, int8_t exponent) {
    if (significand == 0) {
        return exponent == 0;
    }
    int const magnitude = significand < 0 ? -int(significand) : int(significand);
    return exponent >= -126 && magnitude >= 16384 && magnitude <= 32767;
}

/*
 * Derived fixed-significand storage from the IEEE 754 binary32 encoding:
 *   e = ⌊log₂|x|⌋,  q = sign(x) ⌊|x| 2^(14−e)⌋,  x ≈ q 2^(e−14).
 * Symbol directory: x → value; q → result.significand; e → result.exponent.
 * Normal binary32 inputs give 16384 ≤ |q| ≤ 32767 and −126 ≤ e ≤ 127.
 * Discarding the lowest nine fraction bits rounds toward zero without overflow.
 * Zero and subnormal inputs become canonical (q,e)=(0,0); nonfinite inputs use
 * the invalid sentinel (0,−128). Only integer bit operations are performed, so
 * storage conversion does not depend on rounding modes or flush-to-zero flags.
 */
CUDA_EXPORT inline ScaledValue encode(float value) {
    uint32_t const bits = detail::floatBits(value);
    uint32_t const exponent = (bits >> 23) & 0xffU;
    if (exponent == 0) {
        return {};
    }
    if (exponent == 0xffU) {
        return {0, kInvalidExponent};
    }
    int const magnitude = int(0x4000U | ((bits & 0x7fffffU) >> 9));
    int const significand = (bits & 0x80000000U) != 0 ? -magnitude : magnitude;
    return {int16_t(significand), int8_t(int(exponent) - 127)};
}

CUDA_EXPORT inline float decode(int16_t significand, int8_t exponent) {
    if (!isValid(significand, exponent)) {
        return detail::fromFloatBits(0x7fc00000U);
    }
    if (significand == 0) {
        return detail::fromFloatBits(0);
    }
    uint32_t const sign = significand < 0 ? 0x80000000U : 0U;
    uint32_t const magnitude = uint32_t(significand < 0 ? -int(significand) : int(significand));
    uint32_t const fraction = (magnitude - 0x4000U) << 9;
    return detail::fromFloatBits(sign | (uint32_t(int(exponent) + 127) << 23) | fraction);
}

}    // namespace scaled_int16_storage
