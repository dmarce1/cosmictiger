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

namespace float16_storage {
namespace detail {

    /// Read the IEEE binary32 representation without a numerical conversion.
    CUDA_EXPORT inline uint32_t floatBits(float value) {
#ifdef __CUDA_ARCH__
        return __float_as_uint(value);
#else
        return std::bit_cast<uint32_t>(value);
#endif
    }

    /// Construct a float from its IEEE binary32 representation.
    CUDA_EXPORT inline float fromFloatBits(uint32_t bits) {
#ifdef __CUDA_ARCH__
        return __uint_as_float(bits);
#else
        return std::bit_cast<float>(bits);
#endif
    }

    // IEEE 754 binary32/binary16 interchange encodings. Integer rounding makes
    // CPU and CUDA storage independent of the arithmetic mode and flush-to-zero.
    CUDA_EXPORT inline uint16_t encode(float value, bool roundToNearest) {
        uint32_t const bits = floatBits(value);
        uint32_t const sign = (bits >> 16) & 0x8000U;
        uint32_t const exponent = (bits >> 23) & 0xffU;
        uint32_t const fraction = bits & 0x7fffffU;
        if (exponent == 0xffU) {
            return uint16_t(sign | 0x7c00U | (fraction == 0 ? 0 : (fraction >> 13) | 0x200U));
        }
        int const halfExponent = int(exponent) - 112;
        if (halfExponent >= 31) {
            return uint16_t(sign | (roundToNearest ? 0x7c00U : 0x7bffU));
        }
        if (halfExponent <= 0) {
            if (halfExponent < -10) {
                return uint16_t(sign);
            }
            uint32_t const significand = fraction | 0x800000U;
            int const shift = 14 - halfExponent;
            uint32_t rounded = significand >> shift;
            uint32_t const remainder = significand & ((1U << shift) - 1U);
            uint32_t const halfway = 1U << (shift - 1);
            if (roundToNearest &&
                (remainder > halfway || (remainder == halfway && (rounded & 1U)))) {
                ++rounded;
            }
            return uint16_t(sign | rounded);
        }
        uint32_t rounded = (uint32_t(halfExponent) << 10) | (fraction >> 13);
        uint32_t const remainder = fraction & 0x1fffU;
        if (roundToNearest && (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U)))) {
            ++rounded;
        }
        return uint16_t(sign | rounded);
    }

    /// Expand a binary16 encoding to float, preserving signed zero, subnormals,
    /// infinities, and NaN payload bits.
    CUDA_EXPORT inline float decode(uint16_t bits) {
        uint32_t const sign = uint32_t(bits & 0x8000U) << 16;
        uint32_t const exponent = (bits >> 10) & 0x1fU;
        uint32_t fraction = bits & 0x3ffU;
        if (exponent == 0x1fU) {
            return fromFloatBits(sign | 0x7f800000U | (fraction << 13));
        }
        if (exponent != 0) {
            return fromFloatBits(sign | ((exponent + 112U) << 23) | (fraction << 13));
        }
        if (fraction == 0) {
            return fromFloatBits(sign);
        }
        int binaryExponent = -14;
        while ((fraction & 0x400U) == 0) {
            fraction <<= 1;
            --binaryExponent;
        }
        return fromFloatBits(
            sign | (uint32_t(binaryExponent + 127) << 23) | ((fraction & 0x3ffU) << 13));
    }

}    // namespace detail
}    // namespace float16_storage

// Storage only: arithmetic remains float, while assignment rounds to nearest
// binary16 with ties to even. This type has no ROCKSTAR feature dependency.
struct Float16Storage {
    uint16_t bits = 0;

    CUDA_EXPORT Float16Storage() = default;
    /// Encode a float into the storage format using the configured finite-saturation
    /// policy.
    CUDA_EXPORT Float16Storage(float value)
      : bits(float16_storage::detail::encode(value, true)) {}

    /// Wrap an existing binary16 encoding without converting its value.
    CUDA_EXPORT static Float16Storage fromBits(uint16_t rawBits) {
        Float16Storage result;
        result.bits = rawBits;
        return result;
    }

    /// Decode the stored binary16 value to float.
    CUDA_EXPORT operator float() const {
        return float16_storage::detail::decode(bits);
    }

    /// Replace the stored binary16 encoding with the converted float value.
    CUDA_EXPORT Float16Storage& operator=(float value) {
        bits = float16_storage::detail::encode(value, true);
        return *this;
    }
};

static_assert(sizeof(Float16Storage) == 2);
