/*
 * fp16.hpp
 *
 *  Created on: Jul 20, 2022
 *      Author: dmarce1
 */

#ifndef FP16_HPP_
#define FP16_HPP_

#include <cstdint>
#include <limits>
#include <type_traits>

// IEEE 754 binary16 storage. The bits are sign:1, exponent:5, fraction:10.
struct Float16 {
    std::uint16_t bits;
};

static_assert(sizeof(Float16) == 2);
static_assert(std::is_trivial_v<Float16> && std::is_standard_layout_v<Float16>);

// The out-of-line converters use C++23 std::float16_t; this storage header also
// remains usable by the C++20 CUDA compiler. Finite magnitudes above 65504 throw
// std::overflow_error. Conversion follows the default round-to-nearest,
// ties-to-even environment, and supports signed zeros, subnormals, infinities,
// and NaNs according to the standard floating-point implementation.
Float16 double2float16(double value);
double float162double(Float16 value);

namespace std {
template <>
struct numeric_limits<Float16> {
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr float_round_style round_style = round_to_nearest;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr Float16 min() noexcept {
        return {0x0400};
    }
    static constexpr Float16 lowest() noexcept {
        return {0xfbff};
    }
    static constexpr Float16 max() noexcept {
        return {0x7bff};
    }
    static constexpr Float16 epsilon() noexcept {
        return {0x1400};
    }
    static constexpr Float16 round_error() noexcept {
        return {0x3800};
    }
    static constexpr Float16 infinity() noexcept {
        return {0x7c00};
    }
    static constexpr Float16 quiet_NaN() noexcept {
        return {0x7e00};
    }
    static constexpr Float16 signaling_NaN() noexcept {
        return {0x7d00};
    }
    static constexpr Float16 denorm_min() noexcept {
        return {0x0001};
    }
};
}    // namespace std

#endif /* FP16_HPP_ */
