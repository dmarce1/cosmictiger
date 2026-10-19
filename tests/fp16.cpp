/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cosmictiger/math/fp16.hpp>

static_assert(sizeof(Float16) == sizeof(std::uint16_t), "Binary16 must occupy exactly two bytes");
static_assert(
    std::is_trivially_copyable<Float16>::value, "Binary16 records must be copyable as bytes");
static_assert(std::is_standard_layout<Float16>::value, "Binary16 records need a standard layout");

namespace detail {

constexpr std::uint64_t kDoubleSign = UINT64_C(0x8000000000000000);
constexpr std::uint64_t kDoubleExponent = UINT64_C(0x7ff0000000000000);
constexpr std::uint64_t kDoubleFraction = UINT64_C(0x000fffffffffffff);

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t doubleBits(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

double doubleFromBits(std::uint64_t bits) {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Classify representations explicitly: production compilation permits fast-math,
// which can fold std::isnan/std::isinf/signbit into incorrect test predicates.
bool doubleIsNan(double value) {
    std::uint64_t const bits = doubleBits(value);
    return (bits & kDoubleExponent) == kDoubleExponent && (bits & kDoubleFraction) != 0;
}

bool halfIsNan(Float16 value) {
    return (value.bits & 0x7c00U) == 0x7c00U && (value.bits & 0x03ffU) != 0;
}

void expectEncoding(double value, std::uint16_t expected) {
    std::uint16_t const actual = double2float16(value).bits;
    if (actual != expected) {
        std::fprintf(stderr, "Encoding %.17g: expected 0x%04x, got 0x%04x\n", value,
            unsigned(expected), unsigned(actual));
        throw std::runtime_error("Incorrect binary16 encoding");
    }
}

void expectBothSigns(double magnitude, std::uint16_t positiveExpected) {
    expectEncoding(magnitude, positiveExpected);
    expectEncoding(doubleFromBits(doubleBits(magnitude) | kDoubleSign),
        std::uint16_t(positiveExpected | 0x8000U));
}

// Independent mathematical binary16 definition; no encoder shifts, masks, or
// implementation helpers are used to compute the finite numerical values.
double referenceMagnitude(std::uint16_t bits) {
    unsigned const exponent = (bits / 1024U) % 32U;
    unsigned const fraction = bits % 1024U;
    return exponent == 0 ? std::ldexp(double(fraction), -24) :
                           std::ldexp(double(1024U + fraction), int(exponent) - 25);
}

void testKnownEncodings() {
    struct Case {
        double value;
        std::uint16_t bits;
    };
    Case const cases[] = {{0.0, 0x0000}, {1.0, 0x3c00}, {0.5, 0x3800}, {2.0, 0x4000}, {3.0, 0x4200},
        {1.5, 0x3e00}, {65504.0, 0x7bff}, {32768.0, 0x7800}, {0x1p-14, 0x0400}, {0x1p-15, 0x0200},
        {0x1p-24, 0x0001}, {1023.0 * 0x1p-24, 0x03ff}, {0x1p-25, 0x0000}, {3.0 * 0x1p-25, 0x0002},
        {5.0 * 0x1p-25, 0x0002}, {2047.0 * 0x1p-25, 0x0400}, {1.0 + 0x1p-11, 0x3c00},
        {1.0 + 3.0 * 0x1p-11, 0x3c02}, {2.0 - 0x1p-11, 0x4000}, {1.0 - 0x1p-12, 0x3c00}};
    for (Case const& testCase : cases) {
        expectBothSigns(testCase.value, testCase.bits);
    }
    expectBothSigns(doubleFromBits(UINT64_C(0x0000000000000001)), 0x0000);
    expectBothSigns(doubleFromBits(UINT64_C(0x000fffffffffffff)), 0x0000);
    expectBothSigns(doubleFromBits(UINT64_C(0x0010000000000000)), 0x0000);
}

void testAllRepresentations() {
    for (unsigned raw = 0; raw <= 0xffffU; raw++) {
        Float16 const half{std::uint16_t(raw)};
        double const decoded = float162double(half);
        if ((raw & 0x7c00U) == 0x7c00U && (raw & 0x03ffU) != 0) {
            require(doubleIsNan(decoded), "Every binary16 NaN must decode as a binary64 NaN");
            require(halfIsNan(double2float16(decoded)), "A decoded NaN must encode as a NaN");
            continue;
        }
        std::uint64_t expected;
        if ((raw & 0x7fffU) == 0x7c00U) {
            expected = kDoubleExponent;
        } else {
            expected = doubleBits(referenceMagnitude(std::uint16_t(raw)));
        }
        if ((raw & 0x8000U) != 0) {
            expected |= kDoubleSign;
        }
        if (doubleBits(decoded) != expected) {
            std::fprintf(stderr, "Incorrect decoding for binary16 pattern 0x%04x\n", raw);
            throw std::runtime_error("Binary16 decoder disagrees with its mathematical definition");
        }
        expectEncoding(decoded, std::uint16_t(raw));
    }
}

void testAllRoundingBoundaries() {
    // Each adjacent pair has an exactly representable binary64 midpoint.
    // Check the midpoint and its immediate binary64 neighbors for both signs,
    // including underflow, the subnormal/normal transition, and exponent carries.
    for (unsigned lower = 0; lower < 0x7bffU; lower++) {
        double const left = referenceMagnitude(std::uint16_t(lower));
        double const right = referenceMagnitude(std::uint16_t(lower + 1));
        double const midpoint = (left + right) * 0.5;
        std::uint64_t const midpointBits = doubleBits(midpoint);
        std::uint16_t const nearestEven = std::uint16_t(lower + (lower & 1U));
        expectBothSigns(doubleFromBits(midpointBits - 1), std::uint16_t(lower));
        expectBothSigns(midpoint, nearestEven);
        expectBothSigns(doubleFromBits(midpointBits + 1), std::uint16_t(lower + 1));
    }
}

void testSpecialValues() {
    expectEncoding(doubleFromBits(kDoubleExponent), 0x7c00);
    expectEncoding(doubleFromBits(kDoubleSign | kDoubleExponent), 0xfc00);
    std::uint64_t const nanPatterns[] = {UINT64_C(0x7ff8000000000000), UINT64_C(0xfff8000000000000),
        UINT64_C(0x7ff0000000000001), UINT64_C(0xfff0000000000001), UINT64_C(0x7fffffffffffffff)};
    for (std::uint64_t const bits : nanPatterns) {
        Float16 const encoded = double2float16(doubleFromBits(bits));
        require(halfIsNan(encoded), "Quiet and signaling binary64 NaNs must encode as NaNs");
        require(doubleIsNan(float162double(encoded)), "NaNs must remain NaNs after a round trip");
    }
}

void testOverflow() {
    double const magnitudes[] = {doubleFromBits(doubleBits(65504.0) + 1), 65505.0, 65519.0, 65520.0,
        65536.0, 131008.0, 0x1p100, std::numeric_limits<double>::max()};
    for (double const magnitude : magnitudes) {
        for (unsigned sign = 0; sign < 2; sign++) {
            double const value = doubleFromBits(doubleBits(magnitude) | (sign ? kDoubleSign : 0));
            bool rejected = false;
            try {
                (void) double2float16(value);
            } catch (std::overflow_error const&) {
                rejected = true;
            }
            require(rejected, "Every finite magnitude above 65504 must throw overflow_error");
        }
    }
    expectBothSigns(65504.0, 0x7bff);
    expectBothSigns(doubleFromBits(doubleBits(65504.0) - 1), 0x7bff);
}

void testNumericLimits() {
    using Limits = std::numeric_limits<Float16>;
    static_assert(std::is_same<decltype(Limits::max()), Float16>::value,
        "numeric_limits must return the specialized type");
    require(Limits::is_specialized && Limits::is_signed && !Limits::is_integer && !Limits::is_exact,
        "Binary16 numeric_limits type properties are incorrect");
    require(Limits::is_iec559 && Limits::has_infinity && Limits::has_quiet_NaN,
        "Binary16 must advertise IEEE special values");
    require(Limits::radix == 2 && Limits::digits == 11 && Limits::min_exponent == -13 &&
            Limits::max_exponent == 16,
        "Binary16 numeric_limits precision and exponent bounds are incorrect");
    require(
        Limits::has_denorm == std::denorm_present && Limits::round_style == std::round_to_nearest,
        "Binary16 must advertise subnormals and rounding to nearest");
    require(Limits::min().bits == 0x0400 && Limits::max().bits == 0x7bff &&
            Limits::lowest().bits == 0xfbff && Limits::denorm_min().bits == 0x0001 &&
            Limits::epsilon().bits == 0x1400 && Limits::infinity().bits == 0x7c00,
        "Binary16 numeric_limits endpoints are incorrect");
    require(halfIsNan(Limits::quiet_NaN()), "numeric_limits quiet_NaN must return a NaN");
}

}    // namespace detail

int main() {
    try {
        detail::testKnownEncodings();
        detail::testAllRepresentations();
        detail::testAllRoundingBoundaries();
        detail::testSpecialValues();
        detail::testOverflow();
        detail::testNumericLimits();
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Binary16 conversion tests failed: %s\n", error.what());
        return 1;
    }
    std::puts("Binary16 conversions: all representations, rounding boundaries, and overflow checks "
              "passed");
    return 0;
}
