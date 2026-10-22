/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include <cosmictiger/math/fixed.hpp>

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        throw std::runtime_error("Fixed-point arithmetic assertion failed");
    }
}

static_assert(sizeof(Fixed32) == sizeof(uint32_t));
static_assert(sizeof(Fixed64) == sizeof(uint64_t));
static_assert(std::is_trivially_copyable_v<Fixed32>);
static_assert(std::is_trivially_copyable_v<Fixed64>);
static_assert(Fixed32::kToFloatFactor_ == 0x1p-32);
static_assert(Fixed64::kToFloatFactor_ == 0x1p-64);

#if defined(__GLIBC__) && GTEST_HAS_DEATH_TEST
namespace detail {

void fixedFloatingTrapProbe() {
    double volatile inputs[] = {
        0.5, 0.75, std::nextafter(1.0, 0.0), -0.5, -0.75, 0x1p-64, -0x1p-64};
    uint64_t const expected[] = {uint64_t(1) << 63, uint64_t(3) << 62,
        uint64_t(0) - (uint64_t(1) << 11), uint64_t(1) << 63, uint64_t(1) << 62, 1,
        std::numeric_limits<uint64_t>::max()};
    std::feclearexcept(FE_ALL_EXCEPT);
    if (feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW) == -1) {
        std::_Exit(1);
    }
    for (size_t index = 0; index < sizeof(inputs) / sizeof(inputs[0]); ++index) {
        if (Fixed64(inputs[index]).raw() != expected[index]) {
            std::_Exit(2);
        }
    }
    std::_Exit(std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW) == 0 ? 0 : 3);
}

}    // namespace detail

TEST(Fixed, ValidCoordinatesWithFloatingPointTraps) {
    EXPECT_EXIT(detail::fixedFloatingTrapProbe(), ::testing::ExitedWithCode(0), "");
}
#endif

TEST(Fixed, RepresentationAndConversion) {
    EXPECT_EQ(Fixed32(0.5).raw(), uint32_t(1) << 31);
    EXPECT_EQ(Fixed64(0.5).raw(), uint64_t(1) << 63);
    EXPECT_EQ(Fixed64::max().raw(), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(Fixed64::min().raw(), 1U);
    EXPECT_EQ(Fixed64::min().toDouble(), 0x1p-64);
    EXPECT_EQ(Fixed64(0x1p-48).raw(), uint64_t(1) << 16);
    EXPECT_EQ(Fixed64(std::nextafter(1.0, 0.0)).raw(), uint64_t(0) - (uint64_t(1) << 11));
    EXPECT_EQ(Fixed64(0.875).toInt(), 0);
    EXPECT_EQ(Fixed32(0.875).toInt(), 0);

    Fixed32 coarse;
    coarse.raw() = 0xfedcba98U;
    Fixed64 fine(coarse);
    EXPECT_EQ(fine.raw(), uint64_t(0xfedcba98U) << 32);
    fine.raw() += 0x87654321U;
    EXPECT_EQ(Fixed32(fine).raw(), coarse.raw());
    Fixed64 assigned;
    assigned = coarse;
    EXPECT_EQ(assigned.raw(), uint64_t(coarse.raw()) << 32);
    coarse = fine;
    EXPECT_EQ(coarse.raw(), 0xfedcba98U);
}

TEST(Fixed, ModularCoordinatesAndSmallDrifts) {
    EXPECT_EQ(Fixed64(1.0).raw(), 0U);
    EXPECT_EQ(Fixed64(2.0).raw(), 0U);
    EXPECT_EQ(Fixed64(-1.0).raw(), 0U);
    EXPECT_EQ(Fixed64(-0.25).raw(), uint64_t(3) << 62);
    EXPECT_EQ(Fixed32(1.0).raw(), 0U);
    EXPECT_EQ(Fixed32(-0.25).raw(), uint32_t(3) << 30);
    EXPECT_EQ(Fixed64(-0x1p-64).raw(), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(Fixed64(-0x1p-65).raw(), 0U);
    EXPECT_EQ((Fixed64::max() + Fixed64::min()).raw(), 0U);
    EXPECT_EQ((Fixed64(0.0) - Fixed64::min()).raw(), Fixed64::max().raw());
    EXPECT_EQ((-Fixed64::min()).raw(), Fixed64::max().raw());

    Fixed64 position(0.5);
    position.raw() += 17;
    uint64_t const start = position.raw();
    for (int step = 0; step < 1000; ++step) {
        position += Fixed64(0x1p-64);
    }
    EXPECT_EQ(position.raw(), start + 1000);
    for (int step = 0; step < 1000; ++step) {
        position += Fixed64(-0x1p-64);
    }
    EXPECT_EQ(position.raw(), start);
}

TEST(Fixed, DifferencesRetainAllBits) {
    Fixed64 a(0.5);
    Fixed64 b = a;
    a.raw() += 1;
    EXPECT_EQ(a.toDouble(), b.toDouble());
    EXPECT_EQ(distance(a, b), 0x1p-64f);
    EXPECT_EQ(distance(b, a), -0x1p-64f);
    EXPECT_EQ(unwrappedDistance(a, b), 0x1p-64f);
    EXPECT_EQ(unwrappedDoubleDistance(a, b), 0x1p-64);
    EXPECT_EQ(unwrappedDoubleDistance(b, a), -0x1p-64);
    EXPECT_EQ(doubleDistance(a, b), 0x1p-64);
    EXPECT_EQ(distance(a, 0.5), 0x1p-64f);
    EXPECT_EQ(distance(0.5, a), -0x1p-64f);
    EXPECT_EQ(distance(Fixed64(0.0), Fixed64::max()), 0x1p-64f);
    EXPECT_EQ(distance(Fixed64::max(), Fixed64(0.0)), -0x1p-64f);
    EXPECT_EQ(unwrappedDistance(Fixed64::max(), Fixed64(0.0)), 1.0f);

    a.raw() = (uint64_t(1) << 62) + (uint64_t(1) << 32);
    b.raw() = uint64_t(1) << 62;
    EXPECT_EQ(distance(a, b), 0x1p-32f);
    EXPECT_EQ(doubleDistance(a, b), 0x1p-32);
    a.raw() = (uint64_t(1) << 62) + (uint64_t(1) << 35) + 7;
    EXPECT_EQ(doubleDistance(a, b), 0x1p-29 + 7 * 0x1p-64);
}

TEST(Fixed, ArithmeticUsesFullProductAndQuotient) {
    EXPECT_EQ((Fixed64(0.75) * Fixed64(0.5)).raw(), Fixed64(0.375).raw());
    EXPECT_EQ((Fixed32(0.875) * Fixed32(0.875)).raw(), Fixed32(0.765625).raw());
    EXPECT_EQ((Fixed64(0.375) / Fixed64(0.5)).raw(), Fixed64(0.75).raw());
    EXPECT_EQ((Fixed32(0.375) / Fixed32(0.5)).raw(), Fixed32(0.75).raw());
    EXPECT_EQ((Fixed64::max() * Fixed64::max()).raw(), Fixed64::max().raw() - 1);

    std::mt19937_64 random(0x7c45d68eULL);
    for (int sample = 0; sample < 512; ++sample) {
        Fixed64 a;
        Fixed64 b;
        a.raw() = random();
        b.raw() = random() | 1U;
        auto const product = static_cast<unsigned __int128>(a.raw()) * b.raw();
        auto const quotient = (static_cast<unsigned __int128>(a.raw()) << 64) / b.raw();
        EXPECT_EQ((a * b).raw(), static_cast<uint64_t>(product >> 64));
        EXPECT_EQ((a / b).raw(), static_cast<uint64_t>(quotient));
        Fixed64 compound = a;
        compound *= b;
        EXPECT_EQ(compound.raw(), (a * b).raw());
        compound = a;
        compound /= b;
        EXPECT_EQ(compound.raw(), (a / b).raw());
    }
}

TEST(Fixed, SignedRangesAndWrapping) {
    using RangeCoordinate = Fixed<int32_t, 29>;
    using WideRangeCoordinate = Fixed<int64_t, 32>;
    EXPECT_EQ(RangeCoordinate(-1.25).toDouble(), -1.25);
    EXPECT_EQ(RangeCoordinate(-1.25).toInt(), -2);
    EXPECT_EQ(WideRangeCoordinate(-1.25).toDouble(), -1.25);
    EXPECT_EQ(WideRangeCoordinate(RangeCoordinate(-1.25)).toDouble(), -1.25);
    EXPECT_EQ(RangeCoordinate(WideRangeCoordinate(-1.25)).toDouble(), -1.25);
    EXPECT_EQ((RangeCoordinate(-0.5) * RangeCoordinate(0.5)).toDouble(), -0.25);
    EXPECT_EQ((RangeCoordinate(-0.25) / RangeCoordinate(0.5)).toDouble(), -0.5);
    EXPECT_EQ((RangeCoordinate(-0.5) * RangeCoordinate::min()).raw(), -1);

    using Signed64 = Fixed<int64_t, 64>;
    Signed64 minimum;
    minimum.raw() = std::numeric_limits<int64_t>::min();
    EXPECT_EQ((-minimum).raw(), minimum.raw());
    EXPECT_EQ((minimum - Signed64::min()).raw(), std::numeric_limits<int64_t>::max());
    EXPECT_EQ((Signed64::max() + Signed64::min()).raw(), minimum.raw());
    EXPECT_EQ(minimum.toInt(), -1);
}
