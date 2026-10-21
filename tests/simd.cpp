/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/math/simd.hpp>

namespace detail {

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, double tolerance, char const* message) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        std::fprintf(stderr, "%s: actual %.17g, expected %.17g, tolerance %.3g\n", message, actual,
            expected, tolerance);
        throw std::runtime_error(message);
    }
}

template <typename Value, int laneCount>
void testArithmetic() {
    using Pack = SimdVector<Value, laneCount>;
    static_assert(Pack::size() == laneCount);
    Pack const left([](auto lane) { return Value(int(lane) - 3); });
    Pack const right([](auto lane) { return Value(int(lane) + 1); });
    Value expectedSum = 0;
    for (int lane = 0; lane < laneCount; ++lane) {
        expectedSum += Value(lane - 3);
    }
    auto const product = left * right;
    auto const combined = left * right + Pack(Value(2));
    auto const less = left < Pack(Value(0));
    auto const equal = left == Pack(Value(0));
    auto const greaterEqual = left >= Pack(Value(0));
    auto const selected = simdSelect(less, Pack(Value(-1)), Pack(Value(2)));
    for (int lane = 0; lane < laneCount; ++lane) {
        Value const expected = Value((lane - 3) * (lane + 1));
        require(product[lane] == expected, "SIMD multiplication must update every lane");
        require(combined[lane] == expected + Value(2), "SIMD multiply-add differs from scalar");
        require(less[lane] == (lane < 3), "SIMD comparison must preserve each lane's predicate");
        require(equal[lane] == (lane == 3), "SIMD equality must preserve each lane's predicate");
        require(greaterEqual[lane] == (lane >= 3), "SIMD comparison complement is wrong");
        require(selected[lane] == Value(lane < 3 ? -1 : 2), "SIMD masks must select lanes");
    }
    require(simdAllOf(less | greaterEqual), "Complementary SIMD masks must cover every lane");
    require(!simdAnyOf(less & greaterEqual), "Complementary SIMD masks must not overlap");
    require(simdAnyOf(equal) == (laneCount > 3), "SIMD any-of must inspect every lane");
    require(simdSum(left) == expectedSum, "SIMD sum must include all lanes");
    require(simdMaximum(left) == Value(laneCount - 4), "SIMD maximum must include all lanes");
}

template <int laneCount>
void testConversions() {
    using FloatPack = SimdVector<float, laneCount>;
    using DoublePack = SimdVector<double, laneCount>;
    using IntPack = SimdVector<int, laneCount>;
    std::array<float, 12> const arguments = {
        -4.75f, -3.5f, -2.5f, -1.5f, -0.5f, -0.25f, 0.0f, 0.5f, 1.5f, 2.5f, 3.5f, 4.75f};
    std::array<int, 12> const expectedFloor = {-5, -4, -3, -2, -1, -1, 0, 0, 1, 2, 3, 4};
    std::array<float, 12> const expectedRound = {
        -5.0f, -4.0f, -2.0f, -2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 4.0f, 5.0f};
    for (size_t first = 0; first < arguments.size(); first += laneCount) {
        FloatPack const values(
            [&](auto lane) { return arguments[(first + size_t(lane)) % arguments.size()]; });
        auto const wide = simdCast<DoublePack>(values);
        auto const restored = simdCast<FloatPack>(wide);
        auto const integer = simdFloorToInt<IntPack>(values);
        auto const wideInteger = simdFloorToInt<IntPack>(wide);
        auto const floatInteger = simdCast<FloatPack>(integer);
        auto const doubleInteger = simdCast<DoublePack>(integer);
        auto const rounded = simdRound(values);
        auto const wideRounded = simdRound(wide);
        for (int lane = 0; lane < laneCount; ++lane) {
            size_t const index = (first + lane) % arguments.size();
            require(wide[lane] == double(arguments[index]), "Float to double must preserve lanes");
            require(restored[lane] == arguments[index], "Double to float must preserve lanes");
            require(integer[lane] == expectedFloor[index], "Float to int must round downward");
            require(wideInteger[lane] == expectedFloor[index], "Double to int must round downward");
            require(floatInteger[lane] == float(expectedFloor[index]),
                "Integer to float must preserve signed values");
            require(doubleInteger[lane] == double(expectedFloor[index]),
                "Integer to double must preserve signed values");
            require(rounded[lane] == expectedRound[index], "Float round must use ties to even");
            require(wideRounded[lane] == double(expectedRound[index]),
                "Double round must use ties to even");
        }
    }
}

template <typename Value, int laneCount>
void testPrimitives() {
    using Pack = SimdVector<Value, laneCount>;
    Pack lanes{};
    for (int lane = laneCount - 1; lane >= 0; --lane) {
        simdSetLane(lanes, lane, Value(lane + 1));
        for (int other = 0; other < laneCount; ++other) {
            require(lanes[other] == (other >= lane ? Value(other + 1) : Value(0)),
                "Setting a SIMD lane must preserve every other lane");
        }
    }
    auto const roots = simdSqrt(lanes * lanes);
    auto const inverseRoots = simdRsqrt(lanes * lanes);
    Value const epsilon = std::numeric_limits<Value>::epsilon();
    Pack const first([&](auto lane) { return Value(1) + Value(int(lane) + 1) * epsilon; });
    Pack const second([&](auto lane) { return Value(1) - Value(int(lane) + 1) * epsilon; });
    Pack const third(Value(-1));
    auto const fused = simdFma(first, second, third);
    for (int lane = 0; lane < laneCount; ++lane) {
        Value const offset = Value(lane + 1) * epsilon;
        // (1 + offset) * (1 - offset) - 1 is exactly -offset squared;
        // separately rounding the product loses this nonzero result.
        Value const expectedFused = -(offset * offset);
        require(roots[lane] == Value(lane + 1), "SIMD square root must preserve every lane");
        requireNear(inverseRoots[lane], Value(1) / Value(lane + 1), epsilon,
            "SIMD reciprocal square root differs from scalar");
        require(fused[lane] == expectedFused, "SIMD fma must retain fused rounding");
    }
}

template <int laneCount>
void testPeriodicSubtraction() {
    using IntPack = SimdVector<int, laneCount>;
    using FloatPack = SimdVector<float, laneCount>;
    struct PeriodicCase {
        uint32_t source;
        uint32_t sink;
        int32_t difference;
    };
    std::array<PeriodicCase, 8> const cases = {{{0xffffffffU, 0U, -1}, {0U, 0xffffffffU, 1},
        {0x80000000U, 0U, std::numeric_limits<int32_t>::min()},
        {0x7fffffffU, 0xffffffffU, std::numeric_limits<int32_t>::min()},
        {0x80000000U, 0xffffffffU, -2147483647}, {0xffffffffU, 0x80000000U, 2147483647},
        {1U, 0xfffffffeU, 3}, {0xfffffffeU, 1U, -3}}};
    for (size_t first = 0; first < cases.size(); first += laneCount) {
        IntPack const source([&](auto lane) {
            return std::bit_cast<int32_t>(cases[(first + size_t(lane)) % cases.size()].source);
        });
        IntPack const sink([&](auto lane) {
            return std::bit_cast<int32_t>(cases[(first + size_t(lane)) % cases.size()].sink);
        });
        auto const difference = simdPositionDifference(source, sink);
        FloatPack const displacement = simdCast<FloatPack>(difference) * FloatPack(0x1p-32f);
        for (int lane = 0; lane < laneCount; ++lane) {
            auto const& test = cases[(first + lane) % cases.size()];
            require(difference[lane] == test.difference,
                "Periodic position subtraction must wrap modulo 2^32");
            require(displacement[lane] == float(test.difference) * 0x1p-32f,
                "Periodic displacement conversion must preserve the wrapped sign");
        }
    }
    auto const upperBits = simdBroadcast<IntPack>(uint32_t(0xffffffffU));
    auto const signBit = simdBroadcast<IntPack>(uint32_t(0x80000000U));
    IntPack const laneValues([](auto lane) { return int(lane) + 2; });
    auto const squared = laneValues * laneValues;
    for (int lane = 0; lane < laneCount; ++lane) {
        require(upperBits[lane] == -1, "Unsigned integer broadcast must preserve all bits");
        require(signBit[lane] == std::numeric_limits<int32_t>::min(),
            "Unsigned integer broadcast must preserve the sign bit");
        require(squared[lane] == (lane + 2) * (lane + 2),
            "Integer multiplication must update every lane");
    }
}

template <int laneCount>
void testPartialPackets() {
    using Pack = SimdVector<float, laneCount>;
    int const count = 2 * laneCount + 1;
    Pack accumulated(0.0f);
    float expected = 0.0f;
    for (int index = 0; index < count; ++index) {
        float const value = float(index + 1) * 0.125f;
        expected += value * value;
    }
    for (int first = 0; first < count; first += laneCount) {
        Pack const indices([&](auto lane) { return float(first + int(lane)); });
        Pack const values([&](auto lane) {
            int const index = first + int(lane);
            return index < count ? float(index + 1) * 0.125f : 123.0f;
        });
        auto const active = indices < Pack(float(count));
        accumulated += simdSelect(active, values * values, Pack(0.0f));
    }
    require(
        simdSum(accumulated) == expected, "Inactive lanes must not contribute to a partial packet");
}

template <int laneCount>
void testWidth() {
    testArithmetic<float, laneCount>();
    testArithmetic<double, laneCount>();
    testPrimitives<float, laneCount>();
    testPrimitives<double, laneCount>();
    testConversions<laneCount>();
    testPeriodicSubtraction<laneCount>();
    testPartialPackets<laneCount>();
}

void testMathApproximations() {
    std::array<float, 8> const exponentArguments = {
        -20.0f, -8.0f, -1.0f, -0.25f, 0.0f, 0.5f, 3.0f, 20.0f};
    std::array<float, 8> const sineArguments = {-3.0f, -1.5f, -0.25f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    std::array<float, 8> const logarithmArguments = {
        0.0001f, 0.125f, 0.5f, 1.0f, 1.5f, 2.0f, 7.0f, 1.0e12f};
    std::array<float, 8> const errorArguments = {0.0f, 0.125f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f, 8.0f};
    SimdFloat const exponentInput([&](auto lane) { return exponentArguments[size_t(lane)]; });
    SimdFloat const sineInput([&](auto lane) { return sineArguments[size_t(lane)]; });
    SimdFloat const logarithmInput([&](auto lane) { return logarithmArguments[size_t(lane)]; });
    SimdFloat const errorInput([&](auto lane) { return errorArguments[size_t(lane)]; });
    auto const exponential = exp(exponentInput);
    auto const sine = sin(sineInput);
    auto const logarithm = log(logarithmInput);
    SimdFloat errorComplement;
    SimdFloat errorExponential;
    erfcexp(errorInput, &errorComplement, &errorExponential);
    for (int lane = 0; lane < SIMD_FLOAT_SIZE; ++lane) {
        double const expectedExponential = std::exp(double(exponentArguments[lane]));
        requireNear(exponential[lane], expectedExponential, 3.0e-6 * expectedExponential,
            "SIMD exponential approximation");
        requireNear(
            sine[lane], std::sin(double(sineArguments[lane])), 2.0e-6, "SIMD sine approximation");
        requireNear(logarithm[lane], std::log(double(logarithmArguments[lane])), 5.0e-6,
            "SIMD logarithm approximation");
        double const argument = errorArguments[lane];
        double const expectedErrorExponential = std::exp(-argument * argument);
        requireNear(errorComplement[lane], std::erfc(argument), 3.0e-7,
            "SIMD complementary error function approximation");
        requireNear(errorExponential[lane], expectedErrorExponential,
            3.0e-6 * expectedErrorExponential, "SIMD erfcexp exponential approximation");
    }
}

}    // namespace detail

#if defined(COSMICTIGER_SIMD_EXPECT_STANDARD)
static_assert(COSMICTIGER_SIMD_STANDARD == COSMICTIGER_SIMD_EXPECT_STANDARD);
#endif

TEST(Simd, Width2) {
    detail::testWidth<2>();
}

TEST(Simd, Width4) {
    detail::testWidth<4>();
}

TEST(Simd, Width8) {
    detail::testWidth<8>();
}

TEST(Simd, Width16) {
    detail::testWidth<16>();
}

TEST(Simd, MathApproximations) {
    detail::testMathApproximations();
}
