/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/sph/viscosity/balsara.hpp>

TEST(Balsara, BalsaraFactor) {
    // Without the sound-speed term the factor is |div|/(|div| + |curl|).
    EXPECT_EQ(balsara::balsaraFactor(0.0, 0.0, 0.0, 0.0), 0.0);
    EXPECT_EQ(balsara::balsaraFactor(0.0f, 2.0f, 1.0f, 1.0e-4f), 0.0f);
    EXPECT_EQ(balsara::balsaraFactor(-3.0, 0.0, 0.0, 1.0e-4), 1.0);
    EXPECT_NEAR(balsara::balsaraFactor(-3.0, 2.0, 0.0, 1.0e-4), 0.6, 1.0e-15);
    EXPECT_NEAR(balsara::balsaraFactor(3.0f, 2.0f, 0.0f, 1.0e-4f), 0.6f, 1.0e-7f);
    // GADGET-2 equation (14): |div|/(|div| + |curl| + ε_B c/h).
    EXPECT_NEAR(balsara::balsaraFactor(-3.0, 2.0, 1.0e4, 1.0e-4), 0.5, 1.0e-15);
    EXPECT_NEAR(balsara::balsaraFactor(-3.0, 0.0, 30.0, 0.1), 0.5, 1.0e-15);
    // Noise-level divergence in static gas receives almost no viscosity.
    EXPECT_LT(balsara::balsaraFactor(1.0e-6, 0.0, 1.0, 1.0e-4), 0.01);
    // A large coefficient removes viscosity; it never restores the unlimited factor.
    EXPECT_LT(balsara::balsaraFactor(-3.0, 0.0, 1.0, 1.0e6), 1.0e-5);
}

TEST(Balsara, ByteStoragePreservesEveryCodeAndEndpoints) {
    for (unsigned code = 0; code <= 255; ++code) {
        auto const storedFactor = static_cast<uint8_t>(code);
        float const factor = balsara::decodeFactor(storedFactor);
        EXPECT_EQ(balsara::encodeFactor(factor), storedFactor);
        EXPECT_NEAR(double(factor), double(code) / 255.0,
            double(std::numeric_limits<float>::epsilon()) / 2.0);
    }
    EXPECT_EQ(balsara::decodeFactor(0), 0.0f);
    EXPECT_EQ(balsara::decodeFactor(255), 1.0f);
    EXPECT_EQ(balsara::encodeFactor(-1.0f), 0);
    EXPECT_EQ(balsara::encodeFactor(0.0f), 0);
    EXPECT_EQ(balsara::encodeFactor(1.0f), 255);
    EXPECT_EQ(balsara::encodeFactor(2.0f), 255);
    EXPECT_EQ(balsara::encodeFactor(std::numeric_limits<float>::max()), 255);
    EXPECT_EQ(balsara::encodeFactor(-std::numeric_limits<float>::max()), 0);
}

TEST(Balsara, ByteStorageRoundsNearestWithBoundedError) {
    for (unsigned index = 0; index <= 4096; ++index) {
        float const factor = float(index) / 4096.0f;
        auto const storedFactor = balsara::encodeFactor(factor);
        EXPECT_LE(std::abs(double(storedFactor) / 255.0 - double(factor)), 1.0 / 510.0);
    }
    for (unsigned lowerCode = 0; lowerCode < 255; ++lowerCode) {
        double const threshold = (double(lowerCode) + 0.5) / 255.0;
        float const roundedThreshold = static_cast<float>(threshold);
        float const below = std::nextafter(roundedThreshold, 0.0f);
        float const above = std::nextafter(roundedThreshold, 1.0f);
        EXPECT_LT(double(below), threshold);
        EXPECT_GT(double(above), threshold);
        EXPECT_EQ(balsara::encodeFactor(below), lowerCode);
        EXPECT_EQ(balsara::encodeFactor(above), lowerCode + 1);
    }
    // The middle threshold is exactly representable; its tie rounds upward.
    EXPECT_EQ(balsara::encodeFactor(0.5f), 128);
}

TEST(Balsara, ByteStorageRejectsNonfiniteInputsWithoutFloatConversion) {
    for (uint32_t const bits :
        {0x7f800000U, 0xff800000U, 0x7fc00000U, 0x7f800001U, 0xffc00000U, 0xff800001U}) {
        EXPECT_EQ(balsara::encodeFactor(std::bit_cast<float>(bits)), 0);
    }
    EXPECT_EQ(balsara::encodeFactor(std::bit_cast<float>(0x80000000U)), 0);
}
