/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <bit>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/math/float16_storage.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>
#include <cosmictiger/sph/viscosity/hu_etal.hpp>
#include <cosmictiger/sph/viscosity/price_etal.hpp>
#include <tests/sph/test_helpers.hpp>

namespace detail {

template <typename Real>
void testCullenDehnenGradients() {
    Real const velocityGradient[3][3] = {{Real(-0.4), Real(0.7), Real(-0.2)},
        {Real(0.3), Real(-0.8), Real(0.5)}, {Real(-0.1), Real(0.2), Real(-0.6)}};
    Real const accelerationGradient[3][3] = {{Real(0.2), Real(-0.3), Real(0.1)},
        {Real(-0.7), Real(0.4), Real(0.6)}, {Real(0.5), Real(-0.2), Real(-0.1)}};
    // Orthogonal oblique directions exercise collinear, planar, and full-rank
    // neighborhoods without aligning an unresolved axis with a coordinate axis.
    Real const directions[3][3] = {
        {Real(1), Real(2), Real(-1)}, {Real(2), Real(-1), Real(0)}, {Real(-1), Real(-2), Real(-5)}};
    int const permutations[6][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    double const tolerance = 512.0 * std::numeric_limits<Real>::epsilon();
    for (int rank = 1; rank <= 3; ++rank) {
        for (auto const& permutation : permutations) {
            for (Real const weightScale : {Real(1), Real(-7), Real(1.0e-15), Real(1.0e15)}) {
                cullen_and_dehnen::GradientSums<Real> sums;
                for (int sample = 0; sample < rank; ++sample) {
                    Real position[3];
                    Real velocity[3]{};
                    Real acceleration[3]{};
                    for (int i = 0; i < 3; ++i) {
                        position[i] = directions[sample][permutation[i]];
                        for (int j = 0; j < 3; ++j) {
                            velocity[i] += velocityGradient[permutation[i]][permutation[j]] *
                                directions[sample][permutation[j]];
                            acceleration[i] +=
                                accelerationGradient[permutation[i]][permutation[j]] *
                                directions[sample][permutation[j]];
                        }
                    }
                    Real const weight = weightScale * Real(sample + 1);
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            sums.moment[i][j] += weight * position[i] * position[j];
                            sums.velocity[i][j] += weight * velocity[i] * position[j];
                            sums.acceleration[i][j] += weight * acceleration[i] * position[j];
                        }
                    }
                }
                Real reconstructedVelocity[3][3];
                Real reconstructedAcceleration[3][3];
                require(cullen_and_dehnen::reconstructGradients(
                            sums, reconstructedVelocity, reconstructedAcceleration) == rank,
                    "Cullen-Dehnen reconstruction identifies the resolved dimension");
                for (int i = 0; i < 3; ++i) {
                    for (int sample = 0; sample < rank; ++sample) {
                        Real velocity = Real(0);
                        Real acceleration = Real(0);
                        Real expectedVelocity = Real(0);
                        Real expectedAcceleration = Real(0);
                        for (int j = 0; j < 3; ++j) {
                            Real const position = directions[sample][permutation[j]];
                            velocity += reconstructedVelocity[i][j] * position;
                            acceleration += reconstructedAcceleration[i][j] * position;
                            expectedVelocity +=
                                velocityGradient[permutation[i]][permutation[j]] * position;
                            expectedAcceleration +=
                                accelerationGradient[permutation[i]][permutation[j]] * position;
                        }
                        requireClose("affine velocity recovered in the resolved subspace", velocity,
                            expectedVelocity, tolerance);
                        requireClose("affine acceleration recovered in the resolved subspace",
                            acceleration, expectedAcceleration, tolerance);
                    }
                    if (rank == 3) {
                        for (int j = 0; j < 3; ++j) {
                            requireClose("full affine velocity gradient",
                                reconstructedVelocity[i][j],
                                velocityGradient[permutation[i]][permutation[j]], tolerance);
                            requireClose("full affine acceleration gradient",
                                reconstructedAcceleration[i][j],
                                accelerationGradient[permutation[i]][permutation[j]], tolerance);
                        }
                    }
                }
            }
        }
    }
    cullen_and_dehnen::GradientSums<Real> empty;
    auto const result = cullen_and_dehnen::diagnostics(
        empty, Real(1), Real(0), Real(0), Real(0.5), Real(0), Real(2), Real(0.05));
    require(result.divergence == Real(0) && result.targetBulkViscosityCoefficient == Real(0) &&
            result.inverseDecayTime == Real(0),
        "empty cold neighborhood has finite zero diagnostics");
}

template <typename Real>
void testCullenDehnenDiagnostics() {
    double const tolerance = 256.0 * std::numeric_limits<Real>::epsilon();
    cullen_and_dehnen::GradientSums<Real> sums;
    for (int i = 0; i < 3; ++i) {
        sums.moment[i][i] = Real(1);
    }
    auto evaluate = [&](Real signAverage = Real(-1)) {
        return cullen_and_dehnen::diagnostics(
            sums, Real(1), Real(0), signAverage, Real(0.5), Real(2), Real(2), Real(0.05));
    };
    sums.velocity[0][1] = Real(4);
    auto result = evaluate();
    require(result.divergence == Real(0) && result.targetBulkViscosityCoefficient == Real(0),
        "steady pure shear does not activate viscosity");
    sums.velocity[1][0] = Real(-4);
    sums.acceleration[0][0] = sums.acceleration[1][1] = Real(-16);
    result = evaluate();
    require(result.divergence == Real(0) && result.targetBulkViscosityCoefficient == Real(0),
        "solid-body rotation uses trace V squared, not the squared matrix norm");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sums.velocity[i][j] = i == j ? Real(-1) : Real(0);
            sums.acceleration[i][j] = i == j ? Real(1) : Real(0);
        }
    }
    result = evaluate();
    requireClose("homologous contraction divergence", result.divergence, -3.0, tolerance);
    require(result.targetBulkViscosityCoefficient == Real(0),
        "constant homologous contraction has no steepening detector");
    sums.velocity[0][1] = Real(4);
    sums.velocity[1][0] = Real(-4);
    sums.acceleration[0][0] = sums.acceleration[1][1] = Real(-15);
    sums.acceleration[0][1] = Real(-8);
    sums.acceleration[1][0] = Real(8);
    result = evaluate();
    require(result.targetBulkViscosityCoefficient == Real(0),
        "rotating contraction uses trace V squared, not the squared matrix norm");
    sums.velocity[0][1] = sums.velocity[1][0] = Real(0);
    sums.acceleration[0][1] = sums.acceleration[1][0] = Real(0);
    for (int i = 0; i < 3; ++i) {
        sums.acceleration[i][i] = Real(0);
    }
    result = evaluate();
    requireClose("steepening ballistic compression target", result.targetBulkViscosityCoefficient,
        6.0 / 19.0, tolerance);
    requireClose("Cullen-Dehnen decay uses the full support radius", result.inverseDecayTime, 0.4,
        tolerance);
    auto const cold = cullen_and_dehnen::diagnostics(
        sums, Real(1), Real(0), Real(-1), Real(0.5), Real(0), Real(2), Real(0.05));
    requireClose("cold compressing gas reaches maximum viscosity",
        cold.targetBulkViscosityCoefficient, 2.0, tolerance);

    sums.velocity[1][1] = sums.velocity[2][2] = Real(0);
    sums.velocity[0][1] = Real(10);
    for (Real const signAverage : {Real(-1), Real(0), Real(1)}) {
        double const limiter =
            signAverage == Real(-1) ? 384.0 / 403.0 : (signAverage == Real(0) ? 3.0 / 41.0 : 0.0);
        result = evaluate(signAverage);
        requireClose("neighbor compression signs distinguish shocks from shear",
            result.targetBulkViscosityCoefficient, 0.5 * limiter / (4.0 + 0.25 * limiter),
            tolerance);
    }
    // Represent the same physical ballistic contraction at different scale
    // factors, including the background acceleration term in equation (B12).
    for (Real const scaleFactor : {Real(0.3), Real(0.7), Real(1), Real(2)}) {
        Real const scaleSquared = scaleFactor * scaleFactor;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                sums.velocity[i][j] = i == j ? -scaleSquared : Real(0);
                sums.acceleration[i][j] = i == j ? -Real(0.12) * scaleSquared : Real(0);
            }
        }
        result = cullen_and_dehnen::diagnostics(sums, scaleFactor,
            Real(0.12) * scaleSquared * scaleFactor, Real(-1), Real(0.5) / scaleFactor,
            Real(2) * scaleFactor, Real(2), Real(0.05));
        requireClose("physical divergence independent of cosmological storage units",
            result.divergence, -3.0, tolerance);
        requireClose("target viscosity independent of cosmological storage units",
            result.targetBulkViscosityCoefficient, 6.0 / 19.0, tolerance);
        requireClose("decay rate transforms to conformal time", result.inverseDecayTime,
            0.4 * double(scaleFactor), tolerance);
    }
}

}    // namespace detail

TEST(CullenAndDehnen, CullenDehnenGradientsDouble) {
    detail::testCullenDehnenGradients<double>();
}

TEST(CullenAndDehnen, CullenDehnenGradientsFloat) {
    detail::testCullenDehnenGradients<float>();
}

TEST(CullenAndDehnen, CullenDehnenDiagnosticsDouble) {
    detail::testCullenDehnenDiagnostics<double>();
}

TEST(CullenAndDehnen, CullenDehnenDiagnosticsFloat) {
    detail::testCullenDehnenDiagnostics<float>();
}

TEST(CullenAndDehnen, HalfStorageRoundsNearestWithEvenTies) {
    EXPECT_EQ(Float16Storage(1.0f + 1.0f / 2048.0f).bits, 0x3c00U);
    EXPECT_EQ(Float16Storage(1.0f + 3.0f / 2048.0f).bits, 0x3c02U);
    EXPECT_EQ(Float16Storage(-1.0f - 1.0f / 2048.0f).bits, 0xbc00U);
    EXPECT_EQ(Float16Storage(std::ldexp(1.0f, -24)).bits, 0x0001U);
    EXPECT_EQ(Float16Storage(std::ldexp(1.0f, -25)).bits, 0x0000U);
    EXPECT_EQ(Float16Storage(65504.0f).bits, 0x7bffU);
    EXPECT_EQ(Float16Storage(65520.0f).bits, 0x7c00U);
    EXPECT_EQ(Float16Storage(std::bit_cast<float>(0x80000000U)).bits, 0x8000U);
    Float16Storage coefficient = 0.1f;
    EXPECT_EQ(float(coefficient), 0.0999755859375f);
    coefficient = 0.2f;
    EXPECT_EQ(float(coefficient), 0.199951171875f);
}

TEST(CullenAndDehnen, EveryFiniteHalfStorageValueRoundTripsExactly) {
    for (uint32_t raw = 0; raw <= 0xffffU; ++raw) {
        if ((raw & 0x7c00U) == 0x7c00U) {
            continue;
        }
        Float16Storage const original = Float16Storage::fromBits(uint16_t(raw));
        EXPECT_EQ(Float16Storage(float(original)).bits, raw);
    }
}

TEST(CullenAndDehnen, ScaledInt16RetainsNormalFloatRangeAndSign) {
    for (uint32_t exponent = 1; exponent < 255; ++exponent) {
        for (uint32_t const fraction : {0U, 1U, 0x1ffU, 0x200U, 0x3fffffU, 0x7fffffU}) {
            uint32_t const magnitude = (exponent << 23) | fraction;
            for (uint32_t const sign : {0U, 0x80000000U}) {
                uint32_t const originalBits = sign | magnitude;
                auto const stored =
                    scaled_int16_storage::encode(std::bit_cast<float>(originalBits));
                uint32_t const restoredBits = std::bit_cast<uint32_t>(
                    scaled_int16_storage::decode(stored.significand, stored.exponent));
                SCOPED_TRACE(originalBits);
                EXPECT_TRUE(scaled_int16_storage::isValid(stored.significand, stored.exponent));
                EXPECT_EQ(restoredBits & 0x80000000U, sign);
                uint32_t const restoredMagnitude = restoredBits & 0x7fffffffU;
                EXPECT_LE(restoredMagnitude, magnitude);
                EXPECT_GE(restoredMagnitude, 0x00800000U);
                EXPECT_NE(restoredBits & 0x7f800000U, 0x7f800000U);
                double const significand = double(0x800000U | fraction);
                EXPECT_LT(double(magnitude - restoredMagnitude), significand / 16384.0);
            }
        }
    }
}

TEST(CullenAndDehnen, ScaledInt16FlushesSignedZeroAndSubnormals) {
    // Inspect encodings directly so the test also works with flush-to-zero
    // arithmetic enabled, and both signs must produce the same canonical zero.
    for (uint32_t const magnitude : {0U, 1U, 0x1ffU, 0x200U, 0x400000U, 0x7fffffU}) {
        for (uint32_t const sign : {0U, 0x80000000U}) {
            auto const stored =
                scaled_int16_storage::encode(std::bit_cast<float>(sign | magnitude));
            EXPECT_EQ(stored.significand, 0);
            EXPECT_EQ(stored.exponent, 0);
            EXPECT_TRUE(scaled_int16_storage::isValid(stored.significand, stored.exponent));
            EXPECT_EQ(std::bit_cast<uint32_t>(
                          scaled_int16_storage::decode(stored.significand, stored.exponent)),
                0U);
        }
    }
}

TEST(CullenAndDehnen, ScaledInt16RejectsMalformedAndNonfiniteValues) {
    scaled_int16_storage::ScaledValue const malformed[] = {{1, 0}, {-1, 0}, {16383, 0}, {-16383, 0},
        {-32768, 0}, {16384, -127}, {-16384, -128}, {0, 1}, {0, -128}, {0, 127}};
    for (auto const& stored : malformed) {
        EXPECT_FALSE(scaled_int16_storage::isValid(stored.significand, stored.exponent));
        EXPECT_EQ(std::bit_cast<uint32_t>(
                      scaled_int16_storage::decode(stored.significand, stored.exponent)),
            0x7fc00000U);
    }
    for (uint32_t const bits : {0x7f800000U, 0xff800000U, 0x7fc00000U, 0x7f800001U, 0xffc00001U}) {
        auto const stored = scaled_int16_storage::encode(std::bit_cast<float>(bits));
        EXPECT_EQ(stored.significand, 0);
        EXPECT_EQ(stored.exponent, scaled_int16_storage::kInvalidExponent);
        EXPECT_FALSE(scaled_int16_storage::isValid(stored.significand, stored.exponent));
        EXPECT_EQ(std::bit_cast<uint32_t>(
                      scaled_int16_storage::decode(stored.significand, stored.exponent)),
            0x7fc00000U);
    }
    for (int8_t const exponent : {-126, -1, 0, 127}) {
        for (int16_t const significand : {-32767, -16384, 16384, 32767}) {
            EXPECT_TRUE(scaled_int16_storage::isValid(significand, exponent));
        }
    }
}

TEST(CullenAndDehnen, DivergenceStorageUsesIndependentScales) {
    SphCullenDehnenState state{};
    state.sampleRung = 3;
    float const largest = std::numeric_limits<float>::max();
    float const smallest = std::numeric_limits<float>::min();
    state.setDivergences(largest, smallest);
    EXPECT_EQ(state.divergenceExponent, 127);
    EXPECT_EQ(state.sampleDivergenceExponent, -126);
    EXPECT_EQ(std::bit_cast<uint32_t>(state.sampledDivergence()), 0x00800000U);
    uint32_t const largestStored = std::bit_cast<uint32_t>(state.physicalDivergence());
    EXPECT_LE(largestStored, 0x7f7fffffU);
    EXPECT_EQ(largestStored, 0x7f7ffe00U);
    state.setDivergences(-smallest, -largest);
    EXPECT_EQ(std::bit_cast<uint32_t>(state.physicalDivergence()), 0x80800000U);
    EXPECT_EQ(std::bit_cast<uint32_t>(state.sampledDivergence()), largestStored | 0x80000000U);
    EXPECT_EQ(state.sampleRung, 3);
}

TEST(CullenAndDehnen, OwnMidpointSampleIntervalsFollowRungChanges) {
    SphCullenDehnenState state{};
    EXPECT_EQ(sphViscositySampleInterval(state, 2, 1.0, false), 0.0);
    state.sampleRung = -2;
    EXPECT_EQ(sphViscositySampleInterval(state, 2, 1.0, false), 0.125);
    state.sampleRung = 2;
    EXPECT_EQ(sphViscositySampleInterval(state, 2, 1.0, false), 0.25);

    // A common boundary at tau=0.5 joins a rung-1 midpoint at 0.25 to
    // a rung-2 midpoint at 0.625. Reversing the rung change has the same gap.
    state.sampleRung = 1;
    EXPECT_EQ(sphViscositySampleInterval(state, 2, 1.0, false), 0.375);
    state.sampleRung = 2;
    EXPECT_EQ(sphViscositySampleInterval(state, 1, 1.0, false), 0.375);
    EXPECT_EQ(sphViscositySampleInterval(state, 1, 2.0, false), 0.75);
    EXPECT_EQ(sphViscositySampleInterval(state, 1, 0.0, false), 0.0);
    EXPECT_EQ(state.sampleRung, 2);
}

TEST(CullenAndDehnen, GlobalSampleIntervalsUseSharedMidpointSeparation) {
    SphCullenDehnenState state{};
    EXPECT_EQ(sphViscositySampleInterval(state, 0, 0.125, true), 0.0);
    state.sampleRung = -2;
    // The driver supplies the initial half interval; do not halve it again.
    EXPECT_EQ(sphViscositySampleInterval(state, 0, 0.125, true), 0.125);
    state.sampleRung = 0;
    // Consecutive global intervals of 0.25 and 0.5 have midpoint gap 0.375;
    // then intervals of 0.5 and 0.125 have gap 0.3125.
    EXPECT_EQ(sphViscositySampleInterval(state, 0, 0.375, true), 0.375);
    EXPECT_EQ(sphViscositySampleInterval(state, 0, 0.3125, true), 0.3125);
    EXPECT_EQ(sphViscositySampleInterval(state, 0, 0.0, true), 0.0);
    EXPECT_EQ(state.sampleRung, 0);
}

TEST(CullenAndDehnen, DivergenceHistoryUsesOwnMidpointInterval) {
    // The same physical compression has Dθ/Dt = -3 at the latest sample.
    // Startup, unchanged rungs, refinement, and coarsening must agree even
    // though their previous own-midpoint samples have different separations.
    struct SampleCase {
        int8_t previousRung;
        int currentRung;
        double expectedInterval;
    };
    SampleCase const sampleCases[] = {{-2, 2, 0.125}, {2, 2, 0.25}, {1, 2, 0.375}, {2, 1, 0.375}};
    for (float const scaleFactor : {0.25f, 0.5f, 1.0f, 2.0f}) {
        for (auto const& sample : sampleCases) {
            cullen_and_dehnen::VelocityGradientSums<float> sums;
            for (int component = 0; component < 3; ++component) {
                sums.moment[component][component] = 1.0f;
                sums.velocity[component][component] = -scaleFactor * scaleFactor;
            }
            SphCullenDehnenState state{};
            state.sampleRung = sample.previousRung;
            state.setDivergences(-3.0f, float(-3.0 + 3.0 * scaleFactor * sample.expectedInterval));
            double const sampleInterval =
                sphViscositySampleInterval(state, sample.currentRung, 1.0, false);
            unsigned long long operations = 0;
            auto const cd =
                cullen_and_dehnen::diagnosticsFromHistory(sums, scaleFactor, state, sampleInterval,
                    -1.0f, 0.5f / scaleFactor, 2.0f * scaleFactor, 2.0f, 0.05f, operations);
            auto const hu = hu_etal::diagnosticsFromHistory(sums, scaleFactor, state,
                sampleInterval, 0.5f / scaleFactor, 2.0f * scaleFactor, 3.0f * scaleFactor, 1.0f,
                0.05f, operations);
            auto const price = price_etal::diagnosticsFromHistory(sums, scaleFactor, 0.0f, state,
                sampleInterval, 0.5f / scaleFactor, 2.0f * scaleFactor, 4.0f, 0.1f, operations);
            EXPECT_NEAR(cd.targetBulkViscosityCoefficient, 6.0 / 19.0, 2.0e-6);
            EXPECT_NEAR(hu.targetBulkViscosityCoefficient, 3.0 / 19.0, 2.0e-6);
            EXPECT_NEAR(price.targetBulkViscosityCoefficient, 1.875, 2.0e-6);
            EXPECT_NEAR(hu.limiter, 9.0 / (9.0 + 0.0016), 2.0e-6);
            EXPECT_EQ(state.sampleRung, sample.previousRung);
        }
    }
}

TEST(CullenAndDehnen, HistoryHandlesMissingZeroIntervalAndUnresolvedSamples) {
    cullen_and_dehnen::VelocityGradientSums<float> sums;
    for (int component = 0; component < 3; ++component) {
        sums.moment[component][component] = 1.0f;
        sums.velocity[component][component] = -1.0f;
    }
    unsigned long long operations = 0;
    struct SampleCase {
        int8_t previousRung;
        double interval;
    };
    SampleCase const sampleCases[] = {{-1, 0.5}, {0, 0.0}, {0, -0.25}};
    for (auto const& sample : sampleCases) {
        SphCullenDehnenState state{};
        state.sampleRung = sample.previousRung;
        state.setDivergences(100.0f, 100.0f);
        auto const cd = cullen_and_dehnen::diagnosticsFromHistory(
            sums, 1.0f, state, sample.interval, -1.0f, 0.5f, 2.0f, 2.0f, 0.05f, operations);
        auto const hu = hu_etal::diagnosticsFromHistory(
            sums, 1.0f, state, sample.interval, 0.5f, 2.0f, 3.0f, 1.0f, 0.05f, operations);
        auto const price = price_etal::diagnosticsFromHistory(
            sums, 1.0f, 0.0f, state, sample.interval, 0.5f, 2.0f, 1.0f, 0.1f, operations);
        EXPECT_EQ(cd.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_EQ(hu.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_EQ(price.targetBulkViscosityCoefficient, 0.0f);
    }
    SphCullenDehnenState state{};
    state.sampleRung = 0;
    state.setDivergences(100.0f, 100.0f);
    cullen_and_dehnen::VelocityGradientSums<float> empty;
    auto const hu = hu_etal::diagnosticsFromHistory(
        empty, 1.0f, state, 0.5, 0.5f, 2.0f, 3.0f, 1.0f, 0.05f, operations);
    EXPECT_EQ(hu.targetBulkViscosityCoefficient, 0.0f);
}

TEST(CullenAndDehnen, PriceHistorySubtractsHomogeneousExpansion) {
    SphCullenDehnenState state{};
    state.sampleRung = 0;
    state.setDivergences(0.0f, 0.0f);    // No peculiar divergence at the previous time.
    unsigned long long operations = 0;
    for (float const scaleFactor : {0.5f, 1.0f, 2.0f}) {
        cullen_and_dehnen::VelocityGradientSums<float> sums;
        float const expansionRate = 0.25f;
        for (int i = 0; i < 3; ++i) {
            sums.moment[i][i] = 1.0f;
            sums.velocity[i][i] = expansionRate;
        }
        auto const result = price_etal::diagnosticsFromHistory(
            sums, scaleFactor, expansionRate, state, 0.5, 0.5f, 2.0f, 1.0f, 0.1f, operations);
        EXPECT_EQ(result.sampleDivergence, 0.0f);
        EXPECT_EQ(result.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_EQ(result.divergence, 0.75f / (scaleFactor * scaleFactor));
    }
}
