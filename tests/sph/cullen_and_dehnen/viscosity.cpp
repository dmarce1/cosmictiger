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

TEST(CullenAndDehnen, TimestepScaledDivergenceRecoversPreviousSample) {
    struct SampleCase {
        float divergence;
        double timestep;
    };
    SampleCase const samples[] = {{-8.0f, 0.125}, {3.141592f, 0.25},
        {std::numeric_limits<float>::max(), std::ldexp(1.0, -127)},
        {std::numeric_limits<float>::min(), std::ldexp(1.0, 126)}};
    Float16Storage state{};
    for (auto const& sample : samples) {
        ASSERT_TRUE(sphEncodeDivergence(state, sample.divergence, sample.timestep));
        EXPECT_NEAR(sphDecodeDivergence(state, sample.timestep), double(sample.divergence),
            std::abs(double(sample.divergence)) / 2048.0);
        EXPECT_EQ(sphDivergenceSign(state), sample.divergence > 0.0f ? 1.0f : -1.0f);
    }
    ASSERT_TRUE(sphEncodeDivergence(state, 0.0f, 0.125));
    EXPECT_EQ(sphDivergenceSign(state), 0.0f);
    EXPECT_EQ(sphDecodeDivergence(state, 0.125), 0.0);
}

TEST(CullenAndDehnen, TimestepScaledDivergenceRejectsInvalidInputsWithoutClamping) {
    Float16Storage state{};
    ASSERT_TRUE(sphEncodeDivergence(state, -2.0f, 0.25));
    uint16_t const previousBits = state.bits;
    for (double const timestep : {0.0, -1.0, std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN(), 65505.0,
             std::numeric_limits<double>::max()}) {
        EXPECT_FALSE(sphEncodeDivergence(state, 1.0f, timestep));
        EXPECT_EQ(state.bits, previousBits);
    }
    for (float const divergence :
        {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()}) {
        EXPECT_FALSE(sphEncodeDivergence(state, divergence, 1.0));
        EXPECT_EQ(state.bits, previousBits);
    }
    EXPECT_FALSE(sphEncodeDivergence(state, 2.0f, std::numeric_limits<double>::max()));
    EXPECT_EQ(state.bits, previousBits);
    EXPECT_TRUE(sphEncodeDivergence(state, -65504.0f, 1.0));
    EXPECT_EQ(float(state), -65504.0f);
    EXPECT_EQ(sphDecodeDivergence(state, 0.0), 0.0);
}

TEST(CullenAndDehnen, TimestepScaledDivergenceRoundsSmallSamples) {
    Float16Storage state{};
    ASSERT_TRUE(sphEncodeDivergence(state, std::ldexp(1.0f, -24), 1.0));
    EXPECT_EQ(state.bits, 1U);
    ASSERT_TRUE(sphEncodeDivergence(state, std::ldexp(1.0f, -25), 1.0));
    EXPECT_EQ(state.bits, 0U);
    EXPECT_EQ(sphDivergenceSign(state), 0.0f);
}

TEST(CullenAndDehnen, StoredSampleScaleUsesLiveRungOrSharedGlobalStep) {
    EXPECT_EQ(sphViscosityStoredSampleTimestep(3, 2.0, 0.375, false), 0.25);
    EXPECT_EQ(sphViscosityStoredSampleTimestep(3, 2.0, 0.375, true), 0.375);
    EXPECT_EQ(sphViscosityStoredSampleTimestep(0, 2.0, 0.0, true), 2.0);
    EXPECT_EQ(sphViscosityStoredSampleTimestep(3, 0.0, 0.375, false), 0.0);
    EXPECT_EQ(sphViscosityStoredSampleTimestep(-1, 2.0, 0.375, false), 0.0);
}

TEST(CullenAndDehnen, EndpointIntervalsUseCompletedStep) {
    EXPECT_EQ(sphViscositySampleInterval(2, 1.0, false, true), 0.0);
    EXPECT_EQ(sphViscositySampleInterval(2, 1.0, false), 0.25);
    EXPECT_EQ(sphViscositySampleInterval(1, 1.0, false), 0.5);
    EXPECT_EQ(sphViscositySampleInterval(0, 0.375, true), 0.375);
    EXPECT_EQ(sphViscositySampleInterval(0, 0.375, true, true), 0.0);
    EXPECT_EQ(sphViscositySampleInterval(0, 0.0, true), 0.0);
}

TEST(CullenAndDehnen, LiveRungTransitionsRescaleWithoutChangingGradient) {
    Float16Storage state{};
    ASSERT_TRUE(sphEncodeDivergence(state, -3.0f, 0.5));
    ASSERT_TRUE(sphRescaleDivergence(state, 1, 3));
    EXPECT_EQ(sphDecodeDivergence(state, 0.125), -3.0);
    ASSERT_TRUE(sphRescaleDivergence(state, 3, 2));
    EXPECT_EQ(sphDecodeDivergence(state, 0.25), -3.0);
    uint16_t const bits = state.bits;
    ASSERT_TRUE(sphRescaleDivergence(state, 2, 2));
    EXPECT_EQ(state.bits, bits);
    EXPECT_FALSE(sphRescaleDivergence(state, -1, 2));
    EXPECT_EQ(state.bits, bits);
    state = 65504.0f;
    EXPECT_FALSE(sphRescaleDivergence(state, 1, 0));
    EXPECT_EQ(float(state), 65504.0f);
}

TEST(CullenAndDehnen, DivergenceHistoryUsesCompletedEndpointInterval) {
    // The same converted GADGET-4 indicator has a⁻² DG/Dt = -3 at the latest sample.
    // Every live rung recovers the same slope from successive endpoints.
    struct SampleCase {
        int currentRung;
        double expectedInterval;
    };
    SampleCase const sampleCases[] = {{1, 0.5}, {2, 0.25}, {3, 0.125}};
    for (float const scaleFactor : {0.25f, 0.5f, 1.0f, 2.0f}) {
        for (auto const& sample : sampleCases) {
            cullen_and_dehnen::VelocityGradientSums<float> sums;
            for (int component = 0; component < 3; ++component) {
                sums.moment[component][component] = 1.0f;
                sums.velocity[component][component] = -scaleFactor * scaleFactor;
            }
            Float16Storage state{};
            double const previousSampleTimestep =
                sphViscosityStoredSampleTimestep(sample.currentRung, 1.0, 0.0, false);
            ASSERT_TRUE(sphEncodeDivergence(state,
                float(-3.0 * scaleFactor * scaleFactor +
                    3.0 * scaleFactor * scaleFactor * scaleFactor * sample.expectedInterval),
                previousSampleTimestep));
            double const sampleInterval =
                sphViscositySampleInterval(sample.currentRung, 1.0, false);
            unsigned long long operations = 0;
            auto const cd = cullen_and_dehnen::diagnosticsFromHistory(sums, scaleFactor, state,
                sampleInterval, previousSampleTimestep, -1.0f, 0.5f / scaleFactor,
                2.0f * scaleFactor, 2.0f, 0.05f, operations);
            auto const hu = hu_etal::diagnosticsFromHistory(sums, scaleFactor, state,
                sampleInterval, previousSampleTimestep, 0.5f / scaleFactor, 2.0f * scaleFactor,
                3.0f * scaleFactor, 1.0f, 0.05f, operations);
            auto const price = price_etal::diagnosticsFromHistory(sums, scaleFactor, state,
                sampleInterval, previousSampleTimestep, 0.5f / scaleFactor, 2.0f * scaleFactor,
                4.0f, 0.1f, operations);
            EXPECT_NEAR(cd.targetBulkViscosityCoefficient, 6.0 / 19.0, 2.0e-6);
            EXPECT_NEAR(hu.targetBulkViscosityCoefficient, 3.0 / 19.0, 2.0e-6);
            EXPECT_NEAR(price.targetBulkViscosityCoefficient, 1.875, 2.0e-6);
            EXPECT_NEAR(hu.limiter, 9.0 / (9.0 + 0.0016), 2.0e-6);
        }
    }
}

TEST(CullenAndDehnen, HistoryHandlesInitializationZeroIntervalAndUnresolvedSamples) {
    cullen_and_dehnen::VelocityGradientSums<float> sums;
    for (int component = 0; component < 3; ++component) {
        sums.moment[component][component] = 1.0f;
        sums.velocity[component][component] = -1.0f;
    }
    unsigned long long operations = 0;
    for (double const interval : {0.0, -0.25}) {
        Float16Storage state{};
        ASSERT_TRUE(sphEncodeDivergence(state, 100.0f, 1.0));
        auto const cd = cullen_and_dehnen::diagnosticsFromHistory(
            sums, 1.0f, state, interval, 1.0, -1.0f, 0.5f, 2.0f, 2.0f, 0.05f, operations);
        auto const hu = hu_etal::diagnosticsFromHistory(
            sums, 1.0f, state, interval, 1.0, 0.5f, 2.0f, 3.0f, 1.0f, 0.05f, operations);
        auto const price = price_etal::diagnosticsFromHistory(
            sums, 1.0f, state, interval, 1.0, 0.5f, 2.0f, 1.0f, 0.1f, operations);
        EXPECT_EQ(cd.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_EQ(hu.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_EQ(price.targetBulkViscosityCoefficient, 0.0f);
    }
    Float16Storage state{};
    ASSERT_TRUE(sphEncodeDivergence(state, 100.0f, 1.0));
    cullen_and_dehnen::VelocityGradientSums<float> empty;
    auto const hu = hu_etal::diagnosticsFromHistory(
        empty, 1.0f, state, 0.5, 1.0, 0.5f, 2.0f, 3.0f, 1.0f, 0.05f, operations);
    EXPECT_EQ(hu.targetBulkViscosityCoefficient, 0.0f);
}

TEST(CullenAndDehnen, AllHistorySensorsRemainQuietForPureHubbleExpansion) {
    Float16Storage state{};
    ASSERT_TRUE(sphEncodeDivergence(state, 0.0f, 0.25));
    unsigned long long operations = 0;
    for (float const scaleFactor : {0.5f, 1.0f, 2.0f}) {
        // Pure Hubble flow has zero stored peculiar velocity. Production passes
        // these zero peculiar gradients without adding the Hubble background.
        cullen_and_dehnen::VelocityGradientSums<float> sums;
        for (int component = 0; component < 3; ++component) {
            sums.moment[component][component] = 1.0f;
        }
        auto const cd = cullen_and_dehnen::diagnosticsFromHistory(
            sums, scaleFactor, state, 0.375, 0.25, 0.0f, 0.5f, 2.0f, 2.0f, 0.05f, operations);
        auto const hu = hu_etal::diagnosticsFromHistory(
            sums, scaleFactor, state, 0.375, 0.25, 0.5f, 2.0f, 3.0f, 1.0f, 0.05f, operations);
        auto const price = price_etal::diagnosticsFromHistory(
            sums, scaleFactor, state, 0.375, 0.25, 0.5f, 2.0f, 1.0f, 0.1f, operations);
        EXPECT_EQ(cd.divergence, 0.0f);
        EXPECT_EQ(hu.divergence, 0.0f);
        EXPECT_EQ(price.divergence, 0.0f);
        EXPECT_EQ(cd.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_EQ(hu.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_EQ(price.targetBulkViscosityCoefficient, 0.0f);
    }
}

TEST(CullenAndDehnen, GlobalHistorySeparatesSampleScaleFromCompletedInterval) {
    // Changing the global duration changes q's normalization independently of
    // the completed interval used in the derivative. Recover the same slope in
    // both refinement and coarsening cases.
    for (double const previousTimestep : {0.125, 0.25, 0.5}) {
        for (double const currentTimestep : {0.125, 0.25, 0.5}) {
            double const completedInterval = currentTimestep;
            Float16Storage state{};
            ASSERT_TRUE(sphEncodeDivergence(
                state, float(-3.0 + 2.0 * completedInterval), previousTimestep));
            double const sampleScale =
                sphViscosityStoredSampleTimestep(0, 1.0, previousTimestep, true);
            unsigned long long operations = 0;
            EXPECT_DOUBLE_EQ(cullen_and_dehnen::divergenceTimeDerivative(
                                 -3.0, 1.0, state, completedInterval, sampleScale, operations),
                -2.0);
        }
    }
}

TEST(CullenAndDehnen, ConstantStoredVelocityDivergenceHasNoRungChangeTrigger) {
    // The normalized q values change with the rung, while the canonical
    // divergence remains constant. Its GADGET-4 time-difference trigger is zero.
    for (int const previousRung : {0, 1, 2, 3}) {
        for (int const currentRung : {0, 1, 2, 3}) {
            Float16Storage state{};
            double const oldScale = std::ldexp(1.0, -previousRung);
            ASSERT_TRUE(sphEncodeDivergence(state, -3.0f, oldScale));
            ASSERT_TRUE(sphRescaleDivergence(state, previousRung, currentRung));
            double const sampleScale = std::ldexp(1.0, -currentRung);
            double const completedInterval = sphViscositySampleInterval(currentRung, 1.0, false);
            for (float const scaleFactor : {0.25f, 0.5f, 1.0f, 2.0f}) {
                cullen_and_dehnen::VelocityGradientSums<float> sums;
                for (int component = 0; component < 3; ++component) {
                    sums.moment[component][component] = 1.0f;
                    sums.velocity[component][component] = -1.0f;
                }
                unsigned long long operations = 0;
                auto const result =
                    cullen_and_dehnen::diagnosticsFromHistory(sums, scaleFactor, state,
                        completedInterval, sampleScale, -1.0f, 0.5f, 2.0f, 2.0f, 0.05f, operations);
                EXPECT_EQ(result.divergence, -3.0f);
                EXPECT_EQ(result.targetBulkViscosityCoefficient, 0.0f);
            }
        }
    }
}

TEST(CullenAndDehnen, HistoryMatchesGadgetGradientConversionDuringExpansion) {
    // Hold θ_pec constant while a changes. G = a² θ_pec changes, so GADGET-4
    // detects a⁻² DG/Dt rather than cancelling to Dθ_pec/Dt = 0. The finite
    // difference in conformal time becomes a*(G_n-G_previous)/Δτ in the scaled
    // shock indicator consumed by the target formulas.
    float constexpr kPhysicalPeculiarDivergence = -3.0f;
    double constexpr kPreviousTimestep = 0.125;
    double constexpr kCompletedInterval = 0.25;
    for (float const previousScale : {0.25f, 0.5f, 1.0f}) {
        float const currentScale = 2.0f * previousScale;
        float const previousGradient = previousScale * previousScale * kPhysicalPeculiarDivergence;
        float const currentGradient = currentScale * currentScale * kPhysicalPeculiarDivergence;
        Float16Storage state{};
        ASSERT_TRUE(sphEncodeDivergence(state, previousGradient, kPreviousTimestep));
        unsigned long long operations = 0;
        double const expected =
            currentScale * double(currentGradient - previousGradient) / kCompletedInterval;
        EXPECT_DOUBLE_EQ(
            cullen_and_dehnen::divergenceTimeDerivative(double(currentGradient),
                double(currentScale), state, kCompletedInterval, kPreviousTimestep, operations),
            expected);
        cullen_and_dehnen::VelocityGradientSums<float> sums;
        for (int component = 0; component < 3; ++component) {
            sums.moment[component][component] = 1.0f;
            sums.velocity[component][component] = currentGradient / 3.0f;
        }
        auto const cd = cullen_and_dehnen::diagnosticsFromHistory(sums, currentScale, state,
            kCompletedInterval, kPreviousTimestep, -1.0f, 0.5f, 2.0f, 2.0f, 0.05f, operations);
        auto const hu = hu_etal::diagnosticsFromHistory(sums, currentScale, state,
            kCompletedInterval, kPreviousTimestep, 0.5f, 2.0f, 3.0f, 1.0f, 0.05f, operations);
        auto const price = price_etal::diagnosticsFromHistory(sums, currentScale, state,
            kCompletedInterval, kPreviousTimestep, 0.125f, 2.0f, 4.0f, 0.1f, operations);
        EXPECT_EQ(cd.divergence, currentGradient);
        EXPECT_EQ(hu.divergence, currentGradient);
        EXPECT_EQ(price.divergence, currentGradient);
        EXPECT_GT(cd.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_GT(hu.targetBulkViscosityCoefficient, 0.0f);
        EXPECT_GT(price.targetBulkViscosityCoefficient, 0.0f);
    }
}
