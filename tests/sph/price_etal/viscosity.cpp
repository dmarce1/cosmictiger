/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/sph/price_etal/viscosity.hpp>
#include <tests/sph/test_helpers.hpp>

namespace detail {

// Price et al. (2018), equations (46)-(48) and (53), on exact affine gradients with
// unit moment sums, so the reconstructed gradients equal the velocity sums.
template <typename Real>
void testPriceEtAlDiagnostics() {
    double const tolerance = 256.0 * std::numeric_limits<Real>::epsilon();
    cullen_and_dehnen::GradientSums<Real> sums;
    for (int i = 0; i < 3; ++i) {
        sums.moment[i][i] = Real(1);
    }
    auto evaluate = [&](Real soundSpeed = Real(2), Real alphaMaximum = Real(1)) {
        return price_etal::diagnostics(
            sums, Real(1), Real(0), Real(0.5), soundSpeed, alphaMaximum, Real(0.1));
    };
    auto const empty = evaluate();
    require(empty.divergence == Real(0) && empty.targetAlpha == Real(0),
        "uniform flow has no divergence and no target");
    requireClose("decay rate is sigma c_s / h", empty.inverseDecayTime, 0.4, tolerance);
    sums.velocity[0][1] = Real(4);
    auto result = evaluate();
    require(result.divergence == Real(0) && result.targetAlpha == Real(0),
        "steady pure shear does not activate viscosity");
    sums.velocity[1][0] = Real(-4);
    sums.acceleration[0][0] = sums.acceleration[1][1] = Real(-16);
    result = evaluate();
    require(result.divergence == Real(0) && result.targetAlpha == Real(0),
        "solid-body rotation has zero divergence and so zero limiter");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sums.velocity[i][j] = i == j ? Real(-1) : Real(0);
            sums.acceleration[i][j] = i == j ? Real(1) : Real(0);
        }
    }
    result = evaluate();
    requireClose("homologous contraction divergence", result.divergence, -3.0, tolerance);
    require(result.targetAlpha == Real(0),
        "constant homologous contraction has no steepening detector");
    for (int i = 0; i < 3; ++i) {
        sums.acceleration[i][i] = Real(0);
    }
    // Ballistic contraction: A = 3, so 10 h² A / c_s² = 10 (0.25)(3)/4 = 1.875.
    result = evaluate();
    require(result.targetAlpha == Real(1), "equation (48) caps the target at alpha_max");
    requireClose(
        "uncapped equation (48) target", evaluate(Real(2), Real(4)).targetAlpha, 1.875, tolerance);
    requireClose(
        "hotter gas lowers the target", evaluate(Real(4), Real(4)).targetAlpha, 0.46875, tolerance);
    requireClose("decay rate follows the sound speed", evaluate(Real(4), Real(4)).inverseDecayTime,
        0.8, tolerance);
    auto const cold = evaluate(Real(0), Real(0.75));
    require(cold.targetAlpha == Real(0.75) && cold.inverseDecayTime == Real(0),
        "cold compressing gas reaches maximum viscosity without decaying");
    // Sheared compression: div v = -1, |curl v|² = 100, so xi = 1/101 and A = 1/101.
    sums.velocity[1][1] = sums.velocity[2][2] = Real(0);
    sums.velocity[0][1] = Real(10);
    result = evaluate(Real(4), Real(4));
    requireClose("sheared compression divergence", result.divergence, -1.0, tolerance);
    requireClose("equation (47) limits sheared compression", result.targetAlpha,
        10.0 * 0.25 / (101.0 * 16.0), tolerance);
    // Represent the same physical ballistic contraction at different scale factors,
    // including the background acceleration term of equation (53).
    for (Real const scaleFactor : {Real(0.3), Real(0.7), Real(1), Real(2)}) {
        Real const scaleSquared = scaleFactor * scaleFactor;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                sums.velocity[i][j] = i == j ? -scaleSquared : Real(0);
                sums.acceleration[i][j] = i == j ? -Real(0.12) * scaleSquared : Real(0);
            }
        }
        result = price_etal::diagnostics(sums, scaleFactor, Real(0.12) * scaleSquared * scaleFactor,
            Real(0.5) / scaleFactor, Real(2) * scaleFactor, Real(4), Real(0.1));
        requireClose("physical divergence independent of cosmological storage units",
            result.divergence, -3.0, tolerance);
        requireClose("target viscosity independent of cosmological storage units",
            result.targetAlpha, 1.875, tolerance);
        requireClose("decay rate transforms to conformal time", result.inverseDecayTime,
            0.4 * double(scaleFactor), tolerance);
    }
    cullen_and_dehnen::GradientSums<Real> unresolved;
    auto const degenerate = price_etal::diagnostics(
        unresolved, Real(1), Real(0), Real(0.5), Real(2), Real(1), Real(0.1));
    require(degenerate.divergence == Real(0) && degenerate.targetAlpha == Real(0),
        "an unresolved neighborhood has finite zero diagnostics");
}

void testPriceEtAlEvolution() {
    double const tolerance = 4.0 * std::numeric_limits<float>::epsilon();
    require(price_etal::evolvedAlpha(0.2f, 0.5f, 1.0f, 0.1f) == 0.5f,
        "alpha rises immediately to a larger target");
    require(price_etal::evolvedAlpha(0.5f, 0.2f, 1.0f, 0.0f) == 0.5f,
        "a zero step leaves alpha unchanged");
    // Equation (50) with Δt/τ = 1: (1 + 0.25)/2.
    requireClose("backward Euler decay toward the target",
        price_etal::evolvedAlpha(1.0f, 0.25f, 2.0f, 0.5f), 0.625, tolerance);
    requireClose("decay with a tiny step approaches the explicit rate",
        price_etal::evolvedAlpha(1.0f, 0.0f, 1.0f, 1.0e-3f), 1.0 / 1.001, tolerance);
    float const huge = price_etal::evolvedAlpha(1.0f, 0.25f, 1.0f, 1.0e8f);
    require(huge >= 0.25f && huge < 0.2500001f,
        "the implicit step is stable and never undershoots the target");
}

}    // namespace detail

TEST(PriceEtAl, DiagnosticsDouble) {
    detail::testPriceEtAlDiagnostics<double>();
}

TEST(PriceEtAl, DiagnosticsFloat) {
    detail::testPriceEtAlDiagnostics<float>();
}

TEST(PriceEtAl, Evolution) {
    detail::testPriceEtAlEvolution();
}
