/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/sph/viscosity/hu_etal.hpp>

namespace {

template <typename Real>
void checkDiagnostics() {
    double const tolerance = 256 * std::numeric_limits<Real>::epsilon();
    cullen_and_dehnen::GradientSums<Real> sums{};
    for (int i = 0; i < 3; ++i) {
        sums.moment[i][i] = Real(1);
        sums.velocity[i][i] = Real(-1);
    }
    auto evaluate = [&] {
        return hu_etal::diagnostics(
            sums, Real(1), Real(0), Real(0.5), Real(2), Real(3), Real(1), Real(0.05));
    };
    auto result = evaluate();
    // Affine ballistic contraction: theta=-3, Dtheta/Dt=-tr(V^2)=-3.
    EXPECT_NEAR(result.divergence, -3, tolerance);
    EXPECT_NEAR(result.targetBulkViscosityCoefficient, 0.75 / 4.75, tolerance);
    EXPECT_NEAR(result.limiter, 9.0 / (9.0 + 0.0001 * 16), tolerance);
    EXPECT_NEAR(result.inverseDecayTime, 0.6, tolerance);
    Real const unmodifiedTarget = result.targetBulkViscosityCoefficient;
    sums.velocity[0][1] = Real(10);
    result = evaluate();
    EXPECT_NEAR(result.targetBulkViscosityCoefficient, unmodifiedTarget, tolerance);
    EXPECT_NEAR(result.limiter, 9.0 / (9.0 + 100 + 0.0016), tolerance);
    // An arbitrarily strong shock cannot bypass the multiplicative shear limiter.
    sums.acceleration[0][0] = Real(-1.0e8);
    result = evaluate();
    Real const bulkViscosityCoefficient = hu_etal::evolvedBulkViscosityCoefficient(Real(0.8),
        result.targetBulkViscosityCoefficient, result.inverseDecayTime, result.limiter, Real(0.1));
    EXPECT_LE(bulkViscosityCoefficient, result.limiter);
    EXPECT_GE(bulkViscosityCoefficient, Real(0.999999) * result.limiter);

    // Identical physical fields in different cosmological storage units.
    for (double const a : {0.3, 0.7, 1.0, 2.0}) {
        cullen_and_dehnen::GradientSums<Real> scaled{};
        for (int i = 0; i < 3; ++i) {
            scaled.moment[i][i] = Real(1);
            scaled.velocity[i][i] = Real(-a * a);
        }
        auto const physical = hu_etal::diagnostics(
            scaled, Real(a), Real(0), Real(0.5 / a), Real(2 * a), Real(3 * a), Real(1), Real(0.05));
        EXPECT_NEAR(physical.divergence, -3, tolerance);
        EXPECT_NEAR(physical.targetBulkViscosityCoefficient, 0.75 / 4.75, tolerance);
        EXPECT_NEAR(physical.limiter, 9.0 / (9.0 + 0.0016), tolerance);
        EXPECT_NEAR(physical.inverseDecayTime, 0.6 * a, tolerance);
    }
    // Constant compression has Dtheta/Dt=0: no shock target.
    for (int i = 0; i < 3; ++i) {
        sums.acceleration[i][i] = Real(1);
    }
    EXPECT_NEAR(evaluate().targetBulkViscosityCoefficient, 0, tolerance);
    sums = {};
    result = evaluate();
    EXPECT_EQ(result.targetBulkViscosityCoefficient, Real(0));
    EXPECT_EQ(result.limiter, Real(0));
    EXPECT_TRUE(std::isfinite(result.inverseDecayTime));
    for (int i = 0; i < 3; ++i) {
        sums.moment[i][i] = Real(1);
    }
    result = hu_etal::diagnostics(
        sums, Real(1), Real(0), Real(0.5), Real(0), Real(0), Real(1), Real(0.05));
    EXPECT_EQ(result.targetBulkViscosityCoefficient, Real(0));
    EXPECT_EQ(result.limiter, Real(0));
}

}    // namespace

TEST(HuEtAl, DiagnosticsDouble) {
    checkDiagnostics<double>();
}

TEST(HuEtAl, DiagnosticsFloat) {
    checkDiagnostics<float>();
}

TEST(HuEtAl, LimitsWholeUpdate) {
    EXPECT_DOUBLE_EQ(hu_etal::evolvedBulkViscosityCoefficient(0.1, 0.6, 2.0, 0.25, 0.0), 0.15);
    EXPECT_DOUBLE_EQ(hu_etal::evolvedBulkViscosityCoefficient(0.6, 0.6, 2.0, 0.25, 0.0), 0.15);
    EXPECT_NEAR(hu_etal::evolvedBulkViscosityCoefficient(0.9, 0.2, 2.0, 0.25, 0.1),
        0.25 * (0.2 + 0.7 * std::exp(-0.2)), 1.0e-15);
    EXPECT_DOUBLE_EQ(hu_etal::evolvedBulkViscosityCoefficient(0.9, 0.2, 2.0, 0.0, 0.1), 0.0);
    EXPECT_DOUBLE_EQ(hu_etal::evolvedBulkViscosityCoefficient(0.9, 0.2, 2.0, 0.25, 0.0), 0.225);
}
