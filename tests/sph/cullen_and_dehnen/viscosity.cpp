/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/sph/cullen_and_dehnen/viscosity.hpp>
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
    require(result.divergence == Real(0) && result.targetAlpha == Real(0) &&
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
    require(result.divergence == Real(0) && result.targetAlpha == Real(0),
        "steady pure shear does not activate viscosity");
    sums.velocity[1][0] = Real(-4);
    sums.acceleration[0][0] = sums.acceleration[1][1] = Real(-16);
    result = evaluate();
    require(result.divergence == Real(0) && result.targetAlpha == Real(0),
        "solid-body rotation uses trace V squared, not the squared matrix norm");
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
    sums.velocity[0][1] = Real(4);
    sums.velocity[1][0] = Real(-4);
    sums.acceleration[0][0] = sums.acceleration[1][1] = Real(-15);
    sums.acceleration[0][1] = Real(-8);
    sums.acceleration[1][0] = Real(8);
    result = evaluate();
    require(result.targetAlpha == Real(0),
        "rotating contraction uses trace V squared, not the squared matrix norm");
    sums.velocity[0][1] = sums.velocity[1][0] = Real(0);
    sums.acceleration[0][1] = sums.acceleration[1][0] = Real(0);
    for (int i = 0; i < 3; ++i) {
        sums.acceleration[i][i] = Real(0);
    }
    result = evaluate();
    requireClose(
        "steepening ballistic compression target", result.targetAlpha, 6.0 / 19.0, tolerance);
    requireClose("Cullen-Dehnen decay uses the full support radius", result.inverseDecayTime, 0.4,
        tolerance);
    auto const cold = cullen_and_dehnen::diagnostics(
        sums, Real(1), Real(0), Real(-1), Real(0.5), Real(0), Real(2), Real(0.05));
    requireClose(
        "cold compressing gas reaches maximum viscosity", cold.targetAlpha, 2.0, tolerance);

    sums.velocity[1][1] = sums.velocity[2][2] = Real(0);
    sums.velocity[0][1] = Real(10);
    for (Real const signAverage : {Real(-1), Real(0), Real(1)}) {
        double const limiter =
            signAverage == Real(-1) ? 384.0 / 403.0 : (signAverage == Real(0) ? 3.0 / 41.0 : 0.0);
        result = evaluate(signAverage);
        requireClose("neighbor compression signs distinguish shocks from shear", result.targetAlpha,
            0.5 * limiter / (4.0 + 0.25 * limiter), tolerance);
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
            result.targetAlpha, 6.0 / 19.0, tolerance);
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
