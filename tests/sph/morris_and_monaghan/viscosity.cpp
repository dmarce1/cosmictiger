/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/sph/viscosity/morris_and_monaghan.hpp>
#include <tests/sph/test_helpers.hpp>

namespace detail {

template <typename Real>
void testViscosityEvolution() {
    double const tolerance = 64.0 * std::numeric_limits<Real>::epsilon();
    Real const minimumBulkViscosityCoefficient = Real(0.1);
    Real const inverseDecayTime = Real(0.4);
    Real const timeStep = Real(0.5);
    for (Real const divergence : {Real(0), Real(3)}) {
        require(morris_and_monaghan::evolvedBulkViscosityCoefficient(
                    minimumBulkViscosityCoefficient, minimumBulkViscosityCoefficient,
                    inverseDecayTime, divergence, timeStep) == minimumBulkViscosityCoefficient,
            "quiescent and expanding gas remain at the viscosity floor");
        requireClose("backward Euler postshock decay",
            morris_and_monaghan::evolvedBulkViscosityCoefficient(
                Real(0.7), minimumBulkViscosityCoefficient, inverseDecayTime, divergence, timeStep),
            0.6, tolerance);
    }
    requireClose("compression increases viscosity even at the floor",
        morris_and_monaghan::evolvedBulkViscosityCoefficient(minimumBulkViscosityCoefficient,
            minimumBulkViscosityCoefficient, inverseDecayTime, Real(-3), timeStep),
        1.35, tolerance);
    requireClose("cold gas retains compression growth without decay",
        morris_and_monaghan::evolvedBulkViscosityCoefficient(
            Real(4), minimumBulkViscosityCoefficient, Real(0), Real(-3), timeStep),
        5.5, tolerance);
    require(morris_and_monaghan::evolvedBulkViscosityCoefficient(Real(0.7),
                minimumBulkViscosityCoefficient, inverseDecayTime, Real(-3), Real(0)) == Real(0.7),
        "zero interval preserves the coefficient");
    Real const stiff = morris_and_monaghan::evolvedBulkViscosityCoefficient(
        Real(0.7), minimumBulkViscosityCoefficient, inverseDecayTime, Real(0), Real(1e6));
    require(stiff >= minimumBulkViscosityCoefficient && stiff < Real(0.101),
        "stiff decay approaches the floor without overshooting");

    // Morris & Monaghan (1997), equations (24)-(25): constant homologous
    // compression has a steady alpha above the floor, with no upper cap.
    Real const equilibriumAlpha = Real(7.6);
    requireClose("unbounded original M&M compression equilibrium",
        morris_and_monaghan::evolvedBulkViscosityCoefficient(equilibriumAlpha,
            minimumBulkViscosityCoefficient, inverseDecayTime, Real(-3), timeStep),
        7.6, tolerance);
    require(morris_and_monaghan::evolvedBulkViscosityCoefficient(equilibriumAlpha - Real(1),
                minimumBulkViscosityCoefficient, inverseDecayTime, Real(-3),
                timeStep) > equilibriumAlpha - Real(1),
        "compression grows viscosity above later schemes' usual alpha caps");
    require(morris_and_monaghan::evolvedBulkViscosityCoefficient(equilibriumAlpha + Real(1),
                minimumBulkViscosityCoefficient, inverseDecayTime, Real(-3),
                timeStep) < equilibriumAlpha + Real(1),
        "decay returns viscosity toward its compressive equilibrium");

    auto decayError = [minimumBulkViscosityCoefficient](int steps) {
        Real bulkViscosityCoefficient = Real(1.1);
        for (int step = 0; step < steps; ++step) {
            bulkViscosityCoefficient =
                morris_and_monaghan::evolvedBulkViscosityCoefficient(bulkViscosityCoefficient,
                    minimumBulkViscosityCoefficient, Real(1), Real(0), Real(1) / Real(steps));
        }
        return std::fabs(double(bulkViscosityCoefficient) - (0.1 + std::exp(-1.0)));
    };
    double const ratio = decayError(32) / decayError(64);
    require(ratio > 1.8 && ratio < 2.2, "backward Euler decay converges at first order");
}

}    // namespace detail

TEST(MorrisAndMonaghan, ViscosityEvolutionDouble) {
    detail::testViscosityEvolution<double>();
}

TEST(MorrisAndMonaghan, ViscosityEvolutionFloat) {
    detail::testViscosityEvolution<float>();
}
