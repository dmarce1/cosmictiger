/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/sph/morris_and_monaghan/viscosity.hpp>
#include <tests/sph/test_helpers.hpp>

namespace detail {

template <typename Real>
void testViscosityEvolution() {
    double const tolerance = 64.0 * std::numeric_limits<Real>::epsilon();
    Real const alphaMinimum = Real(0.1);
    Real const inverseDecayTime = Real(0.4);
    for (Real const divergence : {Real(0), Real(3)}) {
        require(morris_and_monaghan::viscosityAlphaRate(
                    alphaMinimum, alphaMinimum, inverseDecayTime, divergence) == Real(0),
            "quiescent and expanding gas remain at the viscosity floor");
        requireClose("postshock viscosity decays in both static and expanding gas",
            morris_and_monaghan::viscosityAlphaRate(
                Real(0.7), alphaMinimum, inverseDecayTime, divergence),
            -0.24, tolerance);
    }
    requireClose("compression increases viscosity even at the floor",
        morris_and_monaghan::viscosityAlphaRate(
            alphaMinimum, alphaMinimum, inverseDecayTime, Real(-3)),
        3.0, tolerance);
    requireClose("cold gas retains compression growth without a decay rate",
        morris_and_monaghan::viscosityAlphaRate(Real(4), alphaMinimum, Real(0), Real(-3)), 3.0,
        tolerance);

    // Morris & Monaghan (1997), equations (24)-(25): a constant homologous
    // compression has a steady alpha above the floor, with no imposed upper cap.
    Real const equilibriumAlpha = Real(7.6);
    requireClose("unbounded original M&M compression equilibrium",
        morris_and_monaghan::viscosityAlphaRate(
            equilibriumAlpha, alphaMinimum, inverseDecayTime, Real(-3)),
        0.0, tolerance);
    require(morris_and_monaghan::viscosityAlphaRate(
                equilibriumAlpha - Real(1), alphaMinimum, inverseDecayTime, Real(-3)) > Real(0),
        "compression grows viscosity above later schemes' usual alpha caps");
    require(morris_and_monaghan::viscosityAlphaRate(
                equilibriumAlpha + Real(1), alphaMinimum, inverseDecayTime, Real(-3)) < Real(0),
        "decay returns viscosity toward its compressive equilibrium");
}

}    // namespace detail

TEST(MorrisAndMonaghan, ViscosityEvolutionDouble) {
    detail::testViscosityEvolution<double>();
}

TEST(MorrisAndMonaghan, ViscosityEvolutionFloat) {
    detail::testViscosityEvolution<float>();
}
