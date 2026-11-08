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
    Real const αₘᵢₙ = Real(0.1);
    Real const αₘₐₓ = Real(1.5);
    Real const iτ = Real(0.4);
    Real const Δt = Real(0.5);
    auto const evolve = [αₘᵢₙ, αₘₐₓ](Real α, Real iτ, Real div_v, Real Δt) {
        return morris_and_monaghan::evolvedBulkViscosityCoefficient(α, αₘᵢₙ, αₘₐₓ, iτ, div_v, Δt);
    };
    for (Real const div_v : {Real(0), Real(3)}) {
        require(evolve(αₘᵢₙ, iτ, div_v, Δt) == αₘᵢₙ,
            "quiescent and expanding gas remain at the viscosity floor");
        requireClose(
            "backward Euler postshock decay", evolve(Real(0.7), iτ, div_v, Δt), 0.6, tolerance);
    }
    // α = αₘᵢₙ + (α⁰ − αₘᵢₙ + Δt S (αₘₐₓ − αₘᵢₙ))/(1 + Δt/τ + Δt S), S = 3.
    requireClose("compression increases viscosity even at the floor",
        evolve(αₘᵢₙ, iτ, Real(-3), Δt), 0.1 + 2.1 / 2.7, tolerance);
    requireClose("cold gas grows toward the ceiling without decay",
        evolve(Real(0.9), Real(0), Real(-3), Δt), 1.26, tolerance);
    require(evolve(Real(0.7), iτ, Real(-3), Real(0)) == Real(0.7),
        "zero interval preserves the coefficient");
    Real const stiff = evolve(Real(0.7), iτ, Real(0), Real(1e6));
    require(stiff >= αₘᵢₙ && stiff < Real(0.101),
        "stiff decay approaches the floor without overshooting");

    // Rosswog et al. (2000): constant compression has the steady state
    // α_eq = (αₘᵢₙ/τ + αₘₐₓ S)/(1/τ + S) (derived), below the ceiling.
    Real const α_eq = (αₘᵢₙ * iτ + αₘₐₓ * Real(3)) / (iτ + Real(3));
    requireClose(
        "limited compression equilibrium", evolve(α_eq, iτ, Real(-3), Δt), double(α_eq), tolerance);
    require(evolve(α_eq - Real(0.5), iτ, Real(-3), Δt) > α_eq - Real(0.5),
        "compression grows viscosity toward its equilibrium");
    require(evolve(α_eq + Real(0.1), iτ, Real(-3), Δt) < α_eq + Real(0.1),
        "decay returns viscosity toward its compressive equilibrium");
    Real const saturated = evolve(αₘᵢₙ, iτ, Real(-1e6), Real(1e3));
    require(saturated <= αₘₐₓ && saturated > Real(1.49),
        "strong compression saturates at the ceiling without overshooting");
    for (Real const α : {αₘᵢₙ, Real(0.4), Real(1.1), αₘₐₓ}) {
        for (Real const div_v : {Real(-50), Real(-0.5), Real(0), Real(2)}) {
            for (Real const step : {Real(1e-3), Real(0.5), Real(1e4)}) {
                Real const evolved = evolve(α, iτ, div_v, step);
                require(evolved >= αₘᵢₙ && evolved <= αₘₐₓ, "[αₘᵢₙ, αₘₐₓ] is invariant");
            }
        }
    }
    for (Real const div_v : {Real(-50), Real(0), Real(2)}) {
        require(morris_and_monaghan::evolvedBulkViscosityCoefficient(
                    Real(1), Real(1), Real(1), iτ, div_v, Δt) == Real(1),
            "equal bounds hold alpha fixed");
    }

    auto decayError = [αₘᵢₙ, αₘₐₓ](int steps) {
        Real α = Real(1.1);
        for (int step = 0; step < steps; ++step) {
            α = morris_and_monaghan::evolvedBulkViscosityCoefficient(
                α, αₘᵢₙ, αₘₐₓ, Real(1), Real(0), Real(1) / Real(steps));
        }
        return std::fabs(double(α) - (0.1 + std::exp(-1.0)));
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
