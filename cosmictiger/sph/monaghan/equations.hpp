/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

/*
 * Classical SPH formulae from
 *
 *   J. J. Monaghan, "Smoothed Particle Hydrodynamics",
 *   Annual Review of Astronomy and Astrophysics 30 (1992), 543-574.
 *   doi:10.1146/annurev.aa.30.090192.002551, ADS bibcode 1992ARA&A..30..543M,
 *   https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M
 *
 * Equation numbers and pages below refer to that review. Function bodies use its
 * symbols: particle a is the one being updated and particle b a neighbor; m mass,
 * ρ density, P pressure, u specific internal energy, c sound speed, h smoothing
 * length, r_ab = r_a − r_b, v_ab = v_a − v_b, W_ab = W(r_ab, h) the kernel,
 * W̃_ab the symmetric kernel of p. 554, Π_ab the artificial viscosity and μ_ab its
 * velocity measure, with the constants α, β and η. Function parameters keep
 * descriptive names. The stored smoothing length is the compact support radius H,
 * which equals 2h for the paper's cubic spline of section 7.
 */
#pragma once

#include <cosmictiger/math/simd_math.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_MONAGHAN_EXPORT __host__ __device__
#else
#define COSMICTIGER_MONAGHAN_EXPORT
#endif

namespace monaghan {

// NOLINTBEGIN(readability-identifier-naming)

// The symmetric pressure term P_b/ρ_b² + P_a/ρ_a² shared by the momentum equation 3.3,
// its viscous form 4.1, and the variable-resolution pair equations 6.4-6.5. The same
// factor in the force and the thermal work conserves momentum and energy (p. 549).
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real pressureCoefficient(
    Real rhoI, Real rhoJ, Real pressureI, Real pressureJ) {
    Real const ρ_a = rhoI;
    Real const ρ_b = rhoJ;
    Real const P_a = pressureI;
    Real const P_b = pressureJ;
    return P_a / (ρ_a * ρ_a) + P_b / (ρ_b * ρ_b);
}

// Equation 6.5, p. 554 (equation 3.17 with Π_ab and the symmetric kernel):
//
//   du_a/dt = ½ Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) v_ab·∇_a W̃_ab.
//
// This returns one pair's term without m_b. The velocity factor is split so that the
// pressure work uses the peculiar relative velocity while the viscous heating uses the
// physical one, which lets the caller integrate homogeneous cosmological pressure work
// analytically (docs/monaghan_sph.md); with equal arguments it is the paper's form.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real pairEnergyRate(Real pressurePair, Real viscosity,
    Real peculiarVelocityDotGradient, Real physicalVelocityDotGradient) {
    Real const P_over_ρ2_ab = pressurePair;    // P_b/ρ_b² + P_a/ρ_a²
    Real const Π_ab = viscosity;
    Real const v_ab_dot_gradW̃_ab = peculiarVelocityDotGradient;
    Real const v_ab_dot_gradW̃_ab_physical = physicalVelocityDotGradient;
    return simdBroadcast<Real>(0.5) *
        (P_over_ρ2_ab * v_ab_dot_gradW̃_ab + Π_ab * v_ab_dot_gradW̃_ab_physical);
}

// NOLINTEND(readability-identifier-naming)

}    // namespace monaghan

#undef COSMICTIGER_MONAGHAN_EXPORT
