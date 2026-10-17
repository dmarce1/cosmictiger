/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

/*
 * Classical SPH formulae from Monaghan, ARA&A 30 (1992), 543-574.
 * The stored smoothing length is the compact support radius H = 2 h_paper.
 */
#pragma once

#ifdef __CUDACC__
#define COSMICTIGER_MONAGHAN_EXPORT __host__ __device__
#else
#define COSMICTIGER_MONAGHAN_EXPORT
#endif

namespace monaghan_sph {

// Section 7, pp. 554-555, normalized in three dimensions with q = r/H.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real kernelValue(Real q) {
    constexpr Real kNormalization = Real(8.0 / 3.141592653589793238462643383279502884);
    if (q >= Real(1)) {
        return Real(0);
    }
    if (q < Real(0.5)) {
        return kNormalization * (Real(1) + q * q * (Real(6) * q - Real(6)));
    }
    Real const oneMinusQ = Real(1) - q;
    return Real(2) * kNormalization * oneMinusQ * oneMinusQ * oneMinusQ;
}

template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real kernelDerivative(Real q) {
    constexpr Real kNormalization = Real(8.0 / 3.141592653589793238462643383279502884);
    if (q >= Real(1)) {
        return Real(0);
    }
    if (q < Real(0.5)) {
        return kNormalization * q * (Real(18) * q - Real(12));
    }
    Real const oneMinusQ = Real(1) - q;
    return -Real(6) * kNormalization * oneMinusQ * oneMinusQ;
}

// Equation (4.2). epsilon is eta^2 / hbar^2. The explicit scale factor
// converts stored support radii to the calibrated viscosity radius; its default retains
// the historical cubic-spline convention for standalone Monaghan references.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real viscosityMu(Real radiusSquared, Real velocityDotRadius,
    Real supportI, Real supportJ, Real epsilon, Real supportRadiusPerEffectiveRadius = Real(2),
    Real viscosityWidthScale = Real(1)) {
    if (velocityDotRadius >= Real(0)) {
        return Real(0);
    }
    Real const viscosityRadius =
        Real(0.5) * (supportI + supportJ) / supportRadiusPerEffectiveRadius * viscosityWidthScale;
    return viscosityRadius * velocityDotRadius /
        (radiusSquared + epsilon * viscosityRadius * viscosityRadius);
}

template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real artificialViscosity(
    Real rhoI, Real rhoJ, Real soundI, Real soundJ, Real mu, Real alpha, Real beta) {
    if (mu >= Real(0)) {
        return Real(0);
    }
    Real const meanSound = Real(0.5) * (soundI + soundJ);
    Real const meanDensity = Real(0.5) * (rhoI + rhoJ);
    return (-alpha * meanSound * mu + beta * mu * mu) / meanDensity;
}

// Equation (4.1), using the same symmetric gradient for force and energy.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real pressureCoefficient(
    Real rhoI, Real rhoJ, Real pressureI, Real pressureJ) {
    return pressureI / (rhoI * rhoI) + pressureJ / (rhoJ * rhoJ);
}

// Equation (6.5). Separate peculiar and total velocities allow the caller to
// integrate homogeneous cosmological pressure work analytically while retaining
// artificial-viscosity heating from physical pair convergence.
template <typename Real>
COSMICTIGER_MONAGHAN_EXPORT inline Real pairEnergyRate(Real pressurePair, Real viscosity,
    Real peculiarVelocityDotGradient, Real physicalVelocityDotGradient) {
    return Real(0.5) *
        (pressurePair * peculiarVelocityDotGradient + viscosity * physicalVelocityDotGradient);
}

}    // namespace monaghan_sph

#undef COSMICTIGER_MONAGHAN_EXPORT
