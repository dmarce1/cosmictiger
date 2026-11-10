/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <stdexcept>

#if COSMICTIGER_ENABLE_ROCKSTAR
#include <cosmictiger/analysis/rockstar_properties.hpp>
#endif
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/friedmann.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simulation_units.hpp>

namespace detail {

/// Construct a Friedmann model from the resolved cosmology and simulation unit scales.
cosmology::FriedmannCosmology backgroundCosmology() {
    auto const& opts = getOptions();
    cosmology::BackgroundCosmologyParameters parameters;
    parameters.hubbleConstant = units::normalizedHubble100(opts) * opts.cosmology.hubble;
    parameters.omegaRadiation = opts.cosmology.omegaR;
    parameters.omegaMatter = opts.cosmology.omegaM;
    parameters.omegaDarkEnergy = opts.cosmology.omegaLambda;
    parameters.omegaCurvature = opts.cosmology.omegaK;
    return cosmology::FriedmannCosmology(parameters);
}

}    // namespace detail

/// Advance scale and its physical-time derivative over a conformal interval and return
/// integrated time factors.
CosmosAdvance cosmosAdvance(double& adot, double& a, double dt) {
    auto const result = detail::backgroundCosmology().advanceWithIntegrals(
        {a, adot, a * adot}, dt, cosmology::CosmologicalTime::conformal);
    a = result.state.scaleFactor;
    adot = result.state.physicalDerivative;
    addCpuFlops(result.flops + 3, FlopPrecision::Double);
    return {result.physicalTime, result.driftIntegral};
}

/// Advance the background and refresh its physical-time acceleration.
void cosmosUpdate(double& adotdot, double& adot, double& a, double dt) {
    cosmosAdvance(adot, a, dt);
    adotdot = cosmosD2adt2(a);
}

/// Advance the background and return the interval-averaged inverse scale factor.
double cosmosAinv(double& adot, double& a, double dt) {
    auto const result = cosmosAdvance(adot, a, dt);
    addCpuFlops(1, FlopPrecision::Double);
    return dt == 0.0 ? 1.0 / a : result.driftIntegral / dt;
}

/// Evaluate the approximate linear growth factor for a flat matter-plus-Lambda background.
double cosmosGrowthFactor(double omegaM, float a) {
    double const omegaL = 1.f - omegaM;
    double const a3 = a * sqr(a);
    double const deninv = 1.f / (omegaM + a3 * omegaL);
    double const om = omegaM * deninv;
    double const ol = a3 * omegaL * deninv;
    return a * 2.5 * om / (pow(om, 4.f / 7.f) - ol + (1.f + 0.5f * om) * (1.f + 0.014285714f * ol));
}

/// Return the expanding-branch physical-time derivative of the scale factor.
double cosmosDadt(double a) {
    auto const result = detail::backgroundCosmology().state(a).physicalDerivative;
    addCpuFlops(22, FlopPrecision::Double);
    return result;
}

/// Evaluate the physical-time Friedmann acceleration, validating scale and numerical range.
double cosmosD2adt2(double a) {
    if (!(a > 0.0) || !std::isfinite(a)) {
        throw std::domain_error("Friedmann scale factor must be positive and finite");
    }
    auto const& opts = getOptions();
    double const h = units::normalizedHubble100(opts) * opts.cosmology.hubble;
    double const result = a * h * h *
        (-opts.cosmology.omegaR / (a * a * a * a) - 0.5 * opts.cosmology.omegaM / (a * a * a) +
            opts.cosmology.omegaLambda);
    if (!std::isfinite(result)) {
        throw std::overflow_error("Friedmann acceleration is outside the double-precision range");
    }
    addCpuFlops(14, FlopPrecision::Double);
    return result;
}

/// Return the expanding-branch conformal-time derivative, optionally recording arithmetic
/// cost.
double cosmosDadtau(double a, bool countFlops) {
    double const result = detail::backgroundCosmology().state(a).conformalDerivative;
    if (countFlops) {
        addCpuFlops(22, FlopPrecision::Double);
    }
    return result;
}

/// Evolve an initially expanding background by a conformal interval, retaining turnaround
/// dynamics.
double cosmosTauToScale(double a0, double elapsed) {
    // Initial state is on the expanding branch. Evolution retains its sign
    // through a turnaround; callers needing repeated samples use a trajectory.
    return detail::backgroundCosmology()
        .evolve(a0, elapsed, cosmology::CosmologicalTime::conformal)
        .scaleFactor;
}

/// Integrate the physical-time interval between two scale factors.
double cosmosTime(double a0, double a1) {
    return detail::backgroundCosmology().intervalsBetweenScales(a0, a1).physicalTime;
}

/// Integrate the conformal-time interval between scale factors, optionally recording
/// arithmetic cost.
double cosmosConformalTime(double a0, double a1, bool countFlops) {
    auto const result = detail::backgroundCosmology().intervalsBetweenScales(a0, a1);
    if (countFlops) {
        addCpuFlops(result.flops + 2, FlopPrecision::Double);
    }
    return result.conformalTime;
}

#if COSMICTIGER_ENABLE_ROCKSTAR
/// Estimate the halo scale radius from maximum circular speed, virial radius, and mass.
double cosmosKlypinFit(double vmax, double rvir, double mvir) {
    return rockstar::klypinScaleRadius(vmax, rvir, mvir * getOptions().gravity.gm);
}

/// Fit an NFW profile to the supplied radii and return its scale radius and density.
void cosmosNfwFit(Vector<double> radii, double& rs, double& rhoNfw) {
    auto const fit = rockstar::nfwFit(radii);
    rs = fit.scaleRadius;
    rhoNfw = fit.density;
}
#endif
