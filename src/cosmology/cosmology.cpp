/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <stdexcept>

#include <cosmictiger/analysis/rockstar_properties.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/friedmann.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simulation_units.hpp>

namespace detail {

cosmology::FriedmannCosmology backgroundCosmology() {
    auto const& opts = getOptions();
    return cosmology::FriedmannCosmology(
        {units::normalizedHubble100(opts) * opts.cosmology.hubble, opts.cosmology.omegaR,
            opts.cosmology.omegaM, opts.cosmology.omegaLambda, opts.cosmology.omegaK});
}

}    // namespace detail

CosmosAdvance cosmosAdvance(double& adot, double& a, double dt) {
    auto const result = detail::backgroundCosmology().advanceWithIntegrals(
        {a, adot, a * adot}, dt, cosmology::CosmologicalTime::conformal);
    a = result.state.scaleFactor;
    adot = result.state.physicalDerivative;
    addCpuFlops(result.flops + 3);
    return {result.physicalTime, result.driftIntegral};
}

void cosmosUpdate(double& adotdot, double& adot, double& a, double dt) {
    cosmosAdvance(adot, a, dt);
    adotdot = cosmosD2adt2(a);
}

double cosmosAinv(double& adot, double& a, double dt) {
    auto const result = cosmosAdvance(adot, a, dt);
    addCpuFlops(1);
    return dt == 0.0 ? 1.0 / a : result.driftIntegral / dt;
}

double cosmosGrowthFactor(double omegaM, float a) {
    double const omegaL = 1.f - omegaM;
    double const a3 = a * sqr(a);
    double const deninv = 1.f / (omegaM + a3 * omegaL);
    double const om = omegaM * deninv;
    double const ol = a3 * omegaL * deninv;
    return a * 2.5 * om / (pow(om, 4.f / 7.f) - ol + (1.f + 0.5f * om) * (1.f + 0.014285714f * ol));
}

double cosmosDadt(double a) {
    auto const result = detail::backgroundCosmology().state(a).physicalDerivative;
    addCpuFlops(22);
    return result;
}

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
    addCpuFlops(14);
    return result;
}

double cosmosDadtau(double a, bool countFlops) {
    double const result = detail::backgroundCosmology().state(a).conformalDerivative;
    if (countFlops) {
        addCpuFlops(22);
    }
    return result;
}

double cosmosTauToScale(double a0, double elapsed) {
    // Initial state is on the expanding branch. Evolution retains its sign
    // through a turnaround; callers needing repeated samples use a trajectory.
    return detail::backgroundCosmology()
        .evolve(a0, elapsed, cosmology::CosmologicalTime::conformal)
        .scaleFactor;
}

double cosmosTime(double a0, double a1) {
    return detail::backgroundCosmology().intervalsBetweenScales(a0, a1).physicalTime;
}

double cosmosConformalTime(double a0, double a1, bool countFlops) {
    auto const result = detail::backgroundCosmology().intervalsBetweenScales(a0, a1);
    if (countFlops) {
        addCpuFlops(result.flops + 2);
    }
    return result.conformalTime;
}

double cosmosKlypinFit(double vmax, double rvir, double mvir) {
    return rockstar::klypinScaleRadius(vmax, rvir, mvir * getOptions().gravity.gm);
}

void cosmosNfwFit(Vector<double> radii, double& rs, double& rhoNfw) {
    auto const fit = rockstar::nfwFit(radii);
    rs = fit.scaleRadius;
    rhoNfw = fit.density;
}
