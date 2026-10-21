/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>

#include <cosmictiger/analysis/rockstar_properties.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simulation_units.hpp>

namespace detail {

// Include the multiplication inside normalizedHubble100, as well as the
// explicit Friedmann expressions. Batch the repeated integration stages.
constexpr FlopCount kDadtFlops = 17;
constexpr FlopCount kUpdateStageFlops = 76;

}    // namespace detail

void cosmosUpdate0(double& adotdot, double& adot, double& a, double dt) {
    double const omegaM = getOptions().cosmology.omegaM;
    double const omegaR = getOptions().cosmology.omegaR;
    double const omegaLam = getOptions().cosmology.omegaLambda;
    double const omegaK = getOptions().cosmology.omegaK;
    auto const h = units::normalizedHubble100(getOptions()) * getOptions().cosmology.hubble;
    double const adot0 = adot;

    adotdot = a * sqr(h) * (-omegaR / (sqr(sqr(a))) - 0.5 * omegaM / (sqr(a) * a) + omegaLam);
    double const da1 = adot * a * dt;
    double const dadot1 = adotdot * a * dt;
    a += da1;
    adot += dadot1;

    adotdot = a * sqr(h) * (-omegaR / (sqr(sqr(a))) - 0.5 * omegaM / (sqr(a) * a) + omegaLam);
    double const da2 = adot * a * dt;
    double const dadot2 = adotdot * a * dt;
    a += -da1 + 0.25 * (da1 + da2);
    adot += -dadot1 + 0.25 * (dadot1 + dadot2);

    adotdot = a * sqr(h) * (-omegaR / (sqr(sqr(a))) - 0.5 * omegaM / (sqr(a) * a) + omegaLam);
    double const da3 = adot * a * dt;
    double const dadot3 = adotdot * a * dt;
    a += -0.25 * (da1 + da2) + (1.0 / 6.0) * (da1 + da2 + 4.0 * da3);
    adot += -0.25 * (dadot1 + dadot2) + (1.0 / 6.0) * (dadot1 + dadot2 + 4.0 * dadot3);
}

void cosmosUpdate(double& adotdot, double& adot, double& a, double dt0) {
    constexpr int kN = 128;
    double const dt = dt0 / kN;
    for (int i = 0; i < kN; i++) {
        cosmosUpdate0(adotdot, adot, a, dt);
    }
    addCpuFlops(1 + FlopCount(kN) * detail::kUpdateStageFlops);
}

double cosmosAinv(double& adot, double& a, double dt0) {
    double ainv = 0.0;
    ;
    constexpr int kN = 128;
    double const dt = dt0 / kN;
    double adotdot;
    for (int i = 0; i < kN; i++) {
        double a0 = a;
        cosmosUpdate0(adotdot, adot, a, dt);
        double a1 = a;
        double thisAinv = 0.5 / a0 + 0.5 / a1;
        ainv += thisAinv / kN;
    }
    addCpuFlops(1 + FlopCount(kN) * (detail::kUpdateStageFlops + 5));
    return ainv;
}

double cosmosGrowthFactor(double omegaM, float a) {
    double const omegaL = 1.f - omegaM;
    double const a3 = a * sqr(a);
    double const deninv = 1.f / (omegaM + a3 * omegaL);
    double const om = omegaM * deninv;
    double const ol = a3 * omegaL * deninv;
    return a * 2.5 * om / (pow(om, 4.f / 7.f) - ol + (1.f + 0.5f * om) * (1.f + 0.014285714f * ol));
}

namespace detail {

double cosmosDadtUncounted(double a) {
    auto const h = units::normalizedHubble100(getOptions()) * getOptions().cosmology.hubble;
    auto const omegaM = getOptions().cosmology.omegaM;
    auto const omegaR = getOptions().cosmology.omegaR;
    auto const omegaK = getOptions().cosmology.omegaK;
    auto const omegaLam = getOptions().cosmology.omegaLambda;
    return h * a *
        std::sqrt(omegaR / (a * a * a * a) + omegaM / (a * a * a) + omegaK / (a * a) + omegaLam);
}

}    // namespace detail

double cosmosDadt(double a) {
    addCpuFlops(detail::kDadtFlops);
    return detail::cosmosDadtUncounted(a);
}

double cosmosDadtau(double a, bool countFlops) {
    if (countFlops) {
        addCpuFlops(detail::kDadtFlops + 1);
    }
    return a * detail::cosmosDadtUncounted(a);
}

double cosmosTauToScale(double a0, double t1) {
    constexpr int kN = 1024;
    double a = a0;
    double dt = t1 / kN;
    a += a * detail::cosmosDadtUncounted(a) * 0.5 * dt;
    for (int i = 1; i < kN; i++) {
        a += a * detail::cosmosDadtUncounted(a) * dt;
    }
    a += a * detail::cosmosDadtUncounted(a) * 0.5 * dt;
    // Used only to reconstruct halo lightcone metadata, not to advance forces.
    return a;
}

double cosmosTime(double a0, double a1) {
    double a = a0;
    double t = 0.0;
    int const nValue = 1024 * 1024;
    double const loga0 = log(a0);
    double const loga1 = log(a1);
    double const dloga = (loga1 - loga0) / nValue;
    double loga = loga0;
    a = exp(loga);
    double dadt = detail::cosmosDadtUncounted(a);
    for (int i = 0; i < nValue; i++) {
        t += 0.5 * a / dadt * dloga;
        loga = loga0 + dloga * (i + 1);
        a = exp(loga);
        dadt = detail::cosmosDadtUncounted(a);
        t += 0.5 * a / dadt * dloga;
    }
    // Physical ages are output metadata; exclude these diagnostic integrals.
    return t;
}

double cosmosConformalTime(double a0, double a1, bool countFlops) {
    int nValue = 2;
    double t = 0.0;
    double tlast;
    do {
        tlast = t;
        t = 0.0;
        double a = a0;
        double const loga0 = log(a0);
        double const loga1 = log(a1);
        double const dloga = (loga1 - loga0) / nValue;
        double loga = loga0;
        a = exp(loga);
        double dadt = a * detail::cosmosDadtUncounted(a);
        for (int i = 0; i < nValue; i++) {
            t += 0.5 * a / dadt * dloga;
            loga = loga0 + dloga * (i + 1);
            a = exp(loga);
            dadt = a * detail::cosmosDadtUncounted(a);
            t += 0.5 * a / dadt * dloga;
        }
        // Five setup operations and two termination-test operations per
        // refinement; each trapezoid has eleven plus its scale derivative.
        if (countFlops) {
            addCpuFlops(
                7 + detail::kDadtFlops + 1 + FlopCount(nValue) * (11 + detail::kDadtFlops + 1));
        }
        nValue *= 2;
    } while (fabs(tlast - t) > 1e-9 * t);
    return t;
}

double cosmosKlypinFit(double vmax, double rvir, double mvir) {
    return rockstar::klypinScaleRadius(vmax, rvir, mvir * getOptions().gravity.gm);
}

void cosmosNfwFit(Vector<double> radii, double& rs, double& rhoNfw) {
    auto const fit = rockstar::nfwFit(radii);
    rs = fit.scaleRadius;
    rhoNfw = fit.density;
}
