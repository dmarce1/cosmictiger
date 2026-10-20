/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cosmology/recombination_thermal.hpp>
#include <cosmictiger/cosmology/zero_order.hpp>

namespace detail {

bool isFiniteBackground(double value) {
    constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
    return (std::bit_cast<std::uint64_t>(value) & kExponentMask) != kExponentMask;
}

}    // namespace detail

void ZeroOrderUniverse::computeMatterFractions(double& oc, double& ob, double a) const {
    double omegaM = params.omegaB + params.omegaC;
    double omegaR = params.omegaGam + params.omegaNu;
    double denominator = omegaR / a + omegaM + a * a * a * (1.0 - omegaM - omegaR);
    ob = params.omegaB / denominator;
    oc = params.omegaC / denominator;
}

void ZeroOrderUniverse::computeRadiationFractions(double& ogam, double& onu, double a) const {
    double omegaM = params.omegaB + params.omegaC;
    double omegaR = params.omegaGam + params.omegaNu;
    double denominator = omegaR + a * omegaM + a * a * a * a * (1.0 - omegaM - omegaR);
    ogam = params.omegaGam / denominator;
    onu = params.omegaNu / denominator;
}

double ZeroOrderUniverse::conformalTimeToScaleFactor(double taumax) {
    taumax *= constants::kH0 / cosmic_constants::kH0;
    double dlogtau = 1.0e-3;
    double a = amin;
    double logtaumax = log(taumax);
    auto const hubble = [this](double a) {
        return hubbleFunction(
            a, params.hubble, params.omegaC + params.omegaB, params.omegaGam + params.omegaNu);
    };
    double logtaumin = log(1.f / (a * hubble(a)));
    int nValue = (logtaumax - logtaumin) / dlogtau + 1;
    dlogtau = (logtaumax - logtaumin) / nValue;
    for (int i = 0; i < nValue; i++) {
        double logtau = logtaumin + (double) i * dlogtau;
        double tau = exp(logtau);
        double a0 = a;
        a += tau * a * a * hubble(a) * dlogtau;
        logtau = logtaumin + (double) (i + 1) * dlogtau;
        tau = exp(logtau);
        a = 0.75f * a0 + 0.25f * (a + tau * a * a * hubble(a) * dlogtau);
        logtau = logtaumin + ((double) i + 0.5f) * dlogtau;
        tau = exp(logtau);
        a = 1.f / 3.f * a0 + 2.f / 3.f * (a + tau * a * a * hubble(a) * dlogtau);
    }
    return a;
}

double ZeroOrderUniverse::redshiftToDensity(double z) const {
    if (!detail::isFiniteBackground(z) || !(z > -1.0)) {
        throw std::invalid_argument("Matter density requires finite redshift greater than -1");
    }
    double const a = 1.0 / (1.0 + z);
    double const omegaM = params.omegaB + params.omegaC;
    double const hubbleToday = params.hubble * constants::kH0;
    return omegaM * 3.0 * hubbleToday * hubbleToday / (8.0 * M_PI * constants::kG * a * a * a);
}

double ZeroOrderUniverse::scaleFactorToConformalTime(double a) {
    double amax = a;
    double dloga = 1e-2;
    double logamin = logf(amin);
    double logamax = logf(amax);
    int nValue = (logamax - logamin) / dloga + 1;
    dloga = (logamax - logamin) / (double) nValue;
    auto const hubble = [this](double a) {
        return hubbleFunction(
            a, params.hubble, params.omegaC + params.omegaB, params.omegaGam + params.omegaNu);
    };
    double tau = 1.f / (amin * hubble(amin));
    for (int i = 0; i < nValue; i++) {
        double loga = logamin + (double) i * dloga;
        double a = exp(loga);
        double tau0 = tau;
        tau += dloga / (a * hubble(a));
        loga = logamin + (double) (i + 1) * dloga;
        a = exp(loga);
        tau = 0.75f * tau0 + 0.25f * (tau + dloga / (a * hubble(a)));
        loga = logamin + ((double) i + 0.5f) * dloga;
        a = exp(loga);
        tau = (1.f / 3.f) * tau0 + (2.f / 3.f) * (tau + dloga / (a * hubble(a)));
    }
    tau *= cosmic_constants::kH0 / constants::kH0;
    return tau;
}

double ZeroOrderUniverse::redshiftToTime(double z) const {
    double amax = 1.f / (1.f + z);
    double dloga = 1e-3;
    double logamin = log(amin);
    double logamax = log(amax);
    int nValue = (logamax - logamin) / dloga + 1;
    dloga = (logamax - logamin) / (double) nValue;
    double t = 0.0;
    auto const hubble = [this](double a) {
        return hubbleFunction(
            a, params.hubble, params.omegaC + params.omegaB, params.omegaGam + params.omegaNu);
    };
    for (int i = 0; i < nValue; i++) {
        double loga = logamin + (double) i * dloga;
        double a = exp(loga);
        double t0 = t;
        t += dloga / hubble(a);
        loga = logamin + (double) (i + 1) * dloga;
        a = exp(loga);
        t = 0.75f * t0 + 0.25f * (t + dloga / hubble(a));
        loga = logamin + ((double) i + 0.5f) * dloga;
        a = exp(loga);
        t = (1.f / 3.f) * t0 + (2.f / 3.f) * (t + dloga / hubble(a));
    }
    t *= cosmic_constants::kH0 / constants::kH0;
    return t;
}

void createZeroOrderUniverse(ZeroOrderUniverse* uniPtr, std::function<double(double)> fxe,
    double amax, CosmicParameters cpars, BackgroundThermalHistory thermalHistory) {
    if (uniPtr == nullptr || !fxe || !detail::isFiniteBackground(amax) ||
        !detail::isFiniteBackground(cpars.omegaB) || !(cpars.omegaB > 0.0) ||
        !detail::isFiniteBackground(cpars.omegaC) || cpars.omegaC < 0.0 ||
        !detail::isFiniteBackground(cpars.omegaGam) || !(cpars.omegaGam > 0.0) ||
        !detail::isFiniteBackground(cpars.omegaNu) || cpars.omegaNu < 0.0 ||
        !detail::isFiniteBackground(cpars.neff) || cpars.neff < 0.0 ||
        !detail::isFiniteBackground(cpars.y) || cpars.y < 0.0 || !(cpars.y < 1.0) ||
        !detail::isFiniteBackground(cpars.theta) || !(cpars.theta > 0.0) ||
        !detail::isFiniteBackground(cpars.hubble) || !(cpars.hubble > 0.0)) {
        throw std::invalid_argument("Invalid homogeneous cosmology parameters");
    }
    if (bool(thermalHistory.hydrogenNumberDensity) != bool(thermalHistory.pressurePerHydrogen) ||
        (thermalHistory.hydrogenNumberDensity && !thermalHistory.temperature)) {
        throw std::invalid_argument("Incomplete homogeneous chemistry thermodynamics");
    }
    ZeroOrderUniverse& uni = *uniPtr;
    uni.xe = fxe;
    using namespace constants;
    uni.params = cpars;
    double omegaB = cpars.omegaB;
    double omegaC = cpars.omegaC;
    double omegaGam = cpars.omegaGam;
    double omegaNu = cpars.omegaNu;
    double omegaM = cpars.omegaB + omegaC;
    double omegaR = omegaGam + omegaNu;
    double theta = cpars.theta;
    double littleh = cpars.hubble;
    double neff = cpars.neff;
    double y = cpars.y;
    double amin = theta * kTcmb / (0.07 * 1e6 * kEvtoK);
    if (!(amax > amin) ||
        !(omegaR + omegaM * amax + std::pow(amax, 4) * (1.0 - omegaM - omegaR) > 0.0)) {
        throw std::invalid_argument("Invalid homogeneous cosmology scale-factor interval");
    }
    double logamin = log(amin);
    double logamax = log(amax);
    int nValue = 4 * 1024;
    double dloga = (logamax - logamin) / nValue;
    std::vector<double> k(nValue + 1);
    std::vector<double> thomson(nValue + 1);
    std::vector<double> soundSpeed2(nValue + 1);

    PRINT("\t\tParameters:\n");
    PRINT("\t\t\t h                 = %f\n", littleh);
    PRINT("\t\t\t omega_m           = %f\n", omegaM);
    PRINT("\t\t\t omega_r           = %f\n", omegaR);
    PRINT("\t\t\t omega_lambda      = %f\n", 1 - omegaR - omegaM);
    PRINT("\t\t\t omega_b           = %f\n", omegaB);
    PRINT("\t\t\t omega_c           = %f\n", omegaC);
    PRINT("\t\t\t omega_gam         = %f\n", omegaGam);
    PRINT("\t\t\t omega_nu          = %f\n", omegaNu);
    PRINT("\t\t\t Neff              = %f\n", neff);
    PRINT("\t\t\t temperature today = %f\n\n", 2.73 * theta);

    dloga = (logamax - logamin) / nValue;
    auto const cosmicHubble = [=](double a) {
        using namespace cosmic_constants;
        return littleh * cosmic_constants::kH0 *
            sqrt(omegaR / (a * a * a * a) + omegaM / (a * a * a) + (1 - omegaR - omegaM));
    };
    auto cgsHubble = [=](double a) {
        return constants::kH0 / cosmic_constants::kH0 * cosmicHubble(a);
    };

    auto const rhoBaryon = [=](double a) {
        using namespace constants;
        //		PRINT( "%e\n", omega_b);
        return 3.0 * pow(littleh * kH0, 2) / (8.0 * M_PI * kG) * omegaB / (a * a * a);
    };

    auto const tRadiation = [=](double a) {
        using namespace constants;
        return kTcmb * theta / a;
    };

    double loga;
    double a;

    double rhoB, nH, nHe, ne, tgas, trad;
    double hubble = cgsHubble(amin);

    rhoB = rhoBaryon(amin);
    double nnuc = rhoB / kMh;
    nHe = y * nnuc / 4;
    nH = (1 - y) * nnuc;
    auto const checkedPositive = [](double value) {
        if (!detail::isFiniteBackground(value) || !(value > 0.0)) {
            throw std::invalid_argument("Invalid homogeneous chemistry thermodynamics");
        }
        return value;
    };
    if (thermalHistory.hydrogenNumberDensity) {
        nH = checkedPositive(thermalHistory.hydrogenNumberDensity(amin));
    }
    ne = fxe(amin) * nH;
    trad = tRadiation(amin);
    tgas = thermalHistory.temperature ? checkedPositive(thermalHistory.temperature(amin)) : trad;
    double n = nH + nHe;
    double p = thermalHistory.pressurePerHydrogen ?
        nH * checkedPositive(thermalHistory.pressurePerHydrogen(amin)) :
        kB * (n + ne) * tgas;
    double t = 0.0;
    double dt;
    for (int i = -10 / dloga; i <= 0; i++) {
        loga = logamin + i * dloga;
        dt = dloga / cgsHubble(exp(loga));
        t += dt;
    }
    a = exp(logamin);
    PRINT("%e \n", p);
    //	print_time(t);
    //	PRINT(
    //			", redshift %.0f: Big Bang nucleosynthesis has ended. The Universe is dominated by
    // radiation at a temperature of %8.2e K." 					" \n   Its total matter density is
    // %.1f \% times the density of air at sea level.\n", 1 / a - 1, 			Trad, 100 * rho_b *
    // omega_m / omega_b / 1.274e-3);
    double sigmaC =
        recombination::comptonTemperatureCoupling(a, omegaGam, omegaB, nH, nHe, ne, hubble);
    double sigmaT = kC * kSigmaT * ne / hubble;
    thomson[0] = sigmaT;
    double p1, p2;
    double rho1, rho2;
    double cs2;
    p1 = p2 = p;
    rho1 = rho2 = rhoB;
    k[0] = p / std::pow(rhoB, 5.0 / 3.0);
    double a1;
    for (int i = 1; i <= nValue; i++) {
        loga = logamin + i * dloga;
        a = exp(loga);
        a1 = exp(loga + dloga);
        //		PRINT("%e %e %e %e %e %e\n", a, nH, nHp, nHe, nHep, nHepp);
        p2 = p1;
        p1 = p;
        rho2 = rho1;
        rho1 = rhoB;
        hubble = cgsHubble(a);
        nH /= rhoB;
        nHe /= rhoB;
        ne /= rhoB;
        rhoB = rhoBaryon(a);
        nH *= rhoB;
        nHe *= rhoB;
        ne *= rhoB;
        if (thermalHistory.hydrogenNumberDensity) {
            nH = checkedPositive(thermalHistory.hydrogenNumberDensity(a));
        }
        trad = tRadiation(a);
        double dt = dloga / hubble;
        ne = fxe(a) * nH;
        sigmaC =
            recombination::comptonTemperatureCoupling(a, omegaGam, omegaB, nH, nHe, ne, hubble);
        // Evaluate both the radiation source and the Compton rate at the
        // implicit stage times. Freezing them at the endpoint is only first
        // order through thermal decoupling.
        double const gamma = 1.0 - 1.0 / std::sqrt(2.0);
        auto const thermalDerivative = [&](double stageScale, double stageTemperature) {
            double const stageDensity = rhoBaryon(stageScale);
            double const stageHydrogen = (1.0 - y) * stageDensity / kMh;
            double const stageHelium = y * stageDensity / (4.0 * kMh);
            double const stageElectrons = fxe(stageScale) * stageHydrogen;
            double const coupling = recombination::comptonTemperatureCoupling(stageScale, omegaGam,
                omegaB, stageHydrogen, stageHelium, stageElectrons, cgsHubble(stageScale));
            return (coupling * (tRadiation(stageScale) - stageTemperature) -
                       2.0 * stageTemperature) /
                (1.0 + gamma * dloga * (2.0 + coupling));
        };
        if (thermalHistory.temperature) {
            // A coupled chemistry history already contains the matter
            // temperature. Use it consistently with its electron fraction.
            tgas = checkedPositive(thermalHistory.temperature(a));
        } else {
            double const firstDerivative =
                thermalDerivative(std::exp(loga - (1.0 - gamma) * dloga), tgas);
            double const secondDerivative = thermalDerivative(std::exp(loga - gamma * dloga),
                tgas + (1.0 - 2.0 * gamma) * dloga * firstDerivative);
            tgas += 0.5 * dloga * (firstDerivative + secondDerivative);
        }
        ne = fxe(a) * nH;
        n = nH + nHe;
        p = thermalHistory.pressurePerHydrogen ?
            nH * checkedPositive(thermalHistory.pressurePerHydrogen(a)) :
            kB * (n + ne) * tgas;
        sigmaT = kC * kSigmaT * ne / hubble;
        t += dt;
        if (i == 1) {
            cs2 = (p - p1) / (rhoB - rho1);
        } else {
            cs2 = (p - p2) / (rhoB - rho2);
        }
        soundSpeed2[i - 1] = cs2 / (kC * kC);
        thomson[i] = sigmaT;
        k[i] = p / std::pow(rhoB, 5.0 / 3.0);

        //	printf("%e %e %e %e\n", n, nH, nHe, ne);
        if (1.0 / a - 1.0 < 50.0) {
            //		break;
        }
    }
    cs2 = (p - p1) / (rhoB - rho1);
    // Complete the final endpoint with a one-sided derivative in units of c^2;
    // the loop already supplied the centered derivative at nValue - 1.
    soundSpeed2[nValue] = cs2 / (kC * kC);
    //	print_time(t);
    uni.amin = amin;
    uni.amax = amax;
    buildInterpolationFunction(&uni.k, k, (double) amin, (double) amax);
    buildInterpolationFunction(&uni.sigmaT, thomson, (double) amin, (double) amax);
    buildInterpolationFunction(&uni.cs2, soundSpeed2, (double) amin, (double) amax);
    //	uni.hubble = std::move(cosmic_hubble);
}
