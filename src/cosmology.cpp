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

#include <cmath>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>

void cosmosUpdate0(double& adotdot, double& adot, double& a, double dt) {
    double const omegaM = getOptions().omegaM;
    double const omegaR = getOptions().omegaR;
    double const omegaLam = getOptions().omegaLam;
    double const omegaK = getOptions().omegaK;
    auto const h = constants::kH0 * getOptions().codeToS * getOptions().hubble;
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

double cosmosDadt(double a) {
    auto const h = constants::kH0 * getOptions().codeToS * getOptions().hubble;
    auto const omegaM = getOptions().omegaM;
    auto const omegaR = getOptions().omegaR;
    auto const omegaK = getOptions().omegaK;
    auto const omegaLam = getOptions().omegaLam;
    addCpuFlops(27);
    return h * a *
        std::sqrt(omegaR / (a * a * a * a) + omegaM / (a * a * a) + omegaK / (a * a) + omegaLam);
}

double cosmosDadtau(double a) {
    return a * cosmosDadt(a);
}

double cosmosTauToScale(double a0, double t1) {
    constexpr int kN = 1024;
    double a = a0;
    double dt = t1 / kN;
    a += a * cosmosDadt(a) * 0.5 * dt;
    for (int i = 1; i < kN; i++) {
        a += a * cosmosDadt(a) * dt;
    }
    a += a * cosmosDadt(a) * 0.5 * dt;
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
    double dadt = cosmosDadt(a);
    for (int i = 0; i < nValue; i++) {
        t += 0.5 * a / dadt * dloga;
        loga = loga0 + dloga * (i + 1);
        a = exp(loga);
        dadt = cosmosDadt(a);
        t += 0.5 * a / dadt * dloga;
    }
    return t;
}

double cosmosConformalTime(double a0, double a1) {
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
        double dadt = cosmosDadtau(a);
        for (int i = 0; i < nValue; i++) {
            t += 0.5 * a / dadt * dloga;
            loga = loga0 + dloga * (i + 1);
            a = exp(loga);
            dadt = cosmosDadtau(a);
            t += 0.5 * a / dadt * dloga;
        }
        nValue *= 2;
    } while (fabs(tlast - t) > 1e-9 * t);
    return t;
}

#define Power(a, b) pow(a, b)
#define Log(a) log(a)

double cosmosKlypinFit(double vmax, double rvir, double mvir) {
    double f0 = vmax * vmax * rvir / mvir / getOptions().gm;
    double c;
    if (fabs(f0 - 1.0) < 1e-5) {
        c = 2.16258;
    } else {
        f0 *= 4.62499;
        double err;
        double cinvMax = 1.0 / 2.16258;
        double cinvMin = 0.0;
        auto const func = [f0](double c) { return c / (log(1 + c) - c / (1 + c)) - f0; };
        do {
            double cinvMid = (cinvMax + cinvMin) * 0.5;
            double fMid = func(1.0 / cinvMid);
            double fMax = func(1.0 / cinvMax);
            if (fMid * fMax < 0.0) {
                cinvMin = cinvMid;
            } else {
                cinvMax = cinvMid;
            }
            err = fabs(fMid);
            c = 1.0 / cinvMid;
        } while (err > 1.0e-6);
    }
    return rvir / c;
}

void cosmosNfwFit(Vector<double> radii, double& rs, double& rhoNfw) {
    double err;
    double rmax = radii.back();
    double npart = radii.size();
    int nbin = std::max(1, std::min(50, (int) npart / 15));
    Vector<double> rbnd(nbin + 1);
    rbnd[0] = 0.0;
    for (int n = 1; n < nbin; n++) {
        double j = (double) n / nbin * npart;
        double k = j - (int) j;
        double r0 = radii[j] * (1.0 - k) + radii[j + 1] * k;
        rbnd[n] = r0;
    }
    rbnd[nbin] = rmax;
    double rho0 = npart / (4.0 / 3.0 * M_PI * sqr(rmax) * rmax);

    double cinvMin = 0.0;
    double cinvMax = 1.0;
    double c;

    do {
        double cinvMid = 0.5 * (cinvMax + cinvMin);
        double cmin = 1.0 / cinvMid;
        double cmax = 1.0 / cinvMax;
        double fmin = 0.0;
        double fmax = 0.0;
        for (int i = 0; i < nbin; i++) {
            double dr = (rbnd[i + 1] - rbnd[i]);
            if (dr > 0.0) {
                double vol =
                    4.0 / 3.0 * M_PI * (sqr(rbnd[i + 1]) * rbnd[i + 1] - sqr(rbnd[i]) * rbnd[i]);
                double rho = npart / nbin / vol;
                double x = 0.5 * (rbnd[i] + rbnd[i + 1]) / rmax;
                c = cmin;
                fmin += (-2 * c * rho0 *
                            (c * (2 + c * (3 + c * x)) - 2 * Power(1 + c, 2) * Log(1 + c)) *
                            (c *
                                    (c * (1 + c) * rho0 + 3 * rho * x +
                                        3 * c * rho * Power(x, 2) * (2 + c * x)) -
                                3 * (1 + c) * rho * x * Power(1 + c * x, 2) * Log(1 + c))) /
                    (9. * Power(x, 2) * Power(1 + c * x, 5) * Power(-c + (1 + c) * Log(1 + c), 3));
                c = cmax;
                fmax += (-2 * c * rho0 *
                            (c * (2 + c * (3 + c * x)) - 2 * Power(1 + c, 2) * Log(1 + c)) *
                            (c *
                                    (c * (1 + c) * rho0 + 3 * rho * x +
                                        3 * c * rho * Power(x, 2) * (2 + c * x)) -
                                3 * (1 + c) * rho * x * Power(1 + c * x, 2) * Log(1 + c))) /
                    (9. * Power(x, 2) * Power(1 + c * x, 5) * Power(-c + (1 + c) * Log(1 + c), 3));
            }
        }
        if (fmax * fmin < 0.0) {
            cinvMin = cinvMid;
        } else {
            cinvMax = cinvMid;
        }
        c = 0.5 / cinvMin + 0.5 / cinvMax;
        err = cinvMax / cinvMin - 1.0;
    } while (err > 1.0e-6);
    rs = rmax / c;
    rhoNfw = sqr(c) * c * rho0 / (3.0 * (log(1 + c) - c / (1 + c)));
}
