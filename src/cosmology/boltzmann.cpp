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
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/boltzmann.hpp>
#include <cosmictiger/math/math.hpp>

void einsteinBoltzmannInit(CosState* uptr, ZeroOrderUniverse const* uniPtr, double k,
    double normalization, double a, double ns) {
    CosState& u = *uptr;
    ZeroOrderUniverse const& uni = *uniPtr;
    double ocValue, obValue, ogam, onu, omegaRadiation;
    uni.computeRadiationFractions(ogam, onu, a);
    uni.computeMatterFractions(ocValue, obValue, a);
    omegaRadiation = ogam + onu;
    double hubble = hubbleFunction(a, uni.params.hubble, uni.params.omegaB + uni.params.omegaC,
        uni.params.omegaGam + uni.params.omegaNu);
    double eps = k / (a * hubble);
    double c = normalization * pow(eps, -1.5f) * pow(eps, (ns - 1.f) * 0.5f);
    double rnu = onu / omegaRadiation;
    u[taui] = (double) 1.0 / (a * hubble);
    u[deltanui] = u[deltagami] = -(double) 2.0 / (double) 3.0 * c * eps * eps;
    u[deltabi] = u[deltaci] = (double) 3.0 / (double) 4.0 * u[deltagami];
    u[thetabi] = u[thetagami] = -c / (double) 18.0 * eps * eps * eps;
    u[thetanui] =
        ((double) 23 + (double) 4 * rnu) / ((double) 15 + (double) 4 * rnu) * u[thetagami];
    u[N2i] = (double) 0.5 * ((double) 4.0 * c) / ((double) 3.0 * ((double) 15 + (double) 4 * rnu)) *
        eps * eps;
    u[hdoti] = (double) (double) 2.0 * c * eps * eps;
    u[G0i] = u[G1i] = u[G2i] = u[F2i] = (double) 0.0;
    for (int l = 3; l < LMAX; l++) {
        u[FLi + l] = (double) 0.0;
        u[NLi + l] = (double) 0.0;
        u[GLi + l] = (double) 0.0;
    }
    u[etai] = ((double) 0.5 * u[hdoti] -
                  ((double) 1.5 * (ocValue * u[deltaci] + obValue * u[deltabi]) +
                      (double) 1.5 * (ogam * u[deltagami] + onu * u[deltanui]))) /
        (eps * eps);
}

void einsteinBoltzmanKernel(int myindex, CosState* states, ZeroOrderUniverse const* uniPtr,
    double* ks, double amin, double amax, double norm, int nmax, bool cont, double ns) {
    if (!cont) {
        einsteinBoltzmannInit(states + myindex, uniPtr, ks[myindex], norm, amin, ns);
    }
    einsteinBoltzmann(states + myindex, uniPtr, ks[myindex], amin, amax);
}

void einsteinBoltzmann(
    CosState* uptr, ZeroOrderUniverse const* uniPtr, double k, double amin, double amax) {
    auto const& uni = *uniPtr;
    if (amin < uni.amin && amax > uni.amax) {
        THROW_ERROR("out of range error in einstein_boltzmann\n");
    }
    CosState& u = *uptr;
    CosState u0;
    double loga = log(amin);
    double logamax = log(amax);
    double omegaM = uni.params.omegaB + uni.params.omegaC;
    double omegaR = uni.params.omegaGam + uni.params.omegaNu;
    while (loga < logamax) {
        double ocValue, obValue, ogam, onu, omegaRadiation;
        double a = exp(loga);
        double hubble = hubbleFunction(a, uni.params.hubble, omegaM, omegaR);
        double eps = k / (a * hubble);
        uni.computeRadiationFractions(ogam, onu, a);
        uni.computeMatterFractions(ocValue, obValue, a);
        omegaRadiation = ogam + onu;
        double cs2 = uni.cs2(a);
        double lambdaI = 0.0;
        lambdaI = std::max((double) lambdaI,
            sqrt(((double) LMAX + (double) 1.0) / ((double) LMAX + (double) 3.0)) * eps);
        lambdaI = std::max((double) lambdaI,
            sqrt((double) 3.0 * pow(eps, 4) + (double) 8.0 * eps * eps * omegaRadiation) /
                sqrt((double) 5) / eps);
        double lambdaR = (eps + sqrt(eps * eps + (double) 4.0 * cs2 * pow(eps, (double) 4))) /
            ((double) 2.0 * eps);
        double dlogaI = (double) 2.0 * (double) 1.73 / lambdaI;
        double dlogaR = (double) 2.0 * (double) 2.51 / lambdaR;
        double dloga = std::min(
            std::min((double) 5e-2, std::min((double) 0.9 * dlogaI, (double) 0.1 * dlogaR)),
            logamax - loga);
        double loga0 = loga;
        auto const computeExplicit = [&](int step) {
            u0 = u;
            CosState dudt;
            constexpr double kBeta[3] = {1, 0.25, (2.0 / 3.0)};
            constexpr double kTm[3] = {0, 1, 0.5};
            for (int i = 0; i < 3; i++) {
                loga = loga0 + (double) 0.5 * (kTm[i] + step) * dloga;
                a = exp(loga);
                hubble = hubbleFunction(a, uni.params.hubble, omegaM, omegaR);
                eps = k / (a * hubble);
                uni.computeRadiationFractions(ogam, onu, a);
                uni.computeMatterFractions(ocValue, obValue, a);
                omegaRadiation = ogam + onu;
                cs2 = uni.cs2(a);
                dudt[taui] = (double) 1.0 / (a * hubble);
                dudt[etai] = ((double) 1.5 *
                    ((obValue * u[thetabi]) +
                        ((double) 4.0 / (double) 3.0) * (ogam * u[thetagami] + onu * u[thetanui])) /
                    eps);
                double factor =
                    ((a * omegaM) + (double) 4 * a * a * a * a * ((double) 1 - omegaM - omegaR)) /
                    ((double) 2 * a * omegaM + (double) 2 * omegaR +
                        (double) 2 * a * a * a * a * ((double) 1 - omegaM - omegaR));
                dudt[hdoti] = (-factor * u[hdoti] -
                    ((double) 3.0 * (ocValue * u[deltaci] + obValue * u[deltabi]) +
                        (double) 6.0 * (ogam * u[deltagami] + onu * u[deltanui])));
                dudt[deltaci] = -(double) 0.5 * u[hdoti];
                dudt[deltabi] = -eps * u[thetabi] - (double) 0.5 * u[hdoti];
                dudt[deltagami] = -(double) 4.0 / (double) 3.0 * eps * u[thetagami] -
                    ((double) 2.0 / (double) 3.0) * u[hdoti];
                dudt[deltanui] = -(double) 4.0 / (double) 3.0 * eps * u[thetanui] -
                    ((double) 2.0 / (double) 3.0) * u[hdoti];
                dudt[thetabi] = -u[thetabi] + cs2 * eps * u[deltabi];
                dudt[thetagami] = eps * ((double) 0.25 * u[deltagami] - (double) 0.5 * u[F2i]);
                dudt[thetanui] = eps * ((double) 0.25 * u[deltanui] - (double) 0.5 * u[N2i]);
                dudt[F2i] = ((double) 8.0 / (double) 15.0) * eps * u[thetagami] +
                    ((double) 4.0 / (double) 15.0) * u[hdoti] +
                    ((double) 8.0 / (double) 5.0) * dudt[etai] -
                    ((double) 3.0 / (double) 5.0) * eps * u[FLi + 3];
                dudt[N2i] = ((double) 8.0 / (double) 15.0) * eps * u[thetanui] +
                    ((double) 4.0 / (double) 15.0) * u[hdoti] +
                    ((double) 8.0 / (double) 5.0) * dudt[etai] -
                    ((double) 3.0 / (double) 5.0) * eps * u[NLi + 3];
                dudt[GLi + 0] = -eps * u[GLi + 1];
                dudt[GLi + 1] = eps / (double) (3) * (u[GLi + 0] - (double) 2 * u[GLi + 2]);
                dudt[GLi + 2] =
                    eps / (double) (5) * ((double) 2 * u[GLi + 1] - (double) 3 * u[GLi + 3]);
                for (int l = 3; l < LMAX - 1; l++) {
                    dudt[FLi + l] = eps / (double) (2 * l + 1) *
                        ((double) l * u[FLi - 1 + l] - (double) (l + 1) * u[FLi + 1 + l]);
                    dudt[NLi + l] = eps / (double) (2 * l + 1) *
                        ((double) l * u[NLi - 1 + l] - (double) (l + 1) * u[NLi + 1 + l]);
                    dudt[GLi + l] = eps / (double) (2 * l + 1) *
                        ((double) l * u[GLi - 1 + l] - (double) (l + 1) * u[GLi + 1 + l]);
                }
                dudt[FLi + LMAX - 1] = (eps * u[FLi + LMAX - 2]) / (double) (2 * LMAX - 1);
                dudt[NLi + LMAX - 1] = (eps * u[NLi + LMAX - 2]) / (double) (2 * LMAX - 1);
                dudt[GLi + LMAX - 1] = (eps * u[GLi + LMAX - 2]) / (double) (2 * LMAX - 1);
                for (int f = 0; f < NFIELD; f++) {
                    u[f] = ((double) 1 - kBeta[i]) * u0[f] +
                        kBeta[i] * (u[f] + dudt[f] * dloga * (double) 0.5);
                }
            }
        };

        auto computeImplicitDudt = [&](double loga, double dloga) {
            a = exp(loga);
            double thetab = u[thetabi];
            double thetagam = u[thetagami];
            double f2 = u[F2i];
            double g0 = u[G0i];
            double g1 = u[G1i];
            double g2 = u[G2i];
            double thetab0 = thetab;
            double thetagam0 = thetagam;
            double f20 = f2;
            double g00 = g0;
            double g10 = g1;
            double g20 = g2;
            double sigma = uni.sigmaT(a);

            thetab = -(
                (-(double) 3 * obValue * thetab0 - (double) 3 * dloga * obValue * sigma * thetab0 -
                    (double) 4 * dloga * ogam * sigma * thetagam0) /
                ((double) 3 * obValue + (double) 3 * dloga * obValue * sigma +
                    (double) 4 * dloga * ogam * sigma));
            thetagam = -((-(double) 3 * dloga * obValue * sigma * thetab0 -
                             (double) 3 * obValue * thetagam0 -
                             (double) 4 * dloga * ogam * sigma * thetagam0) /
                ((double) 3 * obValue + (double) 3 * dloga * (double) obValue * sigma +
                    (double) 4 * dloga * ogam * sigma));
            f2 = -((-(double) 10 * f20 - (double) 4 * dloga * f20 * sigma - dloga * g00 * sigma -
                       dloga * g20 * sigma) /
                (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
            g0 = -((-(double) 10 * g00 - (double) 5 * dloga * f20 * sigma -
                       (double) 8 * dloga * g00 * sigma - (double) 5 * dloga * g20 * sigma) /
                (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
            g1 = g10 / ((double) 1 + dloga * sigma);
            g2 = -((-(double) 10 * g20 - dloga * f20 * sigma - dloga * g00 * sigma -
                       (double) 4 * dloga * g20 * sigma) /
                (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
            std::array<double, NFIELD> dudt;
            for (int f = 0; f < NFIELD; f++) {
                dudt[f] = (double) 0.0;
            }
            dudt[thetabi] = (thetab - thetab0) / dloga;
            dudt[thetagami] = (thetagam - thetagam0) / dloga;
            dudt[F2i] = (f2 - f20) / dloga;
            dudt[G0i] = (g0 - g00) / dloga;
            dudt[G1i] = (g1 - g10) / dloga;
            dudt[G2i] = (g2 - g20) / dloga;
            for (int l = 3; l < LMAX - 1; l++) {
                dudt[GLi + l] =
                    u[GLi + l] * ((double) 1 / ((double) 1 + dloga * sigma) - (double) 1) / dloga;
                dudt[FLi + l] =
                    u[FLi + l] * ((double) 1 / ((double) 1 + dloga * sigma) - (double) 1) / dloga;
            }
            dudt[GLi + LMAX - 1] = u[GLi + LMAX - 1] *
                ((double) 1 /
                        ((double) 1 +
                            (sigma +
                                (double) LMAX / (u[taui] * a * hubble) /
                                    ((double) 2 * (double) LMAX - (double) 1))) -
                    (double) 1) /
                dloga;
            dudt[FLi + LMAX - 1] = u[FLi + LMAX - 1] *
                ((double) 1 /
                        ((double) 1 +
                            (sigma +
                                (double) LMAX / (u[taui] * a * hubble) /
                                    ((double) 2 * (double) LMAX - (double) 1))) -
                    (double) 1) /
                dloga;
            return dudt;
        };

        computeExplicit(0);
        double gamma = (double) 1.0 - (double) 1.0 / sqrt((double) 2);

        auto dudt1 = computeImplicitDudt(loga + gamma * dloga, gamma * dloga);
        for (int f = 0; f < NFIELD; f++) {
            u[f] += dudt1[f] * ((double) 1.0 - (double) 2.0 * gamma) * dloga;
        }
        auto dudt2 = computeImplicitDudt(loga + ((double) 1.0 - gamma) * dloga, gamma * dloga);
        for (int f = 0; f < NFIELD; f++) {
            u[f] += (dudt1[f] * ((double) -0.5 + (double) 2.0 * gamma) + dudt2[f] * (double) 0.5) *
                dloga;
        }

        computeExplicit(1);

        loga = loga0 + dloga;
    }
}

void einsteinBoltzmannInterpolationFunction(InterpolationFunctor<double>* darkMatterPowerFunction,
    InterpolationFunctor<double>* baryKFunc, CosState* u, ZeroOrderUniverse* uni, double kmin,
    double kmax, double norm, int n, double astart, double astop, bool cont, double ns) {
    double dlogk = 1.0e-2;
    double logkmin = log(kmin) - dlogk;
    double logkmax = log(kmax) + dlogk;
    dlogk = (logkmax - logkmin) / (double) (n - 1);
    Vector<double> ks(n);
    for (int i = 0; i < n; i++) {
        ks[i] = exp(logkmin + (double) i * dlogk);
    }
    Vector<hpx::future<void>> futs;
    for (int block = 0; block < n; block++) {
        futs.push_back(hpx::async([block, &u, &uni, &ks, astart, astop, norm, n, cont, ns]() {
            einsteinBoltzmanKernel(block, u, uni, ks.data(), astart, astop, norm, n, cont, ns);
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());

    std::vector<double> darkMatterPower(n), baryK(n);
    double oc = uni->params.omegaC;
    double ob = uni->params.omegaB;
    double om = oc + ob;
    oc /= om;
    ob /= om;
    auto const hubble = [uni](double a) {
        return hubbleFunction(a, uni->params.hubble, uni->params.omegaC + uni->params.omegaB,
            uni->params.omegaGam + uni->params.omegaNu);
    };
    double h = hubble(astop);
    for (int i = 0; i < n; i++) {
        double k = ks[i];
        double eps = k / (astop * h);
        darkMatterPower[i] = sqr(u[i][deltaci]);
        baryK[i] = sqr(u[i][deltabi]);
        //		vel_k[i] = sqr((ob * (eps * U[i][thetabi] + (double) 0.5 * U[i][hdoti]) + oc *
        //((double) 0.5 * U[i][hdoti]))); 	printf("%e %e\n", k, vel_k[i]);
    }
    buildInterpolationFunction(
        darkMatterPowerFunction, darkMatterPower, (double) exp(logkmin), (double) exp(logkmax));
    buildInterpolationFunction(baryKFunc, baryK, (double) exp(logkmin), (double) exp(logkmax));
}
