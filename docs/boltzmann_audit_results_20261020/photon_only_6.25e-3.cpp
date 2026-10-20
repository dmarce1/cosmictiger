#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cosmictiger/cosmology/boltzmann.hpp>
#include <cosmictiger/cosmology/recombination_thermal.hpp>
#include <cosmictiger/cosmology/recfast.hpp>
#define THROW_ERROR(...) throw std::runtime_error("domain")
#define PRINT(...) do {} while (false)
void ZeroOrderUniverse::computeMatterFractions(double& oc, double& ob, double a) const {
    double omegaM = params.omegaB + params.omegaC;
    double omegaR = params.omegaGam + params.omegaNu;
    double om = omegaM / (omegaR / a + omegaM + (a * a * a) * ((double) 1.0 - omegaM - omegaR));
    ob = params.omegaB * om / omegaM;
    oc = params.omegaC * om / omegaM;
}

void ZeroOrderUniverse::computeRadiationFractions(double& ogam, double& onu, double a) const {
    double omegaM = params.omegaB + params.omegaC;
    double omegaR = params.omegaGam + params.omegaNu;
    double omegaRadiation =
        omegaR / (omegaR + a * omegaM + (a * a * a * a) * ((double) 1.0 - omegaM - omegaR));
    ogam = params.omegaGam * omegaRadiation / omegaR;
    onu = params.omegaNu * omegaRadiation / omegaR;
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
    double const a = 1.0 / (1.0 + z);
    double omegaM = params.omegaB + params.omegaC;
    double omegaR = params.omegaGam + params.omegaNu;
    double const omegaL = 1.0 - omegaM - omegaR;
    double const hubbleToday = params.hubble * constants::kH0;
    double const h2 =
        hubbleToday * hubbleToday * (omegaR / (a * a * a * a) + omegaM / (a * a * a) + omegaL);
    return omegaM * 3.0 * h2 / (8.0 * M_PI * constants::kG);
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
    double amax, CosmicParameters cpars) {
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
    ne = fxe(amin) * nH;
    trad = tRadiation(amin);
    tgas = trad;
    double n = nH + nHe;
    double p = kB * (n + ne) * tgas;
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
        trad = tRadiation(a);
        double dt = dloga / hubble;
        ne = fxe(a) * nH;
        sigmaC =
            recombination::comptonTemperatureCoupling(a, omegaGam, omegaB, nH, nHe, ne, hubble);
        tgas = recombination::advanceGasTemperature(tgas, trad, dloga, sigmaC);
        ne = fxe(a) * nH;
        n = nH + nHe;
        p = kB * (n + ne) * tgas;
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
            std::min((double) 6.25e-3, std::min((double) 0.9 * dlogaI, (double) 0.1 * dlogaR)),
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


int main(int argc, char** argv) {
    using namespace constants;
    auto const history=recfast::loadIonizationHistory(argv[1]);
    CosmicParameters p{};
    p.omegaB=.049389; p.omegaC=.26503; p.neff=3.046; p.y=.2454006;
    p.theta=2.7255/kTcmb; p.hubble=.6732;
    double const rhoc=3*pow(p.hubble*kH0,2)/(8*M_PI*kG);
    p.omegaGam=4*kSigma/kC*pow(2.7255,4)/(rhoc*kC*kC);
    p.omegaNu=0.; p.neff=0.;
    ZeroOrderUniverse uni;
    createZeroOrderUniverse(&uni,history,1.1,p);
    for (double k: {.1}) {
        CosState u{};
        einsteinBoltzmannInit(&u,&uni,k,1.,uni.amin,.966);
        einsteinBoltzmann(&u,&uni,k,uni.amin,1.);
        printf("cap=6.25e-3 k=%.8g dc=%.12g db=%.12g dg=%.12g vb=%.12g vg=%.12g F2=%.12g\n",k,u[deltaci],u[deltabi],u[deltagami],u[thetabi],u[thetagami],u[F2i]);
    }
}
