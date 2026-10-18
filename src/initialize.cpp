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

#include <cmath>

#include <gsl/gsl_rng.h>

#include <cosmictiger/boltzmann.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/initialize.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/simulation_units.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/zero_order.hpp>

#define RECFAST_N 1000
#define RECFAST_Z0 9990
#define RECFAST_Z1 0
#define RECFAST_DZ 10

std::function<double(double)> runRecfast(CosmicParameters params) {
    std::function<double(double)> func;
    std::string filename = "recfast.in." + std::to_string(hpxRank());
    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp == NULL) {
        THROW_ERROR("Unable to write to %s\n", filename.c_str());
    }
    filename = "recfast.out." + std::to_string(hpxRank());
    FILE* fp1 = fopen(filename.c_str(), "rb");
    if (fp1) {
        fclose(fp1);
        std::string command = "rm " + filename + "\n";
        if (system(command.c_str()) != 0) {
            THROW_ERROR("Unable to execute %s\n", command.c_str());
        }
    }
    fprintf(fp, "%s\n", filename.c_str());
    fprintf(fp, "%f %f %f\n", params.omegaB, params.omegaC, 1.0 - params.omegaB - params.omegaC);
    fprintf(fp, "%f %f %f\n", 100 * params.hubble, params.theta * 2.73, params.y);
    fprintf(fp, "1\n");
    fprintf(fp, "6\n");
    fclose(fp);
    std::string cmd =
        "cat recfast.in." + std::to_string(hpxRank()) + " | ./recfast 1> /dev/null 2> /dev/null";
    if (system(cmd.c_str()) != 0) {
        THROW_ERROR("Unable to run %s\n", cmd.c_str());
    }
    fp = fopen(filename.c_str(), "rb");
    char d1[2];
    char d2[4];
    if (fscanf(fp, " %s %s\n", d1, d2) == 0) {
        THROW_ERROR("unable to read %s\n", filename.c_str());
    }
    std::vector<double> xe;
    for (int i = 0; i < RECFAST_N; i++) {
        float z;
        float thisXe;
        if (fscanf(fp, "%f %f\n", &z, &thisXe) == 0) {
            THROW_ERROR("unable to read %s\n", filename.c_str());
        }
        xe.push_back(thisXe);
    }
    std::vector<double> tmp;
    for (int i = 0; i < RECFAST_N; i++) {
        tmp.push_back(xe.back());
        xe.pop_back();
    }
    xe = std::move(tmp);
    fclose(fp);
    auto interFunc = [xe](double a) {
        double const z = 1.0 / a - 1.0;
        int const i1 = std::min(std::max((int) (z / RECFAST_DZ), 1), RECFAST_N - 2);
        if (i1 == RECFAST_N - 2) {
            return (double) xe.back();
        } else {
            int const i0 = i1 - 1;
            int const i2 = i1 + 1;
            int const i3 = i1 + 2;
            double const t = z / RECFAST_DZ - i1;
            double const y0 = xe[i0];
            double const y1 = xe[i1];
            double const y2 = xe[i2];
            double const y3 = xe[i3];
            double const ct = t * (1.0 - t);
            double const d = -0.5 * ct * ((1.0 - t) * y0 + t * y3);
            double const b = (1.0 - t + ct * (1.0 - 1.5 * t)) * y1;
            double const c = (t + ct * (1.5 * t - 0.5)) * y2;
            return d + b + c;
        }
    };
    return std::move(interFunc);
}

struct PowerSpectrumFunction {
    Vector<float> pValue2;
    float logkmin;
    float logkmax;
    float dlogk;
    float operator()(float k) const {
        float logk = std::log(k);
        float const k0 = (logk - logkmin) / dlogk;
        int i0 = int(k0) - 1;
        if (i0 < 0) {
            PRINT("power.init does not have sufficient range min %e max %e tried %e\n",
                exp(logkmin), exp(logkmax), k);
        }
        if (i0 > pValue2.size() - 4) {
            PRINT("power.init does not have sufficient range min %e max %e tried %e\n",
                exp(logkmin), exp(logkmax), k);
        }
        int i1 = i0 + 1;
        int i2 = i0 + 2;
        int i3 = i0 + 3;
        float x = k0 - i1;
        float y0 = std::log(pValue2[i0]);
        float y1 = std::log(pValue2[i1]);
        float y2 = std::log(pValue2[i2]);
        float y3 = std::log(pValue2[i3]);
        float k1 = (y2 - y0) * 0.5;
        float k2 = (y3 - y1) * 0.5;
        float a = y1;
        float b = k1;
        float c = -2 * k1 - k2 - 3 * y1 + 3 * y2;
        float d = k1 + k2 + 2 * y1 - 2 * y2;
        return std::exp(a + b * x + c * x * x + d * x * x * x);
    }
    float sigma8Integrand(float x) const {
        float r = 8 / getOptions().hubble;
        float const c0 = float(9) / (2. * float(M_PI) * float(M_PI)) / powf(r, 6);
        float k = std::exp(x);
        float pValue = (*this)(k);
        float tmp = (std::sin(k * r) - k * r * std::cos(k * r));
        return c0 * pValue * tmp * tmp * std::pow(k, -3);
    }
    float normalize() {
        int const n = 8 * 1024;
        float sum = 0.0;
        float logkmax = this->logkmax - 2.5 * dlogk;
        float logkmin = this->logkmin + 1.5 * dlogk;
        float dlogk = (logkmax - logkmin) / n;
        sum = (1.0 / 3.0) * (sigma8Integrand(logkmax) + sigma8Integrand(logkmin)) * dlogk;
        for (int i = 1; i < n; i += 2) {
            float logk = logkmin + i * dlogk;
            sum += (4.0 / 3.0) * sigma8Integrand(logk) * dlogk;
        }
        for (int i = 2; i < n; i += 2) {
            float logk = logkmin + i * dlogk;
            sum += (2.0 / 3.0) * sigma8Integrand(logk) * dlogk;
        }
        sum = sqr(getOptions().sigma8) / sum;
        for (int i = 0; i < pValue2.size(); i++) {
            pValue2[i] *= sum;
        }
        if (hpxRank() == 0) {
            PRINT("Normalization = %e\n", sum);
        }
        return sum;
    }
};

static void zeldovichBegin(int dim1, int dim2);
static float zeldovichEnd(int dim, bool, float, float);
static void twolptInit();
static void twolptDestroy();
static void twolpt(int, int);
static void twolptPhase(int phase);
static void twolptCorrection1();
static void twolptF2delta2Inv();
static void twolptCorrection2(int dim);
static PowerSpectrumFunction readPowerSpectrum();

HPX_PLAIN_ACTION(zeldovichBegin, ZeldovichBeginAction);
HPX_PLAIN_ACTION(zeldovichEnd, ZeldovichEndAction);
HPX_PLAIN_ACTION(twolptInit, TwolptInitAction);
HPX_PLAIN_ACTION(twolptPhase, TwolptPhaseAction);
HPX_PLAIN_ACTION(twolptCorrection1, TwolptCorrection1Action);
HPX_PLAIN_ACTION(twolptF2delta2Inv, TwolptF2delta2InvAction);
HPX_PLAIN_ACTION(twolptCorrection2, TwolptCorrection2Action);
HPX_PLAIN_ACTION(twolpt, TwolptAction);
HPX_PLAIN_ACTION(twolptDestroy, TwolptDestroyAction);

static Vector<float> delta2;
static Vector<ComplexFloat> delta2Inv;
static Vector<float> delta2Part;
static Vector<Fixed32> x0Value[NDIM];

static void initX0();
static void clearX0();
HPX_PLAIN_ACTION(initX0, InitX0Action);
HPX_PLAIN_ACTION(clearX0, ClearX0Action);

PowerSpectrumFunction computePowerSpectrum() {
    PowerSpectrumFunction func;
    ZeroOrderUniverse zeroverse;
    double* resultPtr;
    CosmicParameters params;
    InterpolationFunctor<double> mK;
    InterpolationFunctor<double> velK;
    int nk = 1024;
    Vector<CosState> states(nk);

    auto& uni = zeroverse;
    params.omegaB = getOptions().omegaB;
    params.omegaC = getOptions().omegaC;
    params.omegaGam = getOptions().omegaGam;
    params.omegaNu = getOptions().omegaNu;
    params.y = getOptions().y0;
    params.neff = getOptions().neff;
    params.theta = getOptions().thetaCmb;
    params.hubble = getOptions().hubble;
    PRINT("Computing zero order universe...");
    fflush(stdout);
    auto fxe = runRecfast(params);
    createZeroOrderUniverse(&uni, fxe, 1.1, params);
    PRINT("Done.\n");
    auto const ns = getOptions().ns;
    fflush(stdout);
    double kmin;
    double kmax;
    kmin = 1e-5 * params.hubble;
    kmax = 25.271 * params.hubble;
    einsteinBoltzmannInterpolationFunction(
        &mK, &velK, states.data(), &zeroverse, kmin, kmax, 1.0, nk, zeroverse.amin, 1.0, false, ns);

    func.logkmin = log(kmin);
    func.logkmax = log(kmax);
    func.dlogk = (func.logkmax - func.logkmin) / nk;
    for (int i = 0; i < nk; i++) {
        double k = exp(func.logkmin + (double) i * func.dlogk);
        func.pValue2.push_back(mK(k));
    }
    auto const norm = func.normalize();
    if (hpxRank() == 0) {
        int const m = 2 * nk;
        double const logkmin = log(kmin);
        double const logkmax = log(kmax);
        double const dlogk = (logkmax - logkmin) / m;
        FILE* fp = fopen("power.dat", "wt");
        double const lh = params.hubble;
        double const lh3 = lh * lh * lh;
        for (int i = 1; i < m - 2; i++) {
            double k = exp(logkmin + (double) i * dlogk);
            fprintf(fp, "%e %e %e\n", k / lh, norm * mK(k) * lh3, norm * velK(k) * lh3);
        }
        fclose(fp);
    }
    return func;
}

void initialize(double z0) {
    initX0();
    int64_t const n = getOptions().nfour;
    float const omegaM = getOptions().omegaM;
    float const a0 = 1.0 / (z0 + 1.0);
    float const d1Value2 = cosmosGrowthFactor(omegaM, a0) / cosmosGrowthFactor(omegaM, 1.0);
    float const om = omegaM / (omegaM + (a0 * a0 * a0) * (1.0 - omegaM));
    float const f1 = std::pow(om, 5.f / 9.f);
    float const h0 = constants::kH0 * getOptions().codeToS * getOptions().hubble;
    float const hValue = h0 * std::sqrt(omegaM / (a0 * a0 * a0) + 1.0 - omegaM);
    float prefac1 = f1 * hValue * a0 * a0;
    float const d2Value = -3.f * sqr(d1Value2) / 7.f;
    float const f2 = 2.f * std::pow(om, 6.f / 11.f);
    float prefac2 = f2 * hValue * a0 * a0;
    PRINT("D1 = %e\n", d1Value2);
    PRINT("D2 = %e\n", d2Value);
    PRINT("prefac1 = %e\n", prefac1 / a0);
    PRINT("prefac2 = %e\n", prefac2 / a0);
    for (int dim = 0; dim < NDIM; dim++) {
        fft3dInit(n);
        zeldovichBegin(dim, NDIM);
        //		fft3d_force_real();
        fft3dInvExecute();
        float dxmax = zeldovichEnd(dim, dim == 0, d1Value2, prefac1);
        PRINT("dxmax = %e\n", dxmax);
        fft3dDestroy();
    }
    if (getOptions().twolpt) {
        twolptInit();

        PRINT("2LPT phase 1\n");
        twolpt(1, 1);
        twolptPhase(0);
        fft3dDestroy();

        PRINT("2LPT phase 2\n");
        twolpt(2, 2);
        twolptPhase(1);
        fft3dDestroy();

        PRINT("2LPT phase 3\n");
        twolpt(0, 0);
        twolptPhase(2);
        fft3dDestroy();

        PRINT("2LPT phase 4\n");
        twolpt(2, 2);
        twolptPhase(3);
        fft3dDestroy();

        PRINT("2LPT phase 5\n");
        twolpt(0, 0);
        twolptPhase(4);
        fft3dDestroy();

        PRINT("2LPT phase 6\n");
        twolpt(1, 1);
        twolptPhase(5);
        fft3dDestroy();

        PRINT("2LPT phase 7\n");
        twolpt(0, 1);
        twolptPhase(6);
        fft3dDestroy();

        PRINT("2LPT phase 8\n");
        twolpt(0, 2);
        twolptPhase(7);
        fft3dDestroy();

        PRINT("2LPT phase 9\n");
        twolpt(1, 2);
        twolptPhase(8);
        fft3dDestroy();

        twolptCorrection1();
        twolptF2delta2Inv();
        //		fft3d2silo(false);
        //		system( "cp complex.silo complex.3.silo\n");
        fft3dDestroy();

        for (int dim = 0; dim < NDIM; dim++) {
            PRINT("Computing 2LPT correction to %c positions and velocities\n", 'x' + dim);
            twolptCorrection2(dim);
            float dxmax = zeldovichEnd(dim, false, -d2Value, prefac2);
            fft3dDestroy();
            PRINT("%e\n", dxmax);
        }
        twolptDestroy();
    }
    clearX0();
}

void twolptF2delta2Inv() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptF2delta2InvAction>(c));
    }
    auto const box = fft3dComplexRange();
    delta2Inv = fft3dReadComplex(box);
    hpx::wait_all(futs.begin(), futs.end());
}

void twolptCorrection1() {
    int const n = getOptions().nfour;
    if (hpxRank() == 0) {
        fft3dInit(n);
    }
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptCorrection1Action>(c));
    }
    auto const box = fft3dRealRange();
    fft3dAccumulateReal(box, delta2);
    //	fft3d2silo(true);
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        fft3dExecute();
    }
}

void twolptCorrection2(int dim) {
    int const n = getOptions().nfour;
    if (hpxRank() == 0) {
        fft3dInit(n);
    }
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptCorrection2Action>(c, dim));
    }
    float const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    float const factor = std::pow(boxSize, -1.5) * n * n * n;
    auto const box = fft3dComplexRange();
    auto yValue = delta2Inv;
    Array<int64_t, NDIM> iValue;
    Vector<hpx::future<void>> futs2;
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs2.push_back(hpx::async(
            [n, box, boxSize, &yValue, dim, factor](Array<int64_t, NDIM> iValue2) {
                int const i = (iValue2[0] < n / 2 ? iValue2[0] : iValue2[0] - n);
                float const kx = 2.f * (float) M_PI / boxSize * float(i);
                for (iValue2[1] = box.begin[1]; iValue2[1] != box.end[1]; iValue2[1]++) {
                    int const j = (iValue2[1] < n / 2 ? iValue2[1] : iValue2[1] - n);
                    float const ky = 2.f * (float) M_PI / boxSize * float(j);
                    for (iValue2[2] = box.begin[2]; iValue2[2] != box.end[2]; iValue2[2]++) {
                        int const k = (iValue2[2] < n / 2 ? iValue2[2] : iValue2[2] - n);
                        int const i2 = sqr(i, j, k);
                        int64_t const index = box.index(iValue2);
                        if (i2 > 0) {
                            float kz = 2.f * (float) M_PI / boxSize * float(k);
                            float k2 = kx * kx + ky * ky + kz * kz;
                            float const kValue[NDIM] = {kx, ky, kz};
                            yValue[index] = -ComplexFloat(0, 1) * kValue[dim] * yValue[index] / k2;
                            yValue[index] *= cloudFilter(kx * boxSize / n);
                            yValue[index] *= cloudFilter(ky * boxSize / n);
                            yValue[index] *= cloudFilter(kz * boxSize / n);
                        } else {
                            yValue[index] = ComplexFloat(0, 0);
                        }
                    }
                }
            },
            iValue));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    fft3dAccumulateComplex(box, yValue);
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        //		fft3d2silo(false);
        //		std::string cmd = std::string("cp complex.silo complex.") + std::to_string(dim) +
        //".silo\n"; 		system(cmd.c_str());
        fft3dInvExecute();
    }
}

void twolptGenerate(int dim1, int dim2) {
    int64_t const n = getOptions().nfour;
    float const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    float const factor = std::pow(boxSize, -1.5) * n * n * n;
    Vector<ComplexFloat> yValue;
    static auto power = getOptions().usePowerFile ? readPowerSpectrum() : computePowerSpectrum();
    auto const box = fft3dComplexRange();
    yValue.resize(box.volume());
    Array<int64_t, NDIM> iValue;
    constexpr int kRandInitIters = 8;
    Vector<hpx::future<void>> futs2;
    //	const auto filter = get_options().use_glass || get_options().close_pack;
    bool const filter = (dim1 != NDIM || dim2 != NDIM);
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs2.push_back(hpx::async(
            [n, box, boxSize, &yValue, dim1, dim2, factor, filter](Array<int64_t, NDIM> iValue2) {
                int const i = (iValue2[0] < n / 2 ? iValue2[0] : iValue2[0] - n);
                float const kx = 2.f * (float) M_PI / boxSize * float(i);
                for (iValue2[1] = box.begin[1]; iValue2[1] != box.end[1]; iValue2[1]++) {
                    int seed = (iValue2[0] * n + iValue2[1]) * 1234 + 42;
                    gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
                    gsl_rng_set(rndgen, seed);
                    for (int l = 0; l < kRandInitIters; l++) {
                        gsl_rng_uniform_pos(rndgen);
                    }
                    int const j = (iValue2[1] < n / 2 ? iValue2[1] : iValue2[1] - n);
                    float const ky = 2.f * (float) M_PI / boxSize * float(j);
                    for (iValue2[2] = box.begin[2]; iValue2[2] != box.end[2]; iValue2[2]++) {
                        int64_t const index = box.index(iValue2);
                        if (iValue2[2] > 0) {
                            int const k = (iValue2[2] < n / 2 ? iValue2[2] : iValue2[2] - n);
                            int const i2 = sqr(i, j, k);
                            float const kz = 2.f * (float) M_PI / boxSize * float(k);
                            if (i2 > 0) {
                                float const kz = 2.f * (float) M_PI / boxSize * float(k);
                                float const k2 = kx * kx + ky * ky + kz * kz;
                                float const k0Value = sqrtf(kx * kx + ky * ky + kz * kz);
                                float const x = gsl_rng_uniform_pos(rndgen);
                                float const y = gsl_rng_uniform_pos(rndgen);
                                ComplexFloat const kValue[NDIM + 1] = {
                                    {0, -kx}, {0, -ky}, {0, -kz}, {1, 0}};
                                auto const randNormal =
                                    expc(ComplexFloat(0, 1) * 2.f * float(M_PI) * y) *
                                    sqrtf(-logf(fabsf(x)));
                                float const pValue = power(k0Value);
                                yValue[index] = randNormal * sqrtf(pValue) * factor * kValue[dim1] *
                                    kValue[dim2] / k2;
                            } else {
                                yValue[index] = ComplexFloat(0.f, 0.f);
                            }
                            if (filter) {
                                yValue[index] *= cloudFilter(kx * boxSize / n);
                                yValue[index] *= cloudFilter(ky * boxSize / n);
                                yValue[index] *= cloudFilter(kz * boxSize / n);
                            }
                        }
                    }
                    gsl_rng_free(rndgen);
                    iValue2[2] = 0;
                    int64_t const index = box.index(iValue2);
                    int const j0 = iValue2[0] > n / 2 ? n - iValue2[0] : iValue2[0];
                    int const j1Value = iValue2[1] > n / 2 ? n - iValue2[1] : iValue2[1];
                    seed = (j0 * n + j1Value) * 1234 + 42;
                    rndgen = gsl_rng_alloc(gsl_rng_taus);
                    gsl_rng_set(rndgen, seed);
                    for (int l = 0; l < kRandInitIters; l++) {
                        gsl_rng_uniform_pos(rndgen);
                    }
                    int const k = (iValue2[2] < n / 2 ? iValue2[2] : iValue2[2] - n);
                    int const i2 = sqr(i, j, k);
                    float sgn = 1.0;
                    if (dim1 != NDIM) {
                        sgn *= -1.0;
                    }
                    if (dim2 != NDIM) {
                        sgn *= -1.0;
                    }
                    float const kz = 2.f * (float) M_PI / boxSize * float(k);
                    if (i2 > 0) {
                        float const k2 = kx * kx + ky * ky + kz * kz;
                        float const k0Value = sqrtf(kx * kx + ky * ky + kz * kz);
                        float const x = gsl_rng_uniform_pos(rndgen);
                        float const y = gsl_rng_uniform_pos(rndgen);
                        ComplexFloat const kValue[NDIM + 1] = {
                            {0, -kx}, {0, -ky}, {0, -kz}, {1, 0}};
                        auto const randNormal = expc(ComplexFloat(0, 1) * 2.f * float(M_PI) * y) *
                            sqrtf(-logf(fabsf(x)));
                        float const pValue = power(k0Value);
                        yValue[index] =
                            randNormal * sqrtf(pValue) * factor * kValue[dim1] * kValue[dim2] / k2;
                        if (iValue2[0] > n / 2) {
                            yValue[index] = yValue[index].conj() * sgn;
                        } else if (iValue2[0] == 0) {
                            if (iValue2[1] > n / 2) {
                                yValue[index] = yValue[index].conj() * sgn;
                            }
                        }
                    } else {
                        yValue[index] = ComplexFloat(0.f, 0.f);
                    }
                    if (filter) {
                        yValue[index] *= cloudFilter(kx * boxSize / n);
                        yValue[index] *= cloudFilter(ky * boxSize / n);
                        yValue[index] *= cloudFilter(kz * boxSize / n);
                    }
                    gsl_rng_free(rndgen);
                }
            },
            iValue));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    fft3dAccumulateComplex(box, yValue);
}

void twolpt(int dim1, int dim2) {
    int const n = getOptions().nfour;
    if (hpxRank() == 0) {
        fft3dInit(n);
    }
    Vector<hpx::future<void>> futs;
    if (hpxRank() == 0) {
        for (int i = 1; i < hpxSize(); i++) {
            futs.push_back(hpx::async<TwolptAction>(hpxLocalities()[i], dim1, dim2));
        }
    }
    Vector<ComplexFloat> yValue;
    auto const box = fft3dRealRange();
    yValue.resize(box.volume());
    twolptGenerate(dim1, dim2);
    fft3dAccumulateComplex(box, yValue);
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        //		fft3d_force_real();
        fft3dInvExecute();
    }
}

void twolptPhase(int phase) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptPhaseAction>(c, phase));
    }
    auto const box = fft3dRealRange();
    auto const vol = box.volume();
    if (phase > 2 * NDIM) {
        auto thisDelta2Part = fft3dReadReal(box);
        for (int i = 0; i < vol; i++) {
            delta2[i] -= thisDelta2Part[i] * thisDelta2Part[i];
        }
    } else {
        if (phase % 2 == 0) {
            delta2Part = fft3dReadReal(box);
        } else {
            auto delta2ThisPart = fft3dReadReal(box);
            for (int i = 0; i < vol; i++) {
                delta2[i] += delta2Part[i] * delta2ThisPart[i];
            }
        }
    }

    hpx::wait_all(futs.begin(), futs.end());
}

void twolptInit() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptInitAction>(c));
    }
    auto const box = fft3dRealRange();
    delta2.resize(box.volume(), 0.0);
    hpx::wait_all(futs.begin(), futs.end());
}

void twolptDestroy() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptDestroyAction>(c));
    }
    delta2 = decltype(delta2)();
    delta2Inv = decltype(delta2Inv)();
    delta2Part = decltype(delta2Part)();
    hpx::wait_all(futs.begin(), futs.end());
}

static void zeldovichBegin(int dim1, int dim2) {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<ZeldovichBeginAction>(c, dim1, dim2));
    }
    twolptGenerate(dim1, dim2);
    hpx::wait_all(futs.begin(), futs.end());
}

static Range<int64_t> findMyBox(Range<int64_t> box, int begin, int end, int depth) {
    if (end - begin == 1) {
        return box;
    } else {
        int const xdim = depth % NDIM;
        int const mid = (begin + end) / 2;
        float const w = float(mid - begin) / float(end - begin);
        if (hpxRank() < mid) {
            auto left = box;
            left.end[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
            return findMyBox(left, begin, mid, depth + 1);
        } else {
            auto right = box;
            right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
            return findMyBox(right, mid, end, depth + 1);
        }
    }
}

static Range<int64_t> findMyBox() {
    Range<int64_t> box;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = 0;
        box.end[dim] = getOptions().nfour;
    }
    return findMyBox(box, 0, hpxSize(), 0);
}

static void clearX0() {
    Vector<hpx::future<void>> futures;
    if (getOptions().sph) {
        for (auto const& child : hpxChildren()) {
            futures.push_back(hpx::async<ClearX0Action>(child));
        }
    }
    for (int dim = 0; dim < NDIM; dim++) {
        x0Value[dim] = Vector<Fixed32>();
    }
    for (auto& future : futures) {
        future.get();
    }
}

static void initX0() {
    Vector<hpx::future<void>> remoteFutures;
    if (getOptions().sph) {
        for (auto const& child : hpxChildren()) {
            remoteFutures.push_back(hpx::async<InitX0Action>(child));
        }
    }
    Vector<hpx::future<void>> futs;
    double const ninv = 1.0 / getOptions().particlesPerDimension;
    auto const ibox = findMyBox();
    auto const glass = getOptions().useGlass;
    auto const closePack = getOptions().closePack;
    if (!glass) {
        auto box = findMyBox();
        Array<int64_t, NDIM> iValue;
        ParticleInteger index = 0;
        size_t total = box.volume();
        if (closePack) {
            total *= 2;
        }
        for (int dim = 0; dim < NDIM; dim++) {
            x0Value[dim].resize(total);
        }
        for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
            for (iValue[1] = box.begin[1]; iValue[1] != box.end[1]; iValue[1]++) {
                for (iValue[2] = box.begin[2]; iValue[2] != box.end[2]; iValue[2]++) {
                    if (closePack) {
                        for (int dim1 = 0; dim1 < NDIM; dim1++) {
                            double x = iValue[dim1] * ninv;
                            x0Value[dim1][index] = x + (getOptions().sph ? 0.25 * ninv : 0.25);
                            x0Value[dim1][index + 1] = x + 0.75 * ninv;
                        }
                        index += 2;
                    } else {
                        for (int dim1 = 0; dim1 < NDIM; dim1++) {
                            double x = iValue[dim1] * ninv;
                            x0Value[dim1][index] = x;
                        }
                        index++;
                    }
                }
            }
        }
    } else {
        Range<double> box;
        FILE* fp = fopen("glass.bin", "rb");
        if (!fp) {
            THROW_ERROR("Unable to open glass.bin\n");
        }
        int glassParticlesPerDimension;
        int particlesPerDimension = getOptions().particlesPerDimension;
        for (int dim = 0; dim < NDIM; dim++) {
            box.begin[dim] = (double) ibox.begin[dim] / particlesPerDimension;
            box.end[dim] = (double) ibox.end[dim] / particlesPerDimension;
        }
        FREAD(&glassParticlesPerDimension, sizeof(int), 1, fp);
        if (particlesPerDimension % glassParticlesPerDimension != 0) {
            THROW_ERROR("parts_dim (=%i) must be a multiple of the glass dim (%i)\n",
                particlesPerDimension, glassParticlesPerDimension);
        }
        int nf = particlesPerDimension / glassParticlesPerDimension;
        PRINT("glass factor = %i\n", nf);
        size_t numberOfGlassParticles =
            sqr(glassParticlesPerDimension) * glassParticlesPerDimension;
        static Vector<Fixed32> x(numberOfGlassParticles);
        static Vector<Fixed32> y(numberOfGlassParticles);
        static Vector<Fixed32> z(numberOfGlassParticles);
        FREAD(x.data(), sizeof(float), numberOfGlassParticles, fp);
        FREAD(y.data(), sizeof(float), numberOfGlassParticles, fp);
        FREAD(z.data(), sizeof(float), numberOfGlassParticles, fp);
        fclose(fp);

        int const jb = box.begin[XDIM] * nf;
        int const lb = box.begin[ZDIM] * nf;
        int const kb = box.begin[YDIM] * nf;
        int const je = std::min((int) box.end[XDIM] * nf, nf - 1) + 1;
        int const ke = std::min((int) box.end[YDIM] * nf, nf - 1) + 1;
        int const le = std::min((int) box.end[ZDIM] * nf, nf - 1) + 1;
        MutexType mutex;
        Vector<ParticleInteger> counts(je - jb);
        for (int j = jb; j < je; j++) {
            futs.push_back(hpx::async(
                [j, lb, le, jb, kb, ke, numberOfGlassParticles, &mutex, box, nf, &counts]() {
                    ParticleInteger thisCount = 0;
                    for (int k = kb; k < ke; k++) {
                        for (int l = lb; l < le; l++) {
                            for (size_t i = 0; i < numberOfGlassParticles; i++) {
                                double x0 = x[i].toDouble();
                                double y0 = y[i].toDouble();
                                double z0 = z[i].toDouble();
                                double const xValue = (j + x0) / nf;
                                if (xValue >= box.begin[XDIM] && xValue < box.end[XDIM]) {
                                    double const yValue = (k + y0) / nf;
                                    if (yValue >= box.begin[YDIM] && yValue < box.end[YDIM]) {
                                        double const zValue = (l + z0) / nf;
                                        if (zValue >= box.begin[ZDIM] && zValue < box.end[ZDIM]) {
                                            thisCount++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    counts[j - jb] = thisCount;
                }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        futs.resize(0);
        size_t total = 0;
        for (int i = 0; i < counts.size(); i++) {
            auto const thisCount = counts[i];
            counts[i] = total;
            total += thisCount;
        }
        for (int dim = 0; dim < NDIM; dim++) {
            x0Value[dim].resize(total);
        }
        for (int j = jb; j < je; j++) {
            ParticleInteger const indexBegin = counts[j - jb];
            futs.push_back(hpx::async(
                [j, lb, le, kb, ke, numberOfGlassParticles, &mutex, box, nf, indexBegin]() {
                    ParticleInteger iii = indexBegin;
                    for (int k = kb; k < ke; k++) {
                        for (int l = lb; l < le; l++) {
                            for (size_t i = 0; i < numberOfGlassParticles; i++) {
                                double x0 = x[i].toDouble();
                                double y0 = y[i].toDouble();
                                double z0 = z[i].toDouble();
                                double const xValue = (j + x0) / nf;
                                if (xValue >= box.begin[XDIM] && xValue < box.end[XDIM]) {
                                    double const yValue = (k + y0) / nf;
                                    if (yValue >= box.begin[YDIM] && yValue < box.end[YDIM]) {
                                        double const zValue = (l + z0) / nf;
                                        if (zValue >= box.begin[ZDIM] && zValue < box.end[ZDIM]) {
                                            x0Value[XDIM][iii] = xValue;
                                            x0Value[YDIM][iii] = yValue;
                                            x0Value[ZDIM][iii] = zValue;
                                            iii++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }));
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
    particlesResize(x0Value[XDIM].size());
    for (ParticleInteger i = 0; i < x0Value[XDIM].size(); i++) {
        particlesPos(XDIM, i) = x0Value[XDIM][i];
        particlesPos(YDIM, i) = x0Value[YDIM][i];
        particlesPos(ZDIM, i) = x0Value[ZDIM][i];
        particlesVel(XDIM, i) = 0.0;
        particlesVel(YDIM, i) = 0.0;
        particlesVel(ZDIM, i) = 0.0;
        particlesRung(i) = 0;
    }
    if (getOptions().sph) {
        // numberOfParticles counts one species; the two species carry complementary mass fractions.
        ParticleInteger const count = particlesSize();
        sphParticlesResize(count);
        double const gasOffset = (closePack ? 0.25 : 0.5) * ninv;
        for (int dim = 0; dim < NDIM; dim++) {
            x0Value[dim].resize(2 * size_t(count));
            for (ParticleInteger i = 0; i < count; i++) {
                double x = x0Value[dim][i].toDouble() + gasOffset;
                if (x >= 1.0) {
                    x -= 1.0;
                }
                x0Value[dim][count + i] = x;
                particlesPos(dim, count + i) = x;
                particlesVel(dim, count + i) = 0.0;
            }
        }
        double const scale = 1.0 / (1.0 + getOptions().z0);
        auto const unitScales = units::scalesFromOptions(getOptions());
        units::CosmologicalContext const context(scale, getOptions().hubble);
        // The cosmological gas initial condition starts neutral at 1000 K.
        // This fixed composition sets the initial energy; no chemistry is evolved.
        double const neutralNumber = 1.0 - 0.75 * getOptions().y0;
        auto const initialTemperature = units::Temperature::from_value(1000.0);
        units::SpecificEnergy const physicalEnergy =
            1.5 * units::kBoltzmann * units::kParticlesPerGram * neutralNumber * initialTemperature;
        double const storedThermalEnergy = units::toStoredSpecificEnergy(
            units::PhysicalSpecificEnergy(physicalEnergy), unitScales, context)
                                               .value();
        double const density = getOptions().sphMass * getOptions().numberOfParticles;
        float const smoothLength = std::cbrt(
            getOptions().sphNeighborNumber / ((4.0 * M_PI / 3.0) * getOptions().numberOfParticles));
        for (ParticleInteger i = 0; i < count; i++) {
            particlesRung(count + i) = 0;
            sphParticlesThermodynamicState(i).internalEnergy = storedThermalEnergy;
            sphParticlesSupportRadius(i) = smoothLength;
            sphParticlesDensity(i) = density;
        }
    }
    for (auto& future : remoteFutures) {
        future.get();
    }
}

static float zeldovichEnd(int dim, bool initializeParticles, float d1Value, float prefac1) {
    // PRINT( "enter zeldovich_end\n");
    float dxmax = 0.0;
    SpinlockType mutex;
    int64_t const n = getOptions().nfour;
    double const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    auto box = findMyBox();
    bool const glass = getOptions().useGlass;
    for (int dim = 0; dim < NDIM; dim++) {
        box.end[dim] += CLOUD_MAX + (getOptions().sph ? 1 : 0);
        box.begin[dim] += CLOUD_MIN;
    }
    Vector<hpx::future<float>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(
            hpx::async<ZeldovichEndAction>(c, dim, initializeParticles, d1Value, prefac1));
    }
    static Vector<float> yValue;
    yValue = fft3dReadReal(box);
    Array<int64_t, NDIM> iValue;
    float const ninv = 1.0 / n;
    int const numberOfThreads = hpxHardwareConcurrency();
    double const boxSizeInv = 1.0 / boxSize;
    for (ParticleInteger proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(
            hpx::async([proc, numberOfThreads, n, box, boxSizeInv, dim, d1Value, prefac1]() {
                size_t const b = (size_t) proc * particlesSize() / numberOfThreads;
                size_t const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                double dxmax = 0.0;
                for (ParticleInteger i = b; i < e; i++) {
                    double const x0 = x0Value[XDIM][i].toDouble();
                    double const y0 = x0Value[YDIM][i].toDouble();
                    double const z0 = x0Value[ZDIM][i].toDouble();
                    int i1 = x0 * n;
                    int j1 = y0 * n;
                    int k1 = z0 * n;
                    double t[NDIM] = {x0 * n - i1, y0 * n - j1, z0 * n - k1};
                    if (getOptions().sph) {
                        // A positive gas offset may wrap through the upper periodic boundary.
                        if (i1 < box.begin[XDIM] - CLOUD_MIN) {
                            i1 += n;
                        }
                        if (j1 < box.begin[YDIM] - CLOUD_MIN) {
                            j1 += n;
                        }
                        if (k1 < box.begin[ZDIM] - CLOUD_MIN) {
                            k1 += n;
                        }
                    }
                    double dx = 0.0;
                    for (int j = CLOUD_MIN; j <= CLOUD_MAX; j++) {
                        double wtx = cloudWeight(-j + t[XDIM]);
                        for (int k = CLOUD_MIN; k <= CLOUD_MAX; k++) {
                            double wty = cloudWeight(-k + t[YDIM]);
                            for (int l = CLOUD_MIN; l <= CLOUD_MAX; l++) {
                                double wtz = cloudWeight(-l + t[ZDIM]);
                                double wt = wtx * wty * wtz;
                                dx += yValue[box.index(i1 + j, j1 + k, k1 + l)] * wt;
                            }
                        }
                    }
                    dx *= -d1Value * boxSizeInv;
                    double x = particlesPos(dim, i).toDouble();
                    x += dx;
                    if (x >= 1.0) {
                        x -= 1.0;
                    } else if (x < 0.0) {
                        x += 1.0;
                    }
                    particlesPos(dim, i) = x;
                    particlesVel(dim, i) += prefac1 * dx;
                    double const ndim = pow(getOptions().numberOfParticles, 1.0 / NDIM);
                    dxmax = std::max(std::abs(dxmax), dx * ndim);
                }
                return (float) dxmax;
            }));
    }
    for (auto& f : futs) {
        auto const tmp = f.get();
        dxmax = std::max(dxmax, tmp);
    }
    return dxmax;
}

static PowerSpectrumFunction readPowerSpectrum() {
    PowerSpectrumFunction func;
    float const h = getOptions().hubble;
    FILE* fp = fopen("power.init", "rt");
    Vector<std::pair<float, float>> data;
    if (!fp) {
        PRINT("Fatal - unable to open power.init\n");
        abort();
    }
    float kmax = 0.0;
    float kmin = std::numeric_limits<float>::max();
    while (!feof(fp)) {
        float k;
        float p;
        if (fscanf(fp, "%f %f\n", &k, &p) != 0) {
            k *= h;
            p /= (h * h * h);
            func.pValue2.push_back(p);
            kmax = std::max(kmax, (float) k);
            kmin = std::min(kmin, (float) k);
        }
    }
    func.logkmin = std::log(kmin);
    func.logkmax = std::log(kmax);
    func.dlogk = (func.logkmax - func.logkmin) / (func.pValue2.size() - 1);
    fclose(fp);
    func.normalize();
    return func;
}
