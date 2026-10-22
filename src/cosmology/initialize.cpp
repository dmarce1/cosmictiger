/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gsl/gsl_rng.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/boltzmann.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#include <cosmictiger/cosmology/linear_power.hpp>
#include <cosmictiger/cosmology/native_recombination.hpp>
#include <cosmictiger/cosmology/zero_order.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

struct PowerSpectrumFunction {
    linear_power::LogarithmicTable densityAmplitude;
    linear_power::LogarithmicTable derivativeAmplitude;
    double normalizationFactor = 1.0;

    double amplitude(double wavenumber, bool velocity) const {
        return std::sqrt(normalizationFactor) *
            (velocity ? derivativeAmplitude(wavenumber) : densityAmplitude(wavenumber));
    }
};

static std::pair<double, double> initialWavenumberRange() {
    double const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    double const fundamental = 2.0 * std::acos(-1.0) / boxSize;
    double const maximumMode = (getOptions().initial.fourierSize + 1) / 2;
    double const corner = std::sqrt(3.0) * maximumMode * fundamental;
    return {fundamental, corner};
}

static void zeldovichBegin(int dim1, int dim2, bool velocity = false);
static float zeldovichEnd(int dim, bool, float, float, bool velocityOnly = false);
static void twolptInit();
static void twolptDestroy();
static void twolpt(int, int);
static void twolptPhase(int phase);
static void twolptCorrection1();
static void twolptF2delta2Inv();
static void twolptCorrection2(int dim);

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
static Vector<Position> x0Value[NDIM];

static void initX0();
static void clearX0();
HPX_PLAIN_ACTION(initX0, InitX0Action);
HPX_PLAIN_ACTION(clearX0, ClearX0Action);

PowerSpectrumFunction computePowerSpectrum() {
    auto const& options = getOptions();
    if (std::abs(options.cosmology.omegaK) > 1.0e-12) {
        throw std::invalid_argument(
            "The internal Einstein-Boltzmann solver requires a flat universe");
    }
    double const initialScale = 1.0 / (1.0 + options.cosmology.startRedshift);
    if (!linear_power::detail::isFinite(initialScale) || !(initialScale > 0.0)) {
        throw std::invalid_argument("Invalid initial scale factor for the internal power spectrum");
    }
    CosmicParameters params;
    params.omegaB = options.cosmology.omegaB;
    params.omegaC = options.cosmology.omegaC;
    params.omegaGam = options.cosmology.omegaGamma;
    params.omegaNu = options.cosmology.omegaNu;
    params.y = options.cosmology.heliumFraction;
    params.neff = options.cosmology.nEff;
    params.theta = options.cosmology.thetaCmb;
    params.hubble = options.cosmology.hubble;
    if (hpxRank() == 0) {
        PRINT("  Initializing: cosmological background and linear power spectrum\n");
    }
    ZeroOrderUniverse universe;
    createNativeZeroOrderUniverse(&universe, std::max(1.1, 1.1 * initialScale), params);
    terminal::detailPrint("Cosmological background ready\n");

    auto const [fundamental, corner] = initialWavenumberRange();
    double const minimum = std::min(1.0e-5 * params.hubble, 0.95 * fundamental);
    double const maximum = std::max(25.271 * params.hubble, 1.05 * corner);
    double const lower = std::log(minimum);
    double const span = std::log(maximum) - lower;
    size_t const count = std::max(size_t(1024), size_t(std::ceil(span / 0.015)) + 1);
    std::vector<double> wavenumbers(count);
    std::vector<std::vector<BoltzmannTransfer>> transfers(count);
    std::vector<double> scales = {std::min(initialScale, 1.0)};
    if (initialScale != 1.0) {
        scales.push_back(std::max(initialScale, 1.0));
    }
    size_t const initialIndex = initialScale <= 1.0 ? 0 : 1;
    size_t const todayIndex = initialScale < 1.0 ? 1 : 0;
    Vector<hpx::future<void>> futures;
    futures.reserve(count);
    try {
        for (size_t i = 0; i < count; ++i) {
            wavenumbers[i] = std::exp(lower + span * i / (count - 1));
            futures.push_back(hpx::async([&, i]() {
                transfers[i] = solveBoltzmannTransfers(universe, wavenumbers[i], scales);
            }));
        }
    } catch (...) {
        hpx::wait_all(futures.begin(), futures.end());
        throw;
    }
    // Every task captures the background and output arrays by reference. Join
    // all tasks before propagating any exception and destroying those locals.
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }

    double const baryonFraction = params.omegaB / (params.omegaB + params.omegaC);
    double const conformalHubble = initialScale *
        hubbleFunction(initialScale, params.hubble, params.omegaB + params.omegaC,
            params.omegaGam + params.omegaNu);
    std::vector<double> powerToday(count), baryonPowerToday(count);
    std::vector<double> densityAmplitude(count), derivativeAmplitude(count);
    for (size_t i = 0; i < count; ++i) {
        auto const& today = transfers[i][todayIndex];
        auto const& initial = transfers[i][initialIndex];
        // The overall primordial amplitude is fixed below by sigma8 at z=0.
        double const primordial =
            std::pow(wavenumbers[i], 0.5 * (options.cosmology.spectralIndex - 4.0));
        double const matterToday = primordial *
            linear_power::matterTransfer(today.deltaCdm, today.deltaBaryon, baryonFraction);
        powerToday[i] = matterToday * matterToday;
        baryonPowerToday[i] = sqr(primordial * today.deltaBaryon);
        densityAmplitude[i] = primordial *
            linear_power::matterTransfer(initial.deltaCdm, initial.deltaBaryon, baryonFraction);
        derivativeAmplitude[i] = primordial *
            linear_power::matterDerivative(initial.hPrime, initial.thetaBaryonOverK,
                wavenumbers[i] / conformalHubble, baryonFraction);
    }
    PowerSpectrumFunction spectrum;
    linear_power::LogarithmicTable const todaySpectrum(wavenumbers, powerToday, true);
    spectrum.densityAmplitude = linear_power::LogarithmicTable(wavenumbers, densityAmplitude);
    spectrum.derivativeAmplitude = linear_power::LogarithmicTable(wavenumbers, derivativeAmplitude);
    spectrum.normalizationFactor =
        linear_power::normalization(todaySpectrum, options.cosmology.sigma8, params.hubble);
    if (hpxRank() == 0) {
        terminal::detailPrint(
            "Internal P_cb: sigma8(z=0) = %.8g, initial z = %.8g, normalization = %.8e\n",
            options.cosmology.sigma8, options.cosmology.startRedshift,
            spectrum.normalizationFactor);
        std::ofstream output(outputPath("power.dat"));
        output << std::setprecision(12);
        for (size_t i = 0; i < count; ++i) {
            double const factor = spectrum.normalizationFactor * std::pow(params.hubble, 3);
            // Columns retain k, matter power, baryon power in h/Mpc and (Mpc/h)^3.
            output << wavenumbers[i] / params.hubble << ' ' << factor * powerToday[i] << ' '
                   << factor * baryonPowerToday[i] << '\n';
        }
        if (!output) {
            throw std::runtime_error("Unable to write the internal power spectrum");
        }
    }
    return spectrum;
}

void initialize(double z0) {
    if (std::abs(getOptions().cosmology.omegaK) > 1.0e-12) {
        throw std::invalid_argument(
            "The internal Einstein-Boltzmann solver requires a flat universe");
    }
    initX0();
    int64_t const n = getOptions().initial.fourierSize;
    float const omegaM = getOptions().cosmology.omegaM;
    float const a0 = 1.0 / (z0 + 1.0);
    float const om = omegaM / (omegaM + (a0 * a0 * a0) * (1.0 - omegaM));
    float const hValue = cosmosDadt(a0) / a0;
    float const linearVelocityFactor = hValue * a0 * a0;
    // Native amplitudes already include their linear evolution to a0. The
    // retained EdS 2LPT kernel acts on that field without another growth factor.
    float const d2Value = -3.f / 7.f;
    float const f2 = 2.f * std::pow(om, 6.f / 11.f);
    float const prefac2 = f2 * hValue * a0 * a0;
    terminal::detailPrint("D2 = %e\n", d2Value);
    terminal::detailPrint("Linear velocity factor = %e\n", linearVelocityFactor / a0);
    terminal::detailPrint("prefac2 = %e\n", prefac2 / a0);
    for (int dim = 0; dim < NDIM; dim++) {
        fft3dInit(n);
        zeldovichBegin(dim, NDIM);
        //		fft3d_force_real();
        fft3dInvExecute();
        float const dxmax = zeldovichEnd(dim, dim == 0, 1.0f, 0.0f);
        terminal::detailPrint("Initial %c displacement maximum: %e\n", 'x' + dim, dxmax);
        fft3dDestroy();
        fft3dInit(n);
        zeldovichBegin(dim, NDIM, true);
        fft3dInvExecute();
        zeldovichEnd(dim, false, 1.0f, linearVelocityFactor, true);
        fft3dDestroy();
    }
    if (getOptions().initial.twoLpt) {
        twolptInit();

        terminal::detailPrint("2LPT phase 1\n");
        twolpt(1, 1);
        twolptPhase(0);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 2\n");
        twolpt(2, 2);
        twolptPhase(1);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 3\n");
        twolpt(0, 0);
        twolptPhase(2);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 4\n");
        twolpt(2, 2);
        twolptPhase(3);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 5\n");
        twolpt(0, 0);
        twolptPhase(4);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 6\n");
        twolpt(1, 1);
        twolptPhase(5);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 7\n");
        twolpt(0, 1);
        twolptPhase(6);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 8\n");
        twolpt(0, 2);
        twolptPhase(7);
        fft3dDestroy();

        terminal::detailPrint("2LPT phase 9\n");
        twolpt(1, 2);
        twolptPhase(8);
        fft3dDestroy();

        twolptCorrection1();
        twolptF2delta2Inv();
        //		fft3d2silo(false);
        //		system( "cp complex.silo complex.3.silo\n");
        fft3dDestroy();

        for (int dim = 0; dim < NDIM; dim++) {
            terminal::detailPrint(
                "Computing 2LPT correction to %c positions and velocities\n", 'x' + dim);
            twolptCorrection2(dim);
            float dxmax = zeldovichEnd(dim, false, -d2Value, prefac2);
            fft3dDestroy();
            terminal::detailPrint("2LPT %c displacement maximum: %e\n", 'x' + dim, dxmax);
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
    int const n = getOptions().initial.fourierSize;
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
    int const n = getOptions().initial.fourierSize;
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

void twolptGenerate(int dim1, int dim2, bool velocity = false) {
    int64_t const n = getOptions().initial.fourierSize;
    float const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    float const factor = std::pow(boxSize, -1.5) * n * n * n;
    Vector<ComplexFloat> yValue;
    static auto const power = computePowerSpectrum();
    auto const box = fft3dComplexRange();
    yValue.resize(box.volume());
    Array<int64_t, NDIM> iValue;
    constexpr int kRandInitIters = 8;
    Vector<hpx::future<void>> futs2;
    //	const auto filter = get_options().use_glass || get_options().close_pack;
    bool const filter = (dim1 != NDIM || dim2 != NDIM);
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs2.push_back(hpx::async(
            [n, box, boxSize, &yValue, dim1, dim2, factor, filter, velocity](
                Array<int64_t, NDIM> iValue2) {
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
                                float const amplitude = power.amplitude(k0Value, velocity);
                                yValue[index] = randNormal * amplitude * factor * kValue[dim1] *
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
                        float const amplitude = power.amplitude(k0Value, velocity);
                        yValue[index] =
                            randNormal * amplitude * factor * kValue[dim1] * kValue[dim2] / k2;
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
    int const n = getOptions().initial.fourierSize;
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

static void zeldovichBegin(int dim1, int dim2, bool velocity) {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<ZeldovichBeginAction>(c, dim1, dim2, velocity));
    }
    twolptGenerate(dim1, dim2, velocity);
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
        box.end[dim] = getOptions().initial.fourierSize;
    }
    return findMyBox(box, 0, hpxSize(), 0);
}

static void clearX0() {
    Vector<hpx::future<void>> futures;
    if (getOptions().hydro.enabled) {
        for (auto const& child : hpxChildren()) {
            futures.push_back(hpx::async<ClearX0Action>(child));
        }
    }
    for (int dim = 0; dim < NDIM; dim++) {
        x0Value[dim] = Vector<Position>();
    }
    for (auto& future : futures) {
        future.get();
    }
}

static void initX0() {
    Vector<hpx::future<void>> remoteFutures;
    if (getOptions().hydro.enabled) {
        for (auto const& child : hpxChildren()) {
            remoteFutures.push_back(hpx::async<InitX0Action>(child));
        }
    }
    Vector<hpx::future<void>> futs;
    double const ninv = 1.0 / getOptions().initial.particlesPerDimension;
    auto const ibox = findMyBox();
    auto const glass = getOptions().initial.useGlass;
    auto const closePack = getOptions().initial.closePack;
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
                            x0Value[dim1][index] =
                                x + (getOptions().hydro.enabled ? 0.25 * ninv : 0.25);
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
        int particlesPerDimension = getOptions().initial.particlesPerDimension;
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
        terminal::detailPrint("glass factor = %i\n", nf);
        size_t numberOfGlassParticles =
            sqr(glassParticlesPerDimension) * glassParticlesPerDimension;
        static Vector<Fixed32> x(numberOfGlassParticles);
        static Vector<Fixed32> y(numberOfGlassParticles);
        static Vector<Fixed32> z(numberOfGlassParticles);
        FREAD(x.data(), sizeof(Fixed32), numberOfGlassParticles, fp);
        FREAD(y.data(), sizeof(Fixed32), numberOfGlassParticles, fp);
        FREAD(z.data(), sizeof(Fixed32), numberOfGlassParticles, fp);
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
    if (getOptions().hydro.enabled) {
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
        double const scale = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
        auto const unitScales = units::scalesFromOptions(getOptions());
        units::CosmologicalContext const context(scale, getOptions().cosmology.hubble);
        // Without chemistry, the cosmological gas starts neutral at 1000 K.
        // The enabled network replaces this energy/composition below.
        double const neutralNumber = 1.0 - 0.75 * getOptions().cosmology.heliumFraction;
        auto const initialTemperature = units::Temperature::from_value(1000.0);
        units::SpecificEnergy const physicalEnergy =
            1.5 * units::kBoltzmann * units::kParticlesPerGram * neutralNumber * initialTemperature;
        double const storedThermalEnergy = units::toStoredSpecificEnergy(
            units::PhysicalSpecificEnergy(physicalEnergy), unitScales, context)
                                               .value();
        double const density =
            getOptions().initial.gasMass * getOptions().initial.numberOfParticles;
        float const smoothLength = std::cbrt(getOptions().hydro.neighborNumber /
            ((4.0 * M_PI / 3.0) * getOptions().initial.numberOfParticles));
        for (ParticleInteger i = 0; i < count; i++) {
            particlesRung(count + i) = 0;
            sphParticlesThermodynamicState(i).internalEnergy = storedThermalEnergy;
            sphParticlesSupportRadius(i) = smoothLength;
            sphParticlesDensity(i) = density;
            if (getOptions().chemistry.enabled) {
                sphChemistryInitializeParticle(i, scale);
            }
        }
    }
    for (auto& future : remoteFutures) {
        future.get();
    }
}

static float zeldovichEnd(
    int dim, bool initializeParticles, float d1Value, float prefac1, bool velocityOnly) {
    // PRINT( "enter zeldovich_end\n");
    float dxmax = 0.0;
    SpinlockType mutex;
    int64_t const n = getOptions().initial.fourierSize;
    double const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    auto box = findMyBox();
    bool const glass = getOptions().initial.useGlass;
    for (int dim = 0; dim < NDIM; dim++) {
        box.end[dim] += CLOUD_MAX + (getOptions().hydro.enabled ? 1 : 0);
        box.begin[dim] += CLOUD_MIN;
    }
    Vector<hpx::future<float>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<ZeldovichEndAction>(
            c, dim, initializeParticles, d1Value, prefac1, velocityOnly));
    }
    static Vector<float> yValue;
    yValue = fft3dReadReal(box);
    Array<int64_t, NDIM> iValue;
    float const ninv = 1.0 / n;
    int const numberOfThreads = hpxHardwareConcurrency();
    double const boxSizeInv = 1.0 / boxSize;
    for (ParticleInteger proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(
            [proc, numberOfThreads, n, box, boxSizeInv, dim, d1Value, prefac1, velocityOnly]() {
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
                    if (getOptions().hydro.enabled) {
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
                    if (!velocityOnly) {
                        double x = particlesPos(dim, i).toDouble() + dx;
                        x -= std::floor(x);
                        particlesPos(dim, i) = x;
                    }
                    particlesVel(dim, i) += prefac1 * dx;
                    double const ndim = pow(getOptions().initial.numberOfParticles, 1.0 / NDIM);
                    dxmax = std::max(dxmax, std::abs(dx) * ndim);
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
