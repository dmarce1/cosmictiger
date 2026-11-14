/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gsl/gsl_rng.h>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/boltzmann.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#include <cosmictiger/cosmology/linear_power.hpp>
#include <cosmictiger/cosmology/lpt.hpp>
#include <cosmictiger/cosmology/native_recombination.hpp>
#include <cosmictiger/cosmology/zero_order.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/chemistry/sph.hpp>
#endif
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>
#endif

struct PowerSpectrumFunction {
    linear_power::LogarithmicTable densityAmplitude;
    linear_power::LogarithmicTable derivativeAmplitude;
    double normalizationFactor = 1.0;

    /// Return the normalized density or velocity-derivative transfer amplitude at a
    /// wavenumber.
    double amplitude(double wavenumber, bool velocity) const {
        return std::sqrt(normalizationFactor) *
            (velocity ? derivativeAmplitude(wavenumber) : densityAmplitude(wavenumber));
    }
};

/// Determine the fundamental and largest corner wavenumbers for the configured initial
/// Fourier mesh.
static std::pair<double, double> initialWavenumberRange() {
    auto const& opts = getOptions();
    double const boxSize =
        (units::scalesFromOptions(opts).lengthScale() / units::kMegaparsec).value();
    double const fundamental = 2.0 * std::acos(-1.0) / boxSize;
    double const maximumMode = (opts.initial.fourierSize + 1) / 2;
    double const corner = std::sqrt(3.0) * maximumMode * fundamental;
    return {fundamental, corner};
}

static void zeldovichBegin(int dim1, int dim2, bool velocity = false, int latticePhase = 0);
static float zeldovichEnd(int dim, float, float, bool velocityOnly = false, int latticePhase = 0);
static void twolptInit();
static void twolptDestroy();
static void twolpt(int, int);
static void twolptPhase(int phase);
static void twolptCorrection1();
static void twolptF2delta2Inv();
static void twolptCorrection2(int dim, int latticePhase);

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
static Vector<Position> x0Value[kDimensions];

static void initX0();
static void clearX0();
HPX_PLAIN_ACTION(initX0, InitX0Action);
HPX_PLAIN_ACTION(clearX0, ClearX0Action);

/// Solve the background and linear transfers, normalize the initial spectrum, and write
/// spectrum diagnostics.
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
        std::cout << "  Initializing: cosmological background and linear power spectrum\n"
                  << std::flush;
    }
    ZeroOrderUniverse universe;
    {
        profile::ScopedPhase profilePhase(
            "initialize.transferFunctions.background", hpxRank() == 0);
        createNativeZeroOrderUniverse(&universe, std::max(1.1, 1.1 * initialScale), params);
    }
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
    constexpr size_t kProgressInterval = 32;
    bool const showProgress = hpxRank() == 0;
    std::atomic<size_t> completedModes{0};
    Vector<hpx::future<void>> futures;
    futures.reserve(count);
    if (showProgress) {
        std::cout
            << std::format(
                   "  Einstein\u2013Boltzmann: {:d} modes (one dot per {:d} completed modes)\n    ",
                   count, kProgressInterval)
            << std::flush;
    }
    {
        profile::ScopedPhase profilePhase("initialize.transferFunctions.boltzmann", hpxRank() == 0);
        try {
            for (size_t i = 0; i < count; ++i) {
                wavenumbers[i] = std::exp(lower + span * i / (count - 1));
                futures.push_back(hpx::async(hpx::annotated_function(
                    [&, i]() {
                        transfers[i] = solveBoltzmannTransfers(universe, wavenumbers[i], scales);
                        if (showProgress) {
                            size_t const completed =
                                completedModes.fetch_add(1, std::memory_order_relaxed) + 1;
                            if (completed % kProgressInterval == 0 || completed == count) {
                                std::cout << "." << std::flush;
                            }
                        }
                    },
                    "cosmology.solveBoltzmannMode")));
            }
        } catch (...) {
            hpx::wait_all(futures.begin(), futures.end());
            if (showProgress) {
                std::cout << " failed\n" << std::flush;
            }
            throw;
        }
        // Every task captures the background and output arrays by reference. Join
        // all tasks before propagating any exception and destroying those locals.
        hpx::wait_all(futures.begin(), futures.end());
        try {
            for (auto& future : futures) {
                future.get();
            }
        } catch (...) {
            if (showProgress) {
                std::cout << " failed\n" << std::flush;
            }
            throw;
        }
    }
    if (showProgress) {
        std::cout << std::format(" done ({:d}/{:d} modes)\n", count, count) << std::flush;
    }

    profile::ScopedPhase profilePhase("initialize.transferFunctions.normalize", hpxRank() == 0);
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

/// Build cosmological particle initial conditions using transfer functions, Zel'dovich
/// displacement, and optional 2LPT.
void initialize(double z0) {
    auto const& opts = getOptions();
    profile::checkForInterrupt();
    auto const checkInitializationInterrupt = []() {
        if (profile::interruptRequested()) {
            // Call only after distributed phases have joined. These temporaries
            // are process-local storage populated through completed actions.
            fft3dDestroy();
            twolptDestroy();
            clearX0();
            throw profile::Interrupted();
        }
    };
    if (opts.cosmology.sigma8 == 0.0) {
        // A homogeneous lattice needs no transfer functions. The application
        // still requires a spatially flat background.
        profile::ScopedPhase profilePhase("initialize.lattice");
        initX0();
        clearX0();
        return;
    }
    if (std::abs(opts.cosmology.omegaK) > 1.0e-12) {
        throw std::invalid_argument(
            "The internal Einstein-Boltzmann solver requires a flat universe");
    }
    {
        profile::ScopedPhase profilePhase("initialize.lattice");
        initX0();
    }
    checkInitializationInterrupt();
    int64_t const n = opts.initial.fourierSize;
    float const omegaM = opts.cosmology.omegaM;
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
    int const latticePhases =
        interpolatedInitialConditions(opts.initial, opts.scenario.cosmology.useGlass) ?
        1 :
        (opts.initial.closePack ? 2 : 1) * (opts.hydro.enabled ? 2 : 1);
    for (int dim = 0; dim < kDimensions; dim++) {
        for (int phase = 0; phase < latticePhases; ++phase) {
            {
                profile::ScopedPhase profilePhase("initialize.zeldovich.displacement");
                fft3dInit(n);
                zeldovichBegin(dim, kDimensions, false, phase);
                checkInitializationInterrupt();
                fft3dInvExecute();
                float const dxmax = zeldovichEnd(dim, 1.0f, 0.0f, false, phase);
                terminal::detailPrint("Initial %c displacement maximum: %e\n", 'x' + dim, dxmax);
                fft3dDestroy();
                checkInitializationInterrupt();
            }
            {
                profile::ScopedPhase profilePhase("initialize.zeldovich.velocity");
                fft3dInit(n);
                zeldovichBegin(dim, kDimensions, true, phase);
                fft3dInvExecute();
                zeldovichEnd(dim, 1.0f, linearVelocityFactor, true, phase);
                fft3dDestroy();
                checkInitializationInterrupt();
            }
        }
    }
    if (opts.scenario.cosmology.twoLpt) {
        profile::ScopedPhase profilePhase("initialize.twoLpt");
        twolptInit();

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 1\n");
            twolpt(1, 1);
            twolptPhase(0);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 2\n");
            twolpt(2, 2);
            twolptPhase(1);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 3\n");
            twolpt(0, 0);
            twolptPhase(2);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 4\n");
            twolpt(2, 2);
            twolptPhase(3);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 5\n");
            twolpt(0, 0);
            twolptPhase(4);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 6\n");
            twolpt(1, 1);
            twolptPhase(5);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 7\n");
            twolpt(0, 1);
            twolptPhase(6);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 8\n");
            twolpt(0, 2);
            twolptPhase(7);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.source");
            terminal::detailPrint("2LPT phase 9\n");
            twolpt(1, 2);
            twolptPhase(8);
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.sourceTransform");
            twolptCorrection1();
            twolptF2delta2Inv();
            //		fft3d2silo(false);
            //		system( "cp complex.silo complex.3.silo\n");
            fft3dDestroy();
            checkInitializationInterrupt();
        }

        for (int dim = 0; dim < kDimensions; dim++) {
            terminal::detailPrint(
                "Computing 2LPT correction to %c positions and velocities\n", 'x' + dim);
            for (int phase = 0; phase < latticePhases; ++phase) {
                profile::ScopedPhase profilePhase("initialize.twoLpt.apply");
                twolptCorrection2(dim, phase);
                float const dxmax = zeldovichEnd(dim, -d2Value, prefac2, false, phase);
                fft3dDestroy();
                checkInitializationInterrupt();
                terminal::detailPrint("2LPT %c displacement maximum: %e\n", 'x' + dim, dxmax);
            }
        }
        {
            profile::ScopedPhase profilePhase("initialize.twoLpt.cleanup");
            twolptDestroy();
        }
    }
    {
        profile::ScopedPhase profilePhase("initialize.cleanup");
        clearX0();
    }
}

/// Save the Fourier-transformed second-order density source on every locality.
void twolptF2delta2Inv() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptF2delta2InvAction>(c));
    }
    auto const box = fft3dComplexRange();
    delta2Inv = fft3dReadComplex(box);
    hpx::wait_all(futs.begin(), futs.end());
}

/// Load the second-order source into a fresh real mesh and transform it to Fourier space.
void twolptCorrection1() {
    auto const& opts = getOptions();
    int const n = opts.initial.fourierSize;
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

/// Build one Fourier component of the second-order displacement correction and
/// inverse-transform it.
void twolptCorrection2(int dim, int latticePhase) {
    auto const& opts = getOptions();
    int const n = opts.initial.fourierSize;
    if (hpxRank() == 0) {
        fft3dInit(n);
    }
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptCorrection2Action>(c, dim, latticePhase));
    }
    float const boxSize =
        (units::scalesFromOptions(opts).lengthScale() / units::kMegaparsec).value();
    float const factor = std::pow(boxSize, -1.5) * n * n * n;
    auto const box = fft3dComplexRange();
    auto yValue = delta2Inv;
    Array<int64_t, kDimensions> iValue;
    Vector<hpx::future<void>> futs2;
    for (iValue[0] = box.begin[0]; iValue[0] != box.end[0]; iValue[0]++) {
        futs2.push_back(hpx::async(
            hpx::annotated_function(
                [n, box, boxSize, &yValue, dim, factor, latticePhase](
                    Array<int64_t, kDimensions> iValue2) {
                    auto const& opts = getOptions();
                    int const i = (iValue2[0] <= n / 2 ? iValue2[0] : iValue2[0] - n);
                    float const kx = 2.f * std::numbers::pi_v<float> / boxSize * float(i);
                    for (iValue2[1] = box.begin[1]; iValue2[1] != box.end[1]; iValue2[1]++) {
                        int const j = (iValue2[1] <= n / 2 ? iValue2[1] : iValue2[1] - n);
                        float const ky = 2.f * std::numbers::pi_v<float> / boxSize * float(j);
                        for (iValue2[2] = box.begin[2]; iValue2[2] != box.end[2]; iValue2[2]++) {
                            int const k = (iValue2[2] <= n / 2 ? iValue2[2] : iValue2[2] - n);
                            int const i2 = sqr(i, j, k);
                            int64_t const index = box.index(iValue2);
                            if (i2 > 0 && !lpt::isNyquist(iValue2[0], n) &&
                                !lpt::isNyquist(iValue2[1], n) && !lpt::isNyquist(iValue2[2], n)) {
                                float kz = 2.f * std::numbers::pi_v<float> / boxSize * float(k);
                                float k2 = kx * kx + ky * ky + kz * kz;
                                Vector3<float> const kValue = {kx, ky, kz};
                                yValue[index] =
                                    -ComplexFloat(0, 1) * kValue[dim] * yValue[index] / k2;
                                if (interpolatedInitialConditions(
                                        opts.initial, opts.scenario.cosmology.useGlass)) {
                                    yValue[index] *= cloudFilter(kx * boxSize / n);
                                    yValue[index] *= cloudFilter(ky * boxSize / n);
                                    yValue[index] *= cloudFilter(kz * boxSize / n);
                                } else {
                                    yValue[index] *= lpt::latticeShift(iValue2[0], iValue2[1],
                                        iValue2[2], n,
                                        lpt::latticeOffset(latticePhase, opts.initial.closePack));
                                }
                            } else {
                                yValue[index] = ComplexFloat(0, 0);
                            }
                        }
                    }
                },
                "cosmology.twolptCorrection2"),
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

/// Generate deterministic Fourier modes for the requested density/displacement derivative
/// and lattice phase.
void twolptGenerate(int dim1, int dim2, bool velocity = false, int latticePhase = 0) {
    auto const& opts = getOptions();
    int64_t const n = opts.initial.fourierSize;
    float const boxSize =
        (units::scalesFromOptions(opts).lengthScale() / units::kMegaparsec).value();
    float const factor = std::pow(boxSize, -1.5) * n * n * n;
    static auto const power = []() {
        profile::ScopedPhase profilePhase("initialize.transferFunctions", hpxRank() == 0);
        return computePowerSpectrum();
    }();
    auto const box = fft3dComplexRange();
    Vector<ComplexFloat> yValue(box.volume());
    Vector<hpx::future<void>> futures;
    for (int64_t ix = box.begin[0]; ix < box.end[0]; ++ix) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [=, &yValue]() {
                auto const& opts = getOptions();
                for (int64_t iy = box.begin[1]; iy < box.end[1]; ++iy) {
                    for (int64_t iz = box.begin[2]; iz < box.end[2]; ++iz) {
                        auto const random = lpt::gaussianMode(ix, iy, iz, n);
                        int64_t const index = box.index(ix, iy, iz);
                        if (random == ComplexFloat{}) {
                            yValue[index] = {};
                            continue;
                        }
                        auto const wave = [=](int64_t mode) {
                            return 2.f * std::numbers::pi_v<float> / boxSize *
                                float(mode <= n / 2 ? mode : mode - n);
                        };
                        float const kx = wave(ix), ky = wave(iy), kz = wave(iz);
                        float const k2 = kx * kx + ky * ky + kz * kz;
                        ComplexFloat const derivative[kDimensions + 1] = {
                            {0, -kx}, {0, -ky}, {0, -kz}, {1, 0}};
                        yValue[index] = random * float(power.amplitude(std::sqrt(k2), velocity)) *
                            factor * derivative[dim1] * derivative[dim2] / k2;
                        yValue[index] = lpt::interpolationCorrectedDerivative(
                            yValue[index], dim1, dim2, [=, &opts](ComplexFloat value) {
                                if (interpolatedInitialConditions(
                                        opts.initial, opts.scenario.cosmology.useGlass)) {
                                    value *= cloudFilter(kx * boxSize / n);
                                    value *= cloudFilter(ky * boxSize / n);
                                    value *= cloudFilter(kz * boxSize / n);
                                    return value;
                                }
                                return value *
                                    lpt::latticeShift(ix, iy, iz, n,
                                        lpt::latticeOffset(latticePhase, opts.initial.closePack));
                            });
                    }
                }
            },
            "cosmology.twolptGenerate")));
    }
    for (auto& future : futures) {
        future.get();
    }
    fft3dAccumulateComplex(box, yValue);
}

/// Build and inverse-transform a selected second derivative used by the second-order
/// source.
void twolpt(int dim1, int dim2) {
    auto const& opts = getOptions();
    int const n = opts.initial.fourierSize;
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

/// Accumulate one real-space derivative product into the distributed second-order density
/// source.
void twolptPhase(int phase) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptPhaseAction>(c, phase));
    }
    auto const box = fft3dRealRange();
    auto const vol = box.volume();
    auto const thisDelta2Part = fft3dReadReal(box);
    for (int64_t i = 0; i < vol; i++) {
        lpt::accumulateSecondOrderSource(phase, thisDelta2Part[i], delta2Part[i], delta2[i]);
    }

    hpx::wait_all(futs.begin(), futs.end());
}

/// Allocate and zero local second-order source and intermediate arrays across all
/// localities.
void twolptInit() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<TwolptInitAction>(c));
    }
    auto const box = fft3dRealRange();
    delta2.resize(box.volume(), 0.0);
    delta2Part.resize(box.volume(), 0.0);
    hpx::wait_all(futs.begin(), futs.end());
}

/// Release distributed second-order perturbation work arrays.
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

/// Generate the requested Fourier displacement or velocity component across the locality
/// tree.
static void zeldovichBegin(int dim1, int dim2, bool velocity, int latticePhase) {
    Vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<ZeldovichBeginAction>(c, dim1, dim2, velocity, latticePhase));
    }
    twolptGenerate(dim1, dim2, velocity, latticePhase);
    hpx::wait_all(futs.begin(), futs.end());
}

/// Determine this locality's recursively partitioned initial-condition lattice box.
static Range<int64_t> findMyBox(Range<int64_t> box, int begin, int end, int depth) {
    if (end - begin == 1) {
        return box;
    } else {
        int const xdim = depth % kDimensions;
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

/// Determine this locality's recursively partitioned initial-condition lattice box.
static Range<int64_t> findMyBox() {
    auto const& opts = getOptions();
    Range<int64_t> box;
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = 0;
        box.end[dim] = opts.initial.fourierSize;
    }
    return findMyBox(box, 0, hpxSize(), 0);
}

/// Release the saved unperturbed particle coordinates throughout the locality tree.
static void clearX0() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ClearX0Action>(child));
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        x0Value[dim] = Vector<Position>();
    }
    for (auto& future : futures) {
        future.get();
    }
}

/// Construct the unperturbed particle lattice or glass and initialize species-specific
/// state.
static void initX0() {
    Vector<hpx::future<void>> remoteFutures;
    for (auto const& child : hpxChildren()) {
        remoteFutures.push_back(hpx::async<InitX0Action>(child));
    }
    auto const& options = getOptions();
    auto const integerBox = findMyBox();
    bool const interpolate =
        interpolatedInitialConditions(options.initial, options.scenario.cosmology.useGlass);
    bool const identify = options.starFormation.enabled;
    size_t const largestCount =
        std::max(options.run.darkMatterParticleCount, options.run.gasParticleCount);
    if (identify && largestCount > std::numeric_limits<uint64_t>::max() / 2) {
        throw std::overflow_error(
            "Initial particle population exceeds the star-formation ID range");
    }
    std::array<std::vector<Position>, kDimensions> positions;
    std::vector<uint64_t> sourceIds;
    int glassSide = 0;
    size_t glassCount = 0;
    std::array<std::vector<Fixed32>, kDimensions> glassPositions;
    if (options.scenario.cosmology.useGlass) {
        std::ifstream glassFile("glass.bin", std::ios::binary);
        glassFile.read(reinterpret_cast<char*>(&glassSide), sizeof(glassSide));
        if (!glassFile || glassSide <= 0) {
            throw std::ios_base::failure("Unable to read glass.bin dimensions");
        }
        glassCount = particle_count::roundLattice(size_t(glassSide) * glassSide * glassSide).count;
        for (auto& axis : glassPositions) {
            axis.resize(glassCount);
            glassFile.read(reinterpret_cast<char*>(axis.data()), glassCount * sizeof(Fixed32));
        }
        if (!glassFile) {
            throw std::ios_base::failure("Unable to read glass.bin positions");
        }
    }
    auto const appendSpecies = [&](int side, bool gas) {
        if (side == 0) {
            return;
        }
        double const gasOffset = gas ? (options.initial.closePack ? 0.25 : 0.5) / side : 0.0;
        auto const append = [&](std::array<double, kDimensions> position, uint64_t sourceId) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                position[dim] -= std::floor(position[dim]);
                position[dim] = Position(position[dim]).toDouble();
                // Interpolated samples belong to the FFT box containing their
                // actual coordinates. Exact phase samples retain cell ownership.
                if (interpolate &&
                    (position[dim] < double(integerBox.begin[dim]) / options.initial.fourierSize ||
                        position[dim] >=
                            double(integerBox.end[dim]) / options.initial.fourierSize)) {
                    return;
                }
            }
            for (int dim = 0; dim < kDimensions; ++dim) {
                positions[dim].push_back(Position(position[dim]));
            }
            if (identify) {
                sourceIds.push_back(2 * sourceId + (gas ? 2 : 1));
            }
        };
        if (options.scenario.cosmology.useGlass) {
            if (side % glassSide != 0) {
                throw std::invalid_argument(
                    "Resolved particle lattice is not a whole number of glass tiles");
            }
            int const tiles = side / glassSide;
            std::array<std::vector<int>, kDimensions> tileIndices;
            for (int dim = 0; dim < kDimensions; ++dim) {
                int const begin = int(std::floor(
                    (double(integerBox.begin[dim]) / options.initial.fourierSize - gasOffset) *
                    tiles));
                int const end = int(std::ceil(
                    (double(integerBox.end[dim]) / options.initial.fourierSize - gasOffset) *
                    tiles));
                for (int tile = begin; tile < end && tile < begin + tiles; ++tile) {
                    tileIndices[dim].push_back((tile % tiles + tiles) % tiles);
                }
            }
            for (auto const ix : tileIndices[0]) {
                for (auto const iy : tileIndices[1]) {
                    for (auto const iz : tileIndices[2]) {
                        uint64_t const tile = (uint64_t(ix) * tiles + iy) * tiles + iz;
                        for (size_t sample = 0; sample < glassCount; ++sample) {
                            append(
                                {(ix + glassPositions[0][sample].toDouble()) / tiles + gasOffset,
                                    (iy + glassPositions[1][sample].toDouble()) / tiles + gasOffset,
                                    (iz + glassPositions[2][sample].toDouble()) / tiles +
                                        gasOffset},
                                tile * glassCount + sample);
                        }
                    }
                }
            }
        } else {
            int const phases = options.initial.closePack ? 2 : 1;
            for (int phase = 0; phase < phases; ++phase) {
                double const offset = options.initial.closePack ? 0.25 + 0.5 * phase : 0.0;
                double const shiftedOffset = offset + gasOffset * side;
                std::array<int, kDimensions> begin;
                std::array<int, kDimensions> end;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    double const lower =
                        double(integerBox.begin[dim]) * side / options.initial.fourierSize;
                    double const upper =
                        double(integerBox.end[dim]) * side / options.initial.fourierSize;
                    begin[dim] =
                        interpolate ? int(std::floor(lower - shiftedOffset)) - 1 : int(lower);
                    end[dim] = interpolate ? int(std::ceil(upper - shiftedOffset)) + 1 : int(upper);
                }
                for (int ix = begin[0]; ix < end[0] && ix < begin[0] + side; ++ix) {
                    for (int iy = begin[1]; iy < end[1] && iy < begin[1] + side; ++iy) {
                        for (int iz = begin[2]; iz < end[2] && iz < begin[2] + side; ++iz) {
                            uint64_t const cell = (uint64_t((ix % side + side) % side) * side +
                                                      (iy % side + side) % side) *
                                    side +
                                (iz % side + side) % side;
                            append({(ix + shiftedOffset) / side, (iy + shiftedOffset) / side,
                                       (iz + shiftedOffset) / side},
                                cell * phases + phase);
                        }
                    }
                }
            }
        }
    };
    appendSpecies(options.initial.darkMatterLatticeSide, false);
    size_t const darkMatterCount = positions[0].size();
    appendSpecies(options.initial.gasLatticeSide, true);
    size_t const gasCount = positions[0].size() - darkMatterCount;
    if (!interpolate && options.initial.closePack) {
        for (auto const range : {std::pair<size_t, size_t>{0, darkMatterCount},
                 {darkMatterCount, darkMatterCount + gasCount}}) {
            size_t const half = (range.second - range.first) / 2;
            for (auto& axis : positions) {
                auto const copy = axis;
                for (size_t cell = 0; cell < half; ++cell) {
                    axis[range.first + 2 * cell] = copy[range.first + cell];
                    axis[range.first + 2 * cell + 1] = copy[range.first + half + cell];
                }
            }
            if (identify) {
                auto const copy = sourceIds;
                for (size_t cell = 0; cell < half; ++cell) {
                    sourceIds[range.first + 2 * cell] = copy[range.first + cell];
                    sourceIds[range.first + 2 * cell + 1] = copy[range.first + half + cell];
                }
            }
        }
    }
    if (positions[0].size() > size_t(std::numeric_limits<ParticleInteger>::max())) {
        throw std::overflow_error("Local initial particle count exceeds ParticleInteger range");
    }
    particlesResize(darkMatterCount);
#if COSMICTIGER_WITH_SPH
    if (options.hydro.enabled) {
        sphParticlesResize(gasCount);
    }
#endif
    for (int dim = 0; dim < kDimensions; ++dim) {
        x0Value[dim].resize(positions[dim].size());
        for (size_t i = 0; i < positions[dim].size(); ++i) {
            x0Value[dim][i] = positions[dim][i];
            particlesPos(dim, i) = positions[dim][i];
            particlesVel(dim, i) = 0.0;
        }
    }
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        particlesRung(i) = 0;
        if (identify) {
            particlesStarFormationStates[i].id = sourceIds[i];
        }
    }
#if COSMICTIGER_WITH_SPH
    if (options.hydro.enabled) {
        double const scale = 1.0 / (1.0 + options.cosmology.startRedshift);
        auto const unitScales = units::scalesFromOptions(options);
        units::CosmologicalContext const context(scale, options.cosmology.hubble);
        double const neutralNumber = 1.0 - 0.75 * options.cosmology.heliumFraction;
        auto const initialTemperature = units::Temperature::from_value(1000.0);
        units::SpecificEnergy const physicalEnergy =
            1.5 * units::kBoltzmann * units::kParticlesPerGram * neutralNumber * initialTemperature;
        double const storedThermalEnergy = units::toStoredSpecificEnergy(
            units::PhysicalSpecificEnergy(physicalEnergy), unitScales, context)
                                               .value();
        double const density = options.initial.gasMass * options.run.gasParticleCount;
        // h = [m N/(V_D ρ)]^(1/D) with ρ = m N_gas in the unit box; m cancels, so pass unit
        // mass and the particle number density N_gas.
        float const smoothLength = sph_geometry::supportRadius(
            1.0, double(options.run.gasParticleCount), options.hydro.neighborNumber);
        for (ParticleInteger i = 0; i < gasCount; ++i) {
            sphParticlesSetInternalEnergy(i, storedThermalEnergy);
            sphParticlesSmoothingLength(i) = smoothLength;
            sphParticlesDensity(i) = density;
#if COSMICTIGER_ENABLE_CHEMISTRY
            if (options.chemistry.enabled) {
                if (options.chemistry.initializationTemperature > 0.0) {
                    sphParticlesSetInternalEnergy(i, 0.0f);
                }
                sphChemistryInitializeParticle(i, scale);
            }
#endif
        }
    }
#endif
    for (auto& future : remoteFutures) {
        future.get();
    }
}

/// Interpolate the transformed mode onto particles, apply its displacement or velocity, and
/// return maximum displacement.
static float zeldovichEnd(
    int dim, float d1Value, float prefac1, bool velocityOnly, int latticePhase) {
    auto const& opts = getOptions();
    float dxmax = 0.0;
    SpinlockType mutex;
    int64_t const n = opts.initial.fourierSize;
    double const boxSize =
        (units::scalesFromOptions(opts).lengthScale() / units::kMegaparsec).value();
    auto box = findMyBox();
    bool const glass =
        interpolatedInitialConditions(opts.initial, opts.scenario.cosmology.useGlass);
    if (glass) {
        for (int dim = 0; dim < kDimensions; dim++) {
            box.end[dim] += kCloudMax;
            box.begin[dim] += kCloudMin;
        }
    }
    Vector<hpx::future<float>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(
            hpx::async<ZeldovichEndAction>(c, dim, d1Value, prefac1, velocityOnly, latticePhase));
    }
    auto const yValue = fft3dReadReal(box);
    Array<int64_t, kDimensions> iValue;
    float const ninv = 1.0 / n;
    int const numberOfThreads = hpxHardwareConcurrency();
    double const boxSizeInv = 1.0 / boxSize;
    for (ParticleInteger proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, n, box, boxSizeInv, dim, d1Value, prefac1, velocityOnly,
                latticePhase, glass, &yValue]() {
                auto const& opts = getOptions();
                size_t const b = (size_t) proc * particlesSize() / numberOfThreads;
                size_t const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                double dxmax = 0.0;
                for (ParticleInteger i = b; i < e; i++) {
                    double const x0 = x0Value[kXi][i].toDouble();
                    double const y0 = x0Value[kYi][i].toDouble();
                    double const z0 = x0Value[kZi][i].toDouble();
                    double dx = 0.0;
                    if (!glass) {
                        int const sublattices = opts.initial.closePack ? 2 : 1;
                        size_t speciesCount = particlesSize();
#if COSMICTIGER_WITH_SPH
                        if (opts.hydro.enabled) {
                            speciesCount -= sphParticlesSize();
                        }
#endif
                        int const phase =
                            int(i % sublattices) + (size_t(i) >= speciesCount ? sublattices : 0);
                        if (phase != latticePhase) {
                            continue;
                        }
                        double const offset = lpt::latticeOffset(phase, opts.initial.closePack);
                        int64_t const ix = lpt::latticeNode(x0, n, offset);
                        int64_t const iy = lpt::latticeNode(y0, n, offset);
                        int64_t const iz = lpt::latticeNode(z0, n, offset);
                        ALWAYS_ASSERT(ix >= box.begin[0] && ix < box.end[0] && iy >= box.begin[1] &&
                            iy < box.end[1] && iz >= box.begin[2] && iz < box.end[2]);
                        dx = yValue[box.index(ix, iy, iz)];
                    } else {
                        int64_t indices[kDimensions] = {
                            int64_t(x0 * n), int64_t(y0 * n), int64_t(z0 * n)};
                        Vector3<double> const t = {
                            x0 * n - indices[0], y0 * n - indices[1], z0 * n - indices[2]};
                        for (int axis = 0; axis < kDimensions; ++axis) {
                            // Only a gas offset that crosses the periodic upper edge wraps.
                            if (indices[axis] + kCloudMin < box.begin[axis]) {
                                indices[axis] += n;
                            }
                            ALWAYS_ASSERT(indices[axis] + kCloudMin >= box.begin[axis] &&
                                indices[axis] + kCloudMax < box.end[axis]);
                        }
                        for (int j = kCloudMin; j <= kCloudMax; j++) {
                            double const wtx = cloudWeight(-j + t[0]);
                            for (int k = kCloudMin; k <= kCloudMax; k++) {
                                double const wty = cloudWeight(-k + t[1]);
                                for (int l = kCloudMin; l <= kCloudMax; l++) {
                                    double const wtz = cloudWeight(-l + t[2]);
                                    dx += yValue[box.index(
                                              indices[0] + j, indices[1] + k, indices[2] + l)] *
                                        wtx * wty * wtz;
                                }
                            }
                        }
                    }
                    dx *= -d1Value * boxSizeInv;
                    if (!velocityOnly) {
                        double position = particlesPos(dim, i).toDouble() + dx;
                        position -= std::floor(position);
                        particlesPos(dim, i) = position;
                    }
                    particlesVel(dim, i) += prefac1 * dx;
                    double const ndim = pow(opts.initial.numberOfParticles, 1.0 / kDimensions);
                    dxmax = std::max(dxmax, std::abs(dx) * ndim);
                }
                return (float) dxmax;
            },
            "cosmology.zeldovichEnd")));
    }
    for (auto& f : futs) {
        auto const tmp = f.get();
        dxmax = std::max(dxmax, tmp);
    }
    return dxmax;
}
