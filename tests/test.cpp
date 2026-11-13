/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fenv.h>
#include <filesystem>
#include <format>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/math/vector3.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#endif
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_interaction_counts.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/tree.hpp>
#endif
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/drift.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/scenarios/cosmology.hpp>
#include <cosmictiger/scenarios/plummer.hpp>
#endif
#include <tests/application_test.hpp>
#include <tests/test.hpp>

constexpr bool kVerbose = true;

#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_CHEMISTRY
static void nativeChemistryInitializationTest() {
    auto const& options = getOptions();
    if (!options.hydro.enabled || !options.chemistry.enabled ||
        options.run.scenario != "cosmology" || options.checkpoint.readIndex >= 0) {
        throw std::runtime_error(
            "Native chemistry initialization fixture requires fresh cosmological gas");
    }
    DriverParameters state{};
    scenarios::initializeCosmology(state);
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    // Serialize the production initial conditions before any source or gravity
    // step. The runtime checkpoint schedule remains unchanged.
    writeCheckpoint(state);
    std::cout << "Native chemistry initialization checkpoint written at iteration zero\n"
              << std::flush;
}
#endif

#if COSMICTIGER_WITH_GRAVITY
static bool fftFinite(float value) {
    // Keep nonfinite checks effective in the application's fast-math build.
    return (std::bit_cast<uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
}

static void fft1Test() {
    int const n = 100;
    Range<int64_t> box;
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = 0;
        box.end[dim] = n;
    }
    Vector<float> aValue(box.volume());
    for (int i = 0; i < box.volume(); i++) {
        aValue[i] = 0.1 + (double) rand() / RAND_MAX;
    }
    fft3dInit(n);
    fft3dAccumulateReal(box, aValue);
    fft3dExecute();
    fft3dForceReal();
    fft3dInvExecute();
    auto b = fft3dReadReal(box);
    fft3dDestroy();
    double maxError = 0.0;
    for (int i = 0; i < box.volume(); i++) {
        double const err = std::abs(aValue[i] - b[i]) / aValue[i];
        if (!fftFinite(b[i]) || err > 1.0e-5) {
            throw std::runtime_error("FFT random-field round trip exceeded relative tolerance");
        }
        maxError = std::max(maxError, err);
    }
    std::cout << std::format("FFT random-field round trip: maximum relative error {:e}\n", maxError)
              << std::flush;
}

static void fft2Test() {
    int const n = 4;
    Range<int64_t> box;
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = 0;
        box.end[dim] = n;
    }
    Vector<float> aValue(box.volume());
    Array<int64_t, kDimensions> iValue;
    for (iValue[0] = 0; iValue[0] < n; iValue[0]++) {
        for (iValue[1] = 0; iValue[1] < n; iValue[1]++) {
            for (iValue[2] = 0; iValue[2] < n; iValue[2]++) {
                aValue[box.index(iValue)] = cos(std::numbers::pi * iValue[2]);
            }
        }
    }
    fft3dInit(n);
    fft3dAccumulateReal(box, aValue);
    fft3dExecute();
    box.end[kZi] = n / 2 + 1;
    auto b = fft3dReadComplex(box);
    fft3dDestroy();
    auto const checkSpectrum = [&](Vector<ComplexFloat> const& spectrum) {
        double maxError = 0.0;
        for (iValue[0] = 0; iValue[0] < n; iValue[0]++) {
            for (iValue[1] = 0; iValue[1] < n; iValue[1]++) {
                for (iValue[2] = 0; iValue[2] < n / 2 + 1; iValue[2]++) {
                    auto const value = spectrum[box.index(iValue)];
                    double const expected =
                        iValue[0] == 0 && iValue[1] == 0 && iValue[2] == n / 2 ? n * n * n : 0;
                    double const error = std::max(
                        std::abs(double(value.real()) - expected), std::abs(double(value.imag())));
                    if (!fftFinite(value.real()) || !fftFinite(value.imag()) || error > 1.0e-5) {
                        throw std::runtime_error(
                            "FFT cosine spectrum differs from its analytic mode");
                    }
                    maxError = std::max(maxError, error);
                }
            }
        }
        return maxError;
    };
    double maxSpectrumError = checkSpectrum(b);

    // Send complex values through the accumulation actions, including a zero-kz
    // anti-Hermitian perturbation that forceReal must cancel exactly.
    iValue[0] = 1;
    iValue[1] = iValue[2] = 0;
    b[box.index(iValue)] += ComplexFloat(3.0f, 4.0f);
    iValue[0] = n - 1;
    b[box.index(iValue)] += ComplexFloat(-3.0f, 4.0f);
    iValue[0] = 0;
    b[box.index(iValue)] += ComplexFloat(0.0f, 5.0f);
    fft3dInit(n);
    fft3dAccumulateComplex(box, b);
    fft3dForceReal();
    auto const forced = fft3dReadComplex(box);
    fft3dInvExecute();
    box.end[kZi] = n;
    auto const reconstructed = fft3dReadReal(box);
    fft3dDestroy();
    box.end[kZi] = n / 2 + 1;
    maxSpectrumError = std::max(maxSpectrumError, checkSpectrum(forced));
    double maxRoundTripError = 0.0;
    for (size_t index = 0; index < aValue.size(); ++index) {
        double const error = std::abs(double(reconstructed[index]) - aValue[index]);
        if (!fftFinite(reconstructed[index]) || error > 1.0e-5) {
            throw std::runtime_error("FFT cosine reconstruction exceeded absolute tolerance");
        }
        maxRoundTripError = std::max(maxRoundTripError, error);
    }
    std::cout << std::format(
                     "FFT cosine mode: maximum spectrum error {:e}, reconstruction error {:e}\n",
                     maxSpectrumError, maxRoundTripError)
              << std::flush;
}
#endif

static void domainTest() {
    Timer tm;

    tm.start();
    particlesRandomInit();
    tm.stop();
    std::cout << std::format("particles_random_init: {:e} s\n", tm.read()) << std::flush;
    tm.reset();

    tm.start();
    domainsRebound();
    tm.stop();
    std::cout << std::format("domains_rebound: {:e} s\n", tm.read()) << std::flush;
    tm.reset();

    tm.start();
    domainsBegin(0);
    tm.stop();
    std::cout << std::format("domains_begin: {:e} s\n", tm.read()) << std::flush;
    tm.reset();

    tm.start();
    domainsEnd();
    tm.stop();
    std::cout << std::format("domains_end: {:e} s\n", tm.read()) << std::flush;
    tm.reset();
}

void bhTest() {
    /*	int N = 100000;
     int M = 10;
     vector<array<float, NDIM>> x(N);
     vector<array<float, NDIM>> y(M);
     for (int i = 0; i < N; i++) {
     for (int dim = 0; dim < NDIM; dim++) {
     x[i][dim] = rand1();
     }
     }
     for (int i = 0; i < M; i++) {
     for (int dim = 0; dim < NDIM; dim++) {
     y[i][dim] = rand1();
     }
     }
     auto x0 = x;
     //	auto phi0 = direct_evaluate(x0);
     x0 = x;
     timer tm1, tm2;
     tm1.start();
     auto phi1 = bh_evaluate_points(y, x0, false);
     tm1.stop();
     x0 = x;
     tm2.start();

     auto phi2 = bh_evaluate_points(y, x0, true);
     tm2.stop();

     double err_tot = 0.0;
     for (int i = 0; i < phi1.size(); i++) {
     double err = fabs((phi1[i] - phi2[i]) / phi1[i]);

     err_tot += err;
     }
     */
}

#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
void starFormationTest();
void sphSourceTimingTest();
void starFormationParticlesTest();
#endif
void sphChemistryTimestepsTest();
void sphTreeTest();
void sphParentTreeTest();
void sphGhostsTest();
void sphOnlyDiagnosticsTest();
void unitsOutputTest();
void distributedOutputTest();

void rockstarTest();
void bhPotentialTest();
void groupsCatalogueTest();
void groupsCatalogueDistributedTest();
void analysisBackendsTest();

#if COSMICTIGER_WITH_GRAVITY
void fmmTreeTest();
void fmmWalkTest();
#endif

namespace detail {

struct ApplicationTest {
    char const* name;
    void (*run)();
};

ApplicationTest const applicationTests[] = {
#if !COSMICTIGER_WITH_GRAVITY
    {"sphOnlyDiagnostics", sphOnlyDiagnosticsTest},
#endif
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
    {"star_formation", starFormationTest},
    {"sphSourceTiming", sphSourceTimingTest},
    {"starFormationParticles", starFormationParticlesTest},
#endif
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_CHEMISTRY
#if COSMICTIGER_WITH_GRAVITY
    {"nativeChemistryInitialization", nativeChemistryInitializationTest},
#endif
    {"sphChemistryTimesteps", sphChemistryTimestepsTest},
#endif
#if COSMICTIGER_ENABLE_ROCKSTAR
    {"analysisBackends", analysisBackendsTest},
    {"rockstar", rockstarTest},
    {"bhPotential", bhPotentialTest},
    {"groupsCatalogueDistributed", groupsCatalogueDistributedTest},
    {"groupsCatalogue", groupsCatalogueTest},
#endif
#if COSMICTIGER_WITH_GRAVITY
#endif
#if COSMICTIGER_WITH_SPH
    {"unitsOutput", unitsOutputTest},
    {"distributedOutput", distributedOutputTest},
#endif
    {"checkpointState", checkpointStateTest},
#if COSMICTIGER_WITH_GRAVITY
    {"fmmTree", fmmTreeTest},
    {"fmmWalk", fmmWalkTest},
#endif
#if COSMICTIGER_WITH_SPH
    {"sphTree", sphTreeTest},
    {"sphParentTree", sphParentTreeTest},
    {"sphGhosts", sphGhostsTest},
#endif
    {"domain", domainTest},
#if COSMICTIGER_WITH_GRAVITY
    {"fft1", fft1Test},
    {"fft2", fft2Test},
#endif
    {"bh", bhTest},
};

}    // namespace detail

void test(std::string name) {
    for (auto const& entry : detail::applicationTests) {
        if (name != entry.name) {
            continue;
        }
        runApplicationTest(entry.name, entry.run);
        return;
    }
    throw std::invalid_argument(std::format("test {} is not known", name.c_str()));
}
