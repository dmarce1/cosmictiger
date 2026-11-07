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
#if COSMICTIGER_ENABLE_FMM
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
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
#if COSMICTIGER_WITH_GRAVITY
#include <tests/gravity/analytic.hpp>
#endif
#include <tests/test.hpp>

constexpr bool kVerbose = true;

#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_CHEMISTRY
static void nativeChemistryInitializationTest() {
    auto const& options = getOptions();
    if (!options.hydro.enabled || !options.chemistry.enabled ||
        options.run.scenario != "cosmology" || options.checkpoint.readIndex >= 0) {
        THROW_ERROR("Native chemistry initialization fixture requires fresh cosmological gas\n");
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

// 0.7, 0.8
// 0.55, 0.65
// 0.4, 0.5
/*
 float rand_normal() {
 float x = 2.0 * rand1() - 1.0;
 float y = 2.0 * rand1() - 1.0;
 auto normal = expc(cmplx(0, 1) * float(M_PI) * y) * sqrtf(-2.0 * logf(fabsf(x)));
 return normal.real();
 }
 */

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

#if COSMICTIGER_ENABLE_FMM
static void treeTest() {
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

    tm.start();
    TreeCreateParameters tparams(0, 0.7, getOptions().gravity.smoothingLength);
    treeCreate(tparams);
    tm.stop();
    std::cout << std::format("tree_create: {:e} s\n", tm.read()) << std::flush;
    tm.reset();

    tm.start();
    treeDestroy();
    tm.stop();
    std::cout << std::format("tree_destroy: {:e} s\n", tm.read()) << std::flush;
    tm.reset();

    tm.start();
    treeCreate(tparams);
    tm.stop();
    std::cout << std::format("tree_create: {:e} s\n", tm.read()) << std::flush;
    tm.reset();
}

static void reportKickInteractions(double theta);

static void kickTest() {
    Timer tm;

    Timer totalTime;
    double totalFlops = 0.0;
    tm.start();
    if (getOptions().run.scenario == "plummer") {
        scenarios::initializePlummerParticles();
    } else {
        initialize(getOptions().cosmology.startRedshift);
    }
    tm.stop();
    std::cout << std::format("initialize: {:e} s\n", tm.read()) << std::flush;
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
    tm.reset();
    tm.start();
    TreeCreateParameters tparams(
        0, getOptions().gravity.openingAngle, getOptions().gravity.smoothingLength);
    auto sr = treeCreate(tparams);
    tm.stop();
    std::cout << std::format("tree_create: {:e} s\n", tm.read()) << std::flush;
    tm.reset();
    tm.start();
    treeDestroy();
    tm.stop();
    std::cout << std::format("tree_destroy: {:e} s\n", tm.read()) << std::flush;
    totalTime.start();
    tm.reset();
    tm.start();
    treeCreate(tparams);
    tm.stop();
    std::cout << std::format("tree_create: {:e} s\n", tm.read()) << std::flush;
    tm.reset();
    tm.start();
    KickParameters kparams;
    kparams.gpu = true;
    kparams.nodeLoad = 10;
    kparams.minLevel = tparams.minLevel;
    kparams.saveForce = getOptions().analysis.saveForce;
    kparams.gm = getOptions().gravity.gm;
    kparams.gravitySmoothingLength = getOptions().gravity.smoothingLength;
    kparams.eta = getOptions().gravity.timestepEta;
    kparams.a = 1.0;
    kparams.firstCall = true;
    kparams.minRung = 0;
    kparams.t0 = 1.0;
    kparams.theta = getOptions().gravity.openingAngle;
    ExpansionBuffer l;
    l.zero();
    Vector3<Position> pos;
    for (int dim = 0; dim < kDimensions; dim++) {
        pos[dim] = 0.f;
    }
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);
    kickResetInteractionCounts();
    auto kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
    tm.stop();
    std::cout << std::format("tree_kick: {:e} s\n", tm.read()) << std::flush;
    reportKickInteractions(kparams.theta);
    tm.reset();
    tm.start();
    treeDestroy();
    tm.stop();
    std::cout << std::format("tree_destroy: {:e} s\n", tm.read()) << std::flush;
    tm.reset();
    tm.start();
    // drift(1.0, 0.0, 0.0, 0.0, 0.0);
    tm.stop();
    std::cout << std::format("drift: {:e} s\n", tm.read()) << std::flush;
    totalTime.stop();
    std::cout << std::format("avg time per step = {:e}\n", totalTime.read()) << std::flush;
    double const flopsMeasure = totalFlops / totalTime.read() / hpxSize();
    std::cout << std::format("FLOPS / s / locality = {:e}\n", flopsMeasure) << std::flush;
    FILE* fp = fopen("results.txt", "at");
    fprintf(fp, "%i %e %e\n", hpxSize(), totalTime.read(), flopsMeasure);
    fclose(fp);
    kickClearBuffers();
}

// Rank-local CPU walk counters and the net force from the saved accelerations.
static void reportKickInteractions(double theta) {
    auto const counts = kickInteractionCounts();
    std::cout << std::format(
                     "walk theta {:.3f} openLargerCell {:d}: tests {:d} ccDirect {:d} ccEwald {:d} "
                     "cp {:d} pc {:d} ppLeafPairs {:d} ppPairs {:d} opened {:d} deferred {:d}\n",
                     theta, int(getOptions().gravity.openLargerCell), counts.acceptanceTests,
                     counts.cellCellDirect, counts.cellCellEwald, counts.cellParticle,
                     counts.particleCell, counts.particleParticleLeafPairs,
                     counts.particleParticlePairs, counts.sourceOpenings, counts.deferredSources)
              << std::flush;
    std::cout << std::format(
                     "walk theta {:.3f} openLargerCell {:d}: cp at leaf sinks {:d}, cp at internal "
                     "sinks {:d}; pc from leaf sources {:d}, pc from internal sources {:d}\n",
                     theta, int(getOptions().gravity.openLargerCell),
                     counts.cellParticleAtLeafSinks,
                     counts.cellParticle - counts.cellParticleAtLeafSinks,
                     counts.particleCellFromLeafSources,
                     counts.particleCell - counts.particleCellFromLeafSources)
              << std::flush;
    if (getOptions().analysis.saveForce) {
        auto const range = particlesCurrentRange();
        Vector3<double> total{};
        double magnitudeSum = 0.0;
        for (ParticleInteger i = range.first; i < range.second; i++) {
            double squared = 0.0;
            for (int dim = 0; dim < kDimensions; dim++) {
                double const force = particlesMass(i) * particlesGravityAcceleration(dim, i);
                total[dim] += force;
                squared += force * force;
            }
            magnitudeSum += std::sqrt(squared);
        }
        double const residual =
            std::sqrt(total[kXi] * total[kXi] + total[kYi] * total[kYi] + total[kZi] * total[kZi]) /
            std::max(1.0, magnitudeSum);
        std::cout
            << std::format(
                   "walk theta {:.3f} openLargerCell {:d}: rank-local net force residual {:.3e}\n",
                   theta, int(getOptions().gravity.openLargerCell), residual)
            << std::flush;
    }
}

static void forceTest() {
    if (getOptions().hydro.enabled) {
        THROW_ERROR("run.test=force requires hydro.enabled=false; "
                    "use run.test=sphGravity for mixed-particle validation\n");
    }
    Timer tm;
    Timer tmMain;
    constexpr int kNiter = 1;
    tmMain.start();
    auto const readCheck = getOptions().checkpoint.readIndex;
    if (readCheck != -1) {
        readCheckpoint();
    } else if (getOptions().run.scenario == "plummer") {
        // The driver's clustered model, for walks whose cost depends on radius contrast.
        scenarios::initializePlummerParticles();
    } else {
        initialize(getOptions().cosmology.startRedshift);
    }
    // The sweep rebuilds the tree at every angle unless run.treeOpeningAngle is
    // set; then one tree serves every kick angle from there up to 0.95 in steps
    // of 0.025, so the two opening rules can be compared at equal accuracy on
    // the same tree. The tree's Ewald leaf criterion admits every larger angle.
    double const treeTheta = getOptions().run.treeOpeningAngle;
    bool const fixedTree = treeTheta > 0.0;
    Vector<double> thetas;
    if (fixedTree) {
        constexpr double kStep = 0.025;
        for (int k = 0; treeTheta + kStep * k <= 0.95 + 1e-9; k++) {
            thetas.push_back(treeTheta + kStep * k);
        }
    } else {
        for (double const theta : {0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                 0.8, 0.85, 0.9, 0.95}) {
            thetas.push_back(theta);
        }
    }
    auto const buildTree = [](double theta) {
        domainsRebound();
        domainsBegin(0);
        domainsEnd();
        particlesSortByRung(0);
        TreeCreateParameters tparams(0, theta, getOptions().gravity.smoothingLength);
        treeCreate(tparams);
        return tparams;
    };
    TreeCreateParameters tparams(0, treeTheta, getOptions().gravity.smoothingLength);
    if (fixedTree) {
        tparams = buildTree(treeTheta);
        std::cout << std::format("tree built once at theta {:.3f}\n", treeTheta) << std::flush;
    }
    for (int iter = 0; iter < thetas.size(); iter++) {
        tmMain.start();
        if (!fixedTree) {
            tparams = buildTree(thetas[iter]);
        }
        KickParameters kparams;
        kparams.nodeLoad = 10;
        kparams.gpu = true;
        kparams.minLevel = tparams.minLevel;
        kparams.saveForce = getOptions().analysis.saveForce;
        kparams.gm = getOptions().gravity.gm;
        kparams.gravitySmoothingLength = getOptions().gravity.smoothingLength;
        kparams.eta = getOptions().gravity.timestepEta;
        kparams.a = 1.0;
        kparams.firstCall = true;
        kparams.minRung = 0;
        kparams.t0 = 1.0;
        kparams.theta = thetas[iter];
        ExpansionBuffer l;
        l.zero();
        Vector3<Position> pos;
        for (int dim = 0; dim < kDimensions; dim++) {
            pos[dim] = 0.f;
        }
        TreeId rootId;
        rootId.proc = 0;
        rootId.index = 0;
        Vector<TreeId> checklist;
        checklist.push_back(rootId);
        kickResetInteractionCounts();
        Timer tmKick;
        tmKick.start();
        auto kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
        tmKick.stop();
        std::cout << std::format("tree_kick: {:e} s\n", tmKick.read()) << std::flush;
        reportKickInteractions(thetas[iter]);
        if (!fixedTree) {
            treeDestroy(false);
        }
        tmMain.stop();
        tm.reset();
        tm.start();
        auto rc = analyticCompare(1000);
        tm.stop();
        std::cout << std::format("Analytic compare took {:e} s\n", tm.read()) << std::flush;
        std::cout << std::format("{:e} {:e} {:e}\n", thetas[iter], rc.first, rc.second)
                  << std::flush;
        FILE* fp = fopen("force_accuracy.txt", "at");
        fprintf(fp, "%e %e %e\n", thetas[iter], rc.first, rc.second);
        fclose(fp);
    }
    if (fixedTree) {
        treeDestroy(false);
    }

    kickClearBuffers();
    std::cout << std::format("AVERAGE TIME = {:e}\n", tmMain.read() / kNiter) << std::flush;
}

static void bucketTest() {
    int nchecks = 11;
    static char const* checkpoints[] = {"checkpoint.z.48.9", "checkpoint.z.20.0",
        "checkpoint.z.10.5", "checkpoint.z.6.2", "checkpoint.z.4.0", "checkpoint.z.2.6",
        "checkpoint.z.1.7", "checkpoint.z.1.1", "checkpoint.z.0.6", "checkpoint.z.0.3",
        "checkpoint.z.0.1"};
    double const thetas[] = {0.5, 0.65, 0.65, 0.65, 0.65, 0.65, 0.8, 0.8, 0.8, 0.8, 0.8};
    //	for (int ci = 0; ci < nchecks; ci++) {
    //		system("rm -r checkpoint.999999\n");
    //		asprintf(&buffer, "mv %s checkpoint.999999\n", checkpoints[ci]);
    //		system(buffer);
    //		free(buffer);
    for (int bucketSize = 64; bucketSize <= 256; bucketSize += 8) {
        auto opts = getOptions();
        opts.compute.bucketSize = bucketSize;
        opts.checkpoint.readIndex = 999999;
        //			opts.theta = thetas[ci];
        setOptions(opts);
        //	read_checkpoint();
        Timer tm;
        tm.start();
        for (int trial = 0; trial < 10; trial++) {
            particlesRandomInit();
            domainsRebound();
            domainsBegin(0);
            domainsEnd();
            particlesSortByRung(0);
            TreeCreateParameters tparams(
                0, getOptions().gravity.openingAngle, getOptions().gravity.smoothingLength);
            treeCreate(tparams);
            std::cout << "Tree created\n" << std::flush;
            KickParameters kparams;
            kparams.nodeLoad = 10;
            kparams.gpu = true;
            kparams.maxDt = 1.0;
            kparams.minLevel = tparams.minLevel;
            kparams.saveForce = getOptions().analysis.saveForce;
            kparams.gm = getOptions().gravity.gm;
            kparams.gravitySmoothingLength = getOptions().gravity.smoothingLength;
            kparams.eta = getOptions().gravity.timestepEta;
            kparams.a = 1.0;
            kparams.firstCall = true;
            kparams.minRung = 0;
            kparams.t0 = 1.0;
            kparams.doPhi = false;
            kparams.theta = getOptions().gravity.openingAngle;
            kparams.top = true;
            kparams.ascending = kparams.descending = false;
            ExpansionBuffer l;
            l.zero();
            Vector3<Position> pos;
            for (int dim = 0; dim < kDimensions; dim++) {
                pos[dim] = 0.f;
            }
            TreeId rootId;
            rootId.proc = 0;
            rootId.index = 0;
            Vector<TreeId> checklist;
            checklist.push_back(rootId);
            std::cout << "Doing kick\n" << std::flush;
            auto kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
            treeDestroy(false);
        }
        tm.stop();
        FILE* fp = fopen("buckets.txt", "at");
        if (fp == nullptr) {
            THROW_ERROR("Unable to open buckets.txt\n");
        }
        fprintf(fp, "%i %e\n", bucketSize, tm.read());
        fclose(fp);
        /*	}
            asprintf(&buffer, "mv checkpoint.999999 %s\n", checkpoints[ci]);
            system(buffer);
            free(buffer);*/
    }
}

static void speedTest() {
    constexpr int kNchecks = 11;
    static char const* checkpoints[] = {"checkpoint.z.48.9", "checkpoint.z.20.0",
        "checkpoint.z.10.5", "checkpoint.z.6.2", "checkpoint.z.4.0", "checkpoint.z.2.6",
        "checkpoint.z.1.7", "checkpoint.z.1.1", "checkpoint.z.0.6", "checkpoint.z.0.3",
        "checkpoint.z.0.1"};
    double const thetas[] = {0.4, 0.55, 0.55, 0.55, 0.55, 0.55, 0.7, 0.7, 0.7, 0.7, 0.7};
    double const counts[] = {1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    int buckets[kNchecks];
    int bbegin = 96;
    int bend = 168;
    buckets[0] = bbegin;
    for (int i = 6; i < 11; i++) {
        buckets[i] = bend;
    }
    for (int i = 1; i < 6; i++) {
        buckets[i] = (i - 1.0) / 5.0 * (bend - bbegin) + bbegin;
    }
    Timer tm;
    for (int ci = 0; ci < kNchecks; ci++) {
        std::cout << "-----------------------------------------------------------------------------"
                     "-------"
                     "-\n"
                  << std::flush;
        std::cout << std::format("Test {:d} with bucket = {:d}\n", ci, buckets[ci]) << std::flush;
        std::cout << "-----------------------------------------------------------------------------"
                     "-------"
                     "-\n"
                  << std::flush;
        std::filesystem::remove_all("checkpoint.999999");
        std::filesystem::copy(
            checkpoints[ci], "checkpoint.999999", std::filesystem::copy_options::recursive);
        for (int iter = 0; iter < counts[ci]; iter++) {
            auto opts = getOptions();
            opts.compute.bucketSize = buckets[ci];
            opts.checkpoint.readIndex = 999999;
            opts.gravity.openingAngle = thetas[ci];
            setOptions(opts);
            readCheckpoint();
            tm.start();
            domainsRebound();
            domainsBegin(0);
            domainsEnd();
            particlesSortByRung(0);
            TreeCreateParameters tparams(0, thetas[ci], getOptions().gravity.smoothingLength);
            treeCreate(tparams);
            KickParameters kparams;
            kparams.nodeLoad = 10;
            kparams.gpu = true;
            kparams.minLevel = tparams.minLevel;
            kparams.saveForce = getOptions().analysis.saveForce;
            kparams.gm = getOptions().gravity.gm;
            kparams.gravitySmoothingLength = getOptions().gravity.smoothingLength;
            kparams.eta = getOptions().gravity.timestepEta;
            kparams.a = 1.0;
            kparams.firstCall = true;
            kparams.minRung = 0;
            kparams.t0 = 1.0;
            kparams.theta = thetas[ci];
            ExpansionBuffer l;
            l.zero();
            Vector3<Position> pos;
            for (int dim = 0; dim < kDimensions; dim++) {
                pos[dim] = 0.f;
            }
            TreeId rootId;
            rootId.proc = 0;
            rootId.index = 0;
            Vector<TreeId> checklist;
            checklist.push_back(rootId);
            auto kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
            treeDestroy(false);
            drift(1.0, 0.0, 0.0, 0.0, 0.0, 0);
            tm.stop();
        }
    }
    std::cout << std::format("TOTAL TIME = {:e}\n", tm.read()) << std::flush;
}

#endif

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

void sphGravityTest();
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
void starFormationTest();
void sphSourceTimingTest();
void starFormationParticlesTest();
#endif
void sphChemistryTimestepsTest();
void sphTreeTest();
void sphOnlyDiagnosticsTest();
void linearGrowthTest();
void unitsOutputTest();
void distributedOutputTest();
void fmmTreeTest();
void fmmWalkTest();
void meshDomainsTest();
void pmmmTest();
void pmTest();
void p3mTest();
void particleMeshSortTest();
void particleMeshTimestepTest();
void particleMeshDispatchLaunchFailureTest();
void particleMeshDispatchAsyncFailureTest();

void rockstarTest();
void bhPotentialTest();
void groupsCatalogueTest();
void groupsCatalogueDistributedTest();
void analysisBackendsTest();

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
#if COSMICTIGER_ENABLE_FMM
    {"linearGrowth", linearGrowthTest},
    {"fmmWalk", fmmWalkTest},
    {"fmmTree", fmmTreeTest},
#endif
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
    {"meshDomains", meshDomainsTest},
    {"pmmm", pmmmTest},
    {"particleMeshSort", particleMeshSortTest},
    {"particleMeshTimestep", particleMeshTimestepTest},
    {"particleMeshDispatchLaunchFailure", particleMeshDispatchLaunchFailureTest},
    {"particleMeshDispatchAsyncFailure", particleMeshDispatchAsyncFailureTest},
#endif
#if COSMICTIGER_WITH_PARTICLE_MESH
    {"pm", pmTest},
#if !(COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM)
    {"particleMeshSort", particleMeshSortTest},
#endif
#endif
#if COSMICTIGER_WITH_P3M
    {"p3m", p3mTest},
#endif
#if COSMICTIGER_WITH_SPH
    {"unitsOutput", unitsOutputTest},
    {"distributedOutput", distributedOutputTest},
#endif
    {"checkpointState", checkpointStateTest},
#if COSMICTIGER_WITH_SPH
#if COSMICTIGER_ENABLE_FMM
    {"sphGravity", sphGravityTest},
#endif
    {"sphTree", sphTreeTest},
#endif
    {"domain", domainTest},
#if COSMICTIGER_WITH_GRAVITY
    {"fft1", fft1Test},
    {"fft2", fft2Test},
#endif
#if COSMICTIGER_ENABLE_FMM
    {"speed", speedTest},
    {"buckets", bucketTest},
    {"force", forceTest},
    {"kick", kickTest},
    {"tree", treeTest},
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
    THROW_ERROR("test %s is not known\n", name.c_str());
}
