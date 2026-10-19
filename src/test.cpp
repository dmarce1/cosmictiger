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

constexpr bool kVerbose = true;

#include <algorithm>
#include <array>
#include <cmath>
#include <fenv.h>

#include <cosmictiger/analytic.hpp>
#include <cosmictiger/bh.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/driver.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hydro.hpp>
#include <cosmictiger/initialize.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_interaction_counts.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sphere.hpp>
#include <cosmictiger/test.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>

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

static void fft1Test() {
    int const n = 100;
    Range<int64_t> box;
    for (int dim = 0; dim < NDIM; dim++) {
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
    fft3dInvExecute();
    auto b = fft3dReadReal(box);
    fft3dDestroy();
    for (int i = 0; i < box.volume(); i++) {
        double const err = std::abs(aValue[i] - b[i]) / aValue[i];
        if (err > 1.0e-5) {
            PRINT("%e %e %e\n", err, aValue[i], b[i]);
        }
    }
}

static void fft2Test() {
    int const n = 4;
    Range<int64_t> box;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = 0;
        box.end[dim] = n;
    }
    Vector<float> aValue(box.volume());
    Array<int64_t, NDIM> iValue;
    for (iValue[0] = 0; iValue[0] < n; iValue[0]++) {
        for (iValue[1] = 0; iValue[1] < n; iValue[1]++) {
            for (iValue[2] = 0; iValue[2] < n; iValue[2]++) {
                aValue[box.index(iValue)] = cos(M_PI * iValue[2]);
            }
        }
    }
    fft3dInit(n);
    fft3dAccumulateReal(box, aValue);
    fft3dExecute();
    box.end[ZDIM] = n / 2 + 1;
    auto b = fft3dReadComplex(box);
    fft3dDestroy();
    for (iValue[0] = 0; iValue[0] < n; iValue[0]++) {
        for (iValue[1] = 0; iValue[1] < n; iValue[1]++) {
            for (iValue[2] = 0; iValue[2] < n / 2 + 1; iValue[2]++) {
                PRINT("%i %i %i %e %e\n", iValue[0], iValue[1], iValue[2],
                    b[box.index(iValue)].real(), b[box.index(iValue)].imag());
            }
        }
    }
}

static void domainTest() {
    Timer tm;

    tm.start();
    particlesRandomInit();
    tm.stop();
    PRINT("particles_random_init: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsRebound();
    tm.stop();
    PRINT("domains_rebound: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsBegin(0);
    tm.stop();
    PRINT("domains_begin: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsEnd();
    tm.stop();
    PRINT("domains_end: %e s\n", tm.read());
    tm.reset();
}

static void treeTest() {
    Timer tm;

    tm.start();
    particlesRandomInit();
    tm.stop();
    PRINT("particles_random_init: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsRebound();
    tm.stop();
    PRINT("domains_rebound: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsBegin(0);
    tm.stop();
    PRINT("domains_begin: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsEnd();
    tm.stop();
    PRINT("domains_end: %e s\n", tm.read());
    tm.reset();

    tm.start();
    TreeCreateParameters tparams(0, 0.7, getOptions().gravity.supportRadius);
    treeCreate(tparams);
    tm.stop();
    PRINT("tree_create: %e s\n", tm.read());
    tm.reset();

    tm.start();
    treeDestroy();
    tm.stop();
    PRINT("tree_destroy: %e s\n", tm.read());
    tm.reset();

    tm.start();
    treeCreate(tparams);
    tm.stop();
    PRINT("tree_create: %e s\n", tm.read());
    tm.reset();
}

static void reportKickInteractions(double theta);
void plummerInit(double r0);

static void kickTest() {
    Timer tm;

    Timer totalTime;
    double totalFlops = 0.0;
    tm.start();
    if (getOptions().initial.plummer) {
        plummerInit(1e-3);
    } else {
        initialize(getOptions().cosmology.startRedshift);
    }
    tm.stop();
    PRINT("initialize: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsRebound();
    tm.stop();
    PRINT("domains_rebound: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsBegin(0);
    tm.stop();
    PRINT("domains_begin: %e s\n", tm.read());
    tm.reset();

    tm.start();
    domainsEnd();
    tm.stop();
    PRINT("domains_end: %e s\n", tm.read());
    tm.reset();
    tm.reset();
    tm.start();
    TreeCreateParameters tparams(
        0, getOptions().gravity.openingAngle, getOptions().gravity.supportRadius);
    auto sr = treeCreate(tparams);
    tm.stop();
    PRINT("tree_create: %e s\n", tm.read());
    tm.reset();
    tm.start();
    treeDestroy();
    tm.stop();
    PRINT("tree_destroy: %e s\n", tm.read());
    totalTime.start();
    tm.reset();
    tm.start();
    treeCreate(tparams);
    tm.stop();
    PRINT("tree_create: %e s\n", tm.read());
    tm.reset();
    tm.start();
    KickParameters kparams;
    kparams.gpu = true;
    kparams.nodeLoad = 10;
    kparams.minLevel = tparams.minLevel;
    kparams.saveForce = getOptions().analysis.saveForce;
    kparams.gm = getOptions().gravity.gm;
    kparams.gravitySupportRadius = getOptions().gravity.supportRadius;
    kparams.eta = getOptions().gravity.timestepEta;
    kparams.a = 1.0;
    kparams.firstCall = true;
    kparams.minRung = 0;
    kparams.t0 = 1.0;
    kparams.theta = getOptions().gravity.openingAngle;
    ExpansionBuffer l;
    l.zero();
    Array<Fixed32, NDIM> pos;
    for (int dim = 0; dim < NDIM; dim++) {
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
    PRINT("tree_kick: %e s\n", tm.read());
    reportKickInteractions(kparams.theta);
    tm.reset();
    tm.start();
    treeDestroy();
    tm.stop();
    PRINT("tree_destroy: %e s\n", tm.read());
    tm.reset();
    tm.start();
    // drift(1.0, 0.0, 0.0, 0.0, 0.0);
    tm.stop();
    PRINT("drift: %e s\n", tm.read());
    totalTime.stop();
    PRINT("avg time per step = %e\n", totalTime.read());
    double const flopsMeasure = totalFlops / totalTime.read() / hpxSize();
    PRINT("FLOPS / s / locality = %e\n", flopsMeasure);
    FILE* fp = fopen("results.txt", "at");
    fprintf(fp, "%i %e %e\n", hpxSize(), totalTime.read(), flopsMeasure);
    fclose(fp);
    KickWorkspace::clearBuffers();
}

DriverParameters readCheckpoint();

// Rank-local CPU walk counters and the net force from the saved accelerations.
static void reportKickInteractions(double theta) {
    auto const counts = kickInteractionCounts();
    PRINT("walk theta %.3f openLargerCell %i: tests %lld ccDirect %lld ccEwald %lld cp %lld pc "
          "%lld ppLeafPairs %lld ppPairs %lld opened %lld deferred %lld\n",
        theta, int(getOptions().gravity.openLargerCell), counts.acceptanceTests,
        counts.cellCellDirect, counts.cellCellEwald, counts.cellParticle, counts.particleCell,
        counts.particleParticleLeafPairs, counts.particleParticlePairs, counts.sourceOpenings,
        counts.deferredSources);
    PRINT("walk theta %.3f openLargerCell %i: cp at leaf sinks %lld, cp at internal sinks %lld; pc "
          "from leaf sources %lld, pc from internal sources %lld\n",
        theta, int(getOptions().gravity.openLargerCell), counts.cellParticleAtLeafSinks,
        counts.cellParticle - counts.cellParticleAtLeafSinks, counts.particleCellFromLeafSources,
        counts.particleCell - counts.particleCellFromLeafSources);
    if (getOptions().analysis.saveForce) {
        auto const range = particlesCurrentRange();
        std::array<double, NDIM> total{};
        double magnitudeSum = 0.0;
        for (ParticleInteger i = range.first; i < range.second; i++) {
            double squared = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                double const force = particlesMass(i) * particlesGravityAcceleration(dim, i);
                total[dim] += force;
                squared += force * force;
            }
            magnitudeSum += std::sqrt(squared);
        }
        double const residual = std::sqrt(total[XDIM] * total[XDIM] + total[YDIM] * total[YDIM] +
                                    total[ZDIM] * total[ZDIM]) /
            std::max(1.0, magnitudeSum);
        PRINT("walk theta %.3f openLargerCell %i: rank-local net force residual %.3e\n", theta,
            int(getOptions().gravity.openLargerCell), residual);
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
    } else if (getOptions().initial.plummer) {
        // The driver's clustered model, for walks whose cost depends on radius contrast.
        plummerInit(1e-3);
    } else {
        initialize(getOptions().cosmology.startRedshift);
    }
    constexpr int kNthetas = 16;
    double const thetas[kNthetas] = {
        0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
    for (int iter = 0; iter < kNthetas; iter++) {
        tmMain.start();
        domainsRebound();
        domainsBegin(0);
        domainsEnd();
        particlesSortByRung(0);
        TreeCreateParameters tparams(0, thetas[iter], getOptions().gravity.supportRadius);
        treeCreate(tparams);
        KickParameters kparams;
        kparams.nodeLoad = 10;
        kparams.gpu = true;
        kparams.minLevel = tparams.minLevel;
        kparams.saveForce = getOptions().analysis.saveForce;
        kparams.gm = getOptions().gravity.gm;
        kparams.gravitySupportRadius = getOptions().gravity.supportRadius;
        kparams.eta = getOptions().gravity.timestepEta;
        kparams.a = 1.0;
        kparams.firstCall = true;
        kparams.minRung = 0;
        kparams.t0 = 1.0;
        kparams.theta = thetas[iter];
        ExpansionBuffer l;
        l.zero();
        Array<Fixed32, NDIM> pos;
        for (int dim = 0; dim < NDIM; dim++) {
            pos[dim] = 0.f;
        }
        TreeId rootId;
        rootId.proc = 0;
        rootId.index = 0;
        Vector<TreeId> checklist;
        checklist.push_back(rootId);
        kickResetInteractionCounts();
        auto kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
        reportKickInteractions(thetas[iter]);
        treeDestroy(false);
        tmMain.stop();
        tm.reset();
        tm.start();
        auto rc = analyticCompare(1000);
        tm.stop();
        PRINT("Analytic compare took %e s\n", tm.read());
        PRINT("%e %e %e\n", thetas[iter], rc.first, rc.second);
        FILE* fp = fopen("force_accuracy.txt", "at");
        fprintf(fp, "%e %e %e\n", thetas[iter], rc.first, rc.second);
        fclose(fp);
    }

    KickWorkspace::clearBuffers();
    PRINT("AVERAGE TIME = %e\n", tmMain.read() / kNiter);
}

static void bucketTest() {
    int nchecks = 11;
    static char* checkpoints[] = {"checkpoint.z.48.9", "checkpoint.z.20.0", "checkpoint.z.10.5",
        "checkpoint.z.6.2", "checkpoint.z.4.0", "checkpoint.z.2.6", "checkpoint.z.1.7",
        "checkpoint.z.1.1", "checkpoint.z.0.6", "checkpoint.z.0.3", "checkpoint.z.0.1"};
    double const thetas[] = {0.5, 0.65, 0.65, 0.65, 0.65, 0.65, 0.8, 0.8, 0.8, 0.8, 0.8};
    //	for (int ci = 0; ci < nchecks; ci++) {
    char* buffer;
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
                0, getOptions().gravity.openingAngle, getOptions().gravity.supportRadius);
            treeCreate(tparams);
            PRINT("Tree created\n");
            KickParameters kparams;
            kparams.nodeLoad = 10;
            kparams.gpu = true;
            kparams.maxDt = 1.0;
            kparams.minLevel = tparams.minLevel;
            kparams.saveForce = getOptions().analysis.saveForce;
            kparams.gm = getOptions().gravity.gm;
            kparams.gravitySupportRadius = getOptions().gravity.supportRadius;
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
            Array<Fixed32, NDIM> pos;
            for (int dim = 0; dim < NDIM; dim++) {
                pos[dim] = 0.f;
            }
            TreeId rootId;
            rootId.proc = 0;
            rootId.index = 0;
            Vector<TreeId> checklist;
            checklist.push_back(rootId);
            PRINT("Doing kick\n");
            auto kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
            treeDestroy(false);
        }
        tm.stop();
        asprintf(&buffer, "buckets.txt");
        FILE* fp = fopen(buffer, "at");
        free(buffer);
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
    static char* checkpoints[] = {"checkpoint.z.48.9", "checkpoint.z.20.0", "checkpoint.z.10.5",
        "checkpoint.z.6.2", "checkpoint.z.4.0", "checkpoint.z.2.6", "checkpoint.z.1.7",
        "checkpoint.z.1.1", "checkpoint.z.0.6", "checkpoint.z.0.3", "checkpoint.z.0.1"};
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
        PRINT("------------------------------------------------------------------------------------"
              "-\n");
        PRINT("Test %i with bucket = %i\n", ci, buckets[ci]);
        PRINT("------------------------------------------------------------------------------------"
              "-\n");
        char* buffer;
        system("rm -r checkpoint.999999\n");
        asprintf(&buffer, "cp -r %s checkpoint.999999\n", checkpoints[ci]);
        system(buffer);
        free(buffer);
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
            TreeCreateParameters tparams(0, thetas[ci], getOptions().gravity.supportRadius);
            treeCreate(tparams);
            KickParameters kparams;
            kparams.nodeLoad = 10;
            kparams.gpu = true;
            kparams.minLevel = tparams.minLevel;
            kparams.saveForce = getOptions().analysis.saveForce;
            kparams.gm = getOptions().gravity.gm;
            kparams.gravitySupportRadius = getOptions().gravity.supportRadius;
            kparams.eta = getOptions().gravity.timestepEta;
            kparams.a = 1.0;
            kparams.firstCall = true;
            kparams.minRung = 0;
            kparams.t0 = 1.0;
            kparams.theta = thetas[ci];
            ExpansionBuffer l;
            l.zero();
            Array<Fixed32, NDIM> pos;
            for (int dim = 0; dim < NDIM; dim++) {
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
    PRINT("TOTAL TIME = %e\n", tm.read());
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
     PRINT("Doing gpu\n");
     auto phi2 = bh_evaluate_points(y, x0, true);
     tm2.stop();
     PRINT("%e %e\n", tm1.read(), tm2.read());
     double err_tot = 0.0;
     for (int i = 0; i < phi1.size(); i++) {
     double err = fabs((phi1[i] - phi2[i]) / phi1[i]);
     PRINT("%e %e %e\n", phi1[i], phi2[i], err);
     err_tot += err;
     }
     PRINT("%e\n", err_tot / phi1.size());*/
}

void sphGravityTest();
void linearGrowthTest();
void unitsOutputTest();
void fmmTreeTest();
void fmmWalkTest();

#ifdef COSMICTIGER_STF_BENCHMARKS
void stfWalkBenchmark();
#endif

void rockstarTest();
void bhPotentialTest();
void groupsCatalogueTest();
void groupsCatalogueDistributedTest();
void analysisBackendsTest();

void test(std::string test) {
    if (test == "analysisBackends") {
        analysisBackendsTest();
        return;
    }
    if (test == "rockstar") {
        rockstarTest();
        return;
    }
    if (test == "bhPotential") {
        bhPotentialTest();
        return;
    }
    if (test == "groupsCatalogueDistributed") {
        groupsCatalogueDistributedTest();
        return;
    }
    if (test == "groupsCatalogue") {
        groupsCatalogueTest();
        return;
    }
#ifdef COSMICTIGER_STF_BENCHMARKS
    if (test == "stfBenchmark") {
        stfWalkBenchmark();
        return;
    }
#endif
    if (test == "linearGrowth") {
        linearGrowthTest();
        return;
    }
    if (test == "fmmWalk") {
        fmmWalkTest();
        return;
    }
    if (test == "fmmTree") {
        fmmTreeTest();
        return;
    }
    if (test == "unitsOutput") {
        unitsOutputTest();
        return;
    }
    if (test == "checkpointState") {
        checkpointStateTest();
        return;
    }
    if (test == "sphGravity") {
        sphGravityTest();
        return;
    }
    if (test == "domain") {
        domainTest();
    } else if (test == "fft1") {
        fft1Test();
    } else if (test == "fft2") {
        fft2Test();
    } else if (test == "speed") {
        speedTest();
    } else if (test == "buckets") {
        bucketTest();
    } else if (test == "force") {
        forceTest();
    } else if (test == "kick") {
        kickTest();
    } else if (test == "tree") {
        treeTest();
    } else if (test == "rt") {
        hydroRtTest();
    } else if (test == "sod") {
        hydroSodTest();
    } else if (test == "blast") {
        hydroBlastTest();
    } else if (test == "helmholtz") {
        hydroHelmholtzTest();
    } else if (test == "bh") {
        bhTest();
    } else if (test == "star") {
        hydroStarTest();
    } else if (test == "disc") {
        hydroDiscTest();
    } else if (test == "sphere") {
#ifdef USE_CUDA
        if (!cudaEnabled()) {
            THROW_ERROR("Sphere surface test requires compute.cuda=true\n");
        }
        sphereSurfaceTest();
#else
        THROW_ERROR("Sphere surface test requires CUDA support\n");
#endif
    } else if (test == "plummer") {
        hydroPlummer();
    } else {
        THROW_ERROR("test %s is not known\n", test.c_str());
    }
}
