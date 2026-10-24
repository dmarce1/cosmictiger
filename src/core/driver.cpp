/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#define SMOOTHLEN_BUFFER 0.21
#define SCALE_DT 0.01

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>

#include <cosmictiger/analysis/group_tree.hpp>
#include <cosmictiger/analysis/groups.hpp>
#include <cosmictiger/analysis/groups_find.hpp>
#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/analysis/power.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/core/time.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cosmology/background.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/particle_mesh.hpp>
#include <cosmictiger/gravity/particle_mesh_kick.hpp>
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/drift.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/scenarios/cosmology.hpp>
#include <cosmictiger/scenarios/glass.hpp>
#include <cosmictiger/scenarios/plummer.hpp>
#include <cosmictiger/scenarios/scenario.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

double domainTime = 0.0;
double sortTime = 0.0;
double kickTime = 0.0;
double driftTime = 0.0;
double flopsPerNode = 1e6;
double flopsPerParticle = 1e5;
bool usedGpu;
static double synchronizedChemistryPressure = 0.0;

struct Timing {
    double numberOfParticles;
    double time;
    Timing() {
        numberOfParticles = time = 0.0;
    }
};

#define MIN_BUCKET 64
#define MAX_BUCKET 192

Vector<Timing> timings;

static void flushTimings(bool reset) {
    auto const theta = getOptions().gravity.openingAngle;
    char* fname;
    asprintf(&fname, "buckets.%.2f.txt", theta);
    FILE* fp = fopen(outputPath(fname).c_str(), "wt");
    free(fname);
    for (int i = 0; i < timings.size(); i++) {
        if (timings[i].numberOfParticles > 0.0) {
            fprintf(fp, "%i %e\n", i, timings[i].numberOfParticles / timings[i].time);
        }
    }
    if (reset) {
        timings = decltype(timings)();
    }
    fclose(fp);
}

void doGroups(int number, double scale, double years) {
    profilerEnter(__FUNCTION__);
    groupsPrepareHistory(number);

    Timer total;
    total.start();
    terminal::detailPrint("Doing groups\n");
    Timer tm;
    tm.start();
    groupTreeCreate();
    tm.stop();
    terminal::detailPrint("group_tree_create = %e\n", tm.read());
    tm.reset();
    tm.start();
    particlesGroupsInit();
    tm.stop();
    terminal::detailPrint("particles_group_init = %e\n", tm.read());
    tm.reset();
    size_t active;
    int iter = 1;
    do {
        TreeId rootId;
        rootId.proc = 0;
        rootId.index = 0;
        Vector<TreeId> checklist;
        checklist.push_back(rootId);
        tm.start();
        active =
            groupsFind(rootId, std::move(checklist), getOptions().analysis.linkingLength).get();
        tm.stop();
        terminal::detailPrint("%i groups_find = %e active = %li\n", iter, tm.read(), active);
        tm.reset();
        particlesIncGroupCacheEpoch();
        groupTreeIncCacheEpoch();
        iter++;
    } while (active > 0);
    tm.start();
    groupTreeDestroy();
    tm.stop();
    terminal::detailPrint("tree cleanup = %e\n", tm.read());
    Timer reduce;
    reduce.start();
    for (int wave = 0; wave < GROUP_WAVES; wave++) {
        tm.start();
        groupsAddParticles(wave, scale);
        tm.stop();
        terminal::detailPrint("%i groups_add_particles %e\n", wave, tm.read());
        tm.reset();
        tm.start();
        groupsReduce(scale);
        tm.stop();
        terminal::detailPrint("groups_reduce %e\n", tm.read());
        tm.reset();
    }
    reduce.stop();
    tm.start();
    groupsCull();
    tm.stop();
    terminal::detailPrint("groups_cull %e\n", tm.read());
    tm.reset();
    tm.start();
    auto ngroups = groupsSave(number, scale, years);
    tm.stop();
    terminal::detailPrint("groups_save %e\n", tm.read());
    total.stop();
    terminal::detailPrint("Reduction time = %e\n", reduce.read());
    terminal::detailPrint("Total time = %e\n", total.read());
    terminal::detailPrint("Group count = %li of %li candidates\n", ngroups.first, ngroups.second);
    particlesGroupsDestroy();
    profilerExit();
}

SphTreeCreateReturn sphStep1(int minrung, bool verbose) {
    verbose = true;
    if (verbose) {
        terminal::detailPrint("Doing SPH step with minrung = %i\n", minrung);
    }
    SphTreeCreateParameters treeParameters;

    Timer timer;
    treeParameters.minimumRung = minrung;
    treeParameters.supportRadiusMultiplier = (1.0 + SMOOTHLEN_BUFFER);
    SphTreeCreateReturn treeResult;

    timer.start();
    if (verbose) {
        terminal::detailPrint("starting sph_tree_create = %e\n", timer.read());
    }
    profilerEnter("sph_tree_create");
    treeResult = sphTreeCreate(treeParameters);
    profilerExit();
    timer.stop();
    if (verbose) {
        terminal::detailPrint(
            "sph_tree_create time = %e %i\n", timer.read(), treeResult.numberOfActiveParticles);
    }
    timer.reset();
    return treeResult;
}

SphRunReturn sphStep2(int minrung, double scale, double tau, double t0, int phase, double adot,
    EnergiesT* energies, bool verbose, int uniformMinimumRung) {
    float inverseSoundTimestep;
    float inverseViscosityTimestep;
    float inverseAccelerationTimestep;
    verbose = true;
    if (verbose) {
        terminal::detailPrint("Doing SPH step with minrung = %i\n", minrung);
    }
    bool const chemistry = getOptions().chemistry.enabled;
    double const chemistryHalfStep =
        chemistry ? std::ldexp(t0, -1 - getOptions().chemistry.synchronizedRung) : 0.0;
    sphParticlesApplyUpdates(minrung, 0, t0, tau);
    SphRunParameters runParameters;
    if (getOptions().run.scenario == "cosmology") {
        runParameters.maximumTimestep =
            cosmology::expansionTimestep(scale, adot / scale, cosmosD2adt2(scale), SCALE_DT);
    }
    runParameters.maximumTimestep = std::min(runParameters.maximumTimestep,
        std::ldexp(static_cast<float>(t0), -getOptions().run.minRung));
    runParameters.expansionRate = adot;
    runParameters.scaleFactor = scale;
    runParameters.baseTimestep = t0;
    runParameters.minimumRung = minrung;
    bool needsNeighborRefresh;
    SphRunReturn result;

    runParameters.phase = phase;
    SphTreeNeighborParameters neighborParameters;
    neighborParameters.supportRadiusMultiplier = (1.0 + SMOOTHLEN_BUFFER);
    neighborParameters.minimumRung = minrung;
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);

    profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
    neighborParameters.innerSelection = SPH_INTERACTIONS_I;
    neighborParameters.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
    sphTreeNeighbor(neighborParameters, rootId, checklist).get();
    profilerExit();

    runParameters.phase = 0;
    int densityIterations = 0;
    sphParticlesResetConverged();
    do {
        if (++densityIterations > 256) {
            THROW_ERROR("SPH smoothing lengths did not converge after 256 tree rebuilds\n");
        }

        runParameters.runType = SPH_RUN_PREHYDRO1;
        Timer timer;
        timer.start();
        result = sphRun(runParameters, true);
        timer.stop();
        if (verbose) {
            terminal::detailPrint("sph_run(SPH_RUN_PREHYDRO1): tm = %e min_h = %e max_h = %e\n",
                timer.read(), result.minimumSupportRadius, result.maximumSupportRadius);
        }
        timer.reset();
        needsNeighborRefresh = result.needsAnotherPass;
        neighborParameters.supportRadiusMultiplier = (1.0 + SMOOTHLEN_BUFFER);
        neighborParameters.runType = SPH_TREE_NEIGHBOR_BOXES;
        neighborParameters.outerSelection = needsNeighborRefresh ? SPH_SET_ACTIVE : SPH_SET_ALL;
        neighborParameters.innerSelection = SPH_SET_ALL;
        timer.start();
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
        sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
        profilerExit();
        timer.stop();
        terminal::detailPrint("%e\n", timer.read());
        timer.reset();
        timer.start();
        neighborParameters.innerSelection =
            needsNeighborRefresh ? SPH_INTERACTIONS_I : SPH_INTERACTIONS_IJ;
        neighborParameters.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
        sphTreeNeighbor(neighborParameters, rootId, checklist).get();
        profilerExit();
        timer.stop();
        terminal::detailPrint("%e\n", timer.read());
        timer.reset();

        result = SphRunReturn();
    } while (needsNeighborRefresh);
    sphParticlesResetConverged();
    densityIterations = 0;
    do {
        if (++densityIterations > 256) {
            THROW_ERROR("SPH neighbor densities did not converge after 256 tree rebuilds\n");
        }

        runParameters.runType = SPH_RUN_PREHYDRO2;
        Timer timer;
        timer.start();
        result = sphRun(runParameters, true);
        timer.stop();
        if (verbose) {
            terminal::detailPrint("sph_run(SPH_RUN_PREHYDRO2): tm = %e min_h = %e max_h = %e\n",
                timer.read(), result.minimumSupportRadius, result.maximumSupportRadius);
        }
        timer.reset();
        needsNeighborRefresh = result.needsAnotherPass;
        neighborParameters.supportRadiusMultiplier =
            needsNeighborRefresh ? (1.0 + SMOOTHLEN_BUFFER) : 1.01;
        neighborParameters.runType = SPH_TREE_NEIGHBOR_BOXES;
        neighborParameters.outerSelection = SPH_SET_ALL;
        neighborParameters.innerSelection = SPH_SET_ALL;
        timer.start();
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
        sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
        profilerExit();
        timer.stop();
        terminal::detailPrint("%e\n", timer.read());
        timer.reset();
        timer.start();
        neighborParameters.innerSelection = SPH_INTERACTIONS_IJ;
        neighborParameters.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
        sphTreeNeighbor(neighborParameters, rootId, checklist).get();
        profilerExit();
        timer.stop();
        terminal::detailPrint("%e\n", timer.read());
        timer.reset();
        result = SphRunReturn();
    } while (needsNeighborRefresh);
    Timer timer;
    if (chemistry) {
        sphParticlesRefreshChemistryEos(scale);
    }
    runParameters.phase = 0;
    runParameters.runType = SPH_RUN_HYDRO;
    timer.reset();
    timer.start();
    result = sphRun(runParameters, true);
    timer.stop();
    int maxRung = std::max(0, uniformMinimumRung);
    if (verbose) {
        terminal::detailPrint("sph_run(SPH_RUN_HYDRO): tm = %e\n", timer.read());
    }
    timer.reset();

    sphParticlesApplyUpdates(minrung, 1, t0, tau);
    if (chemistry && tau > 0.0) {
        // Complete the previous Strang source half only after the old hydro
        // predictor has been replaced. a at the half-interval midpoint gives
        // second-order conformal-to-physical time conversion.
        energies->heating += sphChemistryPendingHeating() +
            sphChemistryAdvance(scale, scale - 0.5 * adot * chemistryHalfStep, chemistryHalfStep);
        sphChemistryPendingHeating() = 0.0;
    }

    auto eValue = particlesSumEnergies();
    energies->kin = eValue.kin / sqr(scale);
    energies->xmom = eValue.xmom / scale;
    energies->ymom = eValue.ymom / scale;
    energies->zmom = eValue.zmom / scale;
    energies->nmom = eValue.nmom / scale;
    if (minrung == 0) {
        energies->pot = eValue.pot / scale;
    }
    energies->therm = eValue.therm / sqr(scale);

    if (chemistry) {
        synchronizedChemistryPressure = sphChemistryPressureIntegral(scale);
        sphChemistryPendingHeating() =
            sphChemistryAdvance(scale, scale + 0.5 * adot * chemistryHalfStep, chemistryHalfStep);
        sphParticlesRefreshChemistryEos(scale);
    }
    runParameters.phase = 1;
    runParameters.runType = SPH_RUN_HYDRO;
    timer.reset();
    timer.start();
    result = sphRun(runParameters, true);
    inverseSoundTimestep = result.inverseSoundTimestep;
    inverseViscosityTimestep = result.inverseViscosityTimestep;
    inverseAccelerationTimestep = result.inverseAccelerationTimestep;
    timer.stop();
    maxRung = std::max(result.maximumRung, maxRung);
    if (verbose) {
        terminal::detailPrint("sph_run(SPH_RUN_HYDRO): tm = %e max_vsig = %e max_rung = %i, %i\n",
            timer.read(), result.maximumSignalSpeed, result.maximumHydroRung,
            result.maximumGravityRung);
    }
    timer.reset();

    neighborParameters.supportRadiusMultiplier = 1.01;
    neighborParameters.runType = SPH_TREE_NEIGHBOR_BOXES;
    neighborParameters.innerSelection = SPH_SET_ALL;
    neighborParameters.outerSelection = SPH_SET_ACTIVE;
    timer.start();
    profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
    sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
    profilerExit();
    timer.stop();
    timer.reset();
    timer.start();
    neighborParameters.innerSelection = SPH_INTERACTIONS_IJ;
    neighborParameters.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
    profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
    sphTreeNeighbor(neighborParameters, rootId, checklist).get();
    profilerExit();
    timer.stop();
    timer.reset();

    // Uniform PM steps use the global maximum proposal. Neighbor-rung limiting
    // cannot increase that maximum and is unnecessary when every sink is active.
    bool rc = uniformMinimumRung < 0;
    while (rc) {
        runParameters.runType = SPH_RUN_RUNGS;
        timer.start();
        rc = sphRun(runParameters, true).needsAnotherPass;
        //	max_rung = kr.max_rung;
        timer.stop();
        if (verbose) {
            terminal::detailPrint("sph_run(SPH_RUN_RUNGS): tm = %e \n", timer.read());
        }
        timer.reset();
    }
    if (chemistry) {
        if (maxRung > getOptions().chemistry.synchronizedRung) {
            THROW_ERROR("Chemistry synchronized rung %i violates the hydro/gravity timestep bound "
                        "%i; increase chemistry.synchronizedRung\n",
                getOptions().chemistry.synchronizedRung, maxRung);
        }
        sphChemistrySetRungs();
        maxRung = getOptions().chemistry.synchronizedRung;
    }
    if (uniformMinimumRung >= 0) {
        // The dark matter and hydro proposals have both been reduced globally.
        // Set every rung before either species opens its next half-step.
        particleMeshFinishKick(t0, maxRung);
    }
    sphParticlesApplyUpdates(minrung, 2, t0, tau);
    if (verbose) {
        terminal::detailPrint("Completing SPH step with max_rungs = %i, %i\n",
            result.maximumHydroRung, result.maximumGravityRung);
    }

    sphParticlesClearHydroCaches();

    sphTreeDestroy();
    sphParticlesClearCaches();
    sphParticlesResetSemiactive();
    //	PRINT( "%i\n", max_rung);

    float maximumInverseTimestep = std::max(inverseSoundTimestep, inverseViscosityTimestep);
    maximumInverseTimestep = std::max(maximumInverseTimestep, inverseAccelerationTimestep);
    FILE* fp = fopen(outputPath("timestep.txt").c_str(), "at");
    fprintf(fp, "%e %i %e %e %e ", tau, maxRung, inverseSoundTimestep, inverseViscosityTimestep,
        inverseAccelerationTimestep);
    if (maximumInverseTimestep == inverseSoundTimestep) {
        terminal::detailPrint("CFL");
        fprintf(fp, "%s\n", "CFL");
    } else if (maximumInverseTimestep == inverseViscosityTimestep) {
        terminal::detailPrint("VISCOSITY");
        fprintf(fp, "%s\n", "VISCOSITY");
    } else if (maximumInverseTimestep == inverseAccelerationTimestep) {
        terminal::detailPrint("ACCELERATION");
        fprintf(fp, "%s\n", "ACCELERATION");

    } else {
        ALWAYS_ASSERT(false);
    }
    terminal::detailPrint(" LIMITED TIMESTEP\n");
    fclose(fp);
    result.maximumRung = maxRung;
    return result;
}

std::pair<KickReturn, TreeCreateReturn> kickStep(int minrung, double scale, double dadt, double t0,
    double theta, bool firstCall, bool fullEval) {
    Timer tm;
    tm.start();
    terminal::detailPrint("domains_begin\n");
    domainsBegin(minrung);
    terminal::detailPrint("domains_end \n");
    domainsEnd();
    tm.stop();
    domainTime += tm.read();
    tm.reset();
    tm.start();
    bool const sph = getOptions().hydro.enabled;
    // ALWAYS_ASSERT(sph_particles_max_smooth_len() != INFINITY);
    TreeCreateParameters tparams(minrung, theta, 0.f);
    terminal::detailPrint("Create tree %i %e\n", minrung, theta);
    profilerEnter("tree_create");
    auto sr = treeCreate(tparams);
    profilerExit();
    terminal::detailPrint("Done with tree\n");

    tm.stop();
    sortTime += tm.read();
    tm.reset();
    tm.start();
    //	PRINT("numberOfActiveParticles = %li\n", sr.numberOfActiveParticles);
    KickParameters kparams;
    if (getOptions().run.scenario == "cosmology") {
        kparams.maxDt =
            cosmology::expansionTimestep(scale, dadt / scale, cosmosD2adt2(scale), SCALE_DT);
    }
    // The ordinary path visits every starting particle, including rung zero,
    // while the chosen step must still honor the configured starting-rung floor.
    kparams.maxDt =
        std::min(kparams.maxDt, std::ldexp(static_cast<float>(t0), -getOptions().run.minRung));
    kparams.htime = false;
    kparams.ascending = kparams.descending = kparams.top = true;
    kparams.doPhi = fullEval;
    kparams.sign = 1.0;
    kparams.nodeLoad = flopsPerNode / flopsPerParticle;
    kparams.gpu = getOptions().compute.cuda;
    usedGpu = kparams.gpu;
    kparams.minLevel = tparams.minLevel;
    kparams.saveForce = getOptions().analysis.saveForce;
    kparams.gm = getOptions().gravity.enabled ? getOptions().gravity.gm : 0.0;
    kparams.eta = getOptions().gravity.timestepEta;
    kparams.gravitySupportRadius = getOptions().gravity.supportRadius;
    kparams.a = scale;
    kparams.firstCall = firstCall;
    kparams.minRung = minrung;
    kparams.fixedRung =
        getOptions().chemistry.enabled ? getOptions().chemistry.synchronizedRung : 0;
    kparams.t0 = t0;
    kparams.theta = theta;
    ExpansionBuffer l;
    l.zero();
    Array<Position, NDIM> pos;
    for (int dim = 0; dim < NDIM; dim++) {
        pos[dim] = 0.f;
    }
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);
    terminal::detailPrint("Do kick\n");
    profilerEnter("kick");
    KickReturn kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
    profilerExit();
    tm.stop();
    kickTime += tm.read();

    treeDestroy();
    particlesCacheFree();
    terminal::detailPrint("kick done\n");
    return std::make_pair(kr, sr);
}

std::pair<KickReturn, TreeCreateReturn> advanceForces(int minRung, double scale,
    double scaleDerivative, double time, double baseTimestep, double theta, bool firstCall,
    bool fullEvaluation, EnergiesT& energies) {
    std::pair<KickReturn, TreeCreateReturn> result{};
    bool const particleMesh = getOptions().gravity.enabled && getOptions().gravity.method == "pm";
    if (particleMesh) {
        Timer timer;
        timer.start();
        // All particles are sources and sinks at every PM step. Migration must
        // precede cell sorting and the SPH tree's capture of gas indices.
        domainsBegin(0);
        domainsEnd();
        timer.stop();
        domainTime += timer.read();
        timer.reset();
        timer.start();
        particleMeshSolve(true);
        double maximumTimestep = std::ldexp(baseTimestep, -getOptions().run.minRung);
        if (getOptions().run.scenario == "cosmology") {
            maximumTimestep = std::min(maximumTimestep,
                cosmology::expansionTimestep(
                    scale, scaleDerivative / scale, cosmosD2adt2(scale), SCALE_DT));
        }
        result.first =
            particleMeshPrepareKick(scale, baseTimestep, maximumTimestep, firstCall, minRung);
        timer.stop();
        kickTime += timer.read();
    } else if (getOptions().gravity.enabled) {
        result = kickStep(
            minRung, scale, scaleDerivative, baseTimestep, theta, firstCall, fullEvaluation);
    } else {
        // Migration must finish before the SPH tree captures gas particle indices.
        domainsBegin(minRung);
        domainsEnd();
    }
    auto& gravity = result.first;
    if (getOptions().hydro.enabled) {
        int const activeRung = particleMesh ? 0 : minRung;
        sphStep1(activeRung);
        gravity.maxRung = std::max(int(gravity.maxRung),
            sphStep2(activeRung, scale, time, baseTimestep, 1, scaleDerivative, &energies, true,
                particleMesh ? gravity.maxRung : -1)
                .maximumRung);
        if (getOptions().chemistry.enabled &&
            gravity.maxRung > getOptions().chemistry.synchronizedRung) {
            THROW_ERROR("Chemistry synchronized rung violates the gravity timestep bound; "
                        "increase chemistry.synchronizedRung\n");
        }
        if (fullEvaluation || particleMesh) {
            energies.pot = gravity.pot / scale;
        }
        energies.kin += gravity.darkMatterKineticCorrection / sqr(scale);
        energies.xmom += gravity.darkMatterXMomentumCorrection / scale;
        energies.ymom += gravity.darkMatterYMomentumCorrection / scale;
        energies.zmom += gravity.darkMatterZMomentumCorrection / scale;
        energies.nmom += gravity.darkMatterMomentumNormCorrection / scale;
        energies.tckin = energies.kin * sqr(scale);
    } else if (particleMesh) {
        // Preparation left the velocities synchronized for these diagnostics.
        energies.pot = gravity.pot / scale;
        energies.kin = gravity.kin / sqr(scale);
        energies.tckin = gravity.kin;
        energies.xmom = gravity.xmom / scale;
        energies.ymom = gravity.ymom / scale;
        energies.zmom = gravity.zmom / scale;
        energies.nmom = gravity.nmom / scale;
        particleMeshFinishKick(baseTimestep, gravity.maxRung);
    }
    return result;
}

std::pair<KickReturn, TreeCreateReturn> kickStepHierarchical(int& minrung, int maxRung,
    double scale, double adot, double tau, double t0, double theta, EnergiesT* energies,
    int minrung0, bool doPhi, hpx::future<void>* lcFut, bool nocrop, double drag) {
    profilerEnter(__FUNCTION__);
    Timer tm;
    KickReturn kr;
    TreeCreateReturn sr;
    //	minrung = std::max(minrung, 1);
    //	max_rung = std::max(max_rung,minrung);
    // PRINT( "%i %i %i\n", minrung, max_rung, minrung0);
    Vector<int> levels(std::max(maxRung - minrung, 0) + 1);
    int k = 0;
    for (int i = maxRung; i > minrung; i--) {
        levels[k++] = i;
    }
    levels[k++] = minrung;
    bool ascending = true;
    bool top;
    bool clipTop = false;
    //	PRINT("climbing kick ladder\n");
    Timer totalTime;
    totalTime.start();
    double particlesProcessed = 0.0;
    for (int li = 0; li < levels.size(); li++) {
        if (levels[li] == minrung) {
            ascending = false;
            top = true;
        } else {
            top = false;
        }
        if (ascending) {
            //			PRINT("ASCENDING  rung %i\n", levels[li]);
        } else if (top) {
            //			PRINT("AT TOP     rung %i\n", levels[li]);
        } else if (!ascending) {
            //			PRINT("DESCENDING rung %i\n", levels[li]);
        }

        if (!ascending && !top) {
            particlesPushRungs();
        }

        if (ascending || top) {
            tm.reset();
            tm.start();
            domainsBegin(levels[li]);
            domainsEnd();
            tm.stop();
            //			PRINT("Domains took %e\n", tm.read());
        }

        if (!ascending) {
            tm.reset();
            tm.start();
            particlesSortByRung(levels[li]);
            tm.stop();
            //			PRINT("Rung sort took %e\n", tm.read());
        }
        auto rng = particlesCurrentRange();
        particlesProcessed += rng.second - rng.first;
        if (top && minrung == minrung0) {
            auto counts = particlesRungCounts();
            if (counts.size() > minrung0 + 1) {
                auto const total = getOptions().initial.numberOfParticles;
                //				PRINT("Rungs\n");
                for (int i = 0; i < counts.size(); i++) {
                    //					PRINT("%i %li %f %%\n", i, counts[i], 100.0 * counts[i] /
                    // total);
                }
                size_t fast = 0;
                size_t slow = counts[minrung0];
                for (int i = minrung0 + 1; i < counts.size(); i++) {
                    fast += counts[i];
                }
                if (3 * fast > slow) {
                    clipTop = true && !nocrop;
                    //					PRINT("------------------------------------\n");
                    //					PRINT("Setting minimum level to %i\n", minrung + 1);
                    //					PRINT("------------------------------------\n");
                }
            }
        }

        /*		if (ascending || top) {
         tm.reset();
         tm.start();
         const float dt = 0.5 * rung_dt[levels[li]] * t0;
         drift(scale, dt, tau, tau + dt, t0, levels[li]);
         tm.stop();
         terminal::detailPrint("drift = %e\n", tm.read());
         }*/
        TreeCreateParameters tparams(levels[li], theta, 0.f);
        tm.reset();
        tm.start();
        auto thisSr = treeCreate(tparams);
        tm.stop();
        //		PRINT("Tree create took %e\n", tm.read());
        if (top) {
            sr = thisSr;
        }
        KickParameters kparams;
        if (clipTop && levels[li] == minrung0 + 1) {
            kparams.top = true;
        } else {
            kparams.top = top;
        }
        kparams.sign = getOptions().initial.createGlass ? -1.0 : 1.0;
        kparams.ascending = ascending || top;
        if (clipTop && top) {
            kparams.descending = false;
        } else {
            kparams.descending = !ascending || top;
        }
        kparams.velInit = getOptions().initial.plummer;
        kparams.nodeLoad = flopsPerNode / flopsPerParticle;
        kparams.gpu = true;
        kparams.minLevel = tparams.minLevel;
        kparams.saveForce = getOptions().analysis.saveForce;
        kparams.gm = getOptions().gravity.gm;
        kparams.eta = getOptions().gravity.timestepEta;
        kparams.gravitySupportRadius = getOptions().gravity.supportRadius;
        kparams.a = scale;
        kparams.firstCall = tau == 0.0;
        kparams.minRung = levels[li];
        kparams.doPhi = doPhi && top;
        kparams.t0 = t0;
        if (getOptions().run.scenario == "cosmology") {
            kparams.maxDt =
                cosmology::expansionTimestep(scale, adot, cosmosD2adt2(scale), SCALE_DT);
        }
        kparams.theta = theta;
        ExpansionBuffer l;
        l.zero();
        Array<Position, NDIM> pos;
        for (int dim = 0; dim < NDIM; dim++) {
            pos[dim] = 0.f;
        }
        TreeId rootId;
        rootId.proc = 0;
        rootId.index = 0;
        Vector<TreeId> checklist;
        checklist.push_back(rootId);
        tm.reset();
        tm.start();
        if (top) {
            resetGravityCounters();
            setGravityCounterUse(true);
        }
        KickReturn thisKr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
        if (top) {
            setGravityCounterUse(false);
        }
        energies->tckin += thisKr.dkin;
        tm.stop();
        //		PRINT("Kick took %e\n", tm.read());
        if (clipTop && top) {
            particlesSetMinrung(minrung0 + 1);
        }
        if (top) {
            kr = thisKr;
            energies->pot = kr.pot / scale;
            energies->kin = kr.kin / sqr(scale);
            energies->xmom = kr.xmom / scale;
            energies->ymom = kr.ymom / scale;
            energies->zmom = kr.zmom / scale;
            energies->nmom = kr.nmom / scale;
        }
        if (clipTop && top) {
            levels.push_back(minrung0 + 1);
        } else if (!ascending || top) {
            maxRung = thisKr.maxRung;
            kr.maxRung = maxRung;
            if (maxRung > levels[li]) {
                levels.push_back(levels[li] + 1);
            }
        }
        tm.stop();
        if ((!ascending || top) && !(clipTop && top)) {
            tm.reset();
            tm.start();
            double const dt = rungDt[levels[li]] * t0;
            if (lcFut) {
                lcFut->get();
                *lcFut = hpx::make_ready_future();
            }
            double a = scale;
            if (getOptions().run.scenario == "cosmology") {
                a = 1.0 / cosmosAinv(adot, scale, dt);
            }
            //			PRINT( "%e %e %e\n", a, a0, a1);
            drift(a, dt, tau, tau + dt, getOptions().run.numberOfSteps * t0, levels[li], drag);
            tm.stop();
            // PRINT("Drift took %e\n", tm.read());
        }
        treeReset();
        if (ascending && !top) {
            particlesPopRungs();
        }
    }
    //	PRINT("done climbing kick ladder\n");
    if (clipTop) {
        minrung++;
    }
    totalTime.stop();
    int bucketSize = getOptions().compute.bucketSize;
    if (timings.size() <= bucketSize) {
        timings.resize(bucketSize + 1);
    }
    timings[bucketSize].numberOfParticles += particlesProcessed;
    timings[bucketSize].time += totalTime.read();
    profilerExit();
    return std::make_pair(kr, sr);
}

double doPowerSpectrum(int num, double a, double adot) {
    auto const softFilter = [](double kh) {
        if (kh > 0.01) {
            return -15.0 * (3.0 * kh * cos(kh) + ((sqr(kh) - 3.0) * sin(kh))) * pow(kh, -5.0);
        } else {
            return 1.0 - sqr(kh) / 14.0 + sqr(sqr(kh)) / 504.0;
        }
    };
    profilerEnter(__FUNCTION__);
    int const m = 8;
    terminal::detailPrint("Computing power spectrum\n");
    double const h = getOptions().cosmology.hubble;
    auto const glass = getOptions().initial.createGlass;
    double const omegaM = getOptions().cosmology.omegaM;
    double const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    double const n3 = pow((double) getOptions().initial.fourierSize, 3);
    // The historical growth approximation is restricted to an expanding,
    // flat matter/Lambda background. Other models retain measured epoch power.
    auto const& cosmology = getOptions().cosmology;
    bool rescaleGrowth = !glass && adot > 0.0 && omegaM > 0.0 && omegaM <= 1.0 &&
        std::abs(cosmology.omegaK) <= 1.0e-12 && cosmology.omegaLambda >= 0.0;
    double growthRatio = 1.0;
    if (rescaleGrowth) {
        growthRatio = cosmosGrowthFactor(omegaM, a) / cosmosGrowthFactor(omegaM, 1.0);
        if (!std::isfinite(growthRatio) || growthRatio <= 0.0) {
            rescaleGrowth = false;
            growthRatio = 1.0;
        }
    }
    double factor = pow(boxSize, 3) / n3 / (glass ? 1.0 : (n3 * sqr(growthRatio)));
    if (getOptions().initial.closePack) {
        factor *= 0.25;
    }
    double maxpow = 0.0;
    std::string filename = "power." + std::to_string(num) + ".txt";
    FILE* fp = fopen(outputPath(filename).c_str(), "wt");
    if (fp == NULL) {
        THROW_ERROR("Unable to open %s for writing\n", filename.c_str());
    }
    fprintf(fp, "# a=%.17g physical_adot=%.17g growth_rescaling=%s growth_ratio=%.17g\n", a, adot,
        rescaleGrowth ? "flat_lambda_approximation" : "none", growthRatio);
    fprintf(fp,
        "# configured_sigma8=%.17g omega_m=%.17g omega_r=%.17g omega_k=%.17g "
        "omega_lambda=%.17g\n",
        cosmology.sigma8, omegaM, cosmology.omegaR, cosmology.omegaK, cosmology.omegaLambda);
    double kmin = 0.0;
    for (int mfactor = 1; mfactor <= 1; mfactor *= m) {
        auto power0 = powerSpectrumCompute(mfactor);
        double kmax = power0.k.back() * 1.000001;
        //	if (Mfactor != M * M * M) {
        //		kmax = sqrt(power0.k.front() * kmax * M);
        //	}
        double const c1 = h * h * h * factor;
        double const kshot = 2.0 * M_PI * getOptions().initial.particlesPerDimension / boxSize;
        for (int i = 0; i < power0.p.size(); i++) {
            double const k = power0.k[i];
            if (k >= kmin && k <= kmax) {
                double const shotNoise =
                    power0.k[std::min(i + 1, (int) power0.p.size() - 1)] > kshot ?
                    1.0 * factor * n3 :
                    0.0;
                fprintf(fp, "%e %e %e %e %e\n", k / h, (power0.p[i] - power0.perr[i]) * c1,
                    power0.p[i] * c1, (power0.p[i] + power0.perr[i]) * c1, h * h * h * shotNoise);
            }
        }
        kmin = kmax;
    }
    fclose(fp);
    profilerExit();
    return maxpow;
}

void outputTimeFile(DriverParameters const& parameters) {
    profilerEnter(__FUNCTION__);
    double const a0 = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    auto const background = cosmology::simulationBackground(parameters.tauMax);
    bool const expandingBackground = getOptions().run.scenario == "cosmology";
    double const initialTau =
        expandingBackground ? cosmosConformalTime(a0 * 1.0e-6, a0, false) : 0.0;
    double const initialTime = expandingBackground ? cosmosTime(a0 * 1.0e-6, a0) : 0.0;
    int const m = getOptions().run.numberOfSteps;
    FILE* fp = fopen(outputPath("time.txt").c_str(), "wt");
    if (fp == NULL) {
        THROW_ERROR("Unable to open time.txt for writing\n");
    }
    fprintf(
        fp, "%4s %13s %13s %13s %13s\n", "step", "real yrs", "conformal yrs", "scale", "redshift");
    for (int i = 0; i <= m; i++) {
        double const elapsed = parameters.tauMax * (double(i) / m);
        auto const sample = background->sample(elapsed);
        double const a = sample.state.scaleFactor;
        double const z = 1.0 / a - 1.0;
        double const c0 = getOptions().units.timeSeconds / constants::kSpyr;
        fprintf(fp, "%4i %21.14e %21.14e %21.14e %21.14e\n", i,
            (initialTime + sample.physicalTime) * c0, (initialTau + elapsed) * c0, a, z);
    }
    fclose(fp);
    profilerExit();
}

void driver() {
    if (scenarios::definition(getOptions().run.scenario).fixedTime) {
        runFixedScenario();
        return;
    }
    Timer totalTime;
    totalTime.start();
    Timer tmr;
    tmr.start();
    DriverParameters params{};
    Timer buckets50;
    Timer buckets20;
    Timer buckets2;
    hpx::future<void> lcFut = hpx::make_ready_future();
    int buckets[11] = {80, 112, 112, 128, 128, 160, 176, 184, 184, 184, 192};
    double a0 = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    if (getOptions().checkpoint.readIndex != -1) {
        params = readCheckpoint();
        double const coarseInterval = params.tauMax / getOptions().run.numberOfSteps;
        double const endpointTolerance =
            32.0 * std::numeric_limits<double>::epsilon() * params.tauMax;
        if (!std::isfinite(coarseInterval) || !(coarseInterval > 0.0) ||
            !std::isfinite(params.tau) || params.tau < 0.0 ||
            params.tau > params.tauMax + endpointTolerance || params.step < 0) {
            THROW_ERROR("Checkpoint contains an invalid evolution clock\n");
        }
        long double const phase = std::ldexp(static_cast<long double>(params.itime), -64);
        long double const coarsePosition =
            static_cast<long double>(params.tau) / coarseInterval - phase;
        long double const coarseIndex = std::round(coarsePosition);
        long double const clockTolerance = std::max(1.0e-10L,
            256.0L * std::numeric_limits<double>::epsilon() *
                std::max(1.0L, std::abs(coarsePosition)));
        if (std::abs(coarsePosition - coarseIndex) > clockTolerance || coarseIndex < 0.0L ||
            coarseIndex >= std::numeric_limits<int>::max()) {
            THROW_ERROR("Checkpoint conformal time and integer timestep phase are inconsistent; "
                        "retain the original timestep settings\n");
        }
        int const resumedStep = static_cast<int>(coarseIndex);
        if (params.step != resumedStep && !(params.itime == 0 && params.step == resumedStep - 1)) {
            THROW_ERROR("Checkpoint coarse step and conformal time are inconsistent; "
                        "retain the original timestep settings\n");
        }
        // Stopping at a coarse boundary bypasses the outer for-loop increment.
        // Periodic checkpoints taken on the next loop already contain this index.
        // Current headers enforce nominalSteps; the checks above also reject
        // inconsistent timestep settings in older headerless checkpoints.
        if (!cosmology::endTimeReached(params.tau, params.tauMax)) {
            params.step = resumedStep;
        }
        if (getOptions().run.scenario == "cosmology") {
            auto const expected =
                cosmology::simulationBackground(params.tauMax)->sample(params.tau).state;
            double const hubble =
                units::normalizedHubble100(getOptions()) * getOptions().cosmology.hubble;
            double const savedRate = params.a * params.adot / hubble;
            double const expectedRate = expected.conformalDerivative / hubble;
            double const scaleTolerance =
                1.0e-8 * std::max(std::abs(params.a), std::abs(expected.scaleFactor));
            double const rateTolerance =
                1.0e-8 * std::max({1.0, std::abs(savedRate), std::abs(expectedRate)});
            if (!std::isfinite(params.a) || !std::isfinite(savedRate) ||
                std::abs(params.a - expected.scaleFactor) > scaleTolerance ||
                std::abs(savedRate - expectedRate) > rateTolerance) {
                THROW_ERROR("Checkpoint cosmology/background does not match the requested "
                            "initial redshift and cosmological parameters\n");
            }
        }
    } else {
        if (getOptions().run.scenario == "glass") {
            scenarios::initializeGlass(params);
        } else if (getOptions().run.scenario == "plummer") {
            scenarios::initializePlummer(params);
        } else {
            scenarios::initializeCosmology(params);
        }
        if (getOptions().run.endTime > 0.0) {
            params.tauMax = getOptions().run.endTime;
        }
        if (getOptions().analysis.tracers) {
            particlesSetTracers();
        }
        params.step = 0;
        params.flops = 0;
        params.bucketSize = 80;
        terminal::detailPrint("TAU_MAX = %e\n", params.tauMax);
        params.tau = 0.0;
        params.maxRung = 0;
        params.itime = 0;
        params.iter = 0;
        params.minrung0 = 0;
        params.runtime = 0.0;
        params.totalProcessed = 0;
        params.years = getOptions().run.scenario == "cosmology" ?
            cosmosTime(1e-6 * a0, a0) * getOptions().units.timeSeconds / constants::kSpyr :
            0.0;
        auto tmp1 = particlesSumEnergies();
        params.energies.tckin = tmp1.kin;
        //		write_checkpoint(params);
    }
    auto const gadgetFile = getOptions().initial.gadget4Restart;
    if (gadgetFile != "") {
        auto header = particlesReadGadget4(gadgetFile);
        auto const numberOfParticles = header.npartTotal[1];
        auto opts = getOptions();
        opts.units.massG = header.mass[1] * 1e10 / opts.cosmology.hubble;
        opts.units.massG *= constants::kM0;
        opts.units.lengthCm =
            pow(opts.units.massG * (8.0 * M_PI) * numberOfParticles * constants::kG /
                    (3.0 * opts.cosmology.omegaM * sqr(constants::kH0 * opts.cosmology.hubble)),
                1.0 / 3.0);
        opts.units.timeSeconds = opts.units.lengthCm / constants::kC;
        double hValue = constants::kH0 * opts.units.timeSeconds;
        opts.gravity.gm = opts.cosmology.omegaM * 3.0 * sqr(hValue * opts.cosmology.hubble) /
            (8.0 * M_PI) / numberOfParticles;
        opts.cosmology.startRedshift = header.redshift;
        setOptions(opts);
        params.a = 1.0 / (header.redshift + 1.0);
        params.adot = cosmosDadt(params.a);
        params.tauMax = cosmology::configuredDuration(params.a);
        params.years = cosmosTime(1e-6 * params.a, params.a) * getOptions().units.timeSeconds /
            constants::kSpyr;
        terminal::detailPrint("box_size = %e Mpc\n", opts.units.lengthCm / constants::kMpcToCm);
        terminal::detailPrint("code_to_g = %e\n", opts.units.massG);
        terminal::detailPrint("code_to_cm = %e\n", opts.units.lengthCm);
        terminal::detailPrint("code_to_s = %e\n", opts.units.timeSeconds);
    }
    if (getOptions().checkpoint.readIndex == -1) {
        outputTimeFile(params);
    }
    terminal::detailPrint("tau_max = %e\n", params.tauMax);
    auto& years = params.years;
    int& maxRung = params.maxRung;
    auto& a = params.a;
    auto& adot = params.adot;
    auto& tau = params.tau;
    auto& tauMax = params.tauMax;
    auto& energies = params.energies;
    auto& energy0 = params.energy0;
    auto& bucketSize = params.bucketSize;
    auto& itime = params.itime;
    auto& minrung0 = params.minrung0;
    minrung0 = std::max(minrung0, getOptions().run.minRung);
    auto& iter = params.iter;
    int& step = params.step;
    auto& totalProcessed = params.totalProcessed;
    auto& runtime = params.runtime;
    int numberOfSteps = getOptions().run.numberOfSteps;
    double z0 = getOptions().cosmology.startRedshift;
    double dloga = log(z0 + 1.0) / numberOfSteps;
    double pot;
    int thisIter = 0;
    size_t progressRows = 0;
    terminal::section(getOptions().run.scenario.c_str());
    PRINT("  tau is conformal time / coarse interval; dt is in code units.\n");
    double lastTheta = -1.0;
    Timer reset;
    reset.start();
    if (getOptions().lightcone.enabled) {
        lcInit(tau, tauMax);
    }
    double dt = 0.0;
    int jiter = 0;
    if (tau == 0.0) {
        particlesSetMinrung(minrung0);
    }
    auto const checkLc = [&tau, &dt, &tauMax, &a, &iter](bool force) {
        //	profiler_enter("light cone");
        if (force || lcTimeToFlush(tau, tauMax)) {
            Timer tm;
            terminal::detailPrint("Flushing light cone\n");
            tm.start();
            lcBuffer2homes();
            tm.stop();
            terminal::detailPrint("lc_buffer2homes %e\n", tm.read());
            tm.reset();
            tm.start();
            lcParticleBoundaries1();
            tm.stop();
            terminal::detailPrint("lc_particles_boundaries1 %e\n", tm.read());
            tm.reset();
            tm.start();
            double const linkingLength = getOptions().lightcone.linkingLength *
                pow(getOptions().initial.numberOfParticles, -1.0 / NDIM);
            lcFormTrees(tau, linkingLength);
            tm.stop();
            terminal::detailPrint("lc_form_trees %e\n", tm.read());
            tm.reset();
            tm.start();
            size_t cnt;
            lcFindNeighbors();
            terminal::detailPrint("neighbors found\n");
            tm.stop();
            terminal::detailPrint("lc_neighbors_found %e\n", tm.read());
            Timer tm2;
            tm2.start();
            do {
                Timer tm;
                tm.start();
                lcParticleBoundaries2();
                tm.stop();
                terminal::detailPrint("%e ", tm.read());
                tm.reset();
                tm.start();
                cnt = lcFindGroups();
                tm.stop();
                terminal::detailPrint("%e %li\n", cnt, tm.read());
            } while (cnt > 0);
            tm2.stop();
            terminal::detailPrint("loop time = %e\n", tm2.read());
            tm.start();
            lcGroups2homes();
            tm.stop();
            terminal::detailPrint("lc_groups2homes %e\n", tm.read());
            tm.reset();
            tm.start();
            lcParts2groups(a, linkingLength, iter);
            tm.stop();
            terminal::detailPrint("lc_parts2groups %e\n", tm.read());
            tm.reset();
            tm.start();
            lcInit(tau, tauMax);
            tm.stop();
            terminal::detailPrint("lc_init %e\n", tm.read());
            tm.reset();
        }
        //	profiler_exit();
    };
    auto checkpointlist = readCheckpointList();
    double const hsoft0 = getOptions().gravity.supportRadius;
    bool doCheck = false;
    double checkz = 1.0 / a - 1.0;
    int lastCheckpointIteration = getOptions().checkpoint.readIndex >= 0 ? params.iter : -1;
    buckets50.start();
    EnergiesT energies0;
    double glassError = 1.0;
    for (;; step++) {
        if (cosmology::endTimeReached(tau, tauMax)) {
            break;
        }
        //		PRINT("STEP  = %i\n", step);
        double t0 = tauMax / getOptions().run.numberOfSteps;
        do {
            Timer stepTm;
            stepTm.start();
            profilerEnter(__FUNCTION__);
            tmr.stop();
            if ((params.iter > 0 && params.iter != lastCheckpointIteration &&
                    tmr.read() > getOptions().checkpoint.intervalSeconds) ||
                doCheck) {
                totalTime.stop();
                runtime += totalTime.read();
                totalTime.reset();
                totalTime.start();
                terminal::detailPrint(
                    "WRITING CHECKPOINT FOR Z = %e\n", doCheck ? checkz : 1.0 / a - 1.0);
                writeCheckpoint(params);
                lastCheckpointIteration = params.iter;
                tmr.reset();
                if (doCheck && hpxRank() == 0) {
                    detail::checkpointRedshiftAlias(params.iter, checkz);
                }
                doCheck = false;
            }
            tmr.start();
            int minrung = getOptions().chemistry.enabled ? 0 : minRung(itime);
            bool fullEval = minrung == 0;
            if (fullEval) {
                int const number = step;
                if (getOptions().analysis.tracers) {
                    outputTracers(number);
                }
                if (getOptions().analysis.slice) {
                    outputSlice(number, years);
                }
                if (fullEval) {
                    viewOutputViews((tau + 1e-6 * t0) / t0, a, false, years);
                }
                if (getOptions().analysis.views) {
                    Timer tm;
                    tm.start();
                    outputView(number, years);
                    tm.stop();
                    //					PRINT("View %i took %e \n", number, tm.read());
                }
            }
            bool rebound = false;
            if (tau == 0.0) {
                rebound = true;
            } else if (minrung == 0) {
                //				PRINT("Checking imbalance\n");
                double imbalance = domainsGetLoadImbalance();
                rebound = (imbalance > MAX_LOAD_IMBALANCE);
            }
            if (rebound) {
                terminal::detailPrint("Doing rebound\n");
                domainsRebound();
                domainsBegin(0);
                domainsEnd();
                terminal::detailPrint("Rebound done\n");
            }
            double theta;
            double const z = 1.0 / a - 1.0;
            auto opts = getOptions();
            opts.gravity.supportRadius =
                hsoft0;    // Fixed comoving support for every particle type.
            if (getOptions().run.scenario != "cosmology") {
                theta = getOptions().gravity.openingAngle;
            } else {
                /*				if (z > 20.0) {
                 theta = 0.41;
                 } else if (z > 2.0) {
                 theta = 0.51;
                 } else {
                 theta = 0.61;
                 }*/
                if (z > 20.0) {
                    theta = 0.41;
                } else if (z > 2.0) {
                    theta = 0.51;
                } else {
                    theta = 0.61;
                }
            }
            auto const ts = 100 * tau / t0 / getOptions().run.numberOfSteps;
            if (ts <= 10.0) {
                bucketSize = 112;
            } else if (ts < 55.0) {
                bucketSize = 112 + (ts - 10.0) / 45.0 * (184 - 112);
            } else {
                bucketSize = 184;
            }
            // bucket_size = 128;
            opts.compute.bucketSize = bucketSize;
            opts.gravity.openingAngle = theta;
            setOptions(opts);
            lastTheta = theta;
            std::pair<KickReturn, TreeCreateReturn> tmp;
            bool nocrop = false;
            if (int(tau / t0) % 8 == 0 && fullEval) {
                nocrop = true;
                minrung0 = std::max(minrung0 - 1, getOptions().run.minRung);
            }
            int thisMinrung = std::max(minrung, minrung0);
            int om = thisMinrung;
            //				PRINT("MINRUNG0 = %i\n", minrung0);
            //			PRINT( "Doing kick\n");
            if (fullEval) {
                //				kick_workspace::clear_buffers();
                //				tree_destroy(true);
                if (getOptions().analysis.power) {
                    doPowerSpectrum(step, a, adot);
                }
#ifndef CHECK_MUTUAL_SORT
                if (getOptions().analysis.groups) {
                    doGroups(step, a, years);
                }
#endif
            }
            if (getOptions().initial.useGlass && minrung == minrung0) {
                //			particles_displace(rand1(), rand1(), rand1());
            }
            resetFlops();
            if (getOptions().hydro.enabled || getOptions().gravity.method == "pm") {
                tmp = advanceForces(
                    minrung, a, a * adot, tau, t0, theta, tau == 0.0, fullEval, energies);
            } else {
                tmp = kickStepHierarchical(om, maxRung, a, adot, tau, t0, theta, &energies,
                    minrung0, fullEval, &lcFut, nocrop, sqrt(glassError));
            }
            if (tau == 0.0) {
                energies0 = energies;
            }
            double const flops = flopsPerSecond();
            if (om != thisMinrung) {
                minrung0++;
            }
            KickReturn kr = tmp.first;
            int maxRung0 = maxRung;
            // A force pass can be empty after particles move to coarser rungs.
            // Finish the current integer-clock subdivision before taking a
            // coarser step, even when that pass reports no active particle rung.
            maxRung = std::max(kr.maxRung, static_cast<int>(minRung(itime)));

            //			PRINT("GRAVITY max_rung = %i\n", kr.max_rung);
            if (minrung <= 0) {
                double energy = a * (energies.kin + energies.pot + energies.therm) +
                    energies.heating + energies.cosmic;
                if (tau == 0) {
                    energy0 = 0.0;
                    energies.cosmic = -energy;
                    energy = 0.0;
                }
                double const norm = std::max(fabs(energies.kin) + fabs(energies.pot) +
                        fabs(energies.therm) + fabs(energies.heating) + fabs(energies.cosmic),
                    1.0e-30);
                double const err = (energy - energy0) / norm;
                if (tau > 0.0) {
                    glassError = fabs(energies0.pot - a * energies.pot) /
                        std::max(fabs(a * energies.pot), 1.0e-30);
                }
                FILE* fp = fopen(outputPath("energy.txt").c_str(), "at");
                double const momentumNorm = energies.nmom > 0.0 ? energies.nmom : 1.0;
                fprintf(fp, "%e %e %e %e %e %e %e %e %e %e\n", tau / t0, a,
                    energies.xmom / momentumNorm, energies.ymom / momentumNorm,
                    energies.zmom / momentumNorm, a * energies.pot, a * energies.kin,
                    energies.cosmic, err, glassError);
                fclose(fp);
                energies0 = energies;
                energies0.pot *= a;
            }
            dt = std::ldexp(t0, -maxRung);
            TreeCreateReturn sr = tmp.second;
            //			PRINT("Done kicking\n");
            double a0 = a;
            double physicalInterval = a * dt;
            double driftScale = a;
            if (!getOptions().initial.createGlass && !getOptions().initial.plummer) {
                auto const interval = cosmosAdvance(adot, a, dt);
                physicalInterval = interval.physicalTime;
                driftScale = dt / interval.driftIntegral;
            }
            // d[a(K+W+U)]/da = -K+U-3 integral(P dV) + source.
            // For monatomic gamma=5/3 this reduces to the historical -(K+U).
            // Peebles, Large-Scale Structure of the Universe (1980), energy
            // equation; the gas term follows directly from dU=-P dV.
            energies.cosmic += (a - a0) *
                (energies.tckin / (a0 * a0) +
                    (getOptions().chemistry.enabled ?
                            -energies.therm + 3.0 * synchronizedChemistryPressure :
                            energies.therm));
            double const dyears =
                physicalInterval * getOptions().units.timeSeconds / constants::kSpyr;
            auto const z0 = 1.0 / a0 - 1.0;
            auto const z1 = 1.0 / a - 1.0;
            for (auto& z : checkpointlist) {
                if ((z - z0) * (z - z1) < 0.0) {
                    doCheck = true;
                    checkz = z;
                }
            }

            //			PRINT("%e %e\n", a1, a);
            Timer dtm;
            dtm.start();
            if (getOptions().hydro.enabled || getOptions().gravity.method == "pm") {
                drift(driftScale, dt, tau, tau + dt, tauMax);
            }
            dtm.stop();
            driftTime += dtm.read();
            FILE* textfp = fopen(outputPath("progress.txt").c_str(), "at");
            if (textfp == nullptr) {
                THROW_ERROR("unable to open progress.txt for writing\n");
            }
            iter++;
            Timer remainingTime;
            totalTime.stop();
            remainingTime.start();
            runtime += totalTime.read();
            double pps = totalProcessed / runtime;
            //	PRINT( "%e %e %e %e\n", kr.node_flops, kr.part_flops, sr.flops, dr.flops);
            double const numberOfParticles = getOptions().initial.numberOfParticles;
            double close, direct;
            getGravityCounters(close, direct);
            direct += close;
            if (fullEval) {
                fprintf(textfp,
                    "\n%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                    "runtime", "i", "z", "a", "adot", "timestep", "years", "mnr", "mxr", "bs",
                    "Tflops", "time", "close", "direct");
            }
            stepTm.stop();
            TimeType const nextIntegerTime = inc(itime, maxRung);
            double const nextTau = t0 *
                (double(step) +
                    (nextIntegerTime == 0 ? 1.0 : std::ldexp(double(nextIntegerTime), -64)));
            double const nextYears = years + dyears;
            double const outputRedshift = 1.0 / a - 1.0;
            int const displayedMinimumRung =
                getOptions().gravity.method == "pm" ? maxRung : minrung;
            fprintf(textfp,
                "%10.3e %10i %10.3e %10.3e %10.3e %10.3e %10.3e %10i %10i %10i %10.3e %10.3e "
                "%10.3e %10.3e \n",
                runtime, iter - 1, outputRedshift, a, adot, nextTau / t0, nextYears,
                displayedMinimumRung, maxRung, bucketSize, flops * 1e-12, stepTm.read(), close,
                direct);
            if (progressRows == 0) {
                PRINT("  TFLOP/s estimates counted gravity and SPH work during force advancement; "
                      "chemistry is excluded.\n");
            }
            if (progressRows % 25 == 0) {
                PRINT("\n%8s %10s %10s %10s %10s %7s %7s %10s %10s %10s\n", "Step", "z", "a", "tau",
                    "dt", "MinRung", "MaxRung", "Wall[s]", "Step[s]", "TFLOP/s");
            }
            PRINT("%8i %10.3e %10.3e %10.3e %10.3e %7i %7i %10.2f %10.3f %10.3e\n", iter,
                outputRedshift, a, nextTau / t0, dt, displayedMinimumRung, maxRung, runtime,
                stepTm.read(), flops * 1e-12);
            progressRows++;
            fclose(textfp);
            totalTime.reset();
            remainingTime.stop();
            runtime += remainingTime.read();
            totalTime.start();
            itime = nextIntegerTime;
            domainTime = 0.0;
            sortTime = 0.0;
            kickTime = 0.0;
            driftTime = 0.0;
            tau = nextTau;
            thisIter++;
            years = nextYears;
            if (cosmology::endTimeReached(tau, tauMax)) {
                PRINT("\n  Finished: reached the requested end time or redshift.\n");
                break;
            } else if (thisIter > getOptions().run.maxIterations) {
                PRINT("\n  Stopped at the iteration limit.\n");
                break;
            }
            profilerExit();
            profilerOutput();
            profilerEnter(__FUNCTION__);
            jiter++;
            if (jiter > 100) {
                //			abort();
            }
            if (getOptions().lightcone.enabled && minRung(itime) <= minrung0) {
                checkLc(false);
            }
            lcFut = hpx::make_ready_future();

        } while (itime != 0);
        if (cosmology::endTimeReached(tau, tauMax) || thisIter > getOptions().run.maxIterations) {
            break;
        } else if (getOptions().initial.createGlass && std::abs(glassError) < 1.0e-5) {
            scenarios::saveGlass();
            break;
        }
    }
    viewOutputViews(params.step + 1, a, true, years);
    if (getOptions().initial.createGlass) {
        scenarios::saveGlass();
    }
    buckets2.stop();
    FILE* fp = fopen(outputPath("buckets.txt").c_str(), "at");
    fprintf(fp, "%i %e %e %e\n", getOptions().compute.bucketSize, buckets50.read(),
        buckets20.read(), buckets2.read());
    fclose(fp);
    if (getOptions().lightcone.enabled) {
        checkLc(true);
    }
    if (params.iter > 0 && params.iter != lastCheckpointIteration) {
        totalTime.stop();
        runtime += totalTime.read();
        terminal::detailPrint("Writing final checkpoint for iteration %i\n", params.iter);
        writeCheckpoint(params);
        if (doCheck && hpxRank() == 0) {
            detail::checkpointRedshiftAlias(params.iter, checkz);
        }
    }
    if (getOptions().lightcone.enabled) {
        particlesFree();
    }
    KickWorkspace::clearBuffers();
}
