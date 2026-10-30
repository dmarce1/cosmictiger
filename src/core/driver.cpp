/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>

#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/analysis/group_tree.hpp>
#include <cosmictiger/analysis/groups.hpp>
#include <cosmictiger/analysis/groups_find.hpp>
#include <cosmictiger/analysis/lightcone.hpp>
#endif
#include <cosmictiger/analysis/power.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/section_timing.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/core/time.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cosmology/background.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#if COSMICTIGER_WITH_GRAVITY
#if COSMICTIGER_ENABLE_FMM
#include <cosmictiger/gravity/dehnen/distributed_tree.hpp>
#include <cosmictiger/gravity/dehnen/execution.hpp>
#endif
#include <cosmictiger/gravity/error.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
#include <cosmictiger/gravity/nitadori/kick.hpp>
#include <cosmictiger/gravity/nitadori/solver.hpp>
#endif
#include <cosmictiger/gravity/opening_schedule.hpp>
#endif
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
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/scenarios/glass.hpp>
#include <cosmictiger/scenarios/plummer.hpp>
#endif
#include <cosmictiger/scenarios/scenario.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/sph/sph_chemistry.hpp>
#endif
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sph_tree.hpp>
#endif

constexpr double kSmoothLengthBuffer = 0.21;
constexpr double kScaleTimestep = 0.01;

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

#if COSMICTIGER_WITH_GRAVITY
void doGroups(int number, double scale, double years) {
    section_timing::ScopedSection sectionTimer(section_timing::Section::analysis);
    groupsPrepareHistory(number);

    Timer total;
    total.start();
    terminal::detailPrint("Doing groups\n");
    Timer tm;
    tm.start();
    groupTreeCreate();
    tm.stop();
    terminal::detailPrint("group_tree_create = %s\n", terminal::formatDuration(tm.read()).c_str());
    tm.reset();
    tm.start();
    particlesGroupsInit();
    tm.stop();
    terminal::detailPrint(
        "particles_group_init = %s\n", terminal::formatDuration(tm.read()).c_str());
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
        terminal::detailPrint("%i groups_find = %s active = %li\n", iter,
            terminal::formatDuration(tm.read()).c_str(), active);
        tm.reset();
        particlesIncGroupCacheEpoch();
        groupTreeIncCacheEpoch();
        iter++;
    } while (active > 0);
    tm.start();
    groupTreeDestroy();
    tm.stop();
    terminal::detailPrint("tree cleanup = %s\n", terminal::formatDuration(tm.read()).c_str());
    Timer reduce;
    reduce.start();
    for (int wave = 0; wave < kGroupWaves; wave++) {
        tm.start();
        groupsAddParticles(wave, scale);
        tm.stop();
        terminal::detailPrint(
            "%i groups_add_particles %s\n", wave, terminal::formatDuration(tm.read()).c_str());
        tm.reset();
        tm.start();
        groupsReduce(scale);
        tm.stop();
        terminal::detailPrint("groups_reduce %s\n", terminal::formatDuration(tm.read()).c_str());
        tm.reset();
    }
    reduce.stop();
    tm.start();
    groupsCull();
    tm.stop();
    terminal::detailPrint("groups_cull %s\n", terminal::formatDuration(tm.read()).c_str());
    tm.reset();
    tm.start();
    auto ngroups = groupsSave(number, scale, years);
    tm.stop();
    terminal::detailPrint("groups_save %s\n", terminal::formatDuration(tm.read()).c_str());
    total.stop();
    terminal::detailPrint("Reduction time = %s\n", terminal::formatDuration(reduce.read()).c_str());
    terminal::detailPrint("Total time = %s\n", terminal::formatDuration(total.read()).c_str());
    terminal::detailPrint("Group count = %li of %li candidates\n", ngroups.first, ngroups.second);
    particlesGroupsDestroy();
}

#endif

#if COSMICTIGER_WITH_SPH
SphTreeCreateReturn sphStep1(int minrung, bool verbose) {
    verbose = true;
    if (verbose) {
        terminal::detailPrint("Doing SPH step with minrung = %i\n", minrung);
    }
    SphTreeCreateParameters treeParameters;

    Timer timer;
    treeParameters.minimumRung = minrung;
    treeParameters.supportRadiusMultiplier = (1.0 + kSmoothLengthBuffer);
    SphTreeCreateReturn treeResult;

    timer.start();
    if (verbose) {
        terminal::detailPrint(
            "starting sph_tree_create = %s\n", terminal::formatDuration(timer.read()).c_str());
    }
    treeResult = sphTreeCreate(treeParameters);
    timer.stop();
    if (verbose) {
        terminal::detailPrint("sph_tree_create time = %s %i\n",
            terminal::formatDuration(timer.read()).c_str(), treeResult.numberOfActiveParticles);
    }
    timer.reset();
    return treeResult;
}

SphRunReturn sphStep2(int minrung, double scale, double tau, double t0, int phase, double adot,
    EnergiesT* energies, bool verbose, int uniformMinimumRung, bool closingOnly,
    double previousTimestep, double maximumGlobalTimestep) {
    bool const globalTimestep = maximumGlobalTimestep > 0.0;
    double selectedTimestep = maximumGlobalTimestep;
    double const closingTimestep = globalTimestep ? previousTimestep : -1.0;
    float inverseSoundTimestep;
    float inverseViscosityTimestep;
    float inverseAccelerationTimestep;
    verbose = true;
    if (verbose) {
        terminal::detailPrint("Doing SPH step with minrung = %i\n", minrung);
    }
    sphParticlesApplyUpdates(minrung, 0, t0, tau, closingTimestep);
#if COSMICTIGER_ENABLE_CHEMISTRY
    bool const chemistry = getOptions().chemistry.enabled;
    if (chemistry) {
        // Viscosity decay uses the current sound speed during the density pass.
        sphParticlesRefreshChemistryEos(scale);
    }
#endif
    SphRunParameters runParameters;
    if (getOptions().run.scenario == "cosmology") {
        runParameters.maximumTimestep =
            cosmology::expansionTimestep(scale, adot / scale, cosmosD2adt2(scale), kScaleTimestep);
    }
    runParameters.globalTimestep = globalTimestep;
    if (globalTimestep) {
        // The reduction is clipped to the exact double cap below. Round this
        // float kernel cap upward so it cannot invent a shorter final interval.
        float kernelCap = static_cast<float>(maximumGlobalTimestep);
        if (double(kernelCap) < maximumGlobalTimestep) {
            kernelCap = std::nextafter(kernelCap, std::numeric_limits<float>::infinity());
        }
        runParameters.maximumTimestep = std::min(runParameters.maximumTimestep, kernelCap);
    } else {
        runParameters.maximumTimestep = std::min(runParameters.maximumTimestep,
            std::ldexp(static_cast<float>(t0), -getOptions().run.minRung));
    }

    runParameters.expansionRate = adot;
    if (runParameters.gradientSwitch() && getOptions().run.scenario == "cosmology") {
        runParameters.expansionAcceleration = scale * scale * scale * cosmosD2adt2(scale);
    }
    runParameters.scaleFactor = scale;
    runParameters.baseTimestep = t0;
    runParameters.minimumRung = minrung;
    bool needsNeighborRefresh;
    SphRunReturn result;

    runParameters.phase = phase;
    SphTreeNeighborParameters neighborParameters;
    neighborParameters.supportRadiusMultiplier = (1.0 + kSmoothLengthBuffer);
    neighborParameters.minimumRung = minrung;
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);

    neighborParameters.innerSelection = kSphInteractionsI;
    neighborParameters.runType = kSphTreeNeighborNeighbors;
    sphTreeNeighbor(neighborParameters, rootId, checklist).get();

    runParameters.phase = 0;
    int densityIterations = 0;
    sphParticlesResetConverged();
    do {
        if (++densityIterations > 256) {
            THROW_ERROR("SPH smoothing lengths did not converge after 256 tree rebuilds\n");
        }

        runParameters.runType = kSphRunPrehydro1;
        Timer timer;
        timer.start();
        result = sphRun(runParameters, true);
        timer.stop();
        if (verbose) {
            terminal::detailPrint("sph_run(SPH_RUN_PREHYDRO1): tm = %s min_h = %e max_h = %e\n",
                terminal::formatDuration(timer.read()).c_str(), result.minimumSupportRadius,
                result.maximumSupportRadius);
        }
        timer.reset();
        needsNeighborRefresh = result.needsAnotherPass;
        neighborParameters.supportRadiusMultiplier = (1.0 + kSmoothLengthBuffer);
        neighborParameters.runType = kSphTreeNeighborBoxes;
        neighborParameters.outerSelection = needsNeighborRefresh ? kSphSetActive : kSphSetAll;
        neighborParameters.innerSelection = kSphSetAll;
        timer.start();
        sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
        timer.stop();
        terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
        timer.reset();
        timer.start();
        neighborParameters.innerSelection =
            needsNeighborRefresh ? kSphInteractionsI : kSphInteractionsIJ;
        neighborParameters.runType = kSphTreeNeighborNeighbors;
        sphTreeNeighbor(neighborParameters, rootId, checklist).get();
        timer.stop();
        terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
        timer.reset();

        result = SphRunReturn();
    } while (needsNeighborRefresh);
    sphParticlesResetConverged();
    densityIterations = 0;
    do {
        if (++densityIterations > 256) {
            THROW_ERROR("SPH neighbor densities did not converge after 256 tree rebuilds\n");
        }

        runParameters.runType = kSphRunPrehydro2;
        Timer timer;
        timer.start();
        result = sphRun(runParameters, true);
        timer.stop();
        if (verbose) {
            terminal::detailPrint("sph_run(SPH_RUN_PREHYDRO2): tm = %s min_h = %e max_h = %e\n",
                terminal::formatDuration(timer.read()).c_str(), result.minimumSupportRadius,
                result.maximumSupportRadius);
        }
        timer.reset();
        needsNeighborRefresh = result.needsAnotherPass;
        neighborParameters.supportRadiusMultiplier =
            needsNeighborRefresh ? (1.0 + kSmoothLengthBuffer) : 1.01;
        neighborParameters.runType = kSphTreeNeighborBoxes;
        neighborParameters.outerSelection = kSphSetAll;
        neighborParameters.innerSelection = kSphSetAll;
        timer.start();
        sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
        timer.stop();
        terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
        timer.reset();
        timer.start();
        neighborParameters.innerSelection = kSphInteractionsIJ;
        neighborParameters.runType = kSphTreeNeighborNeighbors;
        sphTreeNeighbor(neighborParameters, rootId, checklist).get();
        timer.stop();
        terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
        timer.reset();
        result = SphRunReturn();
    } while (needsNeighborRefresh);
    Timer timer;
    if (runParameters.gradientSwitch()) {
        // Cullen & Dehnen (2010), equation (A15), or Price et al. (2018), equation (50),
        // after density convergence.
        sphParticlesUpdateTargetAlpha(minrung, t0, tau, closingTimestep);
    }
    runParameters.phase = 0;
    runParameters.runType = kSphRunHydro;
    timer.reset();
    timer.start();
    result = sphRun(runParameters, true);
    timer.stop();
    int maxRung = std::max(0, uniformMinimumRung);
    if (verbose) {
        terminal::detailPrint(
            "sph_run(SPH_RUN_HYDRO): tm = %s\n", terminal::formatDuration(timer.read()).c_str());
    }
    timer.reset();

    sphParticlesApplyUpdates(minrung, 1, t0, tau, closingTimestep);
#if COSMICTIGER_ENABLE_CHEMISTRY
    if (chemistry && tau > 0.0) {
        // CKDKC closes C only after the old hydro predictor has been replaced.
        // Each active gas particle still owns its old rung and source interval.
        energies->heating += sphChemistryClose(minrung, scale, adot, t0, closingTimestep);
    }
#endif

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

    if (closingOnly) {
        // The final measurement includes the previous source half-step, but
        // must not solve chemistry or select hydro timesteps beyond the end.
        sphParticlesClearHydroCaches();
        sphTreeDestroy();
        sphParticlesClearCaches();
        sphParticlesResetSemiactive();
        return result;
    }
#if COSMICTIGER_ENABLE_CHEMISTRY
    if (chemistry) {
        synchronizedChemistryPressure = sphChemistryPressureIntegral(scale);
        sphParticlesRefreshChemistryEos(scale);
    }
#endif
    runParameters.phase = 1;
    runParameters.runType = kSphRunHydro;
    result = sphRun(runParameters, true);
    inverseSoundTimestep = result.inverseSoundTimestep;
    inverseViscosityTimestep = result.inverseViscosityTimestep;
    inverseAccelerationTimestep = result.inverseAccelerationTimestep;
    maxRung = std::max(result.maximumRung, maxRung);
    if (globalTimestep) {
        selectedTimestep = std::min(selectedTimestep, result.minimumTimestep);
        ALWAYS_ASSERT(std::isfinite(selectedTimestep) && selectedTimestep > 0.0);
    }

    neighborParameters.supportRadiusMultiplier = 1.01;
    neighborParameters.runType = kSphTreeNeighborBoxes;
    neighborParameters.innerSelection = kSphSetAll;
    neighborParameters.outerSelection = kSphSetActive;
    timer.start();
    sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
    timer.stop();
    timer.reset();
    timer.start();
    neighborParameters.innerSelection = kSphInteractionsIJ;
    neighborParameters.runType = kSphTreeNeighborNeighbors;
    sphTreeNeighbor(neighborParameters, rootId, checklist).get();
    timer.stop();
    timer.reset();

    // A uniform mesh step already uses the global maximum proposal.
    bool repeat = !globalTimestep && uniformMinimumRung < 0;
    while (repeat) {
        runParameters.runType = kSphRunRungs;
        repeat = sphRun(runParameters, true).needsAnotherPass;
    }
#if COSMICTIGER_ENABLE_CHEMISTRY
    if (chemistry) {
        // Chemistry follows the selected SPH interval, subcycling internally.
        sphChemistryOpen(minrung, scale, adot, t0, globalTimestep ? selectedTimestep : -1.0,
            !globalTimestep && uniformMinimumRung >= 0 ? maxRung : -1);
        sphParticlesRefreshChemistryEos(scale);
        // Refresh the opening kick's rates without selecting another timestep.
        // Preserve the pre-source timestep reduction and its diagnostics.
        runParameters.phase = 0;
        runParameters.runType = kSphRunHydro;
        sphRun(runParameters, true);
    }
#endif
    if (!globalTimestep && uniformMinimumRung >= 0) {
        // The dark matter and hydro proposals have both been reduced globally.
        // Set every rung before either species opens its next half-step.
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
        section_timing::measure(
            section_timing::Section::gravity, [&]() { particleMeshFinishKick(t0, maxRung); });
#endif
    }
    sphParticlesApplyUpdates(minrung, 2, t0, tau, globalTimestep ? selectedTimestep : -1.0);
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
    if (globalTimestep) {
        result.minimumTimestep = selectedTimestep;
    }
    return result;
}

#endif

#if COSMICTIGER_ENABLE_FMM
ForceStepResult kickStep(int minrung, double scale, double dadt, double t0, double theta,
    bool firstCall, bool fullEval, double previousTimestep, double maximumGlobalTimestep) {
    Timer tm;
    tm.start();
    terminal::detailPrint("domains_begin\n");
    section_timing::measure(section_timing::Section::domains, [&]() {
        domainsBegin(minrung);
        terminal::detailPrint("domains_end \n");
        domainsEnd();
    });
    tm.stop();
    domainTime += tm.read();
    tm.reset();
    tm.start();
    bool const sph = getOptions().hydro.enabled;
    // ALWAYS_ASSERT(sph_particles_max_smooth_len() != INFINITY);
    TreeCreateParameters tparams(minrung, theta, 0.f);
    terminal::detailPrint("Create tree %i %e\n", minrung, theta);
    section_timing::measure(
        section_timing::Section::treeBuild, [&]() { return treeCreate(tparams); });
    terminal::detailPrint("Done with tree\n");

    tm.stop();
    sortTime += tm.read();
    tm.reset();
    tm.start();
    //	PRINT("numberOfActiveParticles = %li\n", sr.numberOfActiveParticles);
    KickParameters kparams;
    if (getOptions().run.scenario == "cosmology") {
        kparams.maxDt =
            cosmology::expansionTimestep(scale, dadt / scale, cosmosD2adt2(scale), kScaleTimestep);
    }
    // The ordinary path visits every starting particle, including rung zero,
    // while the chosen step must still honor the configured starting-rung floor.
    kparams.globalTimestep = maximumGlobalTimestep > 0.0;
    kparams.previousTimestep = previousTimestep;
    if (kparams.globalTimestep) {
        float kernelCap = static_cast<float>(maximumGlobalTimestep);
        if (double(kernelCap) < maximumGlobalTimestep) {
            kernelCap = std::nextafter(kernelCap, std::numeric_limits<float>::infinity());
        }
        kparams.maxDt = std::min(kparams.maxDt, kernelCap);
        gravityBeginGlobalKick();
    } else {
        kparams.maxDt =
            std::min(kparams.maxDt, std::ldexp(static_cast<float>(t0), -getOptions().run.minRung));
    }
    kparams.htime = false;
    kparams.ascending = kparams.descending = kparams.top = true;
    kparams.doPhi = fullEval;
    kparams.sign = getOptions().initial.createGlass ? -1.0 : 1.0;
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
    kparams.t0 = t0;
    kparams.theta = theta;
    ExpansionBuffer l;
    l.zero();
    Array<Position, kDimensions> pos;
    for (int dim = 0; dim < kDimensions; dim++) {
        pos[dim] = 0.f;
    }
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);
    terminal::detailPrint("Do kick\n");
    KickReturn kr = section_timing::measure(section_timing::Section::gravity,
        [&]() { return kick(kparams, l, pos, rootId, checklist, checklist, nullptr); });
    tm.stop();
    kickTime += tm.read();

    treeDestroy();
    particlesCacheFree();
    terminal::detailPrint("kick done\n");
    return kr;
}

#endif

ForceStepResult advanceForces(int minRung, double scale, double scaleDerivative, double time,
    double baseTimestep, double theta, bool firstCall, bool fullEvaluation, EnergiesT& energies,
    int gravityErrorStep, bool closingOnly, double previousTimestep, double maximumGlobalTimestep) {
    ForceStepResult result{};
    bool const global = maximumGlobalTimestep > 0.0;
    bool const meshGravity = getOptions().gravity.enabled && usesMeshGravity(getOptions().gravity);
    if (global) {
        minRung = 0;
        result.minimumTimestep = maximumGlobalTimestep;
    }
#if COSMICTIGER_WITH_GRAVITY
    if (gravityErrorStep >= 0) {
        section_timing::measure(section_timing::Section::analysis, []() { gravityErrorBegin(); });
    }
    if (getOptions().gravity.enabled) {
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
        if (meshGravity) {
            Timer timer;
            timer.start();
            section_timing::measure(section_timing::Section::domains, [&]() {
                domainsBegin(0);
                domainsEnd();
            });
            timer.stop();
            domainTime += timer.read();
            timer.reset();
            timer.start();
            section_timing::ScopedSection gravityTimer(section_timing::Section::gravity);
            pmmmSolve(true);
            double maximumTimestep = global ? maximumGlobalTimestep :
                                              std::ldexp(baseTimestep, -getOptions().run.minRung);
            if (getOptions().run.scenario == "cosmology") {
                maximumTimestep = std::min(maximumTimestep,
                    cosmology::expansionTimestep(
                        scale, scaleDerivative / scale, cosmosD2adt2(scale), kScaleTimestep));
            }
            result = particleMeshPrepareKick(scale, baseTimestep, maximumTimestep, firstCall,
                minRung, global ? previousTimestep : -1.0);
            timer.stop();
            kickTime += timer.read();
        }
#endif
#if COSMICTIGER_ENABLE_FMM
        if (!meshGravity) {
            result = kickStep(minRung, scale, scaleDerivative, baseTimestep, theta, firstCall,
                fullEvaluation, previousTimestep, maximumGlobalTimestep);
        }
#endif
    } else
#endif
    {
        section_timing::measure(section_timing::Section::domains, [&]() {
            domainsBegin(minRung);
            domainsEnd();
        });
    }
    if (global) {
        result.minimumTimestep = std::min(result.minimumTimestep, maximumGlobalTimestep);
    }
    auto& gravity = result;
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        section_timing::ScopedSection hydroTimer(section_timing::Section::hydrodynamics);
        int const activeRung = meshGravity || global ? 0 : minRung;
        sphStep1(activeRung);
        auto const hydro = sphStep2(activeRung, scale, time, baseTimestep, 1, scaleDerivative,
            &energies, true, meshGravity && !global ? gravity.maxRung : -1, closingOnly,
            previousTimestep, global ? result.minimumTimestep : 0.0);
        gravity.maxRung = std::max(gravity.maxRung, hydro.maximumRung);
        if (global) {
            gravity.minimumTimestep =
                std::min(gravity.minimumTimestep, static_cast<double>(hydro.minimumTimestep));
        }
        if (fullEvaluation || meshGravity) {
            energies.pot = gravity.pot / scale;
        }
        energies.kin += gravity.darkMatterKineticCorrection / sqr(scale);
        energies.xmom += gravity.darkMatterXMomentumCorrection / scale;
        energies.ymom += gravity.darkMatterYMomentumCorrection / scale;
        energies.zmom += gravity.darkMatterZMomentumCorrection / scale;
        energies.nmom += gravity.darkMatterMomentumNormCorrection / scale;
        if (!closingOnly) {
            energies.tckin = particlesSumEnergies().kin;
        }
    }
#endif
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().gravity.enabled && !getOptions().hydro.enabled) {
        if (meshGravity || global || fullEvaluation) {
            energies.pot = gravity.pot / scale;
            energies.kin = gravity.kin / sqr(scale);
            energies.xmom = gravity.xmom / scale;
            energies.ymom = gravity.ymom / scale;
            energies.zmom = gravity.zmom / scale;
            energies.nmom = gravity.nmom / scale;
            energies.tckin = gravity.kin - gravity.darkMatterKineticCorrection;
        } else {
            energies.tckin += gravity.dkin;
        }
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
        if (meshGravity && !global && !closingOnly) {
            energies.tckin = section_timing::measure(section_timing::Section::gravity,
                [&]() { return particleMeshFinishKick(baseTimestep, gravity.maxRung); });
        }
#endif
    }
    if (global && !closingOnly && getOptions().gravity.enabled) {
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
        if (meshGravity) {
            particleMeshFinishGlobalKick(result.minimumTimestep);
        }
#endif
#if COSMICTIGER_ENABLE_FMM
        if (!meshGravity) {
            gravityFinishGlobalKick(result.minimumTimestep);
        }
#endif
        energies.tckin = particlesSumEnergies().kin;
    }
    if (gravityErrorStep >= 0) {
        section_timing::measure(section_timing::Section::analysis,
            [&]() { gravityErrorFinish(gravityErrorStep, time, scale); });
    }
#endif
    return result;
}

#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
// Rung-restricted drift is deliberately locality-local. The PMMM stage entry
// runs only on the root, so dispatch that completed stage to every locality.
void particleMeshDriftRung(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag);
HPX_PLAIN_ACTION(particleMeshDriftRung, ParticleMeshDriftRungAction);

void particleMeshDriftRung(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag) {
    Vector<hpx::future<void>> futures;
    try {
        for (auto const& child : hpxChildren()) {
            futures.push_back(hpx::async<ParticleMeshDriftRungAction>(
                child, scale, dt, tau0, tau1, tauMax, rung, drag));
        }
        drift(scale, dt, tau0, tau1, tauMax, rung, drag);
    } catch (...) {
        hpx::wait_all(futures.begin(), futures.end());
        throw;
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

#endif

ForceStepResult kickStepHierarchical(int& minrung, int maxRung, double scale, double adot,
    double tau, double t0, double theta, EnergiesT* energies, int minrung0, bool doPhi,
    hpx::future<void>* lcFut, bool nocrop, double drag, int gravityErrorStep, bool closingOnly) {
    // A terminal energy evaluation consumes only the outstanding closing
    // ladder. The saved checkpoint retains the original staggered momenta.
    nocrop = nocrop || closingOnly;
    bool const meshGravity = usesMeshGravity(getOptions().gravity);
    Timer tm;
    KickReturn kr;
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
            section_timing::measure(section_timing::Section::domains, [&]() {
                domainsBegin(levels[li]);
                domainsEnd();
            });
            tm.stop();
            //			PRINT("Domains took %e\n", tm.read());
        }

        if (!ascending) {
            tm.reset();
            tm.start();
            section_timing::measure(
                section_timing::Section::particleSort, [&]() { particlesSortByRung(levels[li]); });
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
        int minimumTreeLevel = 0;
        if (top && gravityErrorStep >= 0) {
            section_timing::measure(
                section_timing::Section::analysis, []() { gravityErrorBegin(); });
        }
        tm.reset();
        tm.start();
#if COSMICTIGER_ENABLE_FMM
        if (!meshGravity) {
            TreeCreateParameters tparams(levels[li], theta, 0.f);
            section_timing::measure(
                section_timing::Section::treeBuild, [&]() { return treeCreate(tparams); });
            minimumTreeLevel = tparams.minLevel;
        }
#endif
        tm.stop();
        //		PRINT("Tree create took %e\n", tm.read());
        KickParameters kparams;
        if (clipTop && levels[li] == minrung0 + 1) {
            kparams.top = true;
        } else {
            kparams.top = top;
        }
        kparams.sign = getOptions().initial.createGlass ? -1.0 : 1.0;
        kparams.ascending = ascending || top;
        if (closingOnly || (clipTop && top)) {
            kparams.descending = false;
        } else {
            kparams.descending = !ascending || top;
        }
        kparams.velInit = getOptions().initial.plummer;
        kparams.nodeLoad = flopsPerNode / flopsPerParticle;
        kparams.gpu = getOptions().compute.cuda;
        kparams.minLevel = minimumTreeLevel;
        kparams.saveForce = getOptions().analysis.saveForce;
        kparams.gm = getOptions().gravity.gm;
        kparams.eta = getOptions().gravity.timestepEta;
        kparams.gravitySupportRadius = getOptions().gravity.supportRadius;
        kparams.a = scale;
        kparams.firstCall = tau == 0.0;
        kparams.minRung = levels[li];
        kparams.doPhi = top && (doPhi || saveParticlePotential(getOptions().analysis));
        kparams.t0 = t0;
        if (getOptions().run.scenario == "cosmology") {
            kparams.maxDt =
                cosmology::expansionTimestep(scale, adot, cosmosD2adt2(scale), kScaleTimestep);
        }
        kparams.theta = theta;
        ExpansionBuffer l;
        l.zero();
        Array<Position, kDimensions> pos;
        for (int dim = 0; dim < kDimensions; dim++) {
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
#if COSMICTIGER_ENABLE_FMM || COSMICTIGER_ENABLE_PMMM_FMM
            resetGravityCounters();
            setGravityCounterUse(true);
#endif
        }
        KickReturn thisKr;
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
        if (meshGravity) {
            thisKr = section_timing::measure(
                section_timing::Section::gravity, [&]() { return pmmmKick(kparams); });
        }
#endif
#if COSMICTIGER_ENABLE_FMM
        if (!meshGravity) {
            thisKr = section_timing::measure(section_timing::Section::gravity,
                [&]() { return kick(kparams, l, pos, rootId, checklist, checklist, nullptr); });
        }
#endif
        if (top) {
#if COSMICTIGER_ENABLE_FMM || COSMICTIGER_ENABLE_PMMM_FMM
            setGravityCounterUse(false);
#endif
        }
        if (top && gravityErrorStep >= 0) {
            section_timing::measure(section_timing::Section::analysis,
                [&]() { gravityErrorFinish(gravityErrorStep, tau, scale); });
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
        if (closingOnly && top) {
#if COSMICTIGER_ENABLE_FMM
            if (!meshGravity) {
                treeReset();
            }
#endif
            break;
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
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
            if (meshGravity) {
                section_timing::measure(section_timing::Section::drift, [&]() {
                    particleMeshDriftRung(a, dt, tau, tau + dt, getOptions().run.numberOfSteps * t0,
                        levels[li], drag);
                });
            }
#endif
#if COSMICTIGER_ENABLE_FMM
            if (!meshGravity) {
                section_timing::measure(section_timing::Section::drift, [&]() {
                    drift(a, dt, tau, tau + dt, getOptions().run.numberOfSteps * t0, levels[li],
                        drag);
                });
            }
#endif
            tm.stop();
            // PRINT("Drift took %e\n", tm.read());
        }
#if COSMICTIGER_ENABLE_FMM
        if (!meshGravity) {
            treeReset();
        }
#endif
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
    return kr;
}

#endif

double doPowerSpectrum(int num, double a, double adot) {
    section_timing::ScopedSection sectionTimer(section_timing::Section::analysis);
    auto const softFilter = [](double kh) {
        if (kh > 0.01) {
            return -15.0 * (3.0 * kh * cos(kh) + ((sqr(kh) - 3.0) * sin(kh))) * pow(kh, -5.0);
        } else {
            return 1.0 - sqr(kh) / 14.0 + sqr(sqr(kh)) / 504.0;
        }
    };
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
        double const kshot =
            2.0 * std::numbers::pi * getOptions().initial.particlesPerDimension / boxSize;
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
    return maxpow;
}

void outputTimeFile(DriverParameters const& parameters) {
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
}

static void outputEnergy(double coarseTime, double scale, EnergiesT const& energies,
    double referenceEnergy, double glassError) {
    double const error = cosmology::normalizedEnergyError(scale, energies.kin, energies.pot,
        energies.therm, energies.heating, energies.cosmic, referenceEnergy);
    FILE* fp = fopen(outputPath("energy.txt").c_str(), "at");
    if (fp == nullptr) {
        THROW_ERROR("unable to open energy.txt for writing\n");
    }
    // Label the columns once, when the file is new; restarts append.
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp,
            "# step a px/sum|p| py/sum|p| pz/sum|p| a*W a*K cosmic energy_error "
            "glass_error\n");
    }
    double const momentumNorm = energies.nmom > 0.0 ? energies.nmom : 1.0;
    fprintf(fp, "%e %e %e %e %e %e %e %e %e %e\n", coarseTime, scale, energies.xmom / momentumNorm,
        energies.ymom / momentumNorm, energies.zmom / momentumNorm, scale * energies.pot,
        scale * energies.kin, energies.cosmic, error, glassError);
    fclose(fp);
}

void driver() {
    profile::checkForInterrupt();
    bool const global = usesGlobalIntegration(getOptions().run);
    if (scenarios::definition(getOptions().run.scenario).fixedTime) {
        runFixedScenario();
        return;
    }
    Timer totalTime;
    Timer tmr;
    tmr.start();
    DriverParameters params{};
    hpx::future<void> lcFut = hpx::make_ready_future();
    int buckets[11] = {80, 112, 112, 128, 128, 160, 176, 184, 184, 184, 192};
    double a0 = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    if (getOptions().checkpoint.readIndex != -1) {
        params = section_timing::measure(
            section_timing::Section::checkpointRead, []() { return readCheckpoint(); });
        profile::checkForInterrupt();
        double const coarseInterval = params.tauMax / getOptions().run.numberOfSteps;
        double const endpointTolerance =
            32.0 * std::numeric_limits<double>::epsilon() * params.tauMax;
        if (!std::isfinite(coarseInterval) || !(coarseInterval > 0.0) ||
            !std::isfinite(params.tau) || params.tau < 0.0 ||
            params.tau > params.tauMax + endpointTolerance || params.step < 0) {
            THROW_ERROR("Checkpoint contains an invalid evolution clock\n");
        }
        if (!global) {
            long double const phase = std::ldexp(static_cast<long double>(params.itime), -64);
            long double const coarsePosition =
                static_cast<long double>(params.tau) / coarseInterval - phase;
            long double const coarseIndex = std::round(coarsePosition);
            long double const clockTolerance = std::max(1.0e-10L,
                256.0L * std::numeric_limits<double>::epsilon() *
                    std::max(1.0L, std::abs(coarsePosition)));
            if (std::abs(coarsePosition - coarseIndex) > clockTolerance || coarseIndex < 0.0L ||
                coarseIndex >= std::numeric_limits<int>::max()) {
                THROW_ERROR(
                    "Checkpoint conformal time and integer timestep phase are inconsistent; "
                    "retain the original timestep settings\n");
            }
            int const resumedStep = static_cast<int>(coarseIndex);
            if (params.step != resumedStep &&
                !(params.itime == 0 && params.step == resumedStep - 1)) {
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
        section_timing::ScopedSection initializationTimer(section_timing::Section::initialization);
#if COSMICTIGER_WITH_GRAVITY
        if (getOptions().run.scenario == "glass") {
            scenarios::initializeGlass(params);
        } else if (getOptions().run.scenario == "plummer") {
            scenarios::initializePlummer(params);
        } else
#endif
        {
            scenarios::initializeCosmology(params);
        }
        profile::checkForInterrupt();
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
            pow(opts.units.massG * (8.0 * std::numbers::pi) * numberOfParticles * constants::kG /
                    (3.0 * opts.cosmology.omegaM * sqr(constants::kH0 * opts.cosmology.hubble)),
                1.0 / 3.0);
        opts.units.timeSeconds = opts.units.lengthCm / constants::kC;
        double hValue = constants::kH0 * opts.units.timeSeconds;
        opts.gravity.gm = opts.cosmology.omegaM * 3.0 * sqr(hValue * opts.cosmology.hubble) /
            (8.0 * std::numbers::pi) / numberOfParticles;
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
#if COSMICTIGER_WITH_GRAVITY
    auto const openingSchedule =
        gravity::parseOpeningSchedule(getOptions().gravity.openingSchedule);
    bool const useOpeningSchedule = getOptions().run.scenario == "cosmology" &&
        !usesMeshGravity(getOptions().gravity) && !openingSchedule.empty();
    if (useOpeningSchedule) {
        auto opts = getOptions();
        // Use the same expansion intervals on either branch and after restart.
        // Restarting requires the same requested opening schedule.
        opts.gravity.openingAngle = gravity::openingScheduleTheta(openingSchedule,
            opts.cosmology.startRedshift, 1.0 / params.a - 1.0, opts.gravity.openingAngle);
        setOptions(opts);
        terminal::detailPrint("FMM opening schedule: z=%.17g theta=%.17g\n", 1.0 / params.a - 1.0,
            opts.gravity.openingAngle);
    }
#endif
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
    PRINT("  tau is conformal time / coarse interval; dt is in code units.\n"
          "  Wall durations use hh:mm::ss.ss; initialization is excluded.\n");
    Timer reset;
    reset.start();
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().lightcone.enabled) {
        lcInit(tau, tauMax);
    }
#endif
    double dt = 0.0;
    int jiter = 0;
    if (tau == 0.0) {
        particlesSetMinrung(minrung0);
    }
#if COSMICTIGER_WITH_GRAVITY
    auto const checkLc = [&tau, &dt, &tauMax, &a, &iter](bool force) {
        if (force || lcTimeToFlush(tau, tauMax)) {
            section_timing::ScopedSection analysisTimer(section_timing::Section::analysis);
            Timer tm;
            terminal::detailPrint("Flushing light cone\n");
            tm.start();
            lcBuffer2homes();
            tm.stop();
            terminal::detailPrint(
                "lc_buffer2homes %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            lcParticleBoundaries1();
            tm.stop();
            terminal::detailPrint(
                "lc_particles_boundaries1 %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            double const linkingLength = getOptions().lightcone.linkingLength *
                pow(getOptions().initial.numberOfParticles, -1.0 / kDimensions);
            lcFormTrees(tau, linkingLength);
            tm.stop();
            terminal::detailPrint(
                "lc_form_trees %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            size_t cnt;
            lcFindNeighbors();
            terminal::detailPrint("neighbors found\n");
            tm.stop();
            terminal::detailPrint(
                "lc_neighbors_found %s\n", terminal::formatDuration(tm.read()).c_str());
            Timer tm2;
            tm2.start();
            do {
                Timer tm;
                tm.start();
                lcParticleBoundaries2();
                tm.stop();
                terminal::detailPrint("%s ", terminal::formatDuration(tm.read()).c_str());
                tm.reset();
                tm.start();
                cnt = lcFindGroups();
                tm.stop();
                terminal::detailPrint("%li %s\n", cnt, terminal::formatDuration(tm.read()).c_str());
            } while (cnt > 0);
            tm2.stop();
            terminal::detailPrint("loop time = %s\n", terminal::formatDuration(tm2.read()).c_str());
            tm.start();
            lcGroups2homes();
            tm.stop();
            terminal::detailPrint(
                "lc_groups2homes %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            lcParts2groups(a, linkingLength, iter);
            tm.stop();
            terminal::detailPrint(
                "lc_parts2groups %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            lcInit(tau, tauMax);
            tm.stop();
            terminal::detailPrint("lc_init %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
        }
    };
#endif
    auto checkpointlist = readCheckpointList();
    double const hsoft0 = getOptions().gravity.supportRadius;
    bool doCheck = false;
    double checkz = 1.0 / a - 1.0;
    int lastCheckpointIteration = getOptions().checkpoint.readIndex >= 0 ? params.iter : -1;
    EnergiesT energies0;
    double glassError = 1.0;
    bool interrupted = false;
    double const previousRuntime = runtime;
    Timer evolutionTime;
    evolutionTime.start();
    totalTime.start();
    for (;; step += global ? 0 : 1) {
        if (cosmology::endTimeReached(tau, tauMax)) {
            break;
        }
        //		PRINT("STEP  = %i\n", step);
        double t0 = tauMax / getOptions().run.numberOfSteps;
        do {
            // Stop only between completed substeps, retaining the current clock
            // for the ordinary final checkpoint and distributed cleanup below.
            if (profile::interruptRequested()) {
                interrupted = true;
                break;
            }
            Timer stepTm;
            stepTm.start();
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
                section_timing::measure(
                    section_timing::Section::checkpointWrite, [&]() { writeCheckpoint(params); });
                lastCheckpointIteration = params.iter;
                tmr.reset();
                if (doCheck && hpxRank() == 0) {
                    detail::checkpointRedshiftAlias(params.iter, checkz);
                }
                doCheck = false;
            }
            tmr.start();
            int minrung = global ? 0 : minRung(itime);
            bool fullEval = minrung == 0;
            if (fullEval) {
                section_timing::ScopedSection outputTimer(section_timing::Section::snapshotOutput);
                int const number = step;
                if (getOptions().analysis.tracers) {
                    outputTracers(number);
                }
                if (getOptions().analysis.slice) {
                    outputSlice(number, years);
                }
                if (fullEval) {
                    viewOutputViews(global ? step : (tau + 1e-6 * t0) / t0, a, false, years);
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
                rebound = (imbalance > kMaxLoadImbalance);
            }
            if (rebound) {
                section_timing::ScopedSection domainTimer(section_timing::Section::domains);
                terminal::detailPrint("Doing rebound\n");
                domainsRebound();
                domainsBegin(0);
                domainsEnd();
                terminal::detailPrint("Rebound done\n");
            }
            auto opts = getOptions();
            opts.gravity.supportRadius =
                hsoft0;    // Fixed comoving support for every particle type.
            // Global FMM schedules advance after drift; PMMM retains its angle
            // derived from the ghost width throughout evolution.
            double const theta = opts.gravity.openingAngle;
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
            ForceStepResult tmp;
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
#if COSMICTIGER_WITH_GRAVITY && !defined(CHECK_MUTUAL_SORT)
                if (getOptions().analysis.groups) {
                    doGroups(step, a, years);
                }
#endif
            }
            if (getOptions().initial.useGlass && minrung == minrung0) {
                //			particles_displace(rand1(), rand1(), rand1());
            }
            resetFlops();
            int const errorInterval = getOptions().gravity.errorInterval;
            int const gravityErrorStep =
                itime == 0 && errorInterval > 0 && step % errorInterval == 0 ? step : -1;
            if (!getOptions().run.hierarchicalGravity || !getOptions().gravity.enabled) {
                tmp = advanceForces(minrung, a, a * adot, tau, t0, theta, tau == 0.0, fullEval,
                    energies, gravityErrorStep, false, params.globalTimestep,
                    global ? std::min(t0, tauMax - tau) : 0.0);
#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
            } else {
                tmp = kickStepHierarchical(om, maxRung, a, adot, tau, t0, theta, &energies,
                    minrung0, fullEval, &lcFut, nocrop, sqrt(glassError), gravityErrorStep);
#endif
            }
            if (tau == 0.0) {
                energies0 = energies;
            }
            double const flops = flopsPerSecond();
            if (om != thisMinrung) {
                minrung0++;
            }
            ForceStepResult const& kr = tmp;
            int maxRung0 = maxRung;
            // A force pass can be empty after particles move to coarser rungs.
            // Finish the current integer-clock subdivision before taking a
            // coarser step, even when that pass reports no active particle rung.
            maxRung = global ? 0 : std::max(kr.maxRung, static_cast<int>(minRung(itime)));

            //			PRINT("GRAVITY max_rung = %i\n", kr.max_rung);
            if (minrung <= 0) {
                if (tau == 0) {
                    energy0 = 0.0;
                    energies.cosmic =
                        -a * (energies.kin + energies.pot + energies.therm) - energies.heating;
                }
                if (tau > 0.0) {
                    glassError = fabs(energies0.pot - a * energies.pot) /
                        std::max(fabs(a * energies.pot), 1.0e-30);
                }
                outputEnergy(tau / t0, a, energies, energy0, glassError);
                energies0 = energies;
                energies0.pot *= a;
            }
            dt = global ? std::min(kr.minimumTimestep, tauMax - tau) : std::ldexp(t0, -maxRung);
            if (!(dt > 0.0) || !std::isfinite(dt) || tau + dt == tau) {
                THROW_ERROR("Selected timestep cannot advance the evolution clock\n");
            }
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
            energies.cosmic += cosmology::kineticExpansionWork(a0, a, energies.tckin) +
                (a - a0) *
                    (getOptions().chemistry.enabled ?
                            -energies.therm + 3.0 * synchronizedChemistryPressure :
                            energies.therm);
            double const dyears =
                physicalInterval * getOptions().units.timeSeconds / constants::kSpyr;
            auto const z0 = 1.0 / a0 - 1.0;
            auto const z1 = 1.0 / a - 1.0;
#if COSMICTIGER_WITH_GRAVITY
            if (useOpeningSchedule) {
                double const nextTheta = gravity::openingScheduleTheta(
                    openingSchedule, getOptions().cosmology.startRedshift, z1, theta);
                if (nextTheta != theta) {
                    auto nextOptions = getOptions();
                    nextOptions.gravity.openingAngle = nextTheta;
                    setOptions(nextOptions);
                    terminal::detailPrint(
                        "FMM opening schedule: z=%.17g theta=%.17g\n", z1, nextTheta);
                }
            }
#endif
            for (auto& z : checkpointlist) {
                if ((z - z0) * (z - z1) < 0.0) {
                    doCheck = true;
                    checkz = z;
                }
            }

            //			PRINT("%e %e\n", a1, a);
            Timer dtm;
            dtm.start();
            if (!getOptions().run.hierarchicalGravity || !getOptions().gravity.enabled) {
                section_timing::measure(section_timing::Section::drift,
                    [&]() { drift(driftScale, dt, tau, tau + dt, tauMax); });
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
            double close = 0.0, direct = 0.0;
#if COSMICTIGER_ENABLE_FMM || COSMICTIGER_ENABLE_PMMM_FMM
            if (!usesMeshGravity(getOptions().gravity) ||
                getOptions().gravity.pmmmNearMethod == "fmm") {
                getGravityCounters(close, direct);
            }
#endif
            direct += close;
            if (fullEval) {
                fprintf(textfp,
                    "\n%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                    "runtime", "i", "z", "a", "adot", "timestep", "years", "mnr", "mxr", "bs",
                    "Tflops", "time", "close", "direct");
            }
            stepTm.stop();
            TimeType const nextIntegerTime = global ? 0 : inc(itime, maxRung);
            double const nextTau = global ? tau + dt :
                                            t0 *
                    (double(step) +
                        (nextIntegerTime == 0 ? 1.0 : std::ldexp(double(nextIntegerTime), -64)));
            double const nextYears = years + dyears;
            double const outputRedshift = 1.0 / a - 1.0;
            int const displayedMinimumRung =
                (!getOptions().run.hierarchicalGravity || !getOptions().gravity.enabled) &&
                    usesMeshGravity(getOptions().gravity) ?
                maxRung :
                minrung;
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
                PRINT("\n%8s %10s %10s %10s %10s %7s %7s %13s %13s %10s\n", "Step", "z", "a", "tau",
                    "dt", "MinRung", "MaxRung", "Wall", "Step time", "TFLOP/s");
            }
            PRINT("%8i %10.3e %10.3e %10.3e %10.3e %7i %7i %13s %13s %10.3e\n", iter,
                outputRedshift, a, nextTau / t0, dt, displayedMinimumRung, maxRung,
                terminal::formatDuration(runtime).c_str(),
                terminal::formatDuration(stepTm.read()).c_str(), flops * 1e-12);
            progressRows++;
            fclose(textfp);
            totalTime.reset();
            remainingTime.stop();
            runtime += remainingTime.read();
            totalTime.start();
            itime = nextIntegerTime;
            if (global) {
                params.globalTimestep = dt;
                ++step;
            }
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
            jiter++;
            if (jiter > 100) {
                //			abort();
            }
#if COSMICTIGER_WITH_GRAVITY
            if (getOptions().lightcone.enabled && minRung(itime) <= minrung0) {
                checkLc(false);
            }
#endif
            lcFut = hpx::make_ready_future();

        } while (itime != 0);
        if (interrupted || cosmology::endTimeReached(tau, tauMax) ||
            thisIter > getOptions().run.maxIterations) {
            break;
#if COSMICTIGER_WITH_GRAVITY
        } else if (getOptions().initial.createGlass && std::abs(glassError) < 1.0e-5) {
            scenarios::saveGlass();
            break;
#endif
        }
    }
    lcFut.get();
    if (interrupted) {
        PRINT("\n  Interrupted: saving the completed simulation state before shutdown.\n");
    }
    section_timing::measure(section_timing::Section::snapshotOutput,
        [&]() { viewOutputViews(params.step + 1, a, true, years); });
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().initial.createGlass) {
        scenarios::saveGlass();
    }
    if (getOptions().lightcone.enabled) {
        checkLc(true);
    }
#endif
    if (params.iter > 0 && params.iter != lastCheckpointIteration) {
        totalTime.stop();
        runtime += totalTime.read();
        terminal::detailPrint("Writing final checkpoint for iteration %i\n", params.iter);
        section_timing::measure(
            section_timing::Section::checkpointWrite, [&]() { writeCheckpoint(params); });
        if (doCheck && hpxRank() == 0) {
            detail::checkpointRedshiftAlias(params.iter, checkz);
        }
    }
    if (!interrupted && thisIter > 0 && itime == 0 && cosmology::endTimeReached(tau, tauMax)) {
        // Finish the energy measurement only after all restart state and
        // snapshots have been saved. This terminal force pass may reorder and
        // synchronize particles; nothing subsequently persists or evolves them.
        // Keep the work ledger and driver state at the completed drift, and
        // leave completed-checkpoint restarts alone (thisIter == 0).
        EnergiesT finalEnergies = energies;
        double const t0 = tauMax / getOptions().run.numberOfSteps;
        double const theta = getOptions().gravity.openingAngle;
#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
        if (getOptions().run.hierarchicalGravity && getOptions().gravity.enabled) {
            int minimumRung = minrung0;
            kickStepHierarchical(minimumRung, maxRung, a, adot, tau, t0, theta, &finalEnergies,
                minrung0, true, nullptr, true, 0.0, -1, true);
        } else
#endif
        {
            // Ordinary force advancement records diagnostics between closing
            // and opening kicks. Its terminal particle state is discarded.
            advanceForces(0, a, a * adot, tau, t0, theta, false, true, finalEnergies, -1, true,
                params.globalTimestep, global ? t0 : 0.0);
        }
        double const finalGlassError = std::abs(energies0.pot - a * finalEnergies.pot) /
            std::max(std::abs(a * finalEnergies.pot), 1.0e-30);
        outputEnergy(tau / t0, a, finalEnergies, energy0, finalGlassError);
    }
    if (getOptions().lightcone.enabled) {
        particlesFree();
    }
#if COSMICTIGER_ENABLE_FMM
    kickClearBuffers();
#endif
    evolutionTime.stop();
    PRINT("  Total evolution time (this run, initialization excluded): %s\n",
        terminal::formatDuration(thisIter > 0 ? evolutionTime.read() : 0.0).c_str());
    if (getOptions().checkpoint.readIndex >= 0) {
        PRINT("  Recorded time including previous runs: %s\n",
            terminal::formatDuration(previousRuntime + (thisIter > 0 ? evolutionTime.read() : 0.0))
                .c_str());
    }
}
