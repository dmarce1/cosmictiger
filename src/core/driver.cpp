/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <functional>
#include <ios>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>

#if COSMICTIGER_ENABLE_ROCKSTAR
#include <cosmictiger/analysis/group_tree.hpp>
#include <cosmictiger/analysis/groups.hpp>
#include <cosmictiger/analysis/groups_find.hpp>
#include <cosmictiger/analysis/lightcone.hpp>
#endif
#if !defined(COSMICTIGER_WITH_GRAVITY) || COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/analysis/power.hpp>
#include <cosmictiger/math/vector3.hpp>
#endif
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/rung_merge.hpp>
#include <cosmictiger/core/sph_midpoint.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/core/time.hpp>
#include <cosmictiger/core/timer.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/cosmology/background.hpp>
#include <cosmictiger/cosmology/background_plot.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#include <cosmictiger/gravity/error.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/opening_schedule.hpp>
#include <cosmictiger/gravity/pmmm_kick.hpp>
#include <cosmictiger/gravity/solver.hpp>
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
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/scenarios/cosmology.hpp>
#include <cosmictiger/scenarios/glass.hpp>
#include <cosmictiger/scenarios/plummer.hpp>
#endif
#include <cosmictiger/scenarios/scenario.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/chemistry/sph.hpp>
#endif
#include <cosmictiger/sph/particles.hpp>
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/stars/star_formation.hpp>
#endif
#include <cosmictiger/sph/time_integration.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/sph/volume.hpp>
#endif

constexpr double kScaleTimestep = 0.01;

double domainTime = 0.0;
double sortTime = 0.0;
double kickTime = 0.0;
double driftTime = 0.0;
double flopsPerNode = 1e6;
double flopsPerParticle = 1e5;
bool usedGpu;

struct Timing {
    double numberOfParticles;
    double time;
    /// Initialize the accumulated particle count and elapsed time to zero.
    Timing() {
        numberOfParticles = time = 0.0;
    }
};

Vector<Timing> timings;

/// Write measured throughput by bucket size and optionally clear the accumulated samples.
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

#if COSMICTIGER_ENABLE_ROCKSTAR
/// Run the halo-group tree, linking, catalogue, baryon measurement, and cleanup stages.
void doGroups(int number, double scale, double years) {
    groupsPrepareHistory(number);

    Timer total;
    total.start();
    terminal::detailPrint("Doing groups\n");
    Timer tm;
    tm.start();
    {
        profile::ScopedPhase phase("groups.treeCreate");
        groupTreeCreate();
    }
    tm.stop();
    terminal::detailPrint("group_tree_create = %s\n", terminal::formatDuration(tm.read()).c_str());
    tm.reset();
    tm.start();
    {
        profile::ScopedPhase phase("groups.initialize");
        particlesGroupsInit();
    }
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
        {
            profile::ScopedPhase phase("groups.find");
            active =
                groupsFind(rootId, std::move(checklist), getOptions().analysis.linkingLength).get();
        }
        tm.stop();
        terminal::detailPrint("%i groups_find = %s active = %li\n", iter,
            terminal::formatDuration(tm.read()).c_str(), active);
        tm.reset();
        {
            profile::ScopedPhase phase("groups.cacheEpoch");
            particlesIncGroupCacheEpoch();
            groupTreeIncCacheEpoch();
        }
        iter++;
    } while (active > 0);
    tm.start();
    {
        profile::ScopedPhase phase("groups.treeDestroy");
        groupTreeDestroy();
    }
    tm.stop();
    terminal::detailPrint("tree cleanup = %s\n", terminal::formatDuration(tm.read()).c_str());
    Timer reduce;
    reduce.start();
    for (int wave = 0; wave < kGroupWaves; wave++) {
        tm.start();
        {
            profile::ScopedPhase phase("groups.gather");
            groupsAddParticles(wave, scale);
        }
        tm.stop();
        terminal::detailPrint(
            "%i groups_add_particles %s\n", wave, terminal::formatDuration(tm.read()).c_str());
        tm.reset();
        tm.start();
        {
            profile::ScopedPhase phase("groups.reduce");
            groupsReduce(scale);
        }
        tm.stop();
        terminal::detailPrint("groups_reduce %s\n", terminal::formatDuration(tm.read()).c_str());
        tm.reset();
    }
    reduce.stop();
    tm.start();
    {
        profile::ScopedPhase phase("groups.baryons");
        groupsMeasureBaryons();
    }
    tm.stop();
    terminal::detailPrint("groups_baryons %s\n", terminal::formatDuration(tm.read()).c_str());
    tm.reset();
    tm.start();
    {
        profile::ScopedPhase phase("groups.cull");
        groupsCull();
    }
    tm.stop();
    terminal::detailPrint("groups_cull %s\n", terminal::formatDuration(tm.read()).c_str());
    tm.reset();
    tm.start();
    auto ngroups = [&]() {
        profile::ScopedPhase phase("groups.save");
        return groupsSave(number, scale, years);
    }();
    tm.stop();
    terminal::detailPrint("groups_save %s\n", terminal::formatDuration(tm.read()).c_str());
    total.stop();
    terminal::detailPrint("Reduction time = %s\n", terminal::formatDuration(reduce.read()).c_str());
    terminal::detailPrint("Total time = %s\n", terminal::formatDuration(total.read()).c_str());
    terminal::detailPrint("Group count = %li of %li candidates\n", ngroups.first, ngroups.second);
    {
        profile::ScopedPhase phase("groups.cleanup");
        particlesGroupsDestroy();
    }
}

#endif

#if COSMICTIGER_WITH_SPH
// All-particle counts make the rebuild decision identical on every rank. The
// threshold is per event, not an accumulated number of force evaluations.
bool prepareSphDomainEvent(int minimumRung) {
    auto const counts = particlesRungCounts();
    size_t active = 0;
    size_t total = 0;
    for (size_t rung = 0; rung < counts.size(); ++rung) {
        total += counts[rung];
        if (int(rung) >= minimumRung) {
            active += counts[rung];
        }
    }
    bool const valid = sphTreeAllValid();
    bool const threshold =
        sphTreeShouldRebuild(active, total, getOptions().compute.sphTreeRebuildFraction);
    bool const rebuild = !valid || threshold;
    if (rebuild) {
        terminal::detailPrint("SPH octree rebuild: active=%zu total=%zu reason=%s\n", active, total,
            valid ? "threshold" : "invalidated");
        sphTreeDestroy();
    } else {
        terminal::detailPrint("SPH octree retain: active=%zu total=%zu\n", active, total);
    }
    return rebuild;
}

/// Create the SPH support tree for the requested active rung and report its geometry
/// statistics.
SphTreeCreateReturn sphStep1(int minrung, bool verbose) {
    profile::ScopedPhase profilePhase("hydro.tree.create");
    verbose = true;
    if (verbose) {
        terminal::detailPrint("Doing SPH step with minrung = %i\n", minrung);
    }
    SphTreeCreateParameters treeParameters;

    Timer timer;
    treeParameters.minimumRung = minrung;
    treeParameters.smoothingLengthMultiplier = kSphSupportSearchMultiplier;
    SphTreeCreateReturn treeResult;

    timer.start();
    if (verbose) {
        terminal::detailPrint(
            "starting sph_tree_create = %s\n", terminal::formatDuration(timer.read()).c_str());
    }
    treeResult = sphTreeAllValid() ? sphTreeRefresh(treeParameters) : sphTreeCreate(treeParameters);
    timer.stop();
    if (verbose) {
        terminal::detailPrint("sph_tree_create time = %s %i\n",
            terminal::formatDuration(timer.read()).c_str(), treeResult.numberOfActiveParticles);
        terminal::detailPrint("SPH tree swept/partition volume ratio: min=%.6e max=%.6e\n",
            treeResult.minVolumeRatio, treeResult.maxVolumeRatio);
    }
    timer.reset();
    return treeResult;
}

/// Test whether no particles occupy rungs coarser than the requested boundary.
static bool sphCommonBoundary(int minimumRung) {
    auto const counts = particlesRungCounts();
    for (int rung = 0; rung < minimumRung; ++rung) {
        if (counts[rung] != 0) {
            return false;
        }
    }
    return true;
}

/// Refresh all gas supports, balance neighbor rung proposals, and commit the resulting
/// rungs.
static int sphBalanceBoundaryRungs(double scale) {
    profile::ScopedPhase phase("hydro.rungs.balance");
    if (sphMidpointTargetCount(-1) == 0) {
        return 0;
    }
    // Refresh all supports, including inactive neighbors. Creating this octree
    // reorders only gas records; base-particle-indexed gravity scratch is stable.
    auto const tree = sphStep1(0);
    sphSeedRungProposals();
    auto const balanced = sphBalanceRungs(tree.id, scale);
    int const maximumRung = sphCommitBalancedRungs();
    terminal::detailPrint("SPH rung balance: passes=%d promoted=%zu maximum=%d\n", balanced.passes,
        balanced.changedParticles, maximumRung);
    return maximumRung;
}

/// Evaluate one midpoint hydro event from staged sources and commit its paired increments.
static SphRunReturn sphMidpointEvaluate(int minrung, int maximumHydroRung, double scale, double tau,
    double t0, double adot, double hydroTimestep, double maximumGlobalTimestep,
    double viscosityTimestep = 0.0) {
    bool constexpr verbose = true;
    profile::ScopedPhase profilePhase("hydro.sph");
    bool const globalTimestep = maximumGlobalTimestep > 0.0;
    if (verbose) {
        terminal::detailPrint("Doing SPH step with minrung = %i time=%.17g\n", minrung, tau);
    }
    SphRunParameters runParameters;
    runParameters.midpointHydro = true;
    runParameters.hydroKickTimestep = std::max(0.0, hydroTimestep);
    runParameters.minimumRung = minrung;
    runParameters.maximumRung = maximumHydroRung;
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().run.scenario == "cosmology") {
        runParameters.maximumTimestep =
            cosmology::expansionTimestep(scale, adot / scale, cosmosD2adt2(scale), kScaleTimestep);
    }
#endif
    runParameters.globalTimestep = globalTimestep;
    if (globalTimestep) {
        runParameters.minimumProposedRung = 0;
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
    runParameters.scaleFactor = scale;
    runParameters.baseTimestep = t0;
    // Evolve alpha only at a completed endpoint. For dyadic integration the
    // density pass divides the base duration by each closing particle's rung.
    runParameters.viscosityTimestep = viscosityTimestep;
    bool needsNeighborRefresh;
    SphRunReturn result;

    runParameters.phase = 1;
    SphTreeNeighborParameters neighborParameters;
    neighborParameters.smoothingLengthMultiplier = kSphSupportSearchMultiplier;
    neighborParameters.minimumRung = runParameters.minimumRung;
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);

    auto prepareDensities = [&]() {
        profile::ScopedPhase profilePhase("hydro.density");
        neighborParameters.interactionDirection = SphInteractionDirection::TargetSupport;
        neighborParameters.operation = SphNeighborOperation::FindNeighbors;
        {
            profile::ScopedPhase profilePhase("hydro.neighbors.initial");
            sphTreeNeighbor(neighborParameters, rootId, checklist).get();
        }

        runParameters.phase = 0;
        int densityIterations = 0;
        {
            profile::ScopedPhase profilePhase("hydro.density.reset");
            sphParticlesResetConverged();
        }
        do {
            if (++densityIterations > 256) {
                throw std::runtime_error(
                    "SPH smoothing lengths did not converge after 256 tree rebuilds");
            }

            runParameters.runType = kSphRunSmoothingLengths;
            Timer timer;
            timer.start();
            {
                profile::ScopedPhase profilePhase("hydro.density.activeSolve");
                result = sphRun(runParameters, true);
            }
            timer.stop();
            if (verbose) {
                terminal::detailPrint("sph_run(smoothing lengths): tm = %s min_h = %e max_h = %e\n",
                    terminal::formatDuration(timer.read()).c_str(), result.minimumSmoothingLength,
                    result.maximumSmoothingLength);
            }
            timer.reset();
            needsNeighborRefresh = result.needsAnotherPass;
            neighborParameters.smoothingLengthMultiplier = kSphSupportSearchMultiplier;
            neighborParameters.operation = SphNeighborOperation::RefreshBounds;
            neighborParameters.outerParticleSelection =
                needsNeighborRefresh ? SphParticleSelection::Active : SphParticleSelection::All;
            neighborParameters.innerParticleSelection = SphParticleSelection::All;
            timer.start();
            {
                profile::ScopedPhase profilePhase("hydro.neighbors.boxes");
                sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
            }
            timer.stop();
            terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
            timer.reset();
            timer.start();
            neighborParameters.interactionDirection = needsNeighborRefresh ?
                SphInteractionDirection::TargetSupport :
                SphInteractionDirection::EitherSupport;
            neighborParameters.operation = SphNeighborOperation::FindNeighbors;
            {
                profile::ScopedPhase profilePhase("hydro.neighbors.refresh");
                sphTreeNeighbor(neighborParameters, rootId, checklist).get();
            }
            timer.stop();
            terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
            timer.reset();

            result = SphRunReturn();
        } while (needsNeighborRefresh);
        {
            profile::ScopedPhase profilePhase("hydro.density.reset");
            sphParticlesResetConverged();
        }
        densityIterations = 0;
        SphRunReturn gradHDiagnostics;
        do {
            if (++densityIterations > 256) {
                throw std::runtime_error(
                    "SPH neighbor densities did not converge after 256 tree rebuilds");
            }

            runParameters.runType = kSphRunDensityAndGradients;
            Timer timer;
            timer.start();
            {
                profile::ScopedPhase profilePhase("hydro.density.neighborSolve");
                result = sphRun(runParameters, true);
            }
            gradHDiagnostics += result;
            timer.stop();
            if (verbose) {
                terminal::detailPrint(
                    "sph_run(density and gradients): tm = %s min_h = %e max_h = %e\n",
                    terminal::formatDuration(timer.read()).c_str(), result.minimumSmoothingLength,
                    result.maximumSmoothingLength);
            }
            timer.reset();
            needsNeighborRefresh = result.needsAnotherPass;
            neighborParameters.smoothingLengthMultiplier =
                needsNeighborRefresh ? kSphSupportSearchMultiplier : kSphConvergedSearchMultiplier;
            neighborParameters.operation = SphNeighborOperation::RefreshBounds;
            neighborParameters.outerParticleSelection = SphParticleSelection::All;
            neighborParameters.innerParticleSelection = SphParticleSelection::All;
            timer.start();
            {
                profile::ScopedPhase profilePhase("hydro.neighbors.boxes");
                sphTreeNeighbor(neighborParameters, rootId, Vector<TreeId>()).get();
            }
            timer.stop();
            terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
            timer.reset();
            timer.start();
            neighborParameters.interactionDirection = SphInteractionDirection::EitherSupport;
            neighborParameters.operation = SphNeighborOperation::FindNeighbors;
            {
                profile::ScopedPhase profilePhase("hydro.neighbors.refresh");
                sphTreeNeighbor(neighborParameters, rootId, checklist).get();
            }
            timer.stop();
            terminal::detailPrint("%s\n", terminal::formatDuration(timer.read()).c_str());
            timer.reset();
            result = SphRunReturn();
        } while (needsNeighborRefresh);
        // Free supports report their smallest Ω and largest constraint residual;
        // held supports use ζ = 0. Counts include supports repeated by later passes.
        bool const anyFree = gradHDiagnostics.minimumGradHOmega < std::numeric_limits<float>::max();
        terminal::detailPrint("SPH grad-H: Omega_min = %e residual_max = %e held = %i "
                              "ill-conditioned (Omega < %g) = %i\n",
            anyFree ? gradHDiagnostics.minimumGradHOmega : 1.0f,
            gradHDiagnostics.maximumSupportResidual, gradHDiagnostics.heldSupports,
            sph_volume::kMinimumWellConditionedOmega, gradHDiagnostics.illConditionedSupports);
    };
    prepareDensities();
    auto const clearMidpointSources = []() {
        sphTreeFreeNeighborList();
        sphTreeClearNeighborRanges();
        sphParticlesResetSemiactive();
    };
    if (hydroTimestep < 0.0) {
        // Output density refresh: no force evaluation, viscosity update or entropy kick.
        clearMidpointSources();
        return result;
    }
    // Any later density preparation refreshes sources at this same event time.
    runParameters.viscosityTimestep = 0.0;
    Timer timer;
    // Each pair contributes once on its finer endpoint's midpoint cadence.
    auto const runForces = [&]() {
        profile::ScopedPhase forcePhase("hydro.force.midpoint");
        return sphMidpointForces(runParameters, true);
    };
    timer.reset();
    timer.start();
    result = runForces();
    if (globalTimestep) {
        result.minimumTimestep = std::min(result.minimumTimestep, maximumGlobalTimestep);
    }
    timer.stop();
    if (verbose) {
        terminal::detailPrint(
            "sph_run(SPH_RUN_HYDRO): tm = %s\n", terminal::formatDuration(timer.read()).c_str());
    }
    timer.reset();

    // Reciprocal event-local increments have already been delivered and committed.
    // Only the proposed rung persists after this point.
    clearMidpointSources();
    return result;
}

/// Run an exact-rung midpoint event, or initialize timestep bounds for a zero interval.
SphRunReturn sphMidpointStep(int rung, double time, double timestep, double baseTimestep,
    double scale, double scaleDerivative, EnergiesT& energies, bool global,
    double maximumTimestep) {
    if (sphMidpointTargetCount(rung) == 0) {
        // Gas on coarser rungs still contributes to the expansion-work ledger.
        sphParticlesPrepareThermodynamics(scale, false);
        return {};
    }
    // A zero interval initializes the hydro timestep bounds without kicking.
    // Exact-rung targets own pairs with coarser neighbors; each selected pair
    // updates both endpoints from the same chronological source snapshot.
    if (timestep > 0.0) {
        terminal::detailPrint("SPH midpoint: time=%.17g rung=%d dt=%.17g\n", time, rung, timestep);
    }
    int const minimumRung = std::max(0, rung);
    int const maximumRung = rung < 0 ? kMaxRung - 1 : rung;
    sphMidpointBeginEvent();
    try {
        sphStep1(minimumRung);
        auto const result = sphMidpointEvaluate(minimumRung, maximumRung, scale, time, baseTimestep,
            scaleDerivative, timestep, global ? maximumTimestep : 0.0);
        sphMidpointEndEvent();
        return result;
    } catch (...) {
        sphMidpointEndEvent();
        throw;
    }
}

/// Close endpoint source physics and viscosity relaxation for the selected hydro interval.
SphRunReturn sphCompleteSources(int minimumClosingRung, double time, double baseTimestep,
    double scale, double scaleDerivative, EnergiesT& energies, double explicitTimestep,
    double maximumGlobalTimestep) {
    auto const& options = getOptions();
    bool const sourcePhysics = options.starFormation.enabled || options.chemistry.enabled;
    // Morris-Monaghan viscosity always closes here with its published endpoint relaxation
    // prescription and documented accuracy limit.
    ALWAYS_ASSERT(minimumClosingRung >= 0 && minimumClosingRung < kMaxRung);
    ALWAYS_ASSERT(explicitTimestep != 0.0);
    if (sphMidpointTargetCount(-1) == 0) {
        sphParticlesPrepareThermodynamics(scale, false);
        energies.therm = 0.0;
        return {};
    }
    profile::ScopedPhase phase("hydro.sources.close");
    sphMidpointBeginEvent();
    try {
        // Every pair involving a closing particle has finished its old interval:
        // the pair cadence is the finer endpoint's cadence. Refresh endpoint
        // density before consuming that particle's old rung and full duration.
        sphStep1(minimumClosingRung);
        // Live dyadic rungs recover each closing interval from the base duration;
        // global integration supplies its exact duration.
        double const viscosityTimestep =
            maximumGlobalTimestep > 0.0 ? explicitTimestep : baseTimestep;
        auto const endpointEstimate = sphMidpointEvaluate(minimumClosingRung, kMaxRung - 1, scale,
            time, baseTimestep, scaleDerivative, sourcePhysics ? -1.0 : 0.0, maximumGlobalTimestep,
            viscosityTimestep);
        // Kernel parameters carry a float scale, while the source transaction
        // and checkpoint clock use the exact double endpoint scale.
        sphParticlesPrepareThermodynamics(scale, false);
        if (options.starFormation.enabled) {
#if COSMICTIGER_ENABLE_STAR_FORMATION
            auto const update = starFormationClose(
                minimumClosingRung, scale, scaleDerivative, baseTimestep, explicitTimestep);
            energies.heating += update.heating;
#endif
        } else if (options.chemistry.enabled) {
#if COSMICTIGER_ENABLE_CHEMISTRY
            energies.heating += sphChemistryClose(
                minimumClosingRung, scale, scaleDerivative, baseTimestep, explicitTimestep);
#endif
        }
        SphRunReturn result = endpointEstimate;
        if (sourcePhysics && sphMidpointTargetCount(-1) != 0) {
            // Heating and conversion change neighboring signal speeds too. An
            // all-gas, zero-kick estimate refreshes proposals and the global cap
            // without advancing viscosity or changing live gas rungs.
            // Rung changes still wait for the next common gravity boundary.
            sphStep1(0);
            result = sphMidpointEvaluate(0, kMaxRung - 1, scale, time, baseTimestep,
                scaleDerivative, 0.0, maximumGlobalTimestep);
        }
        sphParticlesPrepareThermodynamics(scale, false);
        energies.therm = particlesSumEnergies().therm / sqr(scale);
        sphMidpointEndEvent();
        return result;
    } catch (...) {
        sphMidpointEndEvent();
        throw;
    }
}

#endif

#if COSMICTIGER_WITH_GRAVITY
namespace detail {

// Called only after the old-membership closing ladder, before migration and
// the fresh far solve. Inner passes and terminal closing-only passes keep theta.
void updateOpeningSchedule(double scale) {
    if (getOptions().gravity.openingSchedule.empty() ||
        getOptions().gravity.openingSchedule == "none") {
        return;
    }
    auto opts = getOptions();
    auto const schedule = gravity::parseOpeningSchedule(opts.gravity.openingSchedule);
    double const theta = gravity::openingScheduleTheta(
        schedule, opts.cosmology.startRedshift, 1.0 / scale - 1.0, opts.gravity.openingAngle);
    if (theta != opts.gravity.openingAngle) {
        terminal::detailPrint("Opening schedule at z=%.17g: theta %.17g -> %.17g\n",
            1.0 / scale - 1.0, opts.gravity.openingAngle, theta);
        // Green spectra and retained FFT plans encode the old near stencil.
        // All old closing work is complete; rebuild them for the next far solve.
        pmmmDestroy();
        opts.gravity.openingAngle = theta;
        setOptions(opts);
    }
}

// PMMM outer interval (docs/pmmm.md): migrate every owned particle to its
// domain at a boundary, rebalancing first when the load has drifted. Inside an
// interval nothing migrates, so each frozen cell keeps its owner.
void pmmmBoundaryMigrate(bool initial) {
    Timer timer;
    timer.start();
    profile::profileBegin("domains");
    if (!initial && domainsGetLoadImbalance() > kMaxLoadImbalance) {
        profile::ScopedPhase phase("domains.rebound");
        terminal::detailPrint("Doing rebound\n");
        domainsRebound();
    }
    {
        profile::ScopedPhase phase("domains.migrate");
        domainsBegin(0);
    }
    {
        profile::ScopedPhase phase("domains.install");
        domainsEnd();
    }
    profile::profileEnd();
    timer.stop();
    domainTime += timer.read();
}

/// Construct gravity kick parameters and timestep bounds for a frozen-near mesh interval.
KickParameters pmmmIntervalParameters(
    int minRung, double scale, double scaleDerivative, double baseTimestep, bool firstCall) {
    double maximumTimestep = std::ldexp(baseTimestep, -getOptions().run.minRung);
    if (getOptions().run.scenario == "cosmology") {
        maximumTimestep = std::min(maximumTimestep,
            cosmology::expansionTimestep(
                scale, scaleDerivative / scale, cosmosD2adt2(scale), kScaleTimestep));
    }
    KickParameters parameters;
    parameters.htime = false;
    parameters.ascending = parameters.descending = parameters.top = true;
    parameters.minRung = minRung;
    parameters.t0 = baseTimestep;
    parameters.a = scale;
    parameters.eta = getOptions().gravity.timestepEta;
    parameters.gravitySmoothingLength = getOptions().gravity.smoothingLength;
    parameters.gm = getOptions().gravity.gm;
    parameters.sign = getOptions().run.scenario == "glass" ? -1.0 : 1.0;
    parameters.firstCall = firstCall;
    parameters.saveForce = getOptions().analysis.saveForce;
    parameters.gpu = getOptions().gravity.cuda;
    parameters.doPhi = false;
    parameters.maxDt = maximumTimestep;
    parameters.frozenNear = true;
    return parameters;
}

// One traditional force pass of the PMMM outer interval. Inside the interval
// it is a near-only pass on the frozen membership. At a boundary
// (schedule.pmmmBoundary) it performs, with S the near field, L the far field,
// T = L + S, H the outer durations and δ the particle's inner durations:
//
//   1. the old-membership closing pass, Δp = (δ_old/2 − H_old/2) S_old,
//      skipped on the first call;
//   2. migration (rebound when unbalanced) and the far solve of every particle
//      on the fresh assignment, which selects H_new;
//   3. the new-membership pass, closing (H_old/2)(L_new + S_new) and opening
//      (H_new/2) L_new + (δ_new/2) S_new with the rungs chosen from S_new.
//
// Together the closing impulse is (δ_old/2) S_old + (H_old/2)(T_new − S_old):
// the old interval's short half-kick and its exact long complement L_close =
// T_new − S_old, so no far expansion is ever evaluated on wandered particles.
// Symbol directory: S_old → near force on the frozen membership; S_new, L_new →
// near and far forces on the fresh assignment; H_old → schedule.closingBaseTimestep;
// H_new → far.nextInterval; δ → rungDt[rung] times the interval.
ForceStepResult pmmmIntervalStep(int minRung, double scale, double scaleDerivative,
    double baseTimestep, bool firstCall, bool closingOnly, bool global,
    MeshKickSchedule const& schedule) {
    if (global) {
        throw std::invalid_argument(
            "The global inner integrator is not available for the PMMM outer interval");
    }
    auto* const boundary = schedule.pmmmBoundary;
    if (boundary == nullptr) {
        auto const parameters =
            pmmmIntervalParameters(minRung, scale, scaleDerivative, baseTimestep, firstCall);
        profile::ScopedPhase phase("pmmmKick.inner");
        return pmmmKick(parameters);
    }
    ForceStepResult closing;
    if (!firstCall) {
        auto parameters =
            pmmmIntervalParameters(0, scale, scaleDerivative, schedule.closingBaseTimestep, false);
        parameters.frozenNearClose = true;
        parameters.closingBaseTimestep = schedule.closingBaseTimestep;
        profile::ScopedPhase phase("pmmmKick.closeOldMembership");
        closing = pmmmKick(parameters);
    }
    if (!closingOnly) {
        updateOpeningSchedule(scale);
    }
    pmmmBoundaryMigrate(firstCall);
    PmBoundaryResult far;
    {
        profile::ScopedPhase phase("pmmmBoundary");
        far = pmmmOuterBoundary(
            scale, scaleDerivative, boundary->time, boundary->coarseInterval, boundary->endTime);
    }
    boundary->nextInterval = far.nextInterval;
    boundary->potentialEnergy = far.potentialEnergy;
    if (!closingOnly && !(far.nextInterval > 0.0)) {
        throw std::logic_error("The PMMM outer interval must be positive before the end time");
    }
    double const nextInterval = closingOnly ? schedule.closingBaseTimestep : far.nextInterval;
    auto parameters = pmmmIntervalParameters(0, scale, scaleDerivative, nextInterval, firstCall);
    parameters.refreshMembership = true;
    parameters.closeShortOverInterval = !firstCall;
    parameters.closingBaseTimestep = schedule.closingBaseTimestep;
    parameters.longCloseTimestep = schedule.longCloseTimestep;
    parameters.longOpenTimestep = closingOnly ? 0.0 : far.nextInterval;
    parameters.doPhi = true;
    ForceStepResult result;
    {
        profile::ScopedPhase phase("pmmmKick.boundary");
        result = pmmmKick(parameters, true);
    }
    result.pot += far.potentialEnergy;
    result.dkin += closing.dkin;
    return result;
}

}    // namespace detail
#endif

/// Coordinate gravity, hydrodynamics, and optional reference-error sampling for the active
/// force step.
ForceStepResult advanceForces(int minRung, double scale, double scaleDerivative, double time,
    double baseTimestep, double theta, bool firstCall, bool fullEvaluation, EnergiesT& energies,
    int gravityErrorStep, bool closingOnly, double previousTimestep, double maximumGlobalTimestep,
    MeshKickSchedule const& schedule, bool deferHydroOpening) {
    ForceStepResult result{};
    bool const global = maximumGlobalTimestep > 0.0;
    bool const meshGravity = getOptions().gravity.enabled;
    [[maybe_unused]] bool const meshInterval =
        getOptions().gravity.enabled && usesOuterInterval(getOptions().gravity);
#if COSMICTIGER_WITH_SPH
    bool const midpointHydro = sphMidpointEnabled();
#endif
    if (global) {
        minRung = 0;
        result.minimumTimestep = maximumGlobalTimestep;
    }
#if COSMICTIGER_WITH_GRAVITY
    if (gravityErrorStep >= 0) {
        profile::profileBegin("gravityError");
        gravityErrorBegin();
        profile::profileEnd();
    }
    if (getOptions().gravity.enabled) {
        if (meshGravity && meshInterval) {
            // Outer interval with frozen near membership: migration happens only
            // inside the boundary pass, inner passes solve no mesh.
            Timer timer;
            timer.start();
            result = detail::pmmmIntervalStep(minRung, scale, scaleDerivative, baseTimestep,
                firstCall, closingOnly, global, schedule);
            timer.stop();
            kickTime += timer.read();
        } else if (meshGravity) {
            // PMMM sorts particles while building its cell forest. Keep hydro
            // serialized until this backend exposes an immutable source stage.
            Timer timer;
            timer.start();
            profile::profileBegin("domains");
            {
                profile::ScopedPhase phase("domains.migrate");
                domainsBegin(0);
            }
            {
                profile::ScopedPhase phase("domains.install");
                domainsEnd();
            }
            profile::profileEnd();
            timer.stop();
            domainTime += timer.read();
            timer.reset();
            timer.start();
            double maximumTimestep = global ? maximumGlobalTimestep :
                                              std::ldexp(baseTimestep, -getOptions().run.minRung);
            if (getOptions().run.scenario == "cosmology") {
                maximumTimestep = std::min(maximumTimestep,
                    cosmology::expansionTimestep(
                        scale, scaleDerivative / scale, cosmosD2adt2(scale), kScaleTimestep));
            }
            if (global) {
                profile::profileBegin("pmmmSolve");
                pmmmSolve(true);
                profile::profileEnd();
                result = particleMeshPrepareKick(
                    scale, baseTimestep, maximumTimestep, firstCall, minRung, previousTimestep);
            } else {
                // Traditional PMMM retains all particles as force sources, but
                // kicks only active sinks on their individual block timesteps.
                KickParameters parameters;
                parameters.htime = false;
                parameters.ascending = parameters.descending = parameters.top = true;
                parameters.minRung = minRung;
                parameters.t0 = baseTimestep;
                parameters.a = scale;
                parameters.eta = getOptions().gravity.timestepEta;
                parameters.gravitySmoothingLength = getOptions().gravity.smoothingLength;
                parameters.gm = getOptions().gravity.gm;
                parameters.sign = getOptions().run.scenario == "glass" ? -1.0 : 1.0;
                parameters.firstCall = firstCall;
                parameters.saveForce = getOptions().analysis.saveForce;
                parameters.gpu = getOptions().gravity.cuda;
                parameters.doPhi = true;
                parameters.maxDt = maximumTimestep;
                profile::profileBegin("pmmmKick");
                result = pmmmKick(parameters);
                profile::profileEnd();
            }
            timer.stop();
            kickTime += timer.read();
        }
    } else
#endif
    {
        profile::profileBegin("domains");
        bool migrate = true;
#if COSMICTIGER_WITH_SPH
        if (getOptions().hydro.enabled) {
            migrate = prepareSphDomainEvent(minRung);
        }
#endif
        if (migrate) {
            profile::ScopedPhase phase("domains.migrate");
            domainsBegin(minRung);
        }
        if (migrate) {
            profile::ScopedPhase phase("domains.install");
            domainsEnd();
        }
        profile::profileEnd();
    }
    if (global) {
        result.minimumTimestep = std::min(result.minimumTimestep, maximumGlobalTimestep);
    }
    auto& gravity = result;
#if COSMICTIGER_WITH_SPH
    if (midpointHydro) {
        if (!getOptions().gravity.enabled && !global && !closingOnly) {
            if (sphCommonBoundary(minRung)) {
                sphParticlesSelectMidpointRungs(minRung);
                result.maxRung = sphBalanceBoundaryRungs(scale);
            } else {
                auto const counts = particlesRungCounts();
                for (int rung = 0; rung < counts.size(); ++rung) {
                    if (counts[rung] != 0) {
                        result.maxRung = rung;
                    }
                }
            }
        }
        // The previous H used its midpoint scale. Boundary diagnostics need
        // the current scale in the entropy-to-energy conversion for general γ.
        sphParticlesPrepareThermodynamics(scale, false);
        auto const gas = particlesSumEnergies();
        energies.therm = gas.therm / sqr(scale);
        if (global || fullEvaluation) {
            bool const selfGravity = getOptions().gravity.enabled;
            energies.kin = (selfGravity ? gravity.kin : gas.kin) / sqr(scale);
            energies.xmom = (selfGravity ? gravity.xmom : gas.xmom) / scale;
            energies.ymom = (selfGravity ? gravity.ymom : gas.ymom) / scale;
            energies.zmom = (selfGravity ? gravity.zmom : gas.zmom) / scale;
            energies.nmom = (selfGravity ? gravity.nmom : gas.nmom) / scale;
            energies.pot = gravity.pot / scale;
        }
        energies.tckin = gas.kin;
    }
#endif
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().gravity.enabled && !getOptions().hydro.enabled) {
        if (global || fullEvaluation) {
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
    }
    if (global && !closingOnly && getOptions().gravity.enabled) {
        {
            profile::ScopedPhase phase("pmmmKick.finishGlobal");
            particleMeshFinishGlobalKick(result.minimumTimestep);
        }
        {
            profile::ScopedPhase phase("diagnostics.energies");
            energies.tckin = particlesSumEnergies().kin;
        }
    }
    if (gravityErrorStep >= 0) {
        profile::profileBegin("gravityError");
        gravityErrorFinish(gravityErrorStep, time, scale);
        profile::profileEnd();
    }
#endif
    return result;
}

#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
// Rung-restricted drift is locality-local; dispatch each completed stage
// from the root to every locality for either gravity solver.
void driftRungEverywhere(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag);
HPX_PLAIN_ACTION(driftRungEverywhere, DriftRungEverywhereAction);

/// Drift the requested rung across the locality tree, draining remote work before
/// propagating errors.
void driftRungEverywhere(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag) {
    Vector<hpx::future<void>> futures;
    try {
        for (auto const& child : hpxChildren()) {
            futures.push_back(hpx::async<DriftRungEverywhereAction>(
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

/// Advance the recursive rung kick/drift schedule and accumulate its force and timing
/// results.
ForceStepResult kickStepHierarchical(int& minrung, int maxRung, double scale, double adot,
    double tau, double t0, double theta, EnergiesT* energies, int minrung0, bool doPhi,
    hpx::future<void>* lcFut, bool nocrop, double drag, int gravityErrorStep, bool closingOnly,
    MeshKickSchedule const& schedule, bool deferHydroOpening) {
    // A terminal energy evaluation consumes only the outstanding closing
    // ladder. The saved checkpoint retains the original staggered momenta.
    nocrop = nocrop || closingOnly;
    bool const selfGravity = getOptions().gravity.enabled;
    bool const meshGravity = selfGravity;
    bool const meshInterval = selfGravity && usesOuterInterval(getOptions().gravity);
    // The PMMM outer interval completes its boundary schedule at the top level.
    MeshKickSchedule meshSchedule = schedule;
    bool const hierarchicalHydro = getOptions().hydro.enabled;
#if COSMICTIGER_WITH_SPH
    bool const balanceHydro = hierarchicalHydro && !closingOnly && sphCommonBoundary(minrung);
    bool hydroBalanced = false;
#endif
    Timer tm;
    KickReturn kr;
    //	minrung = std::max(minrung, 1);
    //	max_rung = std::max(max_rung,minrung);
    Vector<int> levels(std::max(maxRung - minrung, 0) + 1);
    int k = 0;
    for (int i = maxRung; i > minrung; i--) {
        levels[k++] = i;
    }
    levels[k++] = minrung;
    bool ascending = true;
    bool top;
    bool clipTop = false;

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
        } else if (top) {
        } else if (!ascending) {
        }

        if (!ascending && !top) {
            particlesPushRungs();
        }

        bool migrateThisLevel = ascending || top;
        // PMMM outer interval: frozen membership keeps every particle with its
        // owner inside an interval; only the boundary top level migrates, after
        // the ascending levels have closed the old membership.
        bool const meshBoundary = meshInterval && top && schedule.pmmmBoundary != nullptr;
        if (meshBoundary && !closingOnly) {
            detail::updateOpeningSchedule(scale);
            theta = getOptions().gravity.openingAngle;
        }
        if (meshInterval) {
            migrateThisLevel = meshBoundary;
        }
#if COSMICTIGER_WITH_SPH
        if (hierarchicalHydro && migrateThisLevel) {
            migrateThisLevel = prepareSphDomainEvent(levels[li]);
        }
#endif
        if (migrateThisLevel) {
            tm.reset();
            tm.start();
            profile::profileBegin("domains");
            if (meshBoundary) {
                // Every owned particle is in range for the all-rung migration.
                particlesSortByRung(0);
                if (tau > 0.0 && domainsGetLoadImbalance() > kMaxLoadImbalance) {
                    profile::ScopedPhase phase("domains.rebound");
                    terminal::detailPrint("Doing rebound\n");
                    domainsRebound();
                }
            }
            {
                profile::ScopedPhase phase("domains.migrate");
                domainsBegin(meshBoundary ? -1 : levels[li]);
            }
            {
                profile::ScopedPhase phase("domains.install");
                domainsEnd();
            }
            profile::profileEnd();
            tm.stop();
        }

        if (!ascending) {
            tm.reset();
            tm.start();
            profile::profileBegin("sortByRung");
            particlesSortByRung(levels[li]);
            profile::profileEnd();
            tm.stop();
        }
        auto rng = particlesCurrentRange();
        particlesProcessed += rng.second - rng.first;
        if (top && minrung == minrung0 && !nocrop) {
            // Close the lowest rung and reopen it one rung finer when the
            // occupied finer set meets the configured inclusive threshold.
            auto const counts = particlesRungCounts();
            double const fraction = getOptions().run.rungMergeFraction;
            auto const merge = rungMergeDecision(counts, minrung0, fraction);
            if (merge.merge) {
                clipTop = true;
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
        if (selfGravity && top && gravityErrorStep >= 0) {
            profile::profileBegin("gravityError");
            gravityErrorBegin();
            profile::profileEnd();
        }
        if (meshBoundary) {
            // The far solve on the fresh assignment selects the next interval;
            // the top kick then closes the old interval and opens the new one on
            // the refreshed membership (docs/pmmm.md).
            profile::profileBegin("pmmmBoundary");
            auto const far = pmmmOuterBoundary(scale, scale * adot, tau,
                schedule.pmmmBoundary->coarseInterval, schedule.pmmmBoundary->endTime);
            profile::profileEnd();
            schedule.pmmmBoundary->nextInterval = far.nextInterval;
            schedule.pmmmBoundary->potentialEnergy = far.potentialEnergy;
            if (!closingOnly) {
                if (!(far.nextInterval > 0.0)) {
                    throw std::logic_error(
                        "The PMMM outer interval must be positive before the end time");
                }
                t0 = far.nextInterval;
            }
            meshSchedule.longOpenTimestep = closingOnly ? 0.0 : far.nextInterval;
            meshSchedule.longPotentialEnergy = far.potentialEnergy;
        }

        KickParameters kparams;
        if (clipTop && levels[li] == minrung0 + 1) {
            kparams.top = true;
        } else {
            kparams.top = top;
        }
        kparams.sign = getOptions().run.scenario == "glass" ? -1.0 : 1.0;
        kparams.ascending = ascending || top;
        if (closingOnly || (clipTop && top)) {
            kparams.descending = false;
        } else {
            kparams.descending = !ascending || top;
        }
        kparams.velInit = getOptions().run.scenario == "plummer";
        kparams.nodeLoad = flopsPerNode / flopsPerParticle;
        kparams.gpu = getOptions().gravity.cuda;
        kparams.minLevel = 0;
        kparams.saveForce = getOptions().analysis.saveForce;
        kparams.gm = getOptions().gravity.gm;
        kparams.eta = getOptions().gravity.timestepEta;
        kparams.gravitySmoothingLength = getOptions().gravity.smoothingLength;
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
#if COSMICTIGER_WITH_SPH
        if (deferHydroOpening && top) {
            ALWAYS_ASSERT(!balanceHydro && !closingOnly);
            // This is after the final boundary migration. Later ladder levels
            // reorder base particles only; gas-indexed impulse storage survives.
            gravityBeginDeferredGasOpening();
        }
        kparams.deferHydroOpening = deferHydroOpening && !ascending;
        if (hierarchicalHydro) {
            // At a common all-particle boundary the top walk supplies total G.
            // Choose the full gas rung there, close its neighbor constraints,
            // then keep these nested source sets fixed on correction levels.
            // Other boundaries retain gas rungs until every interval closes.
            kparams.fullGasRungSelection =
                balanceHydro && !hydroBalanced && kparams.top && kparams.descending;
            kparams.freezeGasRungs = !kparams.fullGasRungSelection;
        }
#endif
        // Every ascending level closes a substep of the previous outer interval;
        // only the top level applies the long half-kicks of the boundary.
        kparams.closingBaseTimestep = meshSchedule.closingBaseTimestep;
        if (kparams.top) {
            kparams.longCloseTimestep = meshSchedule.longCloseTimestep;
            kparams.longOpenTimestep = closingOnly ? 0.0 : meshSchedule.longOpenTimestep;
        }
        // PMMM outer interval: near-only passes on the frozen membership; the
        // boundary top level re-sorts by position and stores the new membership.
        kparams.frozenNear = meshInterval;
        kparams.refreshMembership = meshBoundary;
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
        tm.reset();
        tm.start();
        if (top) {
            resetGravityCounters();
            setGravityCounterUse(true);
        }
        KickReturn thisKr;
        if (!selfGravity) {
            thisKr.maxRung = top ? 0 : maxRung;
        }
        if (selfGravity && meshGravity) {
            profile::profileBegin("pmmmKick");
            thisKr = pmmmKick(kparams, meshBoundary);
            profile::profileEnd();
            if (meshInterval && top) {
                thisKr.pot += meshSchedule.longPotentialEnergy;
            }
        }
        if (top) {
            setGravityCounterUse(false);
        }
        if (selfGravity && top && gravityErrorStep >= 0) {
            profile::profileBegin("gravityError");
            gravityErrorFinish(gravityErrorStep, tau, scale);
            profile::profileEnd();
        }
        energies->tckin += thisKr.dkin;
        tm.stop();

        if (clipTop && top) {
            particlesSetMinrung(minrung0 + 1);
        }
#if COSMICTIGER_WITH_SPH
        if (kparams.fullGasRungSelection) {
            thisKr.maxRung = std::max(thisKr.maxRung, sphBalanceBoundaryRungs(scale));
            hydroBalanced = true;
        }
#endif
        if (top) {
            kr = thisKr;
            if (kparams.doPhi) {
                energies->pot = kr.pot / scale;
            }
            {
                energies->kin = kr.kin / sqr(scale);
                energies->xmom = kr.xmom / scale;
                energies->ymom = kr.ymom / scale;
                energies->zmom = kr.zmom / scale;
                energies->nmom = kr.nmom / scale;
            }
        }
        if (closingOnly && top) {
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
        if (!hierarchicalHydro && (!ascending || top) && !(clipTop && top)) {
            tm.reset();
            tm.start();
            double const dt = rungDt[levels[li]] * t0;
            if (lcFut) {
                lcFut->get();
                *lcFut = hpx::make_ready_future();
            }
            double a = scale;
            if (getOptions().run.scenario == "cosmology") {
                double scaleWork = scale;
                double adotWork = adot;
                a = 1.0 / cosmosAinv(adotWork, scaleWork, dt);
            }
            profile::profileBegin("drift");
            double const endTime = schedule.endTime > 0.0 ?
                schedule.endTime :
                getOptions().run.outputIntervalCount() * t0;
            driftRungEverywhere(a, dt, tau, tau + dt, endTime, levels[li], drag);
            profile::profileEnd();
            tm.stop();
        }
        if (ascending && !top) {
            particlesPopRungs();
        }
    }

    if (clipTop) {
        minrung++;
    }
#if COSMICTIGER_WITH_SPH
    if (deferHydroOpening) {
        gravitySealDeferredGasOpening();
    }
    if (hierarchicalHydro) {
        sphParticlesPrepareThermodynamics(scale, false);
        auto const gas = particlesSumEnergies();
        if (!closingOnly) {
            energies->tckin = gas.kin;
        }
        energies->therm = gas.therm / sqr(scale);
    }
#endif
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

#if COSMICTIGER_WITH_GRAVITY
/// Measure folded density spectra and write the filtered power and shot-noise diagnostics.
double doPowerSpectrum(int num, double a, double adot) {
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
    auto const glass = getOptions().run.scenario == "glass";
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
        throw std::ios_base::failure(
            std::format("Unable to open {} for writing", filename.c_str()));
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
        double const kshot = 2.0 * std::numbers::pi * getOptions().initial.fourierSize / boxSize;
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

/// Write scheduled output times, scale factors, and redshifts from the background history.
void outputTimeFile(DriverParameters const& parameters) {
    double const a0 = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    auto const background = cosmology::simulationBackground(parameters.tauMax);
    bool const expandingBackground = getOptions().run.scenario == "cosmology";
    double const initialTau =
        expandingBackground ? cosmosConformalTime(a0 * 1.0e-6, a0, false) : 0.0;
    double const initialTime = expandingBackground ? cosmosTime(a0 * 1.0e-6, a0) : 0.0;
    int const m = getOptions().run.outputIntervalCount();
    FILE* fp = fopen(outputPath("time.txt").c_str(), "wt");
    if (fp == NULL) {
        throw std::ios_base::failure("Unable to open time.txt for writing");
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

/// Append normalized momentum and cosmological energy diagnostics, creating the column
/// header once.
static void outputEnergy(double coarseTime, double scale, EnergiesT const& energies,
    double referenceEnergy, double glassError) {
    double const error = cosmology::normalizedEnergyError(scale, energies.kin, energies.pot,
        energies.therm, energies.heating, energies.cosmic, referenceEnergy);
    FILE* fp = fopen(outputPath("energy.txt").c_str(), "at");
    if (fp == nullptr) {
        throw std::ios_base::failure("unable to open energy.txt for writing");
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

#endif

/// Advance the simulation or checkpoint state through its configured integration and output
/// schedule.
void driver() {
    profile::checkForInterrupt();
    bool const global = usesGlobalIntegration(getOptions().run);
#if COSMICTIGER_WITH_SPH
    bool const midpointHydro = sphMidpointEnabled();
#else
    bool const midpointHydro = false;
#endif
    // PMMM outer interval: the integer rung clock subdivides the outer interval
    // instead of the nominal coarse interval, with frozen near membership inside.
    bool const meshInterval =
        getOptions().gravity.enabled && usesOuterInterval(getOptions().gravity);
    bool const outerInterval = meshInterval;
    if (scenarios::definition(getOptions().run.scenario).fixedTime) {
        runFixedScenario();
        return;
    }
#if COSMICTIGER_WITH_GRAVITY
    Timer totalTime;
    Timer tmr;
    tmr.start();
    DriverParameters params{};
    hpx::future<void> lcFut = hpx::make_ready_future();
    int buckets[11] = {80, 112, 112, 128, 128, 160, 176, 184, 184, 184, 192};
    double a0 = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    if (getOptions().checkpoint.readIndex != -1) {
        profile::profileBegin("readCheckpoint");
        params = readCheckpoint();
        profile::profileEnd();
        profile::checkForInterrupt();
        double const coarseInterval = params.tauMax / getOptions().run.outputIntervalCount();
        double const endpointTolerance =
            32.0 * std::numeric_limits<double>::epsilon() * params.tauMax;
        if (!std::isfinite(coarseInterval) || !(coarseInterval > 0.0) ||
            !std::isfinite(params.tau) || params.tau < 0.0 ||
            params.tau > params.tauMax + endpointTolerance || params.step < 0) {
            throw std::runtime_error("Checkpoint contains an invalid evolution clock");
        }
        if (outerInterval) {
            long double const phase = std::ldexp(static_cast<long double>(params.itime), -64);
            long double const expected = static_cast<long double>(params.outerStart) +
                static_cast<long double>(params.outerInterval) * phase;
            long double const clockTolerance =
                1.0e-9L * std::max(static_cast<long double>(coarseInterval), std::abs(expected));
            if ((params.iter > 0 && !(params.outerInterval > 0.0)) ||
                params.outerInterval > coarseInterval * (1.0 + 1.0e-9) ||
                (params.itime != 0 && std::abs(expected - params.tau) > clockTolerance) ||
                (params.itime == 0 && params.iter > 0 &&
                    std::abs(static_cast<long double>(params.outerStart) + params.outerInterval -
                        params.tau) > clockTolerance)) {
                throw std::runtime_error(
                    "Checkpoint outer PM interval, phase and conformal time are inconsistent; "
                    "retain the original mesh and timestep settings");
            }
            params.step = driverOutputStep(params.tau, coarseInterval);
        } else if (!global) {
            long double const phase = std::ldexp(static_cast<long double>(params.itime), -64);
            long double const coarsePosition =
                static_cast<long double>(params.tau) / coarseInterval - phase;
            long double const coarseIndex = std::round(coarsePosition);
            long double const clockTolerance = std::max(1.0e-10L,
                256.0L * std::numeric_limits<double>::epsilon() *
                    std::max(1.0L, std::abs(coarsePosition)));
            if (std::abs(coarsePosition - coarseIndex) > clockTolerance || coarseIndex < 0.0L ||
                coarseIndex >= std::numeric_limits<int>::max()) {
                throw std::runtime_error("Checkpoint conformal time and integer timestep phase are "
                                         "inconsistent; retain the original timestep settings");
            }
            int const resumedStep = static_cast<int>(coarseIndex);
            if (params.step != resumedStep &&
                !(params.itime == 0 && params.step == resumedStep - 1)) {
                throw std::runtime_error("Checkpoint coarse step and conformal time are "
                                         "inconsistent; retain the original timestep settings");
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
            if (getOptions().run.endTime == 0.0) {
                double const requestedDuration = cosmology::configuredDuration(a0);
                double const tolerance = 1.0e-10 * std::max(requestedDuration, params.tauMax);
                if (std::abs(requestedDuration - params.tauMax) > tolerance) {
                    throw std::runtime_error(
                        "Checkpoint duration differs from cosmology.endRedshift");
                }
            }
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
                throw std::runtime_error("Checkpoint cosmology/background does not match the "
                                         "requested initial redshift and cosmological parameters");
            }
        }
    } else {
        profile::profileBegin("initialize");
#if COSMICTIGER_WITH_GRAVITY
        if (getOptions().run.scenario == "glass") {
            scenarios::initializeGlass(params);
        } else if (getOptions().run.scenario == "plummer") {
            scenarios::initializePlummer(params);
        } else
#endif
        {
            if (getOptions().scenario.cosmology.gadget4Restart.empty()) {
                scenarios::initializeCosmology(params);
            } else {
                particlesReadGadget4(getOptions().scenario.cosmology.gadget4Restart);
                params.a = a0;
                params.adot = cosmosDadt(a0);
                params.tauMax = cosmology::configuredDuration(a0);
            }
        }
        profile::profileEnd();
        profile::checkForInterrupt();
        if (getOptions().run.endTime > 0.0) {
            params.tauMax = getOptions().run.endTime;
        }
        if (getOptions().analysis.tracers) {
            particlesSetTracers();
        }
        params.step = 0;
        params.flops = 0;
        params.bucketSize = getOptions().compute.bucketSize;
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
    if (getOptions().checkpoint.readIndex == -1) {
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
        if (getOptions().starFormation.enabled) {
            profile::profileBegin("starFormation");
            starFormationInitialize();
            profile::profileEnd();
        }
#endif
        outputTimeFile(params);
    }
    if (getOptions().run.scenario == "cosmology") {
        profile::ScopedPhase phase("output.background");
        cosmology::writeBackgroundPlot(params.tauMax);
        profile::checkForInterrupt();
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
    minrung0 = global ? 0 : std::max(minrung0, getOptions().run.minRung);
    auto& iter = params.iter;
    int& step = params.step;
    auto& totalProcessed = params.totalProcessed;
    auto& runtime = params.runtime;
    int numberOfSteps = getOptions().run.outputIntervalCount();
    double z0 = getOptions().cosmology.startRedshift;
    double dloga = log(z0 + 1.0) / numberOfSteps;
    double pot;
    int thisIter = 0;
    size_t progressRows = 0;
    std::string progressRungHeader;
    terminal::section(getOptions().run.scenario.c_str());
    std::cout << "  tau is conformal time / coarse interval; dt is in code units.\n"
                 "  Wall durations use hh:mm:ss.ss; initialization is excluded.\n"
              << std::flush;
    std::cout << "  GFLOP/s uses counted gravity and SPH operations divided by force "
                 "advancement wall time.\n"
                 "  FP32/FP64 separate arithmetic precision.\n"
                 "  Columns after GFLOP/s give particle populations by rung.\n"
                 "  Chemistry and star formation operations are excluded; their wall time "
                 "is included.\n"
              << std::flush;
    Timer reset;
    reset.start();
#if COSMICTIGER_ENABLE_ROCKSTAR
    if (getOptions().lightcone.enabled) {
        lcInit(tau, tauMax);
    }
#endif
    double dt = 0.0;
    int jiter = 0;
    if (tau == 0.0) {
        particlesSetMinrung(minrung0);
    }
#if COSMICTIGER_ENABLE_ROCKSTAR
    auto const checkLc = [&tau, &dt, &tauMax, &a, &iter](bool force) {
        if (force || lcTimeToFlush(tau, tauMax)) {
            profile::profileBegin("lightcone");
            Timer tm;
            terminal::detailPrint("Flushing light cone\n");
            tm.start();
            // Assign the buffered shell using its current inner radius, including
            // the final flush where the remaining lightcone radius is zero.
            {
                profile::ScopedPhase phase("lightcone.initialize");
                lcInit(tau, tauMax);
            }
            {
                profile::ScopedPhase phase("lightcone.redistributeParticles");
                lcBuffer2homes();
            }
            tm.stop();
            terminal::detailPrint(
                "lc_buffer2homes %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            {
                profile::ScopedPhase phase("lightcone.particleBoundaries");
                lcParticleBoundaries1();
            }
            tm.stop();
            terminal::detailPrint(
                "lc_particles_boundaries1 %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            double const linkingLength = getOptions().lightcone.linkingLength *
                pow(darkMatterReferenceCount(getOptions()), -1.0 / kDimensions);
            {
                profile::ScopedPhase phase("lightcone.treeCreate");
                lcFormTrees(tau, linkingLength);
            }
            tm.stop();
            terminal::detailPrint(
                "lc_form_trees %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            size_t cnt;
            {
                profile::ScopedPhase phase("lightcone.neighbors");
                lcFindNeighbors();
            }
            terminal::detailPrint("neighbors found\n");
            tm.stop();
            terminal::detailPrint(
                "lc_neighbors_found %s\n", terminal::formatDuration(tm.read()).c_str());
            Timer tm2;
            tm2.start();
            do {
                Timer tm;
                tm.start();
                {
                    profile::ScopedPhase phase("lightcone.groupBoundaries");
                    lcParticleBoundaries2();
                }
                tm.stop();
                terminal::detailPrint("%s ", terminal::formatDuration(tm.read()).c_str());
                tm.reset();
                tm.start();
                {
                    profile::ScopedPhase phase("lightcone.findGroups");
                    cnt = lcFindGroups();
                }
                tm.stop();
                terminal::detailPrint("%li %s\n", cnt, terminal::formatDuration(tm.read()).c_str());
            } while (cnt > 0);
            tm2.stop();
            terminal::detailPrint("loop time = %s\n", terminal::formatDuration(tm2.read()).c_str());
            tm.start();
            {
                profile::ScopedPhase phase("lightcone.redistributeGroups");
                lcGroups2homes();
            }
            tm.stop();
            terminal::detailPrint(
                "lc_groups2homes %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            {
                profile::ScopedPhase phase("lightcone.catalogue");
                lcParts2groups(a, linkingLength, iter);
            }
            tm.stop();
            terminal::detailPrint(
                "lc_parts2groups %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            tm.start();
            {
                profile::ScopedPhase phase("lightcone.initialize");
                lcInit(tau, tauMax);
            }
            tm.stop();
            terminal::detailPrint("lc_init %s\n", terminal::formatDuration(tm.read()).c_str());
            tm.reset();
            profile::profileEnd();
        }
    };
#endif
    auto checkpointlist = readCheckpointList();
    double const hsoft0 = getOptions().gravity.smoothingLength;
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
    double const coarseInterval = tauMax / getOptions().run.outputIntervalCount();
    int lastOutputStep = driverOutputStep(tau, coarseInterval) -
        (driverFinalOutputStep(tau, coarseInterval) == driverOutputStep(tau, coarseInterval));
    for (;; step += global ? 0 : 1) {
        if (cosmology::endTimeReached(tau, tauMax) || thisIter >= getOptions().run.maxIterations) {
            break;
        }

        double t0 = tauMax / getOptions().run.outputIntervalCount();
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
                profile::profileBegin("writeCheckpoint");
                writeCheckpoint(params);
                profile::profileEnd();
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
            int const outputStep = driverOutputStep(tau, coarseInterval);
            if (outerInterval) {
                step = outputStep;
            }
            // Mesh intervals end on coarse boundaries, so an output is due only at
            // the first interval of a coarse step.
            bool const outputDue =
                fullEval && (!(global || outerInterval) || outputStep > lastOutputStep);
            double const globalStepCap =
                global ? std::min(tauMax - tau, (outputStep + 1) * coarseInterval - tau) : 0.0;
#if COSMICTIGER_WITH_SPH
            if (midpointHydro && outputDue && tau > 0.0) {
                sphMidpointStep(
                    -1, tau, -1.0, t0, a, a * adot, energies, global, global ? t0 : 0.0);
            }
#endif
            if (outputDue) {
                lastOutputStep = outputStep;
                profile::profileBegin("output");
                int const number = outputStep;
                if (getOptions().analysis.tracers) {
                    outputTracers(number);
                }
                if (getOptions().analysis.slice) {
                    outputSlice(number, years);
                }
                if (fullEval) {
                    viewOutputViews(outputStep, a, false, years);
                }
                if (getOptions().analysis.views) {
                    Timer tm;
                    tm.start();
                    outputView(number, years);
                    tm.stop();
                }
                profile::profileEnd();
            }
            bool rebound = false;
            if (tau == 0.0) {
                rebound = true;
            } else if (meshInterval) {
                // Frozen membership: rebalancing waits for the boundary pass,
                // after the old membership has been closed.
                rebound = false;
            } else if (minrung == 0) {
                // A full reorder invalidates the saved rung suffix stack. Keep
                // the initial decomposition until a synchronized rebuild exists.
                double imbalance = domainsGetLoadImbalance();
                rebound = (imbalance > kMaxLoadImbalance);
            }
            if (rebound) {
                profile::profileBegin("rebound");
                terminal::detailPrint("Doing rebound\n");
                domainsRebound();
                {
                    profile::ScopedPhase phase("domains.migrate");
                    domainsBegin(0);
                }
                {
                    profile::ScopedPhase phase("domains.install");
                    domainsEnd();
                }
                terminal::detailPrint("Rebound done\n");
                profile::profileEnd();
            }
            MeshKickSchedule schedule;
            schedule.endTime = tauMax;
            PmmmBoundaryRequest pmmmBoundary;
            if (meshInterval && itime == 0) {
                // PMMM outer boundary: the force step closes the frozen
                // membership, migrates, solves the far field once and selects
                // the next interval, reported back through this record.
                pmmmBoundary.time = tau;
                pmmmBoundary.coarseInterval = coarseInterval;
                pmmmBoundary.endTime = tauMax;
                schedule.pmmmBoundary = &pmmmBoundary;
                schedule.closingBaseTimestep = params.outerInterval;
                schedule.longCloseTimestep = params.outerInterval;
            }
            if (meshInterval) {
                t0 = params.outerInterval;
            }
            auto opts = getOptions();
            opts.gravity.smoothingLength =
                hsoft0;    // Fixed comoving support for every particle type.
            // Retain the active interval angle until the boundary closes its old ladder.
            double const theta = opts.gravity.openingAngle;
            bucketSize = opts.compute.bucketSize;
            ForceStepResult tmp;
            bool nocrop = false;
            if ((outerInterval ? params.outerIndex % 8 : uint64_t(step % 8)) == 0 && fullEval) {
                nocrop = true;
                minrung0 = global ? 0 : std::max(minrung0 - 1, getOptions().run.minRung);
            }
            int thisMinrung = std::max(minrung, minrung0);
            int om = thisMinrung;

            if (outputDue) {
                //				kick_workspace::clear_buffers();
                //				tree_destroy(true);
                if (getOptions().analysis.power) {
                    profile::profileBegin("powerSpectrum");
                    doPowerSpectrum(outputStep, a, adot);
                    profile::profileEnd();
                }
#if COSMICTIGER_ENABLE_ROCKSTAR && !defined(CHECK_MUTUAL_SORT)
                if (getOptions().analysis.groups) {
                    profile::profileBegin("groups");
                    doGroups(outputStep, a, years);
                    profile::profileEnd();
                }
#endif
            }
            if (getOptions().scenario.cosmology.useGlass && minrung == minrung0) {
                //			particles_displace(rand1(), rand1(), rand1());
            }
            resetFlops();
            int const errorInterval = getOptions().gravity.errorInterval;
            int const gravityErrorStep =
                outputDue && errorInterval > 0 && outputStep % errorInterval == 0 ? outputStep : -1;
#if COSMICTIGER_WITH_SPH
            if (midpointHydro && tau == 0.0) {
                // With the PMMM outer interval the first interval is chosen by the boundary
                // pass of the first force step, after this estimate. Every interval is at most
                // the coarse interval, so rungs proposed against it are never too coarse.
                double const estimateBase = meshInterval && !(t0 > 0.0) ? coarseInterval : t0;
                auto const estimate = sphMidpointStep(-1, tau, 0.0, estimateBase, a, a * adot,
                    energies, global,
                    global ? estimateBase : std::ldexp(estimateBase, -getOptions().run.minRung));
                params.globalHydroTimestep = estimate.minimumTimestep;
            }
#endif
            double const forceStepCap = global && midpointHydro ?
                std::min(globalStepCap, params.globalHydroTimestep) :
                globalStepCap;
            bool deferHydroOpening = false;
#if COSMICTIGER_WITH_SPH
            int const coincidentHydroRung = global ? -1 : sphMidpointRung(itime);
            deferHydroOpening = midpointHydro && getOptions().gravity.enabled &&
                coincidentHydroRung >= 0 && sphMidpointTargetCount(coincidentHydroRung) != 0;
#endif
            if (!getOptions().run.hierarchicalGravity) {
                tmp = advanceForces(minrung, a, a * adot, tau, t0, theta, tau == 0.0, fullEval,
                    energies, gravityErrorStep, false, params.globalTimestep, forceStepCap,
                    schedule, deferHydroOpening);
#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
            } else {
                tmp = kickStepHierarchical(om, maxRung, a, adot, tau, t0, theta, &energies,
                    minrung0, fullEval, &lcFut, nocrop, sqrt(glassError), gravityErrorStep, false,
                    schedule, deferHydroOpening);
#endif
            }
#if COSMICTIGER_WITH_SPH
            if (midpointHydro && !global) {
                int const hydroRung = sphMidpointRung(itime);
                if (hydroRung >= 0) {
                    // A coarse hydro event lies between the finer gravity
                    // closing and opening half-kicks. Source velocities must
                    // see the closing side only during both RK stages.
                    sphMidpointStep(
                        hydroRung, tau, std::ldexp(t0, -hydroRung), t0, a, a * adot, energies);
#if COSMICTIGER_WITH_GRAVITY
                    if (deferHydroOpening) {
                        gravityFinishDeferredGasOpening();
                    }
#endif
                    auto const afterHydro = particlesSumEnergies();
                    energies.tckin = afterHydro.kin;
                    energies.therm = afterHydro.therm / sqr(a);
                }
            }
#endif
            if (meshInterval && itime == 0) {
                if (!(pmmmBoundary.nextInterval > 0.0)) {
                    throw std::logic_error(
                        "The PMMM outer interval must be positive before the end time");
                }
                params.outerInterval = pmmmBoundary.nextInterval;
                params.outerStart = tau;
                t0 = params.outerInterval;
            }
            if (tau == 0.0) {
                energies0 = energies;
            }
            if (om != thisMinrung) {
                minrung0++;
            }
            ForceStepResult const& kr = tmp;
            int maxRung0 = maxRung;
            // A force pass can be empty after particles move to coarser rungs.
            // Finish the current integer-clock subdivision before taking a
            // coarser step, even when that pass reports no active particle rung.
            maxRung = global ? 0 : std::max(kr.maxRung, static_cast<int>(minRung(itime)));

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
                outputEnergy(tau / coarseInterval, a, energies, energy0, glassError);
                energies0 = energies;
                energies0.pot *= a;
            }
            dt = global ? std::min(kr.minimumTimestep, tauMax - tau) : std::ldexp(t0, -maxRung);
            if (!(dt > 0.0) || !std::isfinite(dt) || tau + dt == tau) {
                throw std::underflow_error("Selected timestep cannot advance the evolution clock");
            }
            TimeType const nextIntegerTime = global ? 0 : inc(itime, maxRung);
            double const integerPhase =
                nextIntegerTime == 0 ? 1.0 : std::ldexp(double(nextIntegerTime), -64);
            double const nextTau = global ? tau + dt :
                                            (outerInterval ? params.outerStart + t0 * integerPhase :
                                                             t0 * (double(step) + integerPhase));

            double a0 = a;
            double physicalInterval = a * dt;
            double driftScale = a;
            double midpointScale = a;
            double midpointScaleDerivative = adot;
            double firstDriftScale = a;
            double secondDriftScale = a;
            if (getOptions().run.scenario != "glass" && getOptions().run.scenario != "plummer") {
                if (midpointHydro) {
                    // Integrate each cosmological half-interval independently;
                    // half of the full drift factor is not generally its value
                    // on either physical half-interval.
                    auto const first = cosmosAdvance(adot, a, 0.5 * dt);
                    midpointScale = a;
                    midpointScaleDerivative = adot;
                    auto const second = cosmosAdvance(adot, a, 0.5 * dt);
                    physicalInterval = first.physicalTime + second.physicalTime;
                    firstDriftScale = 0.5 * dt / first.driftIntegral;
                    secondDriftScale = 0.5 * dt / second.driftIntegral;
                } else {
                    auto const interval = cosmosAdvance(adot, a, dt);
                    physicalInterval = interval.physicalTime;
                    driftScale = dt / interval.driftIntegral;
                }
            }
            // d[a(K+W+U)]/da = -K+U-3 integral(P dV) + source.
            // For monatomic gamma=5/3 this reduces to the historical -(K+U).
            // Peebles, Large-Scale Structure of the Universe (1980), energy
            // equation; the gas term follows directly from dU=-P dV.
            double const driftThermal = energies.therm;
            double const workEndScale = midpointHydro ? midpointScale : a;
            energies.cosmic += cosmology::kineticExpansionWork(a0, workEndScale, energies.tckin) +
                (workEndScale - a0) * (3.0 * getOptions().hydro.heatCapacityRatio - 4.0) *
                    driftThermal;
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

            Timer dtm;
            dtm.start();
#if COSMICTIGER_WITH_SPH
            if (midpointHydro) {
                double const midpointTime = tau + 0.5 * dt;
                {
                    profile::ScopedPhase phase("drift");
                    drift(firstDriftScale, 0.5 * dt, tau, midpointTime, tauMax);
                }
                auto const hydro = sphMidpointStep(global ? -1 : maxRung, midpointTime, dt, t0,
                    midpointScale, midpointScale * midpointScaleDerivative, energies, global,
                    global ? t0 : 0.0);
                if (global) {
                    params.globalHydroTimestep = hydro.minimumTimestep;
                }
                {
                    profile::ScopedPhase phase("drift");
                    drift(secondDriftScale, 0.5 * dt, midpointTime, tau + dt, tauMax);
                }
                auto const afterHydro = particlesSumEnergies();
                energies.tckin = afterHydro.kin;
                energies.cosmic +=
                    cosmology::kineticExpansionWork(midpointScale, a, afterHydro.kin) +
                    (a - midpointScale) * (3.0 * getOptions().hydro.heatCapacityRatio - 4.0) *
                        afterHydro.therm / sqr(midpointScale);
                // Complete each old gas interval before the next gravity
                // boundary changes its rung. Checkpoints therefore contain
                // completed sources and restart never replays this event.
                // Source heat is separate from the adiabatic work above.
                int const minimumClosingRung = global ? 0 : int(minRung(nextIntegerTime));
                auto const source = sphCompleteSources(minimumClosingRung, nextTau, t0, a, a * adot,
                    energies, global ? dt : -1.0, global ? t0 : 0.0);
                if (global) {
                    params.globalHydroTimestep = source.minimumTimestep;
                }
            } else
#endif
                if (!getOptions().run.hierarchicalGravity) {
                profile::profileBegin("drift");
                drift(driftScale, dt, tau, tau + dt, tauMax);
                profile::profileEnd();
            }
            dtm.stop();
            driftTime += dtm.read();
            auto const flopRates = flopRatesPerSecond();
            double const flops = flopRates.total();
            FILE* textfp = fopen(outputPath("progress.txt").c_str(), "at");
            if (textfp == nullptr) {
                throw std::ios_base::failure("unable to open progress.txt for writing");
            }
            iter++;
            Timer remainingTime;
            totalTime.stop();
            remainingTime.start();
            runtime += totalTime.read();
            double pps = totalProcessed / runtime;

            double const numberOfParticles = getOptions().initial.numberOfParticles;
            double close = 0.0, direct = 0.0;
            getGravityCounters(close, direct);
            direct += close;
            if (outputDue) {
                fprintf(textfp,
                    "\n%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s "
                    "%10s\n",
                    "runtime", "i", "z", "a", "adot", "timestep", "years", "mnr", "mxr", "bs",
                    "Gflops", "time", "GPU close", "GPU direct", "Gflops32", "Gflops64");
            }
            stepTm.stop();
            double const nextYears = years + dyears;
            double const outputRedshift = 1.0 / a - 1.0;
            int const displayedMinimumRung = minrung;
            fprintf(textfp,
                "%10.3e %10i %10.3e %10.3e %10.3e %10.3e %10.3e %10i %10i %10i %10.3e %10.3e "
                "%10.3e %10.3e %10.3e %10.3e \n",
                runtime, iter - 1, outputRedshift, a, adot, nextTau / coarseInterval, nextYears,
                displayedMinimumRung, maxRung, bucketSize, flops * 1e-9, stepTm.read(), close,
                direct, flopRates.singlePrecision * 1e-9, flopRates.doublePrecision * 1e-9);
            auto const rungCounts = particlesRungCounts();
            std::span<size_t const> const rungPopulations{rungCounts.data(), rungCounts.size()};
            bool const siloOutput = outputDue && getOptions().output.silo;
            std::string events = siloOutput ? "SILO" : "";
            double const activeTheta = getOptions().gravity.openingAngle;
            if (activeTheta != theta) {
                if (!events.empty()) {
                    events += "; ";
                }
                events += std::format("θ {:.6g} → {:.6g}", theta, activeTheta);
            }
            auto const rungHeader = terminal::formatRungPopulations(rungPopulations, true);
            if (siloOutput || progressRows % 25 == 0 || progressRungHeader != rungHeader) {
                std::cout << std::format("\n{:>8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>7s} {:>7s} "
                                         "{:>13s} {:>13s} {:>10s} {:>10s} {:>10s}",
                                 "Step", "z", "a", "tau", "dt", "MinRung", "MaxRung", "Wall",
                                 "Step time", "FP32 GF/s", "FP64 GF/s", "GFLOP/s")
                          << rungHeader << std::format("  {:>40s}\n", "Events") << std::flush;
            }
            std::cout << std::format("{:8d} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:7d} {:7d} "
                                     "{:>13s} {:>13s} {:10.3e} {:10.3e} {:10.3e}",
                             iter, outputRedshift, a, nextTau / coarseInterval, dt,
                             displayedMinimumRung, maxRung,
                             terminal::formatDuration(runtime).c_str(),
                             terminal::formatDuration(stepTm.read()).c_str(),
                             flopRates.singlePrecision * 1e-9, flopRates.doublePrecision * 1e-9,
                             flops * 1e-9)
                      << terminal::formatRungPopulations(rungPopulations)
                      << std::format("  {:>40s}\n", events) << std::flush;
            progressRungHeader = rungHeader;
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
            if (outerInterval && itime == 0) {
                ++params.outerIndex;
            }
            domainTime = 0.0;
            sortTime = 0.0;
            kickTime = 0.0;
            driftTime = 0.0;
            tau = nextTau;
            thisIter++;
            years = nextYears;
            if (cosmology::endTimeReached(tau, tauMax)) {
                std::cout << "\n  Finished: reached the requested end time or redshift.\n"
                          << std::flush;
                break;
            } else if (thisIter >= getOptions().run.maxIterations) {
                std::cout << "\n  Stopped at the iteration limit.\n" << std::flush;
                break;
            }
            jiter++;
            if (jiter > 100) {
                //			abort();
            }
#if COSMICTIGER_ENABLE_ROCKSTAR
            if (getOptions().lightcone.enabled && minRung(itime) <= minrung0) {
                checkLc(false);
            }
#endif
            lcFut = hpx::make_ready_future();

        } while (itime != 0);
        if (interrupted || cosmology::endTimeReached(tau, tauMax) ||
            thisIter >= getOptions().run.maxIterations) {
            break;
#if COSMICTIGER_WITH_GRAVITY
        } else if (getOptions().run.scenario == "glass" && std::abs(glassError) < 1.0e-5) {
            break;
#endif
        }
    }
    lcFut.get();
    if (interrupted) {
        std::cout << "\n  Interrupted: saving the completed simulation state before shutdown.\n"
                  << std::flush;
    }
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().run.scenario == "glass") {
        scenarios::saveGlass();
    }
#endif
#if COSMICTIGER_ENABLE_ROCKSTAR
    if (getOptions().lightcone.enabled) {
        checkLc(true);
    }
#endif
    if (params.iter > 0 && params.iter != lastCheckpointIteration) {
        totalTime.stop();
        runtime += totalTime.read();
        terminal::detailPrint("Writing final checkpoint for iteration %i\n", params.iter);
        profile::profileBegin("writeCheckpoint");
        writeCheckpoint(params);
        profile::profileEnd();
        if (doCheck && hpxRank() == 0) {
            detail::checkpointRedshiftAlias(params.iter, checkz);
        }
    }
#if COSMICTIGER_WITH_SPH
    if (midpointHydro) {
        // Preserve restart state before refreshing density at the final output
        // positions. This also applies to interrupted and iteration-limit exits.
        sphMidpointStep(-1, tau, -1.0, coarseInterval, a, a * adot, energies, global,
            global ? coarseInterval : 0.0);
    }
#endif
    profile::profileBegin("output");
    viewOutputViews(driverFinalOutputStep(tau, coarseInterval), a, true, years);
    profile::profileEnd();
    if (!interrupted && thisIter > 0 && itime == 0 && cosmology::endTimeReached(tau, tauMax)) {
        // Finish the energy measurement only after all restart state and
        // snapshots have been saved. This terminal force pass may reorder and
        // synchronize particles; nothing subsequently persists or evolves them.
        // Keep the work ledger and driver state at the completed drift, and
        // leave completed-checkpoint restarts alone (thisIter == 0).
        EnergiesT finalEnergies = energies;
        double t0 = tauMax / getOptions().run.outputIntervalCount();
        double const theta = getOptions().gravity.openingAngle;
        MeshKickSchedule schedule;
        schedule.endTime = tauMax;
        PmmmBoundaryRequest terminalBoundary;
        if (meshInterval) {
            // The terminal boundary closes the frozen membership and the last
            // interval's long half-kick on the fresh far solve; nothing opens.
            terminalBoundary.time = tau;
            terminalBoundary.coarseInterval = coarseInterval;
            terminalBoundary.endTime = tauMax;
            schedule.pmmmBoundary = &terminalBoundary;
            schedule.closingBaseTimestep = params.outerInterval;
            schedule.longCloseTimestep = params.outerInterval;
            schedule.longOpenTimestep = 0.0;
            t0 = params.outerInterval;
        }
#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
        if (getOptions().run.hierarchicalGravity) {
            int minimumRung = minrung0;
            kickStepHierarchical(minimumRung, maxRung, a, adot, tau, t0, theta, &finalEnergies,
                minrung0, true, nullptr, true, 0.0, -1, true, schedule);
        } else
#endif
        {
            // Ordinary force advancement records diagnostics between closing
            // and opening kicks. Its terminal particle state is discarded.
            advanceForces(0, a, a * adot, tau, t0, theta, false, true, finalEnergies, -1, true,
                params.globalTimestep, global ? t0 : 0.0, schedule);
        }
        double const finalGlassError = std::abs(energies0.pot - a * finalEnergies.pot) /
            std::max(std::abs(a * finalEnergies.pot), 1.0e-30);
        // Energy rows are labelled on the coarse output clock, not the outer interval.
        outputEnergy(tau / (tauMax / getOptions().run.outputIntervalCount()), a, finalEnergies,
            energy0, finalGlassError);
    }
    if (getOptions().lightcone.enabled) {
        particlesFree();
    }
    evolutionTime.stop();
    if (meshInterval) {
        // One mesh transform per outer boundary (plus the terminal energy pass);
        // inner passes evaluate the near field on the frozen membership only.
        std::cout << std::format("  PMMM outer intervals completed: {:d}; mesh transforms this "
                                 "run: {:d}; force passes this run: {:d}\n",
                         static_cast<unsigned long long>(params.outerIndex),
                         static_cast<unsigned long long>(pmmmDiagnostics().transformCount),
                         thisIter)
                  << std::flush;
    }
    std::cout << std::format("  Total evolution time (this run, initialization excluded): {:s}\n",
                     terminal::formatDuration(thisIter > 0 ? evolutionTime.read() : 0.0).c_str())
              << std::flush;
    if (getOptions().checkpoint.readIndex >= 0) {
        std::cout << std::format("  Recorded time including previous runs: {:s}\n",
                         terminal::formatDuration(
                             previousRuntime + (thisIter > 0 ? evolutionTime.read() : 0.0))
                             .c_str())
                  << std::flush;
    }
#else
    throw std::invalid_argument(std::format(
        "run.scenario={} requires a gravity-enabled build", getOptions().run.scenario.c_str()));
#endif
}
