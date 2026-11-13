#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <ios>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/sph_midpoint.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/core/timer.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/gravity/kick.hpp>
#if COSMICTIGER_WITH_GRAVITY
#endif
#include <cosmictiger/gravity/kick_workspace.hpp>
#endif
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/drift.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/scenarios/scenario.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/stars/star_formation.hpp>
#endif
#endif

namespace detail {
double scenarioMaximumDensity();
}
HPX_PLAIN_ACTION(detail::scenarioMaximumDensity, ScenarioMaximumDensityAction);

namespace detail {

/// Reduce the maximum gas density across the locality tree.
double scenarioMaximumDensity() {
    std::vector<hpx::future<double>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ScenarioMaximumDensityAction>(child));
    }
    double result = 0.0;
#if COSMICTIGER_WITH_SPH
    for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
        result = std::max(result, double(sphParticlesDensity(i)));
    }
#endif
    hpx::wait_all(futures);
    for (auto& future : futures) {
        result = std::max(result, future.get());
    }
    return result;
}

/// Append energy, momentum, and maximum-density diagnostics for a fixed scenario.
void writeScenarioDiagnostics(DriverParameters const& state) {
    auto const& energy = state.energies;
    double const total = energy.kin + energy.therm + energy.pot;
    FILE* file = fopen(outputPath("energy.dat").c_str(), "at");
    if (file == nullptr) {
        throw std::ios_base::failure("Unable to write scenario energy diagnostics");
    }
    double const normalization =
        std::max(std::abs(energy.kin) + std::abs(energy.therm) + std::abs(energy.pot), 1.0e-30);
    fprintf(file, "%e %e %e %e %e %e %e %e %e\n", state.tau, energy.xmom, energy.ymom, energy.zmom,
        energy.kin, energy.therm, total, energy.pot, (total - state.energy0) / normalization);
    fclose(file);
    double const maximumDensity = []() {
        profile::ScopedPhase phase("diagnostics.maximumDensity");
        return scenarioMaximumDensity();
    }();
    file = fopen(outputPath("rho.dat").c_str(), "at");
    if (file == nullptr) {
        throw std::ios_base::failure("Unable to write scenario density diagnostics");
    }
    fprintf(file, "%e %e\n", state.tau, maximumDensity);
    fclose(file);
}

}    // namespace detail

/// Initialize or resume the chosen scenario and advance its planned stages with diagnostics
/// and output.
void runFixedScenario() {
    profile::checkForInterrupt();
    auto const& definition = scenarios::definition(getOptions().run.scenario);
    profile::profileBegin("initialize");
    auto plan = definition.initialize();
    profile::profileEnd();
    profile::checkForInterrupt();
    auto baseOptions = getOptions();
#if COSMICTIGER_WITH_SPH
    baseOptions.run.gasParticleCount = sphParticlesSize();
#else
    baseOptions.run.gasParticleCount = 0;
#endif
    setOptions(baseOptions);
    double const defaultDuration = std::accumulate(plan.stages.begin(), plan.stages.end(), 0.0,
        [](double sum, auto const& stage) { return sum + stage.duration; });
    int const defaultSteps = std::accumulate(plan.stages.begin(), plan.stages.end(), 0,
        [](int sum, auto const& stage) { return sum + stage.numberOfSteps; });
    double const duration =
        baseOptions.run.endTime > 0.0 ? baseOptions.run.endTime : defaultDuration;

    if (baseOptions.run.outputIntervalCount() < int(plan.stages.size())) {
        throw std::invalid_argument(
            "run.numberOfOutputs must allow at least one interval per scenario stage");
    }
    int assignedSteps = 0;
    int cumulativeDefaultSteps = 0;
    for (size_t i = 0; i < plan.stages.size(); i++) {
        auto& stage = plan.stages[i];
        cumulativeDefaultSteps += stage.numberOfSteps;
        int const cumulativeSteps =
            std::clamp(int(std::lround(double(cumulativeDefaultSteps) *
                           baseOptions.run.outputIntervalCount() / defaultSteps)),
                assignedSteps + 1,
                baseOptions.run.outputIntervalCount() - int(plan.stages.size() - i - 1));
        stage.numberOfSteps = cumulativeSteps - assignedSteps;
        assignedSteps += stage.numberOfSteps;
        // Stage transitions fall on the shared coarse clock. This keeps the
        // pending SPH half-step valid when entering/leaving stellar relaxation.
        stage.duration = duration * stage.numberOfSteps / baseOptions.run.outputIntervalCount();
    }
    DriverParameters state{};
    state.a = 1.0;
    state.tauMax = duration;
    state.bucketSize = baseOptions.compute.bucketSize;
    bool const restarting = baseOptions.checkpoint.readIndex >= 0;
    if (restarting) {
        profile::profileBegin("readCheckpoint");
        state = readCheckpoint();
        profile::profileEnd();
        if (state.tauMax != duration) {
            throw std::runtime_error("Checkpoint duration differs from the selected scenario");
        }
    } else {
        particlesSetMinrung(usesGlobalIntegration(baseOptions.run) ? 0 : baseOptions.run.minRung);
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
        if (baseOptions.starFormation.enabled) {
            profile::ScopedPhase phase("starFormation.initialize");
            starFormationInitialize();
        }
#endif
    }
    state.minrung0 = usesGlobalIntegration(baseOptions.run) ?
        0 :
        std::max(state.minrung0, baseOptions.run.minRung);
    bool const hierarchical = baseOptions.run.hierarchicalGravity;
    bool const global = usesGlobalIntegration(baseOptions.run);

    // A restart keeps the decomposition read from the checkpoint: each rank's file holds
    // the particles of its saved domain, and recomputing the boxes here would change
    // which rank owns which particle, and with it the order of every distributed sum.
    if (!restarting) {
        profile::profileBegin("rebound");
        domainsRebound();
        profile::profileEnd();
    }
    Timer checkpointTimer;
    checkpointTimer.start();
    Timer runtime;
    double const restoredRuntime = state.runtime;
    int iterations = 0;
    size_t progressRows = 0;
    std::string progressRungHeader;
    if (hpxRank() == 0) {
        terminal::section(definition.name);
        std::cout << std::format("{:<38s}: {:e} code time units\n", "Scenario duration", duration)
                  << std::flush;
        if (hpxSize() == 1) {
            std::cout << std::format("{:<38s}: {:d}\n", "Initialized particles",
                             static_cast<long long>(particlesSize()))
                      << std::flush;
        }
        std::cout << "  Time and dt are in code units; wall durations use hh:mm:ss.ss and include "
                     "previous runs on restart.\n"
                  << std::flush;
        std::cout << "  GFLOP/s uses counted gravity and SPH operations divided by force "
                     "advancement wall time.\n"
                     "  FP32/FP64 separate arithmetic precision.\n"
                     "  Columns after GFLOP/s give particle populations by rung.\n"
                     "  Chemistry and star formation operations are excluded; their wall "
                     "time is included.\n"
                  << std::flush;
    }
    int lastCheckpoint = restarting ? state.iter : -1;
    auto const checkpoint = [&]() {
        auto const stageOptions = getOptions();
        setOptions(baseOptions);
        profile::profileBegin("writeCheckpoint");
        writeCheckpoint(state);
        profile::profileEnd();
        setOptions(stageOptions);
        lastCheckpoint = state.iter;
        checkpointTimer.reset();
        checkpointTimer.start();
    };
    double elapsedStageTime = 0.0;
    size_t stageIndex = 0;
    bool interrupted = false;
    double const coarseInterval = duration / baseOptions.run.outputIntervalCount();
    int lastOutputStep = driverOutputStep(state.tau, coarseInterval) -
        (driverFinalOutputStep(state.tau, coarseInterval) ==
            driverOutputStep(state.tau, coarseInterval));
    runtime.start();
    for (auto const& stage : plan.stages) {
        double const stageEnd =
            ++stageIndex == plan.stages.size() ? duration : elapsedStageTime + stage.duration;
        elapsedStageTime = stageEnd;
        double const tolerance = 1.0e-12 * duration;
        if (state.tau >= (global ? stageEnd : stageEnd - tolerance)) {
            // A run stopped at a stage end (iteration limit) saved the clock before the snap
            // below; snap it here so its restart follows the uninterrupted run exactly.
            if (!global && state.itime == 0 && std::abs(state.tau - stageEnd) <= tolerance) {
                state.tau = stageEnd;
            }
            continue;
        }
        auto stageOptions = baseOptions;
        stageOptions.hydro.damping = stage.damping;
        if (stage.bulkViscosityCoefficient >= 0.0) {
            // Equal bounds hold alpha constant; beta_ab = 2 alpha_ab as for fixed Monaghan
            // coefficients. Checkpoints keep the configured bounds of baseOptions.
            stageOptions.hydro.minimumBulkViscosityCoefficient = stage.bulkViscosityCoefficient;
            stageOptions.hydro.maximumBulkViscosityCoefficient = stage.bulkViscosityCoefficient;
            stageOptions.hydro.vonNeumannRichtmyerCoefficient = 2.0;
        }
        setOptions(stageOptions);
#if COSMICTIGER_WITH_SPH
        if (stageOptions.hydro.enabled && stage.bulkViscosityCoefficient >= 0.0) {
            sphParticlesSetBulkViscosityCoefficients(float(stage.bulkViscosityCoefficient));
        }
        bool const midpointHydro = sphMidpointEnabled();
#else
        bool const midpointHydro = false;
#endif
        double const baseTimestep = duration / baseOptions.run.outputIntervalCount();
        while (state.tau < (global ? stageEnd : stageEnd - tolerance) &&
            iterations < baseOptions.run.maxIterations) {
            if (profile::interruptRequested()) {
                interrupted = true;
                break;
            }
            Timer stepTimer;
            stepTimer.start();
            int const minimumRung = global ? 0 : minRung(state.itime);
            bool const fullEvaluation = minimumRung == 0;
            int const outputStep = driverOutputStep(state.tau, baseTimestep);

            bool const outputDue = fullEvaluation && outputStep > lastOutputStep;
            // The rung clock uses the fixed output interval as its base step.
            double t0 = baseTimestep;
#if COSMICTIGER_WITH_SPH
            if (midpointHydro) {
                if (state.iter == 0) {
                    auto const estimate = sphMidpointStep(-1, state.tau, 0.0, baseTimestep, 1.0,
                        0.0, state.energies, global,
                        global ? baseTimestep : std::ldexp(baseTimestep, -baseOptions.run.minRung));
                    state.globalHydroTimestep = estimate.minimumTimestep;
                } else if (outputDue) {
                    // Refresh density at the output coordinates without a force
                    // evaluation, viscosity update, or entropy kick.
                    sphMidpointStep(-1, state.tau, -1.0, baseTimestep, 1.0, 0.0, state.energies,
                        global, global ? baseTimestep : 0.0);
                }
            }
#endif
            if (outputDue) {
                lastOutputStep = outputStep;
                profile::profileBegin("output");
                viewOutputViews(outputStep, 1.0, false,
                    state.tau * getOptions().units.timeSeconds * constants::kSecondsToYears);
                profile::profileEnd();
            }
            resetFlops();
            int const errorInterval = baseOptions.gravity.errorInterval;
            int const gravityErrorStep =
                outputDue && errorInterval > 0 && outputStep % errorInterval == 0 ? outputStep : -1;
            double const globalStepCap = global ?
                std::min(stageEnd - state.tau, (outputStep + 1) * baseTimestep - state.tau) :
                0.0;
            double const forceStepCap = global && midpointHydro ?
                std::min(globalStepCap, state.globalHydroTimestep) :
                globalStepCap;
            ForceStepResult result;
            bool deferHydroOpening = false;
#if COSMICTIGER_WITH_SPH
            int const coincidentHydroRung = global ? -1 : sphMidpointRung(state.itime);
            deferHydroOpening = midpointHydro && baseOptions.gravity.enabled &&
                coincidentHydroRung >= 0 && sphMidpointTargetCount(coincidentHydroRung) != 0;
#endif
            ForceStepSchedule schedule;
            schedule.endTime = stageEnd;

#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
            if (hierarchical) {
                int activeRung = std::max(minimumRung, state.minrung0);
                result = kickStepHierarchical(activeRung, state.maxRung, 1.0, 0.0, state.tau, t0,
                    baseOptions.gravity.openingAngle, &state.energies, state.minrung0,
                    fullEvaluation, nullptr, true, 0.0, gravityErrorStep, false, schedule,
                    deferHydroOpening);
            } else
#endif
            {
                result = advanceForces(minimumRung, 1.0, 0.0, state.tau, t0,
                    baseOptions.gravity.openingAngle, state.iter == 0, fullEvaluation,
                    state.energies, gravityErrorStep, false, state.globalTimestep, forceStepCap,
                    schedule, deferHydroOpening);
            }
#if COSMICTIGER_WITH_SPH
            if (midpointHydro && !global) {
                int const hydroRung = sphMidpointRung(state.itime);
                if (hydroRung >= 0) {
                    // Keep the coincident coarse hydro between finer gravity
                    // closing and opening half-kicks in the driver sequence.
                    sphMidpointStep(hydroRung, state.tau, std::ldexp(t0, -hydroRung), t0, 1.0, 0.0,
                        state.energies);
#if COSMICTIGER_WITH_GRAVITY
                    if (deferHydroOpening) {
                        gravityFinishDeferredGasOpening();
                        state.energies.tckin = particlesSumEnergies().kin;
                    }
#endif
                }
            }
#endif

            state.maxRung =
                global ? 0 : std::max({int(result.maxRung), baseOptions.run.minRung, minimumRung});
            int const displayedMinimumRung = minimumRung;
            if (state.iter == 0) {
                state.energy0 = state.energies.kin + state.energies.therm + state.energies.pot;
            }
            if (outputDue) {
                profile::profileBegin("diagnostics");
                detail::writeScenarioDiagnostics(state);
                profile::profileEnd();
            }
            double const timestep = global ?
                std::min(result.minimumTimestep, stageEnd - state.tau) :
                std::ldexp(t0, -state.maxRung);
            if (!std::isfinite(timestep) || !(timestep > 0.0) ||
                state.tau + timestep == state.tau) {
                throw std::underflow_error("Scenario timestep cannot advance time");
            }
#if COSMICTIGER_WITH_SPH
            if (midpointHydro) {
                double const midpointTime = state.tau + 0.5 * timestep;
                {
                    profile::ScopedPhase phase("drift");
                    drift(1.0, 0.5 * timestep, state.tau, midpointTime, duration);
                }
                auto const hydro = sphMidpointStep(global ? -1 : state.maxRung, midpointTime,
                    timestep, t0, 1.0, 0.0, state.energies, global, global ? baseTimestep : 0.0);
                if (global) {
                    state.globalHydroTimestep = hydro.minimumTimestep;
                }
                {
                    profile::ScopedPhase phase("drift");
                    drift(1.0, 0.5 * timestep, midpointTime, state.tau + timestep, duration);
                }
                int const closingRung = global ? 0 : minRung(inc(state.itime, state.maxRung));
                auto const sourceEstimate =
                    sphCompleteSources(closingRung, state.tau + timestep, t0, 1.0, 0.0,
                        state.energies, global ? timestep : -1.0, global ? baseTimestep : 0.0);
                if (global) {
                    state.globalHydroTimestep = sourceEstimate.minimumTimestep;
                }
                state.energies.tckin = particlesSumEnergies().kin;
            }
#endif
            if (!midpointHydro && !hierarchical) {
                profile::profileBegin("drift");
                drift(1.0, timestep, state.tau, state.tau + timestep, duration);
                profile::profileEnd();
            }
            auto const flopRates = flopRatesPerSecond();
            double const flops = flopRates.total();
            TimeType const nextIntegerTime = global ? 0 : inc(state.itime, state.maxRung);

            state.tau += timestep;

            state.itime = nextIntegerTime;

            if (global) {
                state.globalTimestep = timestep;
            }
            state.iter++;
            iterations++;
            if (state.itime == 0) {
                state.step++;
            }
            runtime.stop();
            state.runtime = restoredRuntime + runtime.read();
            runtime.start();
            FILE* progress = fopen(outputPath("progress.txt").c_str(), "at");
            if (progress == nullptr) {
                throw std::ios_base::failure("Unable to write scenario progress");
            }
            fprintf(progress, "%i %e %e %i %i\n", state.iter, state.tau, timestep,
                displayedMinimumRung, state.maxRung);
            fclose(progress);
            checkpointTimer.stop();
            if (checkpointTimer.read() > baseOptions.checkpoint.intervalSeconds) {
                checkpoint();
            } else {
                checkpointTimer.start();
            }
            stepTimer.stop();
            if (hpxRank() == 0) {
                runtime.stop();
                double const displayedRuntime = restoredRuntime + runtime.read();
                runtime.start();
                auto const rungCounts = particlesRungCounts();
                std::span<size_t const> const rungPopulations{rungCounts.data(), rungCounts.size()};
                bool const siloOutput = outputDue && getOptions().output.silo;
                std::string events = siloOutput ? "SILO" : "";
                auto const rungHeader = terminal::formatRungPopulations(rungPopulations, true);
                if (siloOutput || progressRows % 25 == 0 || progressRungHeader != rungHeader) {
                    std::cout << std::format("\n{:>8s} {:>13s} {:>13s} {:>7s} {:>13s} {:>13s} "
                                             "{:>13s} {:>10s} {:>10s} {:>10s}",
                                     "Step", "Time", "dt", "Done %", "Rung(min/max)", "Wall",
                                     "Step time", "FP32 GF/s", "FP64 GF/s", "GFLOP/s")
                              << rungHeader << std::format("  {:>40s}\n", "Events") << std::flush;
                }
                std::cout << std::format("{:8d} {:13.6e} {:13.6e} {:7.2f} {:6d}/{:<6d} {:>13s} "
                                         "{:>13s} {:10.3e} {:10.3e} {:10.3e}",
                                 state.iter, state.tau, timestep,
                                 std::clamp(100.0 * state.tau / duration, 0.0, 100.0),
                                 displayedMinimumRung, state.maxRung,
                                 terminal::formatDuration(displayedRuntime).c_str(),
                                 terminal::formatDuration(stepTimer.read()).c_str(),
                                 flopRates.singlePrecision * 1e-9, flopRates.doublePrecision * 1e-9,
                                 flops * 1e-9)
                          << terminal::formatRungPopulations(rungPopulations)
                          << std::format("  {:>40s}\n", events) << std::flush;
                progressRungHeader = rungHeader;
                progressRows++;
            }
        }
        if (interrupted || iterations >= baseOptions.run.maxIterations) {
            break;
        }
        state.tau = stageEnd;
        state.itime = 0;
    }
    // Preserve the completed integrator state before refreshing density for
    // final views; this keeps stopped and uninterrupted checkpoints identical.
    auto finalOptions = getOptions();
    setOptions(baseOptions);
    if (state.iter > 0 && state.iter != lastCheckpoint) {
        profile::profileBegin("writeCheckpoint");
        writeCheckpoint(state);
        profile::profileEnd();
    }
    bool const completed = state.tau >= duration * (1.0 - 1.0e-12);
    if (completed) {
        finalOptions = baseOptions;
        finalOptions.hydro.damping = plan.stages.back().damping;
        if (plan.stages.back().bulkViscosityCoefficient >= 0.0) {
            double const α = plan.stages.back().bulkViscosityCoefficient;
            finalOptions.hydro.minimumBulkViscosityCoefficient = α;
            finalOptions.hydro.maximumBulkViscosityCoefficient = α;
            finalOptions.hydro.vonNeumannRichtmyerCoefficient = 2.0;
        }
    }
    setOptions(finalOptions);
#if COSMICTIGER_WITH_SPH
    if (sphMidpointEnabled()) {
        sphMidpointStep(-1, state.tau, -1.0, coarseInterval, 1.0, 0.0, state.energies, global,
            global ? coarseInterval : 0.0);
    }
#endif
    // Use the next cycle for a final partial coarse step, preserving earlier snapshots.
    profile::profileBegin("output");
    viewOutputViews(driverFinalOutputStep(state.tau, coarseInterval), 1.0, true,
        state.tau * getOptions().units.timeSeconds * constants::kSecondsToYears);
    profile::profileEnd();
    if (completed && !interrupted && iterations > 0) {
        // Only terminal diagnostics consume the outstanding gravity closing
        // half-step. They reuse the density already refreshed for final views.
        ForceStepSchedule terminalSchedule;
        terminalSchedule.endTime = duration;
        double terminalBase = coarseInterval;

#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
        if (hierarchical) {
            int minimumRung = state.minrung0;
            kickStepHierarchical(minimumRung, state.maxRung, 1.0, 0.0, state.tau, terminalBase,
                baseOptions.gravity.openingAngle, &state.energies, state.minrung0, true, nullptr,
                true, 0.0, -1, true, terminalSchedule);
        } else
#endif
        {
            advanceForces(0, 1.0, 0.0, state.tau, terminalBase, baseOptions.gravity.openingAngle,
                false, true, state.energies, -1, true, state.globalTimestep,
                global ? coarseInterval : 0.0, terminalSchedule);
        }
        {
            profile::ScopedPhase phase("diagnostics");
            detail::writeScenarioDiagnostics(state);
        }
    }
#if COSMICTIGER_WITH_GRAVITY

#endif
    setOptions(baseOptions);
    // Scenario diagnostics require the synchronized terminal closing pass.
    if (completed && !interrupted && iterations > 0 && plan.finish != nullptr) {
        profile::ScopedPhase phase("diagnostics.scenario");
        plan.finish(state.tau);
    }
    runtime.stop();
    if (hpxRank() == 0) {
        std::cout << std::format("\n  {:s}: {:s} at t={:.6e} after {:d} steps.\n", definition.name,
                         completed ?
                             "finished" :
                             (interrupted ? "interrupted" : "paused at the iteration limit"),
                         state.tau, state.iter)
                  << std::flush;
        double const evolutionSeconds = iterations > 0 ? runtime.read() : 0.0;
        std::cout << std::format(
                         "  Total evolution time (this run, initialization excluded): {:s}\n",
                         terminal::formatDuration(evolutionSeconds).c_str())
                  << std::flush;
        if (restarting) {
            std::cout << std::format("  Recorded time including previous runs: {:s}\n",
                             terminal::formatDuration(restoredRuntime + evolutionSeconds).c_str())
                      << std::flush;
        }
    }
}
