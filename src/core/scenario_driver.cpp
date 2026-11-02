#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/core/timer.hpp>
#if COSMICTIGER_WITH_GRAVITY
#if COSMICTIGER_ENABLE_FMM
#include <cosmictiger/gravity/dehnen/execution.hpp>
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
#include <cosmictiger/sph/sph_particles.hpp>
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/sph/star_formation.hpp>
#endif
#endif

namespace detail {
double scenarioMaximumDensity();
}
HPX_PLAIN_ACTION(detail::scenarioMaximumDensity, ScenarioMaximumDensityAction);

namespace detail {

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

void writeScenarioDiagnostics(DriverParameters const& state) {
    auto const& energy = state.energies;
    double const total = energy.kin + energy.therm + energy.pot;
    FILE* file = fopen(outputPath("energy.dat").c_str(), "at");
    if (file == nullptr) {
        THROW_ERROR("Unable to write scenario energy diagnostics\n");
    }
    double const normalization =
        std::max(std::abs(energy.kin) + std::abs(energy.therm) + std::abs(energy.pot), 1.0e-30);
    fprintf(file, "%e %e %e %e %e %e %e %e %e\n", state.tau, energy.xmom, energy.ymom, energy.zmom,
        energy.kin, energy.therm, total, energy.pot, (total - state.energy0) / normalization);
    fclose(file);
    double const maximumDensity = scenarioMaximumDensity();
    file = fopen(outputPath("rho.dat").c_str(), "at");
    if (file == nullptr) {
        THROW_ERROR("Unable to write scenario density diagnostics\n");
    }
    fprintf(file, "%e %e\n", state.tau, maximumDensity);
    fclose(file);
}

}    // namespace detail

void runFixedScenario() {
    profile::checkForInterrupt();
    auto const& definition = scenarios::definition(getOptions().run.scenario);
    profile::profileBegin("initialize");
    auto plan = definition.initialize();
    profile::profileEnd();
    profile::checkForInterrupt();
    auto const baseOptions = getOptions();
    double const defaultDuration = std::accumulate(plan.stages.begin(), plan.stages.end(), 0.0,
        [](double sum, auto const& stage) { return sum + stage.duration; });
    int const defaultSteps = std::accumulate(plan.stages.begin(), plan.stages.end(), 0,
        [](int sum, auto const& stage) { return sum + stage.numberOfSteps; });
    double const duration =
        baseOptions.run.endTime > 0.0 ? baseOptions.run.endTime : defaultDuration;
    if (baseOptions.run.numberOfSteps < int(plan.stages.size())) {
        THROW_ERROR("run.numberOfSteps must allow at least one coarse step per scenario stage\n");
    }
    int assignedSteps = 0;
    int cumulativeDefaultSteps = 0;
    for (size_t i = 0; i < plan.stages.size(); i++) {
        auto& stage = plan.stages[i];
        cumulativeDefaultSteps += stage.numberOfSteps;
        int const cumulativeSteps = std::clamp(int(std::lround(double(cumulativeDefaultSteps) *
                                                   baseOptions.run.numberOfSteps / defaultSteps)),
            assignedSteps + 1, baseOptions.run.numberOfSteps - int(plan.stages.size() - i - 1));
        stage.numberOfSteps = cumulativeSteps - assignedSteps;
        assignedSteps += stage.numberOfSteps;
        // Stage transitions fall on the shared coarse clock. This keeps the
        // pending SPH half-step valid when entering/leaving stellar relaxation.
        stage.duration = duration * stage.numberOfSteps / baseOptions.run.numberOfSteps;
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
            THROW_ERROR("Checkpoint duration differs from the selected scenario\n");
        }
    } else {
        particlesSetMinrung(baseOptions.run.minRung);
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
        if (baseOptions.starFormation.enabled) {
            starFormationInitialize();
        }
#endif
    }
    state.minrung0 = std::max(state.minrung0, baseOptions.run.minRung);
    bool const hierarchicalMesh =
        baseOptions.run.hierarchicalGravity && baseOptions.gravity.enabled;
    bool const global = usesGlobalIntegration(baseOptions.run);
    profile::profileBegin("rebound");
    domainsRebound();
    profile::profileEnd();
    Timer checkpointTimer;
    checkpointTimer.start();
    Timer runtime;
    double const restoredRuntime = state.runtime;
    int iterations = 0;
    size_t progressRows = 0;
    if (hpxRank() == 0) {
        terminal::section(definition.name);
        PRINT("%-38s: %e code time units\n", "Scenario duration", duration);
        if (hpxSize() == 1) {
            PRINT(
                "%-38s: %lld\n", "Initialized particles", static_cast<long long>(particlesSize()));
        }
        PRINT("  Time and dt are in code units; wall durations use hh:mm:ss.ss and include "
              "previous runs on restart.\n");
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
    int stageEndStep = 0;
    bool interrupted = false;
    double const coarseInterval = duration / baseOptions.run.numberOfSteps;
    int lastOutputStep = driverOutputStep(state.tau, coarseInterval) -
        (driverFinalOutputStep(state.tau, coarseInterval) ==
            driverOutputStep(state.tau, coarseInterval));
    runtime.start();
    for (auto const& stage : plan.stages) {
        stageEndStep += stage.numberOfSteps;
        double const stageEnd = stageEndStep == baseOptions.run.numberOfSteps ?
            duration :
            duration * (double(stageEndStep) / baseOptions.run.numberOfSteps);
        double const tolerance = 1.0e-12 * duration;
        if (state.tau >= (global ? stageEnd : stageEnd - tolerance)) {
            continue;
        }
        auto stageOptions = baseOptions;
        stageOptions.hydro.damping = stage.damping;
        if (stage.viscosityAlpha >= 0.0) {
            stageOptions.hydro.viscosityAlpha = stage.viscosityAlpha;
            stageOptions.hydro.viscosityBeta = 2.0 * stage.viscosityAlpha;
            stageOptions.hydro.viscosityEvolution = "none";
        }
        setOptions(stageOptions);
        double const baseTimestep = duration / baseOptions.run.numberOfSteps;
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
            bool const outputDue = fullEvaluation && (!global || outputStep > lastOutputStep);
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
            ForceStepResult result;
#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
            if (hierarchicalMesh) {
                if (minimumRung == 0) {
                    state.minrung0 = std::max(state.minrung0 - 1, baseOptions.run.minRung);
                }
                int activeRung = std::max(minimumRung, state.minrung0);
                int const previousActiveRung = activeRung;
                result = kickStepHierarchical(activeRung, state.maxRung, 1.0, 0.0, state.tau,
                    baseTimestep, baseOptions.gravity.openingAngle, &state.energies, state.minrung0,
                    fullEvaluation, nullptr, true, 0.0, gravityErrorStep);
                if (activeRung != previousActiveRung) {
                    ++state.minrung0;
                }
            } else
#endif
            {
                result = advanceForces(minimumRung, 1.0, 0.0, state.tau, baseTimestep,
                    baseOptions.gravity.openingAngle, state.iter == 0, fullEvaluation,
                    state.energies, gravityErrorStep, false, state.globalTimestep,
                    global ? std::min(stageEnd - state.tau,
                                 (outputStep + 1) * baseTimestep - state.tau) :
                             0.0);
            }
            auto const flopRates = flopRatesPerSecond();
            double const flops = flopRates.total();
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
                std::ldexp(baseTimestep, -state.maxRung);
            if (!std::isfinite(timestep) || !(timestep > 0.0) ||
                state.tau + timestep == state.tau) {
                THROW_ERROR("Scenario timestep cannot advance time\n");
            }
            if (!hierarchicalMesh) {
                profile::profileBegin("drift");
                drift(1.0, timestep, state.tau, state.tau + timestep, duration);
                profile::profileEnd();
            }
            state.tau += timestep;
            state.itime = global ? 0 : inc(state.itime, state.maxRung);
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
                THROW_ERROR("Unable to write scenario progress\n");
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
                if (progressRows == 0) {
                    PRINT("  TFLOP/s uses counted gravity, SPH, and Hopkins operations divided by "
                          "force "
                          "advancement wall time.\n"
                          "  FP32/FP64 separate arithmetic precision; Mixed is reserved for "
                          "unclassified diagnostic counts (normally zero).\n"
                          "  Chemistry and star formation operations are excluded; their wall "
                          "time is included.\n");
                }
                if (progressRows % 25 == 0) {
                    PRINT("\n%8s %13s %13s %7s %13s %13s %13s %10s %10s %10s %10s\n", "Step",
                        "Time", "dt", "Done %", "Rung(min/max)", "Wall", "Step time", "FP32 TF/s",
                        "FP64 TF/s", "Mixed TF/s", "TFLOP/s");
                }
                PRINT("%8i %13.6e %13.6e %7.2f %6i/%-6i %13s %13s %10.3e %10.3e %10.3e %10.3e\n",
                    state.iter, state.tau, timestep,
                    std::clamp(100.0 * state.tau / duration, 0.0, 100.0), displayedMinimumRung,
                    state.maxRung, terminal::formatDuration(displayedRuntime).c_str(),
                    terminal::formatDuration(stepTimer.read()).c_str(),
                    flopRates.singlePrecision * 1e-12, flopRates.doublePrecision * 1e-12,
                    flopRates.unclassified * 1e-12, flops * 1e-12);
                progressRows++;
            }
        }
        if (interrupted || iterations >= baseOptions.run.maxIterations) {
            break;
        }
        state.tau = stageEnd;
        state.itime = 0;
    }
    // Use the next cycle for a final partial coarse step, preserving earlier snapshots.
    profile::profileBegin("output");
    viewOutputViews(driverFinalOutputStep(state.tau, coarseInterval), 1.0, true,
        state.tau * getOptions().units.timeSeconds * constants::kSecondsToYears);
    profile::profileEnd();
    setOptions(baseOptions);
    if (state.iter > 0 && state.iter != lastCheckpoint) {
        profile::profileBegin("writeCheckpoint");
        writeCheckpoint(state);
        profile::profileEnd();
    }
    bool const completed = state.tau >= duration * (1.0 - 1.0e-12);
    if (completed && !interrupted && iterations > 0) {
        // Restart state remains at the completed drift with a pending closing
        // half-step. Only the final diagnostics consume the synchronized state.
        auto finalOptions = baseOptions;
        finalOptions.hydro.damping = plan.stages.back().damping;
        if (plan.stages.back().viscosityAlpha >= 0.0) {
            finalOptions.hydro.viscosityAlpha = plan.stages.back().viscosityAlpha;
            finalOptions.hydro.viscosityBeta = 2.0 * plan.stages.back().viscosityAlpha;
            finalOptions.hydro.viscosityEvolution = "none";
        }
        setOptions(finalOptions);
#if COSMICTIGER_WITH_GRAVITY && COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
        if (hierarchicalMesh) {
            int activeRung = state.minrung0;
            kickStepHierarchical(activeRung, state.maxRung, 1.0, 0.0, state.tau, coarseInterval,
                baseOptions.gravity.openingAngle, &state.energies, state.minrung0, true, nullptr,
                true, 0.0, -1, true);
        } else
#endif
        {
            advanceForces(0, 1.0, 0.0, state.tau, coarseInterval, baseOptions.gravity.openingAngle,
                false, true, state.energies, -1, true, state.globalTimestep,
                global ? coarseInterval : 0.0);
        }
        detail::writeScenarioDiagnostics(state);
        setOptions(baseOptions);
    }
    if (completed && plan.finish != nullptr) {
        plan.finish(state.tau);
    }
#if COSMICTIGER_ENABLE_FMM
    kickClearBuffers();
#endif
    runtime.stop();
    if (hpxRank() == 0) {
        PRINT("\n  %s: %s at t=%.6e after %i steps.\n", definition.name,
            completed ? "finished" :
                        (interrupted ? "interrupted" : "paused at the iteration limit"),
            state.tau, state.iter);
        double const evolutionSeconds = iterations > 0 ? runtime.read() : 0.0;
        PRINT("  Total evolution time (this run, initialization excluded): %s\n",
            terminal::formatDuration(evolutionSeconds).c_str());
        if (restarting) {
            PRINT("  Recorded time including previous runs: %s\n",
                terminal::formatDuration(restoredRuntime + evolutionSeconds).c_str());
        }
    }
}
