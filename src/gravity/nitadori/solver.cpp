/* CosmicTiger - Nitadori PMMM solver; BSL-1.0. */

#include "actions.hpp"

namespace pmmm::detail {

MeshCache cache;
SolveState state;
MutexType solveMutex;
PmmmDiagnostics globalDiagnostics;
uint64_t nextGeneration = 0;
KickReturn globalKickResult;
std::function<void(PmmmCellEvent, uint64_t)> cellObserver;

void observe(PmmmCellEvent event, uint64_t key) {
    if (cellObserver) {
        cellObserver(event, key);
    }
}

void prepare(SolveRequest const& request) {
    initialize();
    initializeGreen();
    prepareCollectively([&]() {
        state.request = request;
        state.activeRange =
            request.applyKick ? particlesCurrentRange() : Pair<ParticleInteger>(0, particlesSize());
        state.cells = particleMeshSort(cache.meshSize, state.activeRange);
        state.stageRungs.resize(state.activeRange.second - state.activeRange.first);
        state.moments.assign(cache.cellCount() * cache.sourceCount, 0.0);
        state.sources.clear();
        state.sourceOwners.clear();
        state.ownedCells.clear();
        state.sourceCells.clear();
        state.sourceCellsByKey.clear();
        state.sinkSources.clear();
        state.unresolvedSources.clear();
        state.readySinks.clear();
        state.tasks.clear();
        state.failure = {};
        state.kickResult = {};
        state.remainingSinks = state.cells.size();
        state.pendingGhosts = 0;
        state.farPromise = std::make_shared<hpx::promise<void>>();
        state.farReady = state.farPromise->get_future().share();
#if COSMICTIGER_PMMM_NEAR_FMM
        // Only the FMM pipeline fulfills these per-cell promises. Creating them
        // for PP would report a broken promise for every cell on the next solve.
        for (size_t index = 0; index < state.cells.size(); ++index) {
            auto source = std::make_shared<SourceCell>();
            source->range = state.cells[index];
            source->owner = hpxRank();
            state.ownedCells.emplace(source->range.key, index);
            state.sourceCellsByKey[source->range.key].push_back(index);
            state.sourceCells.push_back(std::move(source));
        }
#endif
        state.radialTrace = 0;
        auto const previous = state.diagnostics;
        state.diagnostics = {};
        state.diagnostics.planBuilds = previous.planBuilds;
        state.diagnostics.greenBuilds = previous.greenBuilds;
        state.diagnostics.solveCount = previous.solveCount;
        if (request.applyKick) {
            Vector<Array<float, kDimensions>>().swap(state.acceleration);
            Vector<float>().swap(state.potential);
        } else {
            state.acceleration.resize(particlesSize());
            if (request.doPotential) {
                state.potential.assign(particlesSize(), 0.0f);
            } else {
                Vector<float>().swap(state.potential);
            }
        }
        state.diagnostics.storedAccelerationParticles = state.acceleration.size();
        state.diagnostics.storedPotentialParticles = state.potential.size();
        state.background = {};
        for (ParticleInteger i = state.activeRange.first; i < state.activeRange.second; ++i) {
            state.diagnostics.sourceMass += particlesMass(i);
        }
    });
}

PmmmDiagnostics diagnostics() {
    return state.diagnostics;
}

KickReturn kickResult() {
    return state.kickResult;
}

template <class Action, class... Arguments>
void allLocalities(Arguments... arguments) {
    if constexpr (std::is_same_v<Action, PmmmPrepareAction> ||
        std::is_same_v<Action, PmmmTransformAction> || std::is_same_v<Action, PmmmDestroyAction>) {
        particle_mesh::detail::dispatchCollectively(hpxLocalities().size(),
            [&](size_t index) { return hpx::async<Action>(hpxLocalities()[index], arguments...); });
    } else {
        std::vector<hpx::future<void>> futures;
        futures.reserve(hpxLocalities().size());
        std::exception_ptr failure;
        try {
            for (auto const& locality : hpxLocalities()) {
                futures.push_back(hpx::async<Action>(locality, arguments...));
            }
        } catch (...) {
            failure = std::current_exception();
        }
        for (auto& future : futures) {
            future.wait();
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        for (auto& future : futures) {
            future.get();
        }
    }
}

void solve(SolveRequest request) {
    std::unique_lock<MutexType> lock(solveMutex);
    request.generation = ++nextGeneration;
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        sphParticlesClearCaches();
    }
#endif
    allLocalities<PmmmPrepareAction>(request);
    std::exception_ptr failure;
#if COSMICTIGER_PMMM_NEAR_FMM
    bool started = false;
#endif
    try {
        allLocalities<PmmmDepositAction>();
#if COSMICTIGER_PMMM_NEAR_FMM
        allLocalities<PmmmPlanGhostsAction>();
        started = true;
        allLocalities<PmmmStartPipelineAction>();
#endif
        allLocalities<PmmmTransformAction>();
#if COSMICTIGER_PMMM_NEAR_FMM
        allLocalities<PmmmFinishPipelineAction>();
#else
        allLocalities<PmmmEvaluateAction>(request.doPotential);
#endif
    } catch (...) {
        failure = std::current_exception();
    }
#if COSMICTIGER_PMMM_NEAR_FMM
    auto cleanup = [&](auto&& operation) {
        try {
            operation();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    };
    if (failure && started) {
        cleanup([]() { allLocalities<PmmmAbortPipelineAction>(); });
        cleanup([]() { allLocalities<PmmmFinishPipelineAction>(); });
    }
    cleanup([]() { treeDestroy(); });
    cleanup([]() { particlesCacheFree(); });
    cleanup([]() { allLocalities<PmmmClearGhostsAction>(); });
#endif
    if (failure) {
        std::rethrow_exception(failure);
    }
    globalDiagnostics = {};
    globalKickResult = {};
    std::vector<hpx::future<PmmmDiagnostics>> diagnosticsFutures;
    std::vector<hpx::future<KickReturn>> kickFutures;
    for (auto const& locality : hpxLocalities()) {
        diagnosticsFutures.push_back(hpx::async<PmmmDiagnosticsAction>(locality));
        kickFutures.push_back(hpx::async<PmmmKickResultAction>(locality));
    }
    hpx::wait_all(diagnosticsFutures);
    hpx::wait_all(kickFutures);
    for (auto& future : diagnosticsFutures) {
        auto const value = future.get();
        globalDiagnostics.sourceMass += value.sourceMass;
        globalDiagnostics.depositedMass += value.depositedMass;
        globalDiagnostics.nearCpuAcceptedMultipoles += value.nearCpuAcceptedMultipoles;
        globalDiagnostics.nearCpuDirectPairs += value.nearCpuDirectPairs;
        globalDiagnostics.ghostParticles += value.ghostParticles;
        globalDiagnostics.ghostCells += value.ghostCells;
        globalDiagnostics.ghostBytes += value.ghostBytes;
        globalDiagnostics.nearRemoteTreeFetches += value.nearRemoteTreeFetches;
        globalDiagnostics.nearRemoteParticleFetches += value.nearRemoteParticleFetches;
        globalDiagnostics.nearBatches += value.nearBatches;
        globalDiagnostics.peakBatchParticles =
            std::max(globalDiagnostics.peakBatchParticles, value.peakBatchParticles);
        globalDiagnostics.completedWhileGhostsPending += value.completedWhileGhostsPending;
        globalDiagnostics.storedAccelerationParticles += value.storedAccelerationParticles;
        globalDiagnostics.storedPotentialParticles += value.storedPotentialParticles;
        globalDiagnostics.planBuilds = std::max(globalDiagnostics.planBuilds, value.planBuilds);
        globalDiagnostics.greenBuilds = std::max(globalDiagnostics.greenBuilds, value.greenBuilds);
        globalDiagnostics.solveCount = std::max(globalDiagnostics.solveCount, value.solveCount);
    }
    for (auto& future : kickFutures) {
        globalKickResult += future.get();
    }
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmPrepareAction, PmmmPrepareAction);
HPX_REGISTER_ACTION(PmmmDiagnosticsAction, PmmmDiagnosticsAction);
HPX_REGISTER_ACTION(PmmmKickResultAction, PmmmKickResultAction);

void pmmmSolve(bool doPotential) {
    pmmm::detail::SolveRequest request;
    request.doPotential = doPotential;
    pmmm::detail::solve(std::move(request));
}

KickReturn pmmmKick(KickParameters const& parameters) {
    pmmm::detail::SolveRequest request;
    request.doPotential = parameters.doPhi;
    request.applyKick = true;
    request.parameters = parameters;
    pmmm::detail::solve(std::move(request));
    return pmmm::detail::globalKickResult;
}

void pmmmDestroy() {
    using namespace pmmm::detail;
    std::unique_lock<MutexType> lock(solveMutex);
    allLocalities<PmmmDestroyAction>();
}

PmmmDiagnostics pmmmDiagnostics() {
    return pmmm::detail::globalDiagnostics;
}

Array<float, kDimensions> const& pmmmAcceleration(ParticleInteger index) {
    return pmmm::detail::state.acceleration.at(index);
}

float pmmmPotential(ParticleInteger index) {
    return pmmm::detail::state.potential.empty() ? 0.0f : pmmm::detail::state.potential.at(index);
}

double pmmmEffectiveRadius() {
    return getOptions().gravity.supportRadius /
        selectedGravityKernel().supportRadiusPerEffectiveRadius;
}

void pmmmSetCellObserver(std::function<void(PmmmCellEvent, uint64_t)> observer) {
    pmmm::detail::cellObserver = std::move(observer);
}
