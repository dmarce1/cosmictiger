/* CosmicTiger - Nitadori PMMM solver; BSL-1.0. */

#include <cosmictiger/gravity/nitadori/mesh_size.hpp>
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

Pair<ParticleInteger> selectedRange(bool applyKick) {
    ParticleInteger const owned = particlesOwnedSize();
    auto const range = applyKick ? particlesCurrentRange() : Pair<ParticleInteger>(0, owned);
    if (range.first < 0 || range.first > owned || range.second < range.first) {
        throw std::runtime_error("Invalid PMMM selected particle range");
    }
    return {range.first, std::min(range.second, owned)};
}

std::array<uint64_t, 2> particleCounts(bool applyKick) {
    auto const range = selectedRange(applyKick);
    return {uint64_t(range.second - range.first), uint64_t(particlesOwnedSize())};
}

void prepare(SolveRequest const& request) {
    initialize(request.meshSize);
    initializeGreen();
    prepareCollectively([&]() {
        state.request = request;
        state.activeRange = selectedRange(request.applyKick);
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
#if COSMICTIGER_ENABLE_PMMM_NEAR_FMM
        // Only the FMM pipeline fulfills these per-cell promises. Creating them
        // for PP would report a broken promise for every cell on the next solve.
        if (getOptions().gravity.pmmmNearMethod == "fmm") {
            for (size_t index = 0; index < state.cells.size(); ++index) {
                auto source = std::make_shared<SourceCell>();
                source->range = state.cells[index];
                source->owner = hpxRank();
                state.ownedCells.emplace(source->range.key, index);
                state.sourceCellsByKey[source->range.key].push_back(index);
                state.sourceCells.push_back(std::move(source));
            }
        }
#endif
        state.radialTrace = 0;
        auto const previous = state.diagnostics;
        state.diagnostics = {};
        state.diagnostics.planBuilds = previous.planBuilds;
        state.diagnostics.greenBuilds = previous.greenBuilds;
        state.diagnostics.solveCount = previous.solveCount;
        state.diagnostics.meshSize = request.meshSize;
        state.diagnostics.activeParticles = request.activeParticles;
        state.diagnostics.totalParticles = request.totalParticles;
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
    std::vector<hpx::future<std::array<uint64_t, 2>>> counts;
    counts.reserve(hpxLocalities().size());
    try {
        for (auto const& locality : hpxLocalities()) {
            counts.push_back(hpx::async<PmmmParticleCountsAction>(locality, request.applyKick));
        }
    } catch (...) {
        hpx::wait_all(counts);
        throw;
    }
    hpx::wait_all(counts);
    for (auto& future : counts) {
        auto const value = future.get();
        if (value[0] > std::numeric_limits<uint64_t>::max() - request.activeParticles ||
            value[1] > std::numeric_limits<uint64_t>::max() - request.totalParticles) {
            throw std::runtime_error("PMMM global particle count overflow");
        }
        request.activeParticles += value[0];
        request.totalParticles += value[1];
    }
    auto const& options = getOptions().gravity;
    request.meshSize = request.applyKick ?
        hierarchicalMeshSize(options.meshSize, options.pmmmNearWidth, request.activeParticles,
            request.totalParticles, options.pmmmMaxPrimeFactor) :
        fftMeshSize(options.meshSize, options.pmmmMaxPrimeFactor);
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        sphParticlesClearCaches();
    }
#endif
    allLocalities<PmmmPrepareAction>(request);
    std::exception_ptr failure;
#if COSMICTIGER_ENABLE_PMMM_NEAR_FMM
    bool started = false;
#endif
    try {
        allLocalities<PmmmDepositAction>();
#if COSMICTIGER_ENABLE_PMMM_NEAR_FMM
        if (options.pmmmNearMethod == "fmm") {
            allLocalities<PmmmPlanGhostsAction>();
            started = true;
            allLocalities<PmmmStartPipelineAction>();
        }
#endif
        allLocalities<PmmmTransformAction>();
#if COSMICTIGER_ENABLE_PMMM_NEAR_FMM
        if (options.pmmmNearMethod == "fmm") {
            allLocalities<PmmmFinishPipelineAction>();
        }
#endif
#if COSMICTIGER_ENABLE_PMMM_NEAR_PP
        if (options.pmmmNearMethod == "pp") {
            allLocalities<PmmmEvaluateAction>(request.doPotential);
        }
#endif
    } catch (...) {
        failure = std::current_exception();
    }
#if COSMICTIGER_ENABLE_PMMM_NEAR_FMM
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
    if (options.pmmmNearMethod == "fmm") {
        cleanup([]() { treeDestroy(); });
        cleanup([]() { particlesCacheFree(); });
        cleanup([]() { allLocalities<PmmmClearGhostsAction>(); });
    }
#endif
    if (failure) {
        std::rethrow_exception(failure);
    }
    globalDiagnostics = {};
    globalDiagnostics.meshSize = request.meshSize;
    globalDiagnostics.activeParticles = request.activeParticles;
    globalDiagnostics.totalParticles = request.totalParticles;
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
HPX_REGISTER_ACTION(PmmmParticleCountsAction, PmmmParticleCountsAction);
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
