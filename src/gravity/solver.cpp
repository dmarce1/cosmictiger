/* CosmicTiger - Nitadori PMMM solver; BSL-1.0. */

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/gravity/mesh_size.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/math/vector3.hpp>
#include "actions.hpp"

namespace pmmm::detail {

MeshCache cache;
SolveState state;
MutexType solveMutex;
PmmmDiagnostics globalDiagnostics;
uint64_t nextGeneration = 0;
KickReturn globalKickResult;
std::function<void(PmmmCellEvent, uint64_t)> cellObserver;

/// Invoke the optional cell-event observer for solver diagnostics or tests.
void observe(PmmmCellEvent event, uint64_t key) {
    if (cellObserver) {
        cellObserver(event, key);
    }
}

/// Select and validate the owned particle range participating in the requested rung solve.
Pair<ParticleInteger> selectedRange(bool rungSelected) {
    ParticleInteger const owned = particlesOwnedSize();
    auto const range = rungSelected ? particlesCurrentRange() : Pair<ParticleInteger>(0, owned);
    if (range.first < 0 || range.first > owned || range.second < range.first) {
        throw std::runtime_error("Invalid PMMM selected particle range");
    }
    return {range.first, std::min(range.second, owned)};
}

/// Return selected and total owned-particle counts for distributed solve planning.
std::array<uint64_t, 2> particleCounts(bool rungSelected) {
    auto const range = selectedRange(rungSelected);
    return {uint64_t(range.second - range.first), uint64_t(particlesOwnedSize())};
}

/// Prepare mesh caches, particle membership, source storage, dependencies, and diagnostics
/// for a solve generation.
void prepare(SolveRequest const& request) {
    initialize(request.meshSize);
    if (!frozenNearRequest(request)) {
        initializeGreen();
    }
    prepareCollectively([&]() {
        if (request.reuseBoundaryTree) {
            if (!state.boundaryTreeReady || !frozenNearRequest(request) ||
                !request.parameters.refreshMembership ||
                state.boundaryMeshSize != request.meshSize || state.activeRange.first != 0 ||
                state.activeRange.second != particlesOwnedSize()) {
                throw std::runtime_error("PMMM opening kick has no matching endpoint tree");
            }
            state.boundaryTreeReady = false;
            state.request = request;
            for (auto const& cell : state.cells) {
                for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
                    particlesFrozenCell(i) = cell.key;
                    state.stageRungs[i] = particlesRung(i);
                }
            }
            state.moments.clear();
            state.radialTrace = 0;
            state.background = {};
            state.kickResult = {};
            state.diagnostics.evaluatedParticles = 0;
            state.diagnostics.depositedMass = 0;
            state.farPromise = std::make_shared<hpx::promise<void>>();
            state.farReady = state.farPromise->get_future().share();
            return;
        }
        state.boundaryTreeReady = false;
        state.request = request;
        bool const frozen = frozenNearRequest(request);
        state.activeRange = selectedRange(request.applyKick && request.parameters.htime);
        if (frozen && !request.parameters.refreshMembership) {
            // Inner pass of the outer interval: group the selected range by the
            // membership stored when the interval opened, whatever the current
            // positions are.
            state.cells = particleMeshSortByKey(state.activeRange);
        } else {
            state.cells = particleMeshSort(cache.meshSize, state.activeRange);
            if (frozen) {
                // Boundary pass: this fresh position sort of every owned particle
                // is the membership frozen for the coming interval.
                if (state.activeRange.first != 0 ||
                    state.activeRange.second != particlesOwnedSize()) {
                    throw std::runtime_error(
                        "The PMMM membership refresh requires every owned particle");
                }
                for (auto const& cell : state.cells) {
                    for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
                        particlesFrozenCell(i) = cell.key;
                    }
                }
            }
        }
        state.stageRungs.resize(state.activeRange.second - state.activeRange.first);
        if (frozen) {
            // No deposit or transform: the far field is not evaluated.
            std::vector<double>().swap(state.moments);
        } else {
            state.moments.assign(cache.cellCount() * cache.sourceCount, 0.0);
        }
        state.ownedCells.clear();
        for (size_t index = 0; index < state.cells.size(); ++index) {
            state.ownedCells.emplace(state.cells[index].key, index);
        }
        state.sourceCells.clear();
        state.sourceCellsByKey.clear();
        state.sinkSources.clear();
        state.unresolvedSources.clear();
        state.readySinks.clear();
        state.tasks.clear();
        state.ownedTasks.clear();
        state.failure = {};
        state.kickResult = {};
        state.remainingSinks = state.cells.size();
        state.pendingGhosts = 0;
        state.farPromise = std::make_shared<hpx::promise<void>>();
        state.farReady = state.farPromise->get_future().share();
        {
            for (size_t index = 0; index < state.cells.size(); ++index) {
                auto source = std::make_shared<SourceCell>();
                source->range = state.cells[index];
                source->owner = hpxRank();
                state.sourceCellsByKey.emplace(source->range.key, index);
                state.sourceCells.push_back(std::move(source));
            }
        }
        state.radialTrace = 0;
        auto const previous = state.diagnostics;
        state.diagnostics = {};
        state.diagnostics.planBuilds = previous.planBuilds;
        state.diagnostics.greenBuilds = previous.greenBuilds;
        state.diagnostics.solveCount = previous.solveCount;
        state.diagnostics.transformCount = previous.transformCount;
        state.diagnostics.meshSize = request.meshSize;
        state.diagnostics.activeParticles = request.activeParticles;
        state.diagnostics.totalParticles = request.totalParticles;
        if (request.applyKick || request.farOnly) {
            Vector<Vector3<float>>().swap(state.acceleration);
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

/// Return this locality's current mesh-solver diagnostics.
PmmmDiagnostics diagnostics() {
    return state.diagnostics;
}

/// Return this locality's accumulated mesh kick diagnostics.
KickReturn kickResult() {
    return state.kickResult;
}

/// Dispatch a solver phase to every locality and join submissions before propagating
/// failure.
template <class Action, class... Arguments>
void allLocalities(Arguments... arguments) {
    if constexpr (std::is_same_v<Action, PmmmPrepareAction> ||
        std::is_same_v<Action, PmmmTransformAction> || std::is_same_v<Action, PmmmDestroyAction>) {
        pm::detail::dispatchCollectively(hpxLocalities().size(),
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

/// Serialize mesh solves and coordinate preparation, deposition, transforms, near work,
/// cleanup, and reductions.
void solve(SolveRequest request) {
    std::unique_lock<MutexType> lock(solveMutex);
    bool const coordinator = hpxRank() == 0;
    request.generation = ++nextGeneration;
    {
        profile::ScopedPhase phase(
            request.applyKick ? "pmmmKick.particleCounts" : "pmmmSolve.particleCounts",
            coordinator);
        std::vector<hpx::future<std::array<uint64_t, 2>>> counts;
        counts.reserve(hpxLocalities().size());
        try {
            for (auto const& locality : hpxLocalities()) {
                counts.push_back(hpx::async<PmmmParticleCountsAction>(
                    locality, request.applyKick && request.parameters.htime));
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
    }
    auto const& options = getOptions().gravity;
    bool const frozen = frozenNearRequest(request);
    // The outer interval keeps the full mesh: frozen keys and the boundary far
    // solve both refer to it, whatever fraction of the particles is active.
    request.meshSize = request.applyKick && request.parameters.htime && !frozen ?
        hierarchicalMeshSize(options.meshSize,
            resolvedCoarseMeshSize(
                options.meshSize, options.pmmmCoarseMeshSize, options.pmmmNearWidth),
            request.activeParticles, request.totalParticles) :
        options.meshSize;
    bool const farField = !frozen;
    bool const nearField = !request.farOnly;
    {
        profile::ScopedPhase phase(
            request.applyKick ? "pmmmKick.prepare" : "pmmmSolve.prepare", coordinator);
        allLocalities<PmmmPrepareAction>(request);
    }
    std::exception_ptr failure;
    bool started = false;
    try {
        {
            if (!request.reuseBoundaryTree) {
                profile::ScopedPhase phase(
                    request.applyKick ? "pmmmKick.planGhosts" : "pmmmSolve.planGhosts",
                    coordinator);
                allLocalities<PmmmPlanGhostsAction>();
            }
            started = true;
            profile::ScopedPhase phase(
                request.applyKick ? "pmmmKick.startNearPipeline" : "pmmmSolve.startNearPipeline",
                coordinator);
            allLocalities<PmmmStartPipelineAction>();
        }
        if (farField) {
            // FMM producers deposit completed owned roots. This wait excludes
            // near consumers, which may already be waiting for farReady.
            profile::ScopedPhase phase(
                request.applyKick ? "pmmmKick.deposit" : "pmmmSolve.deposit", coordinator);
            allLocalities<PmmmDepositAction>();
        }
        if (farField) {
            // The FMM near-field pipeline overlaps the transform. These phases measure
            // coordinator wall intervals; finishNearPipeline is only the remaining wait.
            profile::ScopedPhase phase(
                request.applyKick ? "pmmmKick.transform" : "pmmmSolve.transform", coordinator);
            allLocalities<PmmmTransformAction>();
        } else {
            allLocalities<PmmmSkipTransformAction>();
        }
        {
            profile::ScopedPhase phase(
                request.applyKick ? "pmmmKick.finishNearPipeline" : "pmmmSolve.finishNearPipeline",
                coordinator);
            allLocalities<PmmmFinishPipelineAction>();
        }
        if (request.farOnly) {
            profile::ScopedPhase phase("pmmmBoundary.evaluateFar", coordinator);
            allLocalities<PmmmEvaluateFarAction>();
        }
    } catch (...) {
        failure = std::current_exception();
    }
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
        profile::ScopedPhase phase(
            request.applyKick ? "pmmmKick.abortPipeline" : "pmmmSolve.abortPipeline", coordinator);
        cleanup([]() { allLocalities<PmmmAbortPipelineAction>(); });
        cleanup([]() { allLocalities<PmmmFinishPipelineAction>(); });
    }
    if (nearField || failure) {
        profile::ScopedPhase phase(
            request.applyKick ? "pmmmKick.cleanup" : "pmmmSolve.cleanup", coordinator);
        cleanup([]() { treeDestroy(); });
        cleanup([]() { particlesCacheFree(); });
        cleanup([]() { allLocalities<PmmmClearGhostsAction>(); });
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    profile::ScopedPhase phase(
        request.applyKick ? "pmmmKick.diagnostics" : "pmmmSolve.diagnostics", coordinator);
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
        globalDiagnostics.evaluatedParticles += value.evaluatedParticles;
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
        globalDiagnostics.transformCount =
            std::max(globalDiagnostics.transformCount, value.transformCount);
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

/// Compute and retain particle accelerations and optional potentials without applying a
/// kick.
void pmmmSolve(bool doPotential) {
    pmmm::detail::SolveRequest request;
    request.doPotential = doPotential;
    pmmm::detail::solve(std::move(request));
}

/// Run the mesh solver in kick mode and return globally reduced kick diagnostics.
KickReturn pmmmKick(KickParameters const& parameters, bool reuseBoundaryTree) {
    pmmm::detail::SolveRequest request;
    request.doPotential = parameters.doPhi;
    request.applyKick = true;
    request.parameters = parameters;
    request.reuseBoundaryTree = reuseBoundaryTree;
    pmmm::detail::solve(std::move(request));
    return pmmm::detail::globalKickResult;
}

/// Serialize and perform mesh-solver teardown on every locality.
void pmmmDestroy() {
    using namespace pmmm::detail;
    std::unique_lock<MutexType> lock(solveMutex);
    allLocalities<PmmmDestroyAction>();
}

/// Return the globally reduced diagnostics from the latest mesh solve.
PmmmDiagnostics pmmmDiagnostics() {
    return pmmm::detail::globalDiagnostics;
}

/// Return a retained local acceleration with bounds checking.
Vector3<float> const& pmmmAcceleration(ParticleInteger index) {
    return pmmm::detail::state.acceleration.at(index);
}

/// Return a retained potential with bounds checking, or zero when potential storage was not
/// requested.
float pmmmPotential(ParticleInteger index) {
    return pmmm::detail::state.potential.empty() ? 0.0f : pmmm::detail::state.potential.at(index);
}

/// Install or clear the process-local cell-event observer.
void pmmmSetCellObserver(std::function<void(PmmmCellEvent, uint64_t)> observer) {
    pmmm::detail::cellObserver = std::move(observer);
}
