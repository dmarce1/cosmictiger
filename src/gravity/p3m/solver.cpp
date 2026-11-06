/* CosmicTiger - P3M short-range solver coordination; BSL-1.0. */

#include <cmath>
#include <limits>
#include <unordered_set>

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/gravity/p3m/pp/short_pp.hpp>
#include <cosmictiger/gravity/pm/geometry.hpp>
#include <cosmictiger/gravity/pm/kick.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif
#include "state.hpp"

namespace p3m::detail {

SolveState state;
MutexType solveMutex;
P3mDiagnostics globalDiagnostics;
KickReturn globalKickResult;
uint64_t nextGeneration = 0;

int periodicIndex(int index) {
    int const n = state.meshSize;
    return (index % n + n) % n;
}

std::array<int, 3> coordinates(uint64_t key) {
    uint64_t const n = state.meshSize;
    return {int(key / (n * n)), int(key / n % n), int(key % n)};
}

ShortParticle particle(ParticleInteger index) {
    ShortParticle result;
    for (int axis = 0; axis < kDimensions; ++axis) {
        result.position[axis] = positionDomainCoordinate(particlesPos(axis, index));
    }
    result.mass = particlesMass(index);
    result.rank = hpxRank();
    result.index = index;
    return result;
}

bool activeSink(ParticleInteger index) {
    if (!state.request.applyKick) {
        return true;
    }
    return particlesRung(index) >= state.request.parameters.minRung;
}

static Pair<ParticleInteger> selectedRange(bool rungSelected) {
    ParticleInteger const owned = particlesOwnedSize();
    auto const range = rungSelected ? particlesCurrentRange() : Pair<ParticleInteger>(0, owned);
    if (range.first < 0 || range.first > owned || range.second < range.first) {
        throw std::runtime_error("Invalid P3M selected particle range");
    }
    return {range.first, std::min(range.second, owned)};
}

std::array<uint64_t, 2> particleCounts(bool rungSelected) {
    auto const range = selectedRange(rungSelected);
    return {uint64_t(range.second - range.first), uint64_t(particlesOwnedSize())};
}

void prepare(SolveRequest const& request) {
    auto const& gravity = getOptions().gravity;
    if (!usesSplitParticleMesh(gravity) || gravity.p3mMeshSize < 2 * gravity.pmmmNearWidth + 1) {
        throw std::runtime_error("P3M short solver requires gravity.method.far=p3m geometry");
    }
    state.meshSize = gravity.p3mMeshSize;
    state.request = request;
    state.activeRange = selectedRange(request.applyKick && request.parameters.htime);
    state.cells = particleMeshSort(state.meshSize, state.activeRange);
    state.ownedCells.clear();
    for (size_t index = 0; index < state.cells.size(); ++index) {
        state.ownedCells.emplace(state.cells[index].key, index);
    }
    state.sourceCells.clear();
    state.sourceByKey.clear();
    state.kickResult = {};
    auto const previous = state.diagnostics;
    state.diagnostics = {};
    state.diagnostics.solveCount = previous.solveCount;
    state.diagnostics.meshSize = state.meshSize;
    if (request.applyKick || particlesGravityDiagnostics != nullptr) {
        Vector<Vector3<float>>().swap(state.acceleration);
        Vector<float>().swap(state.potential);
    } else {
        state.acceleration.resize(particlesOwnedSize());
        state.potential.assign(particlesOwnedSize(), 0.0f);
    }
    state.diagnostics.storedAccelerationParticles = state.acceleration.size();
}

std::vector<CellParticles> readCellParticles(
    uint64_t generation, std::vector<uint64_t> const& keys) {
    if (generation != state.request.generation) {
        throw std::runtime_error("P3M cell request belongs to another solve");
    }
    std::vector<CellParticles> result;
    result.reserve(keys.size());
    FlopCount flops = 0;
    for (auto key : keys) {
        auto const found = state.ownedCells.find(key);
        if (found == state.ownedCells.end()) {
            continue;
        }
        auto const& cell = state.cells[found->second];
        CellParticles particles;
        particles.key = key;
        particles.particles.reserve(cell.end - cell.begin);
        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
            particles.particles.push_back(particle(i));
        }
        flops += 4 * (cell.end - cell.begin);
        result.push_back(std::move(particles));
    }
    addCpuFlops(flops, FlopPrecision::Double);
    return result;
}

}    // namespace p3m::detail

HPX_PLAIN_ACTION(p3m::detail::particleCounts, P3mParticleCountsAction);
HPX_PLAIN_ACTION(p3m::detail::prepare, P3mPrepareAction);
HPX_PLAIN_ACTION(p3m::detail::readCellParticles, P3mReadCellParticlesAction);
HPX_PLAIN_ACTION(p3m::detail::gatherSources, P3mGatherSourcesAction);
HPX_PLAIN_ACTION(p3m::detail::evaluate, P3mEvaluateAction);
HPX_PLAIN_ACTION(p3m::detail::diagnostics, P3mDiagnosticsAction);
HPX_PLAIN_ACTION(p3m::detail::kickResult, P3mKickResultAction);
HPX_PLAIN_ACTION(p3m::detail::destroy, P3mDestroyAction);

namespace p3m::detail {

// Every search-mesh cell lies in one locality's domain: domain cuts follow the
// coarse mesh planes, so the cell center identifies the particle owner.
static int cellOwner(uint64_t key) {
    auto const cell = coordinates(key);
    Vector3<double> center;
    for (int axis = 0; axis < kDimensions; ++axis) {
        center[axis] = (cell[axis] + 0.5) / state.meshSize;
    }
    return domainsFindRank(center);
}

void gatherSources() {
    int const width = getOptions().gravity.pmmmNearWidth;
    std::unordered_map<int, std::vector<uint64_t>> requests;
    for (size_t index = 0; index < state.cells.size(); ++index) {
        auto const& cell = state.cells[index];
        if (state.sourceByKey.emplace(cell.key, state.sourceCells.size()).second) {
            SourceCell source;
            source.key = cell.key;
            source.owned = true;
            source.ownedIndex = index;
            state.sourceCells.push_back(std::move(source));
        }
    }
    int const n = state.meshSize;
    for (auto const& cell : state.cells) {
        auto const center = coordinates(cell.key);
        for (int x = -width; x <= width; ++x) {
            for (int y = -width; y <= width; ++y) {
                for (int z = -width; z <= width; ++z) {
                    uint64_t const key = (uint64_t(periodicIndex(center[0] + x)) * n +
                                             periodicIndex(center[1] + y)) *
                            n +
                        periodicIndex(center[2] + z);
                    if (state.sourceByKey.count(key) != 0) {
                        continue;
                    }
                    int const owner = cellOwner(key);
                    if (owner == hpxRank()) {
                        // An unoccupied owned cell contributes nothing.
                        continue;
                    }
                    state.sourceByKey.emplace(key, state.sourceCells.size());
                    SourceCell source;
                    source.key = key;
                    state.sourceCells.push_back(std::move(source));
                    requests[owner].push_back(key);
                }
            }
        }
    }
    std::vector<std::pair<int, hpx::future<std::vector<CellParticles>>>> futures;
    futures.reserve(requests.size());
    std::exception_ptr failure;
    try {
        for (auto& [owner, keys] : requests) {
            std::sort(keys.begin(), keys.end());
            futures.emplace_back(owner,
                hpx::async<P3mReadCellParticlesAction>(
                    hpxLocalities()[owner], state.request.generation, keys));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& [owner, future] : futures) {
        if (future.valid()) {
            future.wait();
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    for (auto& [owner, future] : futures) {
        for (auto& packet : future.get()) {
            auto& source = state.sourceCells.at(state.sourceByKey.at(packet.key));
            source.particles = std::move(packet.particles);
            ++state.diagnostics.ghostCells;
            state.diagnostics.ghostParticles += source.particles.size();
        }
    }
    // Owned source cells copy their records once so the backends see one layout.
    parallelFor(state.sourceCells.size(), "gravity.p3m.copyOwnedSources", [&](size_t index) {
        auto& source = state.sourceCells[index];
        if (!source.owned) {
            return;
        }
        auto const& cell = state.cells[source.ownedIndex];
        source.particles.reserve(cell.end - cell.begin);
        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
            source.particles.push_back(particle(i));
        }
    });
    state.diagnostics.activeParticles = state.activeRange.second - state.activeRange.first;
    state.diagnostics.totalParticles = particlesOwnedSize();
}

void consumeField(ParticleInteger index, std::array<double, 4> const& field, KickReturn& result,
    FlopCount& doubleFlops, FlopCounts& kickFlops) {
    Vector3<float> acceleration;
    double const gm = getOptions().gravity.gm;
    for (int component = 0; component < 4; ++component) {
        if (!std::isfinite(field[component]) ||
            std::abs(gm * field[component]) > std::numeric_limits<float>::max()) {
            throw std::runtime_error("P3M short field is not representable");
        }
        if (component != 0) {
            acceleration[component - 1] = gm * field[component];
        }
    }
    float const potential = state.request.doPotential ? gm * field[0] : 0.0f;
    doubleFlops += 2 * (kDimensions + (state.request.doPotential ? 1 : 0));
    if (particlesGravityDiagnostics != nullptr) {
        auto& output = particlesGravityDiagnostics[index];
        output.nearAcceleration = acceleration;
        output.nearPotential = potential;
        for (int dim = 0; dim < kDimensions; ++dim) {
            output.acceleration[dim] = output.longAcceleration[dim] + acceleration[dim];
        }
        output.potential = output.longPotential + potential;
    } else if (state.request.applyKick) {
        result += particleMeshApplyField(
            index, acceleration, potential, state.request.parameters, nullptr, &kickFlops);
    } else {
        state.acceleration[index] = acceleration;
        if (!state.potential.empty()) {
            state.potential[index] = potential;
        }
    }
}

void evaluate() {
    auto const& gravity = getOptions().gravity;
    if (gravity.pmmmNearMethod == "fmm") {
#if COSMICTIGER_ENABLE_P3M_FMM
        evaluateTrees();
#else
        throw std::runtime_error("FMM-PM is not compiled into this executable");
#endif
    } else {
#if COSMICTIGER_ENABLE_P3M_PP
        evaluatePairs();
#else
        throw std::runtime_error("PP-PM is not compiled into this executable");
#endif
    }
    ++state.diagnostics.solveCount;
    state.sourceCells.clear();
    state.sourceByKey.clear();
}

P3mDiagnostics diagnostics() {
    return state.diagnostics;
}

KickReturn kickResult() {
    return state.kickResult;
}

void destroy() {
#if defined(USE_CUDA) && COSMICTIGER_ENABLE_P3M_PP
    cosmictiger::gravity::p3m::pp::clearShortCudaWorkspaces();
#endif
    state.cells.clear();
    state.ownedCells.clear();
    state.sourceCells.clear();
    state.sourceByKey.clear();
    state.acceleration.clear();
    state.potential.clear();
}

template <class Action, class... Arguments>
static void allLocalities(Arguments... arguments) {
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

static void solve(SolveRequest request) {
    std::unique_lock<MutexType> lock(solveMutex);
    bool const coordinator = hpxRank() == 0;
    request.generation = ++nextGeneration;
    bool const kick = request.applyKick;
    {
        profile::ScopedPhase phase(kick ? "p3mKick.prepare" : "p3mSolve.prepare", coordinator);
#if COSMICTIGER_WITH_SPH
        if (getOptions().hydro.enabled) {
            sphParticlesClearCaches();
        }
#endif
        allLocalities<P3mPrepareAction>(request);
    }
    {
        profile::ScopedPhase phase(kick ? "p3mKick.gather" : "p3mSolve.gather", coordinator);
        allLocalities<P3mGatherSourcesAction>();
    }
    {
        profile::ScopedPhase phase(kick ? "p3mKick.evaluate" : "p3mSolve.evaluate", coordinator);
        allLocalities<P3mEvaluateAction>();
    }
    profile::ScopedPhase phase(kick ? "p3mKick.diagnostics" : "p3mSolve.diagnostics", coordinator);
    globalDiagnostics = {};
    globalKickResult = {};
    std::vector<hpx::future<P3mDiagnostics>> diagnosticsFutures;
    std::vector<hpx::future<KickReturn>> kickFutures;
    for (auto const& locality : hpxLocalities()) {
        diagnosticsFutures.push_back(hpx::async<P3mDiagnosticsAction>(locality));
        kickFutures.push_back(hpx::async<P3mKickResultAction>(locality));
    }
    hpx::wait_all(diagnosticsFutures);
    hpx::wait_all(kickFutures);
    for (auto& future : diagnosticsFutures) {
        auto const value = future.get();
        globalDiagnostics.meshSize = value.meshSize;
        globalDiagnostics.activeParticles += value.activeParticles;
        globalDiagnostics.totalParticles += value.totalParticles;
        globalDiagnostics.evaluatedParticles += value.evaluatedParticles;
        globalDiagnostics.ghostCells += value.ghostCells;
        globalDiagnostics.ghostParticles += value.ghostParticles;
        globalDiagnostics.directPairs += value.directPairs;
        globalDiagnostics.acceptedMultipoles += value.acceptedMultipoles;
        globalDiagnostics.cutoffRejections += value.cutoffRejections;
        globalDiagnostics.sourceTreeNodes += value.sourceTreeNodes;
        globalDiagnostics.batches += value.batches;
        globalDiagnostics.cudaBatches += value.cudaBatches;
        globalDiagnostics.storedAccelerationParticles += value.storedAccelerationParticles;
        globalDiagnostics.solveCount = std::max(globalDiagnostics.solveCount, value.solveCount);
    }
    for (auto& future : kickFutures) {
        globalKickResult += future.get();
    }
}

}    // namespace p3m::detail

KickReturn p3mKick(KickParameters const& parameters) {
    p3m::detail::SolveRequest request;
    request.doPotential = parameters.doPhi;
    request.applyKick = true;
    request.parameters = parameters;
    p3m::detail::solve(std::move(request));
    return p3m::detail::globalKickResult;
}

void p3mSolve(bool doPotential) {
    p3m::detail::SolveRequest request;
    request.doPotential = doPotential;
    p3m::detail::solve(std::move(request));
}

void p3mDestroy() {
    std::unique_lock<MutexType> lock(p3m::detail::solveMutex);
    p3m::detail::allLocalities<P3mDestroyAction>();
}

P3mDiagnostics p3mDiagnostics() {
    return p3m::detail::globalDiagnostics;
}

Vector3<float> const& p3mAcceleration(ParticleInteger index) {
    return p3m::detail::state.acceleration.at(index);
}

float p3mPotential(ParticleInteger index) {
    return p3m::detail::state.potential.empty() ? 0.0f : p3m::detail::state.potential.at(index);
}
