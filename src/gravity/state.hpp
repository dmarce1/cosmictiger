/* CosmicTiger - private Nitadori PMMM state; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fftw3-mpi.h>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/run_as_os_thread.hpp>
#include <hpx/serialization/array.hpp>
#include <hpx/serialization/vector.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/fftw_lock.hpp>
#include <cosmictiger/gravity/collective_dispatch.hpp>
#include <cosmictiger/gravity/fmm/cell_tree.hpp>
#include <cosmictiger/gravity/fmm/fmm_batch.hpp>
#include <cosmictiger/gravity/fmm/ghost_storage.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/pmmm_kernels.hpp>
#include <cosmictiger/gravity/pmmm_kick.hpp>
#include <cosmictiger/gravity/solver.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif

namespace pmmm::detail {

struct SolveRequest {
    uint64_t generation = 0;
    int meshSize = 0;
    uint64_t activeParticles = 0;
    uint64_t totalParticles = 0;
    bool doPotential = false;
    bool applyKick = false;
    // Outer-interval boundary far solve: deposit, transform and evaluate the far
    // field of every owned particle into particlesLongAcceleration, with the
    // far potential energy and the acceleration timestep bound reduced through
    // the kick result; no near field, velocity or rung changes.
    bool farOnly = false;
    bool reuseBoundaryTree = false;
    double scale = 1.0;
    double eta = 0.0;
    KickParameters parameters;

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & generation & meshSize & activeParticles & totalParticles & doPotential &
            applyKick & farOnly & reuseBoundaryTree & scale & eta;
        if (applyKick) {
            archive & parameters;
        }
    }
};

// A near-only pass on the frozen cell membership of the PMMM outer interval.
inline bool frozenNearRequest(SolveRequest const& request) {
    return request.applyKick && request.parameters.frozenNear;
}

// Moment packet from a cell's particle owner to its FFT slab owner.
struct CellData {
    uint64_t key = 0;
    std::vector<double> coefficients;
    double radialTrace = 0;

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & key & coefficients & radialTrace;
    }
};

struct SourceCell {
    ParticleMeshCell range;
    int owner;
    std::vector<size_t> dependents;
    hpx::promise<TreeCellRoot> promise;
    hpx::shared_future<TreeCellRoot> ready = promise.get_future().share();
    std::atomic<bool> published{false};
};

// Persistent FFT allocation and immutable configuration.
struct MeshCache {
    int meshSize = 0;
    int order = 0;
    int nearWidth = 0;
    int sourceCount = 0;
    int localCount = 0;
    MPI_Comm communicator = MPI_COMM_NULL;
    ptrdiff_t localSize = 0;
    ptrdiff_t localBegin = 0;
    ptrdiff_t spectralSize = 0;
    ptrdiff_t spectralBegin = 0;
    ptrdiff_t sourceAllocation = 0;
    ptrdiff_t localAllocation = 0;
    float* sourceReal = nullptr;
    float* localReal = nullptr;
    fftwf_complex* sourceSpectrum = nullptr;
    fftwf_complex* localSpectrum = nullptr;
    fftwf_complex* greenSpectrum = nullptr;
    fftwf_plan forward = nullptr;
    fftwf_plan inverse = nullptr;
    fftwf_plan greenForward = nullptr;
    FlopCount forwardFlops = 0;
    FlopCount inverseFlops = 0;
    FlopCount greenForwardFlops = 0;
    // Global FFT buffer bytes; all localities use the same eviction decisions.
    double workspaceBytes = 0;
    uint64_t lastUsed = 0;
    bool initialized = false;
    bool greenReady = false;
    Vector<hpx::id_type> owners;
    std::vector<int> slabOwner;
    std::vector<ContractionTerm> terms;

    /// Return the number of real mesh cells in this locality's FFT slab.
    size_t cellCount() const {
        return size_t(localSize) * meshSize * meshSize;
    }

    /// Convert a global cell key to a local slab offset, rejecting cells owned by
    /// another slab.
    size_t cellIndex(uint64_t key) const {
        uint64_t const first = uint64_t(localBegin) * meshSize * meshSize;
        if (key < first || key - first >= cellCount()) {
            throw std::runtime_error("PMMM cell reached the wrong FFT slab");
        }
        return key - first;
    }
};

// Transient solve state; all users drain before a new stage is prepared.
struct SolveState {
    Vector<ParticleMeshCell> cells;
    SolveRequest request;
    Pair<ParticleInteger> activeRange;
    // A synchronized endpoint forest can serve the following opening short pass.
    // Reuse is explicit in the driver, never inferred across particle mutation.
    bool boundaryTreeReady = false;
    int boundaryMeshSize = 0;
    int boundaryOrder = 0;
    std::vector<char> stageRungs;
    std::unordered_map<uint64_t, size_t> ownedCells;
    std::vector<std::shared_ptr<SourceCell>> sourceCells;
    std::unordered_map<uint64_t, size_t> sourceCellsByKey;
    std::vector<std::vector<size_t>> sinkSources;
    std::vector<size_t> unresolvedSources;
    std::deque<size_t> readySinks;
    std::vector<hpx::future<void>> tasks;
    // Source producers never wait for farReady. Deposition waits only for
    // these tasks, allowing ghost receivers and near consumers to overlap FFT.
    std::vector<hpx::shared_future<void>> ownedTasks;
    std::shared_ptr<hpx::promise<void>> farPromise;
    hpx::shared_future<void> farReady;
    std::atomic<size_t> pendingGhosts{0};
    uint64_t treeFetchStart = 0;
    uint64_t particleFetchStart = 0;
    size_t remainingSinks = 0;
    std::exception_ptr failure;
    MutexType queueMutex;
    hpx::condition_variable queueChanged;
    KickReturn kickResult;
    std::vector<double> moments;
    double radialTrace = 0;
    std::array<double, 5> background{};
    Vector<Vector3<float>> acceleration;
    Vector<float> potential;
    PmmmDiagnostics diagnostics;
    MutexType accumulationMutex;
};

extern MeshCache cache;
extern SolveState state;
extern MutexType solveMutex;
extern PmmmDiagnostics globalDiagnostics;
extern uint64_t nextGeneration;
extern KickReturn globalKickResult;
extern std::function<void(PmmmCellEvent, uint64_t)> cellObserver;

void observe(PmmmCellEvent event, uint64_t key);
void mpiCheck(int status, char const* operation);
void collectiveCheck(bool okay, char const* message);
int periodicIndex(int index);
int signedIndex(int index);
std::array<int, 3> coordinates(uint64_t key);
void initialize(int meshSize);
std::array<uint64_t, 2> particleCounts(bool applyKick);
void initializeGreen();
void prepare(SolveRequest const& request);
void accumulate(std::vector<CellData> const& cells);
// The locality owning every particle of a cell, from the particle-domain tree.
int cellOwner(uint64_t key);
std::vector<ParticleInteger> readCellCounts(uint64_t generation, std::vector<uint64_t> const& keys);
Vector<ParticleGhost> readGhostParticles(uint64_t generation, uint64_t key);
std::vector<double> readExpansion(uint64_t key);
std::vector<uint64_t> neighbors(uint64_t key);
void planGhosts();
void recordFailure(std::exception_ptr failure);
void publishSource(size_t index, TreeCellRoot root);
void failSource(size_t index, std::exception_ptr failure);
void clearGhosts();
void deposit();
CellData rootMoments(size_t cellIndex, TreeCreateReturn const& root);
void depositCells(std::vector<CellData> cells);
/// Test whether this solve can reuse owned near-FMM roots for mesh moment deposition.
inline bool reuseOwnedRoots() {
    return !frozenNearRequest(state.request);
}
void transform();
// Frozen near passes skip the transform; release waiters on the far field.
void skipTransform();
// Boundary far solve: local expansions evaluated on every owned particle.
void evaluateFar();
void consumeField(ParticleInteger index, std::array<double, 4> const& field, KickReturn& result,
    FlopCount& doubleFlops, FlopCounts& kickFlops);
std::vector<double> localExpansion(uint64_t key);
void computeReadyCells();
void startPipeline();
void finishPipeline();
void abortPipeline();
PmmmDiagnostics diagnostics();
KickReturn kickResult();
void destroy();
void solve(SolveRequest request);
void receiveGhostCell(size_t index);

/// Run local preparation and establish collective success before entering the next MPI
/// phase.
template <class Function>
void prepareCollectively(Function function) {
    std::exception_ptr failure;
    try {
        function();
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        hpx::run_as_os_thread(hpx::annotated_function(
                                  [&]() {
                                      collectiveCheck(!failure,
                                          "A locality failed while preparing a PMMM collective");
                                  },
                                  "gravity.pmmm.checkCollectivePreparation"))
            .get();
    } catch (...) {
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw;
    }
}

/// Distribute indices dynamically over bounded HPX tasks and join all work before
/// propagating errors.
template <class Function>
void parallelFor(size_t count, char const* taskName, Function function) {
    size_t const workers = std::min(count, std::max(size_t(1), hpx::get_os_thread_count()));
    std::atomic<size_t> next{0};
    std::vector<hpx::future<void>> futures;
    futures.reserve(workers);
    std::exception_ptr failure;
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async(hpx::annotated_function(
                [&]() {
                    for (;;) {
                        size_t const index = next.fetch_add(1, std::memory_order_relaxed);
                        if (index >= count) {
                            break;
                        }
                        function(index);
                    }
                },
                taskName)));
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

}    // namespace pmmm::detail
