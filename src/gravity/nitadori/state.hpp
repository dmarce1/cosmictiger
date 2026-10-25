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

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/fftw_lock.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/nitadori/cell_tree.hpp>
#include <cosmictiger/gravity/nitadori/collective_dispatch.hpp>
#include <cosmictiger/gravity/nitadori/fmm_batch.hpp>
#include <cosmictiger/gravity/nitadori/ghost_storage.hpp>
#include <cosmictiger/gravity/nitadori/kernels.hpp>
#include <cosmictiger/gravity/nitadori/kick.hpp>
#include <cosmictiger/gravity/nitadori/near_pp.hpp>
#include <cosmictiger/gravity/nitadori/solver.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace pmmm::detail {

struct SolveRequest {
    uint64_t generation = 0;
    bool doPotential = false;
    bool applyKick = false;
    KickParameters parameters;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & generation & doPotential & applyKick;
        if (applyKick) {
            archive & parameters;
        }
    }
};

struct CellOwner {
    int rank;
    ParticleInteger count;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & rank & count;
    }
};

struct CellData {
    uint64_t key = 0;
    std::vector<double> coefficients;
    std::vector<PmmmNearParticle> sources;
    int sourceRank = -1;
    ParticleInteger particleCount = 0;
    double radialTrace = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & key & coefficients & sources & sourceRank & particleCount & radialTrace;
    }
};

struct CellOwners {
    uint64_t key;
    std::vector<CellOwner> owners;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & key & owners;
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
    bool greenReady = false;
    Vector<hpx::id_type> owners;
    std::vector<int> slabOwner;
    std::vector<ContractionTerm> terms;

    size_t cellCount() const {
        return size_t(localSize) * meshSize * meshSize;
    }

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
    std::vector<char> stageRungs;
    std::unordered_map<uint64_t, size_t> ownedCells;
    std::vector<std::shared_ptr<SourceCell>> sourceCells;
    std::unordered_map<uint64_t, std::vector<size_t>> sourceCellsByKey;
    std::vector<std::vector<size_t>> sinkSources;
    std::vector<size_t> unresolvedSources;
    std::deque<size_t> readySinks;
    std::vector<hpx::future<void>> tasks;
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
    std::unordered_map<uint64_t, std::vector<PmmmNearParticle>> sources;
    std::unordered_map<uint64_t, std::vector<CellOwner>> sourceOwners;
    double radialTrace = 0;
    std::array<double, 5> background{};
    Vector<Array<float, NDIM>> acceleration;
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
PmmmNearParticle particle(ParticleInteger index, std::array<int, 3> const& cell);
void initialize();
void initializeGreen();
void prepare(SolveRequest const& request);
void accumulate(std::vector<CellData> const& cells);
std::vector<CellOwners> readOwners(std::vector<uint64_t> const& keys);
Vector<ParticleGhost> readGhostParticles(uint64_t generation, uint64_t key);
std::vector<CellData> readCells(std::vector<uint64_t> const& keys, uint64_t expansionKey);
std::vector<uint64_t> neighbors(uint64_t key);
void planGhosts();
void recordFailure(std::exception_ptr failure);
void publishSource(size_t index, TreeCellRoot root);
void failSource(size_t index, std::exception_ptr failure);
void clearGhosts();
void deposit();
void transform();
void consumeField(ParticleInteger index, std::array<double, 4> const& field, KickReturn& result);
std::vector<double> localExpansion(uint64_t key);
void computeReadyCells();
void startPipeline();
void finishPipeline();
void abortPipeline();
void evaluateParticles(bool doPotential);
PmmmDiagnostics diagnostics();
KickReturn kickResult();
void destroy();
void solve(SolveRequest request);
void receiveGhostCell(size_t index);

template <class Function>
void prepareCollectively(Function function) {
    std::exception_ptr failure;
    try {
        function();
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        hpx::run_as_os_thread([&]() {
            collectiveCheck(!failure, "A locality failed while preparing a PMMM collective");
        }).get();
    } catch (...) {
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw;
    }
}

template <class Function>
void parallelFor(size_t count, Function function) {
    size_t const workers = std::min(count, std::max(size_t(1), hpx::get_os_thread_count()));
    std::atomic<size_t> next{0};
    std::vector<hpx::future<void>> futures;
    futures.reserve(workers);
    std::exception_ptr failure;
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async([&]() {
                for (;;) {
                    size_t const index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= count) {
                        break;
                    }
                    function(index);
                }
            }));
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
