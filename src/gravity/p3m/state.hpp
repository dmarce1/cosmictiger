/* CosmicTiger - private P3M short-range solver state; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/serialization/vector.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/p3m/fmm/short_fmm.hpp>
#include <cosmictiger/gravity/p3m/short_pair.hpp>
#include <cosmictiger/gravity/p3m/solver.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>

namespace p3m::detail {

using cosmictiger::gravity::p3m::ShortParticle;

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

// Source records of one search-mesh cell, served by the cell's particle owner.
struct CellParticles {
    uint64_t key = 0;
    std::vector<ShortParticle> particles;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & key & particles;
    }
};

// A source cell needed by at least one owned sink cell: owned cells reference
// the sorted particle arrays, ghost cells hold fetched copies.
struct SourceCell {
    uint64_t key = 0;
    bool owned = false;
    size_t ownedIndex = 0;
    std::vector<ShortParticle> particles;
    std::shared_ptr<cosmictiger::gravity::p3m::fmm::ShortTree> tree;
};

struct SolveState {
    int meshSize = 0;
    SolveRequest request;
    Pair<ParticleInteger> activeRange;
    Vector<ParticleMeshCell> cells;
    std::unordered_map<uint64_t, size_t> ownedCells;
    std::vector<SourceCell> sourceCells;
    std::unordered_map<uint64_t, size_t> sourceByKey;
    KickReturn kickResult;
    Vector<Vector3<float>> acceleration;
    Vector<float> potential;
    P3mDiagnostics diagnostics;
    MutexType mutex;
};

extern SolveState state;
extern MutexType solveMutex;
extern P3mDiagnostics globalDiagnostics;
extern KickReturn globalKickResult;
extern uint64_t nextGeneration;

std::array<uint64_t, 2> particleCounts(bool rungSelected);
void prepare(SolveRequest const& request);
std::vector<CellParticles> readCellParticles(
    uint64_t generation, std::vector<uint64_t> const& keys);
void gatherSources();
void evaluate();
P3mDiagnostics diagnostics();
KickReturn kickResult();
void destroy();

int periodicIndex(int index);
std::array<int, 3> coordinates(uint64_t key);
ShortParticle particle(ParticleInteger index);
// pairCurvature is the sink's direct-pair coefficient without G (gravity/timestep.hpp).
void consumeField(ParticleInteger index, std::array<double, 4> const& field, KickReturn& result,
    FlopCount& doubleFlops, FlopCounts& kickFlops, double pairCurvature);
bool activeSink(ParticleInteger index);
void evaluatePairs();
void evaluateTrees();

template <class Function>
void parallelFor(size_t count, char const* taskName, Function function) {
    if (count == 0) {
        return;
    }
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

}    // namespace p3m::detail
