/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fftw3-mpi.h>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/run_as_os_thread.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/fftw_lock.hpp>
#include <cosmictiger/gravity/particle_mesh.hpp>
#include <cosmictiger/gravity/particle_mesh_cloud.hpp>
#include <cosmictiger/gravity/particle_mesh_dispatch.hpp>
#include <cosmictiger/gravity/particle_mesh_exchange.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace particle_mesh::detail {

constexpr int kTileWidth = 8;
constexpr int kHalo = 2;

struct MeshPatch {
    Vector<uint64_t> keys;
    Vector<double> values;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & keys & values;
    }
};

struct Tile {
    std::array<int, NDIM> begin{};
    std::array<int, NDIM> extent{};
    Vector<size_t> cells;

    size_t index(int x, int y, int z) const {
        return (size_t(x - begin[0]) * extent[1] + y - begin[1]) * extent[2] + z - begin[2];
    }

    size_t size() const {
        return size_t(extent[0]) * extent[1] * extent[2];
    }
};

struct MeshState {
    int meshSize = 0;
    int order = 0;
    MPI_Comm communicator = MPI_COMM_NULL;
    ptrdiff_t localSize = 0;
    ptrdiff_t localBegin = 0;
    ptrdiff_t allocated = 0;
    float* real = nullptr;
    fftwf_complex* densitySpectrum = nullptr;
    fftwf_complex* workSpectrum = nullptr;
    fftwf_plan forward = nullptr;
    fftwf_plan inverse = nullptr;
    Vector<hpx::id_type> owners;
    Vector<int> slabOwner;
    Vector<double> inverseWindowSquared;
    Vector<double> squaredModes;
    Vector<double> derivativeWaves;
    Vector<double> density;
    Vector<Array<float, 4>> fields;
    Vector<ParticleMeshCell> cells;
    Vector<Tile> tiles;
    Vector<Array<float, NDIM>> acceleration;
    Vector<float> potential;
    std::array<double, 64> selfGreen{};
    bool selfGreenReady = false;
    ParticleMeshDiagnostics diagnostics;
    MutexType accumulationMutex;

    size_t compactSize() const {
        return size_t(localSize) * meshSize * meshSize;
    }

    size_t compactIndex(uint64_t key) const {
        return key - uint64_t(localBegin) * meshSize * meshSize;
    }
};

MeshState state;
MutexType solveMutex;
ParticleMeshDiagnostics globalDiagnostics;

class CollectiveAbortWorker {
    std::atomic<int> request_{0};
    std::atomic<bool> parked_{false};
    std::thread worker_;

public:
    CollectiveAbortWorker()
      : worker_([this]() {
          request_.wait(0);
          if (request_.load() == 2) {
              std::fputs("Particle-mesh collective dispatch failed; aborting MPI job\n", stderr);
              std::fflush(stderr);
              MPI_Abort(MPI_COMM_WORLD, 86);
              std::_Exit(86);
          }
      }) {}

    ~CollectiveAbortWorker() {
        request_.store(1);
        request_.notify_one();
        worker_.join();
    }

    [[noreturn]] void abort() noexcept {
        request_.store(2);
        request_.notify_one();
        // The native worker is already running, so failure handling needs no
        // allocation or HPX I/O worker (which may be blocked inside MPI).
        for (;;) {
            parked_.wait(false);
        }
    }
};

CollectiveAbortWorker& collectiveAbortWorker() {
    static CollectiveAbortWorker worker;
    return worker;
}

void prepareCollectiveDispatch() {
    (void) collectiveAbortWorker();
}

[[noreturn]] void abortCollectiveDispatch() noexcept {
    collectiveAbortWorker().abort();
}

template <class Function>
void parallelFor(size_t count, Function function) {
    if (count == 0) {
        return;
    }
    size_t const workers = std::max(size_t(1), std::min(count, hpx::get_os_thread_count()));
    std::atomic<size_t> next{0};
    Vector<hpx::future<void>> futures;
    futures.reserve(workers);
    std::exception_ptr launchFailure;
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
        launchFailure = std::current_exception();
    }
    for (auto& future : futures) {
        future.wait();
    }
    if (launchFailure) {
        std::rethrow_exception(launchFailure);
    }
    for (auto& future : futures) {
        future.get();
    }
}

void mpiCheck(int result, char const* operation) {
    if (result != MPI_SUCCESS) {
        throw std::runtime_error(std::string("Particle-mesh MPI failure: ") + operation);
    }
}

void collectiveCheck(bool okay, char const* message) {
    int local = okay ? 1 : 0;
    int global = 0;
    mpiCheck(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, state.communicator),
        "failure agreement");
    if (!global) {
        throw std::runtime_error(message);
    }
}

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
            collectiveCheck(!failure,
                "A locality failed while preparing the distributed particle-mesh transform");
        }).get();
    } catch (...) {
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw;
    }
}

void initialize() {
    auto const& gravity = getOptions().gravity;
    int const n = gravity.meshSize;
    int const order = cloudOrder(gravity.cloud);
    if (state.communicator != MPI_COMM_NULL) {
        if (state.meshSize != n || state.order != order) {
            throw std::runtime_error("Particle-mesh geometry changed without destruction");
        }
        return;
    }
    state.meshSize = n;
    state.order = order;
    int const locality = hpxRank();
    int const localityCount = hpxSize();
    Vector<int> localityIds;
    Vector<int64_t> slabs;
    hpx::run_as_os_thread([&]() {
        mpiCheck(MPI_Comm_dup(MPI_COMM_WORLD, &state.communicator), "communicator duplication");
        mpiCheck(MPI_Comm_set_errhandler(state.communicator, MPI_ERRORS_RETURN), "error handler");
        int mpiSize = 0;
        mpiCheck(MPI_Comm_size(state.communicator, &mpiSize), "communicator size");
        collectiveCheck(
            mpiSize == localityCount, "Particle mesh requires one HPX locality per MPI process");
        bool mappingAllocated = true;
        try {
            localityIds.resize(mpiSize);
            slabs.resize(2 * mpiSize);
        } catch (...) {
            mappingAllocated = false;
        }
        collectiveCheck(mappingAllocated, "Unable to allocate particle-mesh rank mapping");
        mpiCheck(MPI_Allgather(
                     &locality, 1, MPI_INT, localityIds.data(), 1, MPI_INT, state.communicator),
            "locality mapping");
        {
            std::lock_guard<std::mutex> lock(fftwPlannerMutex());
            fftwf_mpi_init();
            state.allocated = fftwf_mpi_local_size_3d(
                n, n, n / 2 + 1, state.communicator, &state.localSize, &state.localBegin);
            ptrdiff_t const allocation = std::max(ptrdiff_t(1), state.allocated);
            state.real = fftwf_alloc_real(2 * allocation);
            state.densitySpectrum = fftwf_alloc_complex(allocation);
            state.workSpectrum = fftwf_alloc_complex(allocation);
        }
        collectiveCheck(state.real && state.densitySpectrum && state.workSpectrum,
            "Unable to allocate distributed particle-mesh FFT workspace");
        {
            std::lock_guard<std::mutex> lock(fftwPlannerMutex());
            state.forward = fftwf_mpi_plan_dft_r2c_3d(
                n, n, n, state.real, state.densitySpectrum, state.communicator, FFTW_ESTIMATE);
            state.inverse = fftwf_mpi_plan_dft_c2r_3d(
                n, n, n, state.workSpectrum, state.real, state.communicator, FFTW_ESTIMATE);
        }
        collectiveCheck(
            state.forward && state.inverse, "Unable to create distributed particle-mesh FFT plans");
        int64_t const localSlab[2] = {state.localBegin, state.localSize};
        mpiCheck(MPI_Allgather(
                     localSlab, 2, MPI_INT64_T, slabs.data(), 2, MPI_INT64_T, state.communicator),
            "slab distribution");
    }).get();
    state.owners.resize(localityCount);
    state.slabOwner.assign(n, -1);
    for (int rank = 0; rank < localityCount; ++rank) {
        auto const owner = std::find_if(
            hpxLocalities().begin(), hpxLocalities().end(), [&](hpx::id_type const& id) {
                return hpx::naming::get_locality_id_from_id(id) == unsigned(localityIds[rank]);
            });
        if (owner == hpxLocalities().end()) {
            throw std::runtime_error("Cannot map an MPI rank to an HPX locality");
        }
        state.owners[rank] = *owner;
        for (int64_t x = slabs[2 * rank]; x < slabs[2 * rank] + slabs[2 * rank + 1]; ++x) {
            state.slabOwner[x] = rank;
        }
    }
    if (std::find(state.slabOwner.begin(), state.slabOwner.end(), -1) != state.slabOwner.end()) {
        throw std::runtime_error("FFTW slab distribution does not cover the mesh");
    }
    state.density.resize(state.compactSize());
    state.fields.resize(state.compactSize());
    state.inverseWindowSquared.resize(n);
    state.squaredModes.resize(n);
    state.derivativeWaves.resize(n);
    for (int index = 0; index < n; ++index) {
        int const mode = signedMode(index, n);
        double const window = assignmentWindow(mode, n, order);
        state.inverseWindowSquared[index] = 1 / (window * window);
        state.squaredModes[index] = double(mode) * mode;
        state.derivativeWaves[index] = derivativeWavenumber(index, n);
    }
}

double potentialFactor(int x, int y, int z) {
    double const norm = state.squaredModes[x] + state.squaredModes[y] + state.squaredModes[z];
    return norm == 0 ? 0 :
                       -state.inverseWindowSquared[x] * state.inverseWindowSquared[y] *
            state.inverseWindowSquared[z] / (std::numbers::pi * norm);
}

void prepare() {
    initialize();
    state.cells = particleMeshSort(state.meshSize);
    state.acceleration.resize(particlesSize());
    state.potential.assign(particlesSize(), 0);
    std::fill(state.density.begin(), state.density.end(), 0);
    state.diagnostics = {};
    state.tiles.clear();
    std::unordered_map<uint64_t, size_t> tileIndices;
    int const n = state.meshSize;
    int const tileCount = (n + kTileWidth - 1) / kTileWidth;
    for (size_t cellIndex = 0; cellIndex < state.cells.size(); ++cellIndex) {
        auto const key = state.cells[cellIndex].key;
        int const coordinates[NDIM] = {
            int(key / (uint64_t(n) * n)), int(key / n % n), int(key % n)};
        uint64_t const tileKey =
            (uint64_t(coordinates[0] / kTileWidth) * tileCount + coordinates[1] / kTileWidth) *
                tileCount +
            coordinates[2] / kTileWidth;
        auto const [iterator, inserted] = tileIndices.emplace(tileKey, state.tiles.size());
        if (inserted) {
            Tile tile;
            for (int axis = 0; axis < NDIM; ++axis) {
                int const start = coordinates[axis] / kTileWidth * kTileWidth;
                tile.begin[axis] = start - kHalo;
                tile.extent[axis] = std::min(kTileWidth, n - start) + 2 * kHalo;
            }
            state.tiles.push_back(std::move(tile));
        }
        state.tiles[iterator->second].cells.push_back(cellIndex);
    }
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        state.diagnostics.sourceMass += particlesMass(index);
    }
}

void accumulate(MeshPatch const& patch) {
    std::unique_lock<MutexType> lock(state.accumulationMutex);
    for (size_t i = 0; i < patch.keys.size(); ++i) {
        size_t const index = state.compactIndex(patch.keys[i]);
        if (index >= state.density.size()) {
            throw std::runtime_error("Particle-mesh deposition reached the wrong slab");
        }
        state.density[index] += patch.values[i];
    }
}

Vector<Array<float, 4>> readFields(Vector<uint64_t> const& keys) {
    Vector<Array<float, 4>> result(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        size_t const index = state.compactIndex(keys[i]);
        if (index >= state.fields.size()) {
            throw std::runtime_error("Particle-mesh interpolation reached the wrong slab");
        }
        result[i] = state.fields[index];
    }
    return result;
}

}    // namespace particle_mesh::detail

HPX_PLAIN_ACTION(particle_mesh::detail::accumulate, ParticleMeshAccumulateAction);
HPX_PLAIN_ACTION(particle_mesh::detail::readFields, ParticleMeshReadFieldsAction);

namespace particle_mesh::detail {

std::array<CloudWeights, NDIM> weights(ParticleInteger particle) {
    std::array<CloudWeights, NDIM> result;
    for (int axis = 0; axis < NDIM; ++axis) {
        result[axis] =
            cloudWeightsFromFixed(particlesPos(axis, particle).raw(), state.meshSize, state.order);
    }
    return result;
}

template <class Function>
void stencil(std::array<CloudWeights, NDIM> const& cloud, Function function) {
    for (int x = 0; x < state.order; ++x) {
        for (int y = 0; y < state.order; ++y) {
            for (int z = 0; z < state.order; ++z) {
                function(cloud[0].first + x, cloud[1].first + y, cloud[2].first + z,
                    cloud[0].weights[x] * cloud[1].weights[y] * cloud[2].weights[z]);
            }
        }
    }
}

template <class Function>
void visitTile(Tile const& tile, Function function) {
    int const n = state.meshSize;
    for (int x = tile.begin[0]; x < tile.begin[0] + tile.extent[0]; ++x) {
        int const periodicX = periodicIndex(x, n);
        int const owner = state.slabOwner[periodicX];
        for (int y = tile.begin[1]; y < tile.begin[1] + tile.extent[1]; ++y) {
            for (int z = tile.begin[2]; z < tile.begin[2] + tile.extent[2]; ++z) {
                uint64_t const key =
                    (uint64_t(periodicX) * n + periodicIndex(y, n)) * n + periodicIndex(z, n);
                function(owner, key, tile.index(x, y, z));
            }
        }
    }
}

void deposit() {
    parallelFor(state.tiles.size(), [&](size_t tileIndex) {
        auto const& tile = state.tiles[tileIndex];
        Vector<double> density(tile.size(), 0);
        for (size_t cellIndex : tile.cells) {
            auto const& cell = state.cells[cellIndex];
            for (ParticleInteger particle = cell.begin; particle < cell.end; ++particle) {
                double const mass = particlesMass(particle);
                stencil(weights(particle), [&](int x, int y, int z, double weight) {
                    density[tile.index(x, y, z)] += mass * weight;
                });
            }
        }
        std::unordered_map<int, MeshPatch> patches;
        visitTile(tile, [&](int owner, uint64_t key, size_t index) {
            if (density[index] != 0) {
                patches[owner].keys.push_back(key);
                patches[owner].values.push_back(density[index]);
            }
        });
        Vector<hpx::future<void>> futures;
        futures.reserve(patches.size());
        std::exception_ptr launchFailure;
        try {
            for (auto& [owner, patch] : patches) {
                futures.push_back(hpx::async<ParticleMeshAccumulateAction>(
                    state.owners[owner], std::move(patch)));
            }
        } catch (...) {
            launchFailure = std::current_exception();
        }
        // Remote writes must finish before failed-solve cleanup can clear the
        // density slab. Leave the catch before waiting: HPX can migrate a task.
        for (auto& future : futures) {
            future.wait();
        }
        if (launchFailure) {
            std::rethrow_exception(launchFailure);
        }
        for (auto& future : futures) {
            future.get();
        }
    });
}

template <class Function>
void spectralRows(Function function) {
    int const n = state.meshSize;
    prepareCollectively([&]() {
        parallelFor(size_t(state.localSize) * n, [&](size_t row) {
            int const x = int(row / n + state.localBegin);
            int const y = int(row % n);
            for (int z = 0; z <= n / 2; ++z) {
                size_t const index = row * (n / 2 + 1) + z;
                function(index, x, y, z);
            }
        });
    });
}

void inverseTransform() {
    hpx::run_as_os_thread([]() { fftwf_execute(state.inverse); }).get();
}

void initializeSelfGreen() {
    if (state.selfGreenReady) {
        return;
    }
    int const n = state.meshSize;
    int const order = state.order;
    double const meshVolume = double(n) * n * n;
    spectralRows([&](size_t index, int x, int y, int z) {
        state.workSpectrum[index][0] = meshVolume * potentialFactor(x, y, z);
        state.workSpectrum[index][1] = 0;
    });
    inverseTransform();
    std::array<double, 64> local{};
    for (int x = 0; x < order; ++x) {
        if (x < state.localBegin || x >= state.localBegin + state.localSize) {
            continue;
        }
        for (int y = 0; y < order; ++y) {
            for (int z = 0; z < order; ++z) {
                size_t const index = (size_t(x - state.localBegin) * n + y) * (2 * (n / 2 + 1)) + z;
                local[(x * order + y) * order + z] = state.real[index] / meshVolume;
            }
        }
    }
    hpx::run_as_os_thread([&]() {
        mpiCheck(MPI_Allreduce(local.data(), state.selfGreen.data(), int(local.size()), MPI_DOUBLE,
                     MPI_SUM, state.communicator),
            "self-potential Green table");
    }).get();
    state.selfGreenReady = true;
}

void transform(bool doPotential) {
    int const n = state.meshSize;
    double const meshVolume = double(n) * n * n;
    if (doPotential) {
        initializeSelfGreen();
    }
    state.diagnostics.depositedMass = 0;
    for (double mass : state.density) {
        state.diagnostics.depositedMass += mass;
    }
    prepareCollectively([&]() {
        parallelFor(size_t(state.localSize) * n, [&](size_t row) {
            size_t const paddedRow = row * 2 * (n / 2 + 1);
            for (int z = 0; z < n; ++z) {
                state.real[paddedRow + z] = state.density[row * n + z] * meshVolume;
            }
            for (int z = n; z < 2 * (n / 2 + 1); ++z) {
                state.real[paddedRow + z] = 0;
            }
        });
    });
    hpx::run_as_os_thread([]() { fftwf_execute(state.forward); }).get();
    double const gm = getOptions().gravity.gm;
    for (int component = 0; component < (doPotential ? 4 : 3); ++component) {
        spectralRows([&](size_t index, int x, int y, int z) {
            double const factor = gm * potentialFactor(x, y, z);
            double const real = state.densitySpectrum[index][0] * factor;
            double const imaginary = state.densitySpectrum[index][1] * factor;
            if (component == 3) {
                state.workSpectrum[index][0] = real;
                state.workSpectrum[index][1] = imaginary;
            } else {
                int const coordinates[NDIM] = {x, y, z};
                double const wave = state.derivativeWaves[coordinates[component]];
                state.workSpectrum[index][0] = wave * imaginary;
                state.workSpectrum[index][1] = -wave * real;
            }
        });
        inverseTransform();
        prepareCollectively([&]() {
            parallelFor(size_t(state.localSize) * n, [&](size_t row) {
                for (int z = 0; z < n; ++z) {
                    state.fields[row * n + z][component] =
                        state.real[row * 2 * (n / 2 + 1) + z] / meshVolume;
                    if (!doPotential) {
                        state.fields[row * n + z][3] = 0;
                    }
                }
            });
        });
    }
}

double selfPotential(std::array<CloudWeights, NDIM> const& cloud) {
    double correlations[NDIM][4]{};
    for (int axis = 0; axis < NDIM; ++axis) {
        for (int offset = 0; offset < state.order; ++offset) {
            for (int index = 0; index + offset < state.order; ++index) {
                correlations[axis][offset] +=
                    cloud[axis].weights[index] * cloud[axis].weights[index + offset];
            }
            if (offset != 0) {
                correlations[axis][offset] *= 2;
            }
        }
    }
    double value = 0;
    for (int x = 0; x < state.order; ++x) {
        for (int y = 0; y < state.order; ++y) {
            for (int z = 0; z < state.order; ++z) {
                value += correlations[0][x] * correlations[1][y] * correlations[2][z] *
                    state.selfGreen[(x * state.order + y) * state.order + z];
            }
        }
    }
    return value;
}

void interpolate(bool doPotential) {
    parallelFor(state.tiles.size(), [&](size_t tileIndex) {
        auto const& tile = state.tiles[tileIndex];
        Vector<Array<float, 4>> fields(tile.size());
        struct Request {
            Vector<uint64_t> keys;
            Vector<size_t> indices;
            hpx::future<Vector<Array<float, 4>>> future;
        };
        std::unordered_map<int, Request> requests;
        visitTile(tile, [&](int owner, uint64_t key, size_t index) {
            requests[owner].keys.push_back(key);
            requests[owner].indices.push_back(index);
        });
        exchangeFields(
            requests,
            [](int owner, Request& request) {
                return hpx::async<ParticleMeshReadFieldsAction>(
                    state.owners[owner], std::move(request.keys));
            },
            [&](Request const& request, Vector<Array<float, 4>> const& values) {
                for (size_t i = 0; i < request.indices.size(); ++i) {
                    fields[request.indices[i]] = values[i];
                }
            });
        for (size_t cellIndex : tile.cells) {
            auto const& cell = state.cells[cellIndex];
            for (ParticleInteger particle = cell.begin; particle < cell.end; ++particle) {
                auto const cloud = weights(particle);
                double value[4]{};
                stencil(cloud, [&](int x, int y, int z, double weight) {
                    auto const& field = fields[tile.index(x, y, z)];
                    for (int component = 0; component < (doPotential ? 4 : 3); ++component) {
                        value[component] += field[component] * weight;
                    }
                });
                for (int axis = 0; axis < NDIM; ++axis) {
                    state.acceleration[particle][axis] = value[axis];
                }
                if (doPotential) {
                    // The same discrete compensated Green function is used for
                    // the source and sink cloud, including subcell dependence.
                    state.potential[particle] = value[3] -
                        getOptions().gravity.gm * particlesMass(particle) * selfPotential(cloud);
                }
            }
        }
    });
}

ParticleMeshDiagnostics diagnostics() {
    return state.diagnostics;
}

void destroy() {
    if (state.communicator == MPI_COMM_NULL) {
        return;
    }
    hpx::run_as_os_thread([]() {
        // Failed planning can leave a plan on only some ranks. Destruction of
        // MPI plans is collective, so only destroy plans present everywhere.
        // A partially created plan is deliberately abandoned on this error
        // shutdown path, rather than deadlocking the other localities.
        int const present[2] = {state.forward != nullptr, state.inverse != nullptr};
        int allPresent[2]{};
        mpiCheck(MPI_Allreduce(present, allPresent, 2, MPI_INT, MPI_MIN, state.communicator),
            "plan destruction agreement");
        {
            std::lock_guard<std::mutex> lock(fftwPlannerMutex());
            if (allPresent[0]) {
                fftwf_destroy_plan(state.forward);
            }
            if (allPresent[1]) {
                fftwf_destroy_plan(state.inverse);
            }
            state.forward = nullptr;
            state.inverse = nullptr;
            fftwf_free(state.real);
            fftwf_free(state.densitySpectrum);
            fftwf_free(state.workSpectrum);
            state.real = nullptr;
            state.densitySpectrum = nullptr;
            state.workSpectrum = nullptr;
            // Do not call fftwf_cleanup: the initial-condition and analysis
            // transforms share FFTW's process-global planner state.
        }
        mpiCheck(MPI_Comm_free(&state.communicator), "communicator destruction");
    }).get();
    state.owners.clear();
    state.slabOwner.clear();
    state.inverseWindowSquared.clear();
    state.squaredModes.clear();
    state.derivativeWaves.clear();
    state.density.clear();
    state.fields.clear();
    state.cells.clear();
    state.tiles.clear();
    state.acceleration.clear();
    state.potential.clear();
    state.selfGreenReady = false;
}

}    // namespace particle_mesh::detail

HPX_PLAIN_ACTION(particle_mesh::detail::prepare, ParticleMeshPrepareAction);
HPX_PLAIN_ACTION(particle_mesh::detail::deposit, ParticleMeshDepositAction);
HPX_PLAIN_ACTION(particle_mesh::detail::transform, ParticleMeshTransformAction);
HPX_PLAIN_ACTION(particle_mesh::detail::interpolate, ParticleMeshInterpolateAction);
HPX_PLAIN_ACTION(particle_mesh::detail::diagnostics, ParticleMeshDiagnosticsAction);
HPX_PLAIN_ACTION(particle_mesh::detail::destroy, ParticleMeshDestroyAction);

namespace particle_mesh::detail {

template <class Action, class... Arguments>
void allLocalities(Arguments... arguments) {
    if constexpr (std::is_same_v<Action, ParticleMeshPrepareAction> ||
        std::is_same_v<Action, ParticleMeshTransformAction> ||
        std::is_same_v<Action, ParticleMeshDestroyAction>) {
        dispatchCollectively(hpxLocalities().size(),
            [&](size_t index) { return hpx::async<Action>(hpxLocalities()[index], arguments...); });
        return;
    }
    Vector<hpx::future<void>> futures;
    futures.reserve(hpxLocalities().size());
    std::exception_ptr launchFailure;
    try {
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<Action>(locality, arguments...));
        }
    } catch (...) {
        launchFailure = std::current_exception();
    }
    for (auto& future : futures) {
        future.wait();
    }
    if (launchFailure) {
        std::rethrow_exception(launchFailure);
    }
    for (auto& future : futures) {
        future.get();
    }
}

}    // namespace particle_mesh::detail

void particleMeshSolve(bool doPotential) {
    using namespace particle_mesh::detail;
    std::unique_lock<MutexType> lock(solveMutex);
    if (getOptions().hydro.enabled) {
        sphParticlesClearCaches();
    }
    allLocalities<ParticleMeshPrepareAction>();
    allLocalities<ParticleMeshDepositAction>();
    allLocalities<ParticleMeshTransformAction>(doPotential);
    allLocalities<ParticleMeshInterpolateAction>(doPotential);
    Vector<hpx::future<ParticleMeshDiagnostics>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<ParticleMeshDiagnosticsAction>(locality));
    }
    globalDiagnostics = {};
    for (auto& future : futures) {
        auto const value = future.get();
        globalDiagnostics.sourceMass += value.sourceMass;
        globalDiagnostics.depositedMass += value.depositedMass;
    }
}

void particleMeshDestroy() {
    using namespace particle_mesh::detail;
    std::unique_lock<MutexType> lock(solveMutex);
    allLocalities<ParticleMeshDestroyAction>();
}

ParticleMeshDiagnostics particleMeshDiagnostics() {
    return particle_mesh::detail::globalDiagnostics;
}

Array<float, NDIM> const& particleMeshAcceleration(ParticleInteger index) {
    return particle_mesh::detail::state.acceleration.at(index);
}

float particleMeshPotential(ParticleInteger index) {
    return particle_mesh::detail::state.potential.at(index);
}

double particleMeshEffectiveRadius() {
    auto const& gravity = getOptions().gravity;
    return particle_mesh::effectiveRadius(
        gravity.meshSize, particle_mesh::cloudOrder(gravity.cloud));
}
