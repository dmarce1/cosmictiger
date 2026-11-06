/* CosmicTiger - scalar particle-mesh (PM) solve; BSL-1.0. */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fftw3-mpi.h>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/run_as_os_thread.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/fftw_lock.hpp>
#include <cosmictiger/gravity/pm/cloud.hpp>
#include <cosmictiger/gravity/pm/collective_dispatch.hpp>
#include <cosmictiger/gravity/pm/fft_flops.hpp>
#include <cosmictiger/gravity/pm/geometry.hpp>
#include <cosmictiger/gravity/pm/solver.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif

namespace pm::detail {

// Deposition and interpolation tiles: whole-tile halos cover every cloud point
// (PCS reaches cell-1 through cell+2) of the particles sorted into the tile.
constexpr int kTileWidth = 8;
constexpr int kHalo = 2;

struct MeshPatch {
    std::vector<uint64_t> keys;
    std::vector<double> values;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & keys & values;
    }
};

struct Tile {
    std::array<int, kDimensions> begin{};
    std::array<int, kDimensions> extent{};
    std::vector<size_t> cells;

    size_t index(int x, int y, int z) const {
        return (size_t(x - begin[0]) * extent[1] + y - begin[1]) * extent[2] + z - begin[2];
    }

    size_t size() const {
        return size_t(extent[0]) * extent[1] * extent[2];
    }
};

struct LocalSolveReturn {
    double potentialEnergy = 0.0;
    double minimumTimestep = 1.0e30;
    double sourceMass = 0.0;
    uint64_t sourceParticles = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & potentialEnergy & minimumTimestep & sourceMass & sourceParticles;
    }
};

struct MeshState {
    int meshSize = 0;
    int order = 0;
    double splitRadius = 0;
    MPI_Comm communicator = MPI_COMM_NULL;
    ptrdiff_t localSize = 0;
    ptrdiff_t localBegin = 0;
    ptrdiff_t allocated = 0;
    float* real = nullptr;
    fftwf_complex* densitySpectrum = nullptr;
    fftwf_complex* workSpectrum = nullptr;
    fftwf_plan forward = nullptr;
    fftwf_plan inverse = nullptr;
    FlopCount forwardFlops = 0;
    FlopCount inverseFlops = 0;
    std::vector<hpx::id_type> owners;
    std::vector<int> slabOwner;
    std::vector<double> inverseWindowSquared;
    std::vector<double> squaredModes;
    std::vector<double> derivativeWaves;
    std::vector<double> density;
    std::vector<std::array<float, 4>> fields;
    Vector<ParticleMeshCell> cells;
    std::vector<Tile> tiles;
    Vector<Vector3<float>> acceleration;
    Vector<float> potential;
    std::array<double, 64> selfGreen{};
    bool selfGreenReady = false;
    PmSolveRequest request;
    LocalSolveReturn localReturn;
    PmDiagnostics diagnostics;
    uint64_t transformCount = 0;
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
PmDiagnostics globalDiagnostics;

template <class Function>
void parallelFor(size_t count, char const* taskName, Function function) {
    if (count == 0) {
        return;
    }
    size_t const workers = std::max(size_t(1), std::min(count, hpx::get_os_thread_count()));
    std::atomic<size_t> next{0};
    std::vector<hpx::future<void>> futures;
    futures.reserve(workers);
    std::exception_ptr launchFailure;
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
        hpx::run_as_os_thread(hpx::annotated_function(
                                  [&]() {
                                      collectiveCheck(!failure,
                                          "A locality failed while preparing the distributed "
                                          "particle-mesh transform");
                                  },
                                  "gravity.pm.checkCollectivePreparation"))
            .get();
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
    double const splitRadius = usesSplitParticleMesh(gravity) ? pm::splitRadius(gravity) : 0.0;
    if (n < std::max(2, order)) {
        throw std::runtime_error("Particle-mesh size must be at least max(2, cloud points)");
    }
    if (state.communicator != MPI_COMM_NULL) {
        if (state.meshSize != n || state.order != order || state.splitRadius != splitRadius) {
            throw std::runtime_error("Particle-mesh geometry changed without destruction");
        }
        return;
    }
    state.meshSize = n;
    state.order = order;
    state.splitRadius = splitRadius;
    int const locality = hpxRank();
    int const localityCount = hpxSize();
    int const fftThreads = int(
        std::clamp(hpx::get_os_thread_count(), size_t(1), size_t(std::numeric_limits<int>::max())));
    std::vector<int> localityIds;
    std::vector<int64_t> slabs;
    hpx::run_as_os_thread(
        hpx::annotated_function(
            [&]() {
                mpiCheck(
                    MPI_Comm_dup(MPI_COMM_WORLD, &state.communicator), "communicator duplication");
                mpiCheck(MPI_Comm_set_errhandler(state.communicator, MPI_ERRORS_RETURN),
                    "error handler");
                int mpiSize = 0;
                mpiCheck(MPI_Comm_size(state.communicator, &mpiSize), "communicator size");
                collectiveCheck(mpiSize == localityCount,
                    "Particle mesh requires one HPX locality per MPI process");
                bool mappingAllocated = true;
                try {
                    localityIds.resize(mpiSize);
                    slabs.resize(2 * mpiSize);
                } catch (...) {
                    mappingAllocated = false;
                }
                collectiveCheck(mappingAllocated, "Unable to allocate particle-mesh rank mapping");
                mpiCheck(MPI_Allgather(&locality, 1, MPI_INT, localityIds.data(), 1, MPI_INT,
                             state.communicator),
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
                    "Unable to allocate the distributed particle-mesh FFT workspace");
                {
                    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
                    fftwf_plan_with_nthreads(fftThreads);
                    state.forward = fftwf_mpi_plan_dft_r2c_3d(n, n, n, state.real,
                        state.densitySpectrum, state.communicator, FFTW_ESTIMATE);
                    state.inverse = fftwf_mpi_plan_dft_c2r_3d(
                        n, n, n, state.workSpectrum, state.real, state.communicator, FFTW_ESTIMATE);
                    fftwf_plan_with_nthreads(1);
                }
                collectiveCheck(state.forward && state.inverse,
                    "Unable to create the distributed particle-mesh FFT plans");
                bool counted = true;
                try {
                    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
                    state.forwardFlops = fftPlanFlops(state.forward);
                    state.inverseFlops = fftPlanFlops(state.inverse);
                    // FFTW's MPI cost hook already sums all ranks; submit on rank zero only.
                    if (hpxRank() != 0) {
                        state.forwardFlops = state.inverseFlops = 0;
                    }
                } catch (...) {
                    counted = false;
                }
                collectiveCheck(counted, "PM FFT operation count exceeds the supported range");
                int64_t const localSlab[2] = {state.localBegin, state.localSize};
                mpiCheck(MPI_Allgather(localSlab, 2, MPI_INT64_T, slabs.data(), 2, MPI_INT64_T,
                             state.communicator),
                    "slab distribution");
            },
            "gravity.pm.initializeFft"))
        .get();
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
    state.selfGreenReady = false;
    ++state.diagnostics.planBuilds;
    if (hpxRank() == 0) {
        terminal::detailPrint(
            "PM mesh %i^3, cloud order %i, split radius %e\n", n, order, splitRadius);
    }
}

// Unit box, excluding gm and inverse-FFT normalization. Both assignment passes
// are compensated by 1/W^2; the Gaussian factor exp(-k^2 r_s^2) with the
// physical wave number k=2*pi*m keeps only the smooth long-range potential.
double potentialFactor(int x, int y, int z) {
    double const norm = state.squaredModes[x] + state.squaredModes[y] + state.squaredModes[z];
    if (norm == 0) {
        return 0;
    }
    double factor = -state.inverseWindowSquared[x] * state.inverseWindowSquared[y] *
        state.inverseWindowSquared[z] / (std::numbers::pi * norm);
    if (state.splitRadius > 0) {
        double const kSquared = 4 * std::numbers::pi * std::numbers::pi * norm;
        factor *= std::exp(-kSquared * state.splitRadius * state.splitRadius);
    }
    return factor;
}

void prepare(PmSolveRequest const& request) {
    initialize();
    prepareCollectively([&]() {
        state.request = request;
        state.cells = particleMeshSort(state.meshSize, {0, particlesOwnedSize()});
        if (request.storeFields) {
            state.acceleration.resize(particlesOwnedSize());
            state.potential.assign(particlesOwnedSize(), 0.0f);
        } else {
            Vector<Vector3<float>>().swap(state.acceleration);
            Vector<float>().swap(state.potential);
        }
        std::fill(state.density.begin(), state.density.end(), 0.0);
        auto const previous = state.diagnostics;
        state.diagnostics = {};
        state.diagnostics.planBuilds = previous.planBuilds;
        state.diagnostics.solveCount = previous.solveCount;
        state.diagnostics.meshSize = state.meshSize;
        state.diagnostics.cloudOrder = state.order;
        state.diagnostics.splitRadius = state.splitRadius;
        state.localReturn = {};
        state.tiles.clear();
        std::unordered_map<uint64_t, size_t> tileIndices;
        int const n = state.meshSize;
        int const tileCount = (n + kTileWidth - 1) / kTileWidth;
        for (size_t cellIndex = 0; cellIndex < state.cells.size(); ++cellIndex) {
            auto const key = state.cells[cellIndex].key;
            int const coordinates[kDimensions] = {
                int(key / (uint64_t(n) * n)), int(key / n % n), int(key % n)};
            uint64_t const tileKey =
                (uint64_t(coordinates[0] / kTileWidth) * tileCount + coordinates[1] / kTileWidth) *
                    tileCount +
                coordinates[2] / kTileWidth;
            auto const [iterator, inserted] = tileIndices.emplace(tileKey, state.tiles.size());
            if (inserted) {
                Tile tile;
                for (int axis = 0; axis < kDimensions; ++axis) {
                    int const start = coordinates[axis] / kTileWidth * kTileWidth;
                    tile.begin[axis] = start - kHalo;
                    tile.extent[axis] = std::min(kTileWidth, n - start) + 2 * kHalo;
                }
                state.tiles.push_back(std::move(tile));
            }
            state.tiles[iterator->second].cells.push_back(cellIndex);
        }
        for (ParticleInteger index = 0; index < particlesOwnedSize(); ++index) {
            state.diagnostics.sourceMass += particlesMass(index);
        }
        state.diagnostics.sourceParticles = particlesOwnedSize();
        state.localReturn.sourceMass = state.diagnostics.sourceMass;
        state.localReturn.sourceParticles = state.diagnostics.sourceParticles;
    });
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
    addCpuFlops(FlopCount(patch.keys.size()), FlopPrecision::Double);
}

std::vector<std::array<float, 4>> readFields(std::vector<uint64_t> const& keys) {
    std::vector<std::array<float, 4>> result(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        size_t const index = state.compactIndex(keys[i]);
        if (index >= state.fields.size()) {
            throw std::runtime_error("Particle-mesh interpolation reached the wrong slab");
        }
        result[i] = state.fields[index];
    }
    return result;
}

}    // namespace pm::detail

HPX_PLAIN_ACTION(pm::detail::accumulate, PmAccumulateAction);
HPX_PLAIN_ACTION(pm::detail::readFields, PmReadFieldsAction);

namespace pm::detail {

std::array<CloudWeights, kDimensions> weights(ParticleInteger particle) {
    std::array<CloudWeights, kDimensions> result;
    for (int axis = 0; axis < kDimensions; ++axis) {
        result[axis] =
            cloudWeightsFromFixed(particlesPos(axis, particle).raw(), state.meshSize, state.order);
    }
    return result;
}

template <class Function>
void stencil(std::array<CloudWeights, kDimensions> const& cloud, Function function) {
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
    parallelFor(state.tiles.size(), "gravity.pm.deposit", [&](size_t tileIndex) {
        auto const& tile = state.tiles[tileIndex];
        std::vector<double> density(tile.size(), 0);
        FlopCount flops = 0;
        for (size_t cellIndex : tile.cells) {
            auto const& cell = state.cells[cellIndex];
            for (ParticleInteger particle = cell.begin; particle < cell.end; ++particle) {
                double const mass = particlesMass(particle);
                stencil(weights(particle), [&](int x, int y, int z, double weight) {
                    density[tile.index(x, y, z)] += mass * weight;
                });
                // Three one-dimensional clouds, their products, and the accumulation.
                flops +=
                    3 * (state.order + 2) + FlopCount(state.order) * state.order * state.order * 4;
            }
        }
        addCpuFlops(flops, FlopPrecision::Double);
        std::unordered_map<int, MeshPatch> patches;
        visitTile(tile, [&](int owner, uint64_t key, size_t index) {
            if (density[index] != 0) {
                patches[owner].keys.push_back(key);
                patches[owner].values.push_back(density[index]);
            }
        });
        std::vector<hpx::future<void>> futures;
        futures.reserve(patches.size());
        std::exception_ptr launchFailure;
        try {
            for (auto& [owner, patch] : patches) {
                if (state.owners[owner] == hpx::find_here()) {
                    accumulate(patch);
                } else {
                    futures.push_back(
                        hpx::async<PmAccumulateAction>(state.owners[owner], std::move(patch)));
                }
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
        parallelFor(size_t(state.localSize) * n, "gravity.pm.spectralRows", [&](size_t row) {
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
    hpx::run_as_os_thread(
        hpx::annotated_function([]() { fftwf_execute(state.inverse); }, "gravity.pm.inverseFft"))
        .get();
    addCpuFlops(state.inverseFlops, FlopPrecision::Single);
    ++state.transformCount;
    ++state.diagnostics.inverseTransforms;
}

// The discrete response of the compensated mesh to one unit source cloud. The
// potential interpolated at a particle contains this self term; it is removed
// so the reported potential approximates the pairwise long-range sum.
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
    hpx::run_as_os_thread(
        hpx::annotated_function(
            [&]() {
                mpiCheck(MPI_Allreduce(local.data(), state.selfGreen.data(), int(local.size()),
                             MPI_DOUBLE, MPI_SUM, state.communicator),
                    "self-potential Green table");
            },
            "gravity.pm.selfGreen"))
        .get();
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
        parallelFor(size_t(state.localSize) * n, "gravity.pm.fillDensity", [&](size_t row) {
            size_t const paddedRow = row * 2 * (n / 2 + 1);
            for (int z = 0; z < n; ++z) {
                state.real[paddedRow + z] = state.density[row * n + z] * meshVolume;
            }
            for (int z = n; z < 2 * (n / 2 + 1); ++z) {
                state.real[paddedRow + z] = 0;
            }
        });
    });
    hpx::run_as_os_thread(
        hpx::annotated_function([]() { fftwf_execute(state.forward); }, "gravity.pm.forwardFft"))
        .get();
    addCpuFlops(state.forwardFlops, FlopPrecision::Single);
    ++state.transformCount;
    ++state.diagnostics.forwardTransforms;
    addCpuFlops(FlopCount(state.compactSize()) * 2, FlopPrecision::Double);
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
                int const coordinates[kDimensions] = {x, y, z};
                double const wave = state.derivativeWaves[coordinates[component]];
                state.workSpectrum[index][0] = wave * imaginary;
                state.workSpectrum[index][1] = -wave * real;
            }
        });
        // Green factor (window products, norm, Gaussian), scaling and the derivative multiply.
        addCpuFlops(FlopCount(state.localSize) * n * (n / 2 + 1) * (12 + (component == 3 ? 2 : 4)),
            FlopPrecision::Double);
        inverseTransform();
        prepareCollectively([&]() {
            parallelFor(size_t(state.localSize) * n, "gravity.pm.readFields", [&](size_t row) {
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

double selfPotential(std::array<CloudWeights, kDimensions> const& cloud) {
    double correlations[kDimensions][4]{};
    for (int axis = 0; axis < kDimensions; ++axis) {
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

LocalSolveReturn interpolate(bool doPotential) {
    auto const request = state.request;
    double const gm = getOptions().gravity.gm;
    double const effectiveRadius = pmEffectiveRadius();
    bool const writeParticles =
        request.applyToParticles && particlesHasLongAcceleration(getOptions());
    std::vector<LocalSolveReturn> partial(state.tiles.size());
    parallelFor(state.tiles.size(), "gravity.pm.interpolate", [&](size_t tileIndex) {
        auto const& tile = state.tiles[tileIndex];
        std::vector<std::array<float, 4>> fields(tile.size());
        struct Request {
            std::vector<uint64_t> keys;
            std::vector<size_t> indices;
            hpx::future<std::vector<std::array<float, 4>>> future;
        };
        std::unordered_map<int, Request> requests;
        visitTile(tile, [&](int owner, uint64_t key, size_t index) {
            requests[owner].keys.push_back(key);
            requests[owner].indices.push_back(index);
        });
        std::exception_ptr failure;
        try {
            for (auto& [owner, fieldRequest] : requests) {
                fieldRequest.future = state.owners[owner] == hpx::find_here() ?
                    hpx::make_ready_future(readFields(fieldRequest.keys)) :
                    hpx::async<PmReadFieldsAction>(state.owners[owner], fieldRequest.keys);
            }
        } catch (...) {
            failure = std::current_exception();
        }
        for (auto& [owner, fieldRequest] : requests) {
            if (fieldRequest.future.valid()) {
                fieldRequest.future.wait();
            }
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        for (auto& [owner, fieldRequest] : requests) {
            auto const values = fieldRequest.future.get();
            for (size_t i = 0; i < fieldRequest.indices.size(); ++i) {
                fields[fieldRequest.indices[i]] = values[i];
            }
        }
        LocalSolveReturn local;
        FlopCount flops = 0;
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
                flops += 3 * (state.order + 2) +
                    FlopCount(state.order) * state.order * state.order * (doPotential ? 9 : 7);
                Vector3<float> acceleration;
                double accelerationSquared = 0.0;
                for (int axis = 0; axis < kDimensions; ++axis) {
                    acceleration[axis] = float(value[axis]);
                    accelerationSquared += value[axis] * value[axis];
                }
                if (particlesGravityDiagnostics != nullptr) {
                    auto& output = particlesGravityDiagnostics[particle];
                    output.longAcceleration = output.acceleration = acceleration;
                }
                if (writeParticles) {
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        particlesLongAcceleration(axis, particle) = acceleration[axis];
                    }
                }
                if (!state.acceleration.empty()) {
                    state.acceleration[particle] = acceleration;
                }
                double const mass = particlesMass(particle);
                if (doPotential) {
                    // The same discrete compensated Green function is used for
                    // the source and sink cloud, including subcell dependence.
                    double const potential = value[3] - gm * mass * selfPotential(cloud);
                    if (particlesGravityDiagnostics != nullptr) {
                        auto& output = particlesGravityDiagnostics[particle];
                        output.longPotential = output.potential = float(potential);
                    }
                    local.potentialEnergy += 0.5 * mass * potential;
                    if (!state.potential.empty()) {
                        state.potential[particle] = float(potential);
                    }
                    flops += 4 + 3 * state.order * (state.order + 1) +
                        FlopCount(state.order) * state.order * state.order * 4;
                }
                if (request.eta > 0.0) {
                    double const timestep = std::min(request.maximumTimestep,
                        request.eta *
                            std::sqrt(request.scale * effectiveRadius /
                                std::max(std::sqrt(accelerationSquared), 1.0e-30)));
                    if (!(timestep > 0.0) || !std::isfinite(timestep)) {
                        throw std::runtime_error("Particle-mesh timestep is not finite/positive");
                    }
                    local.minimumTimestep = std::min(local.minimumTimestep, timestep);
                    flops += 8;
                }
            }
        }
        addCpuFlops(flops, FlopPrecision::Double);
        partial[tileIndex] = local;
    });
    LocalSolveReturn result = state.localReturn;
    for (auto const& local : partial) {
        result.potentialEnergy += local.potentialEnergy;
        result.minimumTimestep = std::min(result.minimumTimestep, local.minimumTimestep);
    }
    ++state.diagnostics.solveCount;
    state.localReturn = result;
    return result;
}

PmDiagnostics diagnostics() {
    return state.diagnostics;
}

void destroy() {
    if (state.communicator == MPI_COMM_NULL) {
        return;
    }
    hpx::run_as_os_thread(
        hpx::annotated_function(
            []() {
                // Destruction of MPI plans is collective, so
                // only destroy plans present everywhere.
                int const present[2] = {state.forward != nullptr, state.inverse != nullptr};
                int allPresent[2]{};
                mpiCheck(
                    MPI_Allreduce(present, allPresent, 2, MPI_INT, MPI_MIN, state.communicator),
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
                    // No fftwf_cleanup: other users share the planner.
                }
                mpiCheck(MPI_Comm_free(&state.communicator), "communicator destruction");
            },
            "gravity.pm.destroyFft"))
        .get();
    auto const diagnostics = state.diagnostics;
    auto const transforms = state.transformCount;
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
    state.meshSize = 0;
    state.diagnostics = {};
    state.diagnostics.planBuilds = diagnostics.planBuilds;
    state.diagnostics.solveCount = diagnostics.solveCount;
    state.transformCount = transforms;
}

}    // namespace pm::detail

HPX_PLAIN_ACTION(pm::detail::prepare, PmPrepareAction);
HPX_PLAIN_ACTION(pm::detail::deposit, PmDepositAction);
HPX_PLAIN_ACTION(pm::detail::transform, PmTransformAction);
HPX_PLAIN_ACTION(pm::detail::interpolate, PmInterpolateAction);
HPX_PLAIN_ACTION(pm::detail::diagnostics, PmDiagnosticsAction);
HPX_PLAIN_ACTION(pm::detail::destroy, PmDestroyAction);

namespace pm::detail {

template <class Action, class... Arguments>
auto allLocalities(Arguments... arguments) {
    using Result = decltype(hpx::async<Action>(hpx::find_here(), arguments...).get());
    if constexpr (std::is_same_v<Action, PmPrepareAction> ||
        std::is_same_v<Action, PmTransformAction> || std::is_same_v<Action, PmDestroyAction>) {
        static_assert(std::is_void_v<Result>);
        dispatchCollectively(hpxLocalities().size(),
            [&](size_t index) { return hpx::async<Action>(hpxLocalities()[index], arguments...); });
        return;
    } else {
        std::vector<hpx::future<Result>> futures;
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
        if constexpr (std::is_void_v<Result>) {
            for (auto& future : futures) {
                future.get();
            }
            return;
        } else {
            std::vector<Result> results;
            results.reserve(futures.size());
            for (auto& future : futures) {
                results.push_back(future.get());
            }
            return results;
        }
    }
}

}    // namespace pm::detail

PmSolveResult pmSolve(PmSolveRequest const& request) {
    using namespace pm::detail;
    std::unique_lock<MutexType> lock(solveMutex);
    bool const coordinator = hpxRank() == 0;
#if COSMICTIGER_WITH_SPH
    if (getOptions().hydro.enabled) {
        sphParticlesClearCaches();
    }
#endif
    {
        profile::ScopedPhase phase("pmSolve.prepare", coordinator);
        allLocalities<PmPrepareAction>(request);
    }
    {
        profile::ScopedPhase phase("pmSolve.deposit", coordinator);
        allLocalities<PmDepositAction>();
    }
    {
        profile::ScopedPhase phase("pmSolve.transform", coordinator);
        allLocalities<PmTransformAction>(request.doPotential);
    }
    PmSolveResult result;
    {
        profile::ScopedPhase phase("pmSolve.interpolate", coordinator);
        auto const locals = allLocalities<PmInterpolateAction>(request.doPotential);
        for (auto const& local : locals) {
            result.potentialEnergy += local.potentialEnergy;
            result.minimumTimestep = std::min(result.minimumTimestep, local.minimumTimestep);
            result.sourceMass += local.sourceMass;
            result.sourceParticles += local.sourceParticles;
        }
    }
    {
        profile::ScopedPhase phase("pmSolve.diagnostics", coordinator);
        auto const values = allLocalities<PmDiagnosticsAction>();
        globalDiagnostics = {};
        for (auto const& value : values) {
            globalDiagnostics.meshSize = value.meshSize;
            globalDiagnostics.cloudOrder = value.cloudOrder;
            globalDiagnostics.splitRadius = value.splitRadius;
            globalDiagnostics.sourceMass += value.sourceMass;
            globalDiagnostics.depositedMass += value.depositedMass;
            globalDiagnostics.sourceParticles += value.sourceParticles;
            globalDiagnostics.planBuilds = std::max(globalDiagnostics.planBuilds, value.planBuilds);
            globalDiagnostics.solveCount = std::max(globalDiagnostics.solveCount, value.solveCount);
            globalDiagnostics.forwardTransforms =
                std::max(globalDiagnostics.forwardTransforms, value.forwardTransforms);
            globalDiagnostics.inverseTransforms =
                std::max(globalDiagnostics.inverseTransforms, value.inverseTransforms);
        }
        result.depositedMass = globalDiagnostics.depositedMass;
    }
    return result;
}

void pmDestroy() {
    using namespace pm::detail;
    std::unique_lock<MutexType> lock(solveMutex);
    allLocalities<PmDestroyAction>();
}

PmDiagnostics pmDiagnostics() {
    return pm::detail::globalDiagnostics;
}

uint64_t pmTransformCount() {
    return pm::detail::state.transformCount;
}

void pmResetTransformCount() {
    pm::detail::state.transformCount = 0;
}

Vector3<float> const& pmAcceleration(ParticleInteger index) {
    return pm::detail::state.acceleration.at(index);
}

float pmPotential(ParticleInteger index) {
    return pm::detail::state.potential.at(index);
}

double pmEffectiveRadius() {
    return pm::longForceEffectiveRadius(getOptions().gravity);
}
