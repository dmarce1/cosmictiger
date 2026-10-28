/* CosmicTiger - Nitadori PMMM fft; BSL-1.0. */

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/gravity/nitadori/fft_flops.hpp>
#include "actions.hpp"

namespace pmmm::detail {

void mpiCheck(int status, char const* operation) {
    if (status != MPI_SUCCESS) {
        throw std::runtime_error(std::string("PMMM MPI failure: ") + operation);
    }
}

void collectiveCheck(bool okay, char const* message) {
    int local = okay ? 1 : 0;
    int global = 0;
    mpiCheck(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, cache.communicator),
        "failure agreement");
    if (!global) {
        throw std::runtime_error(message);
    }
}

void initialize() {
    auto const& options = getOptions().gravity;
    if (options.meshSize < 3 || options.pmmmNearWidth < 1 || options.pmmmNearWidth > 4 ||
        options.pmmmNearWidth > (options.meshSize - 1) / 2 || options.pmmmOrder < kFmmMinOrder ||
        options.pmmmOrder > kFmmMaxOrder || !(options.supportRadius > 0) ||
        options.supportRadius > double(options.pmmmNearWidth) / options.meshSize) {
        throw std::runtime_error("Invalid PMMM mesh, order, near width or softening support");
    }
    if (cache.communicator != MPI_COMM_NULL) {
        if (cache.meshSize != options.meshSize || cache.order != options.pmmmOrder ||
            cache.nearWidth != options.pmmmNearWidth) {
            throw std::runtime_error("PMMM configuration changed without destroying its plans");
        }
        return;
    }
    cache.meshSize = options.meshSize;
    cache.order = gravity::kBuildFmmOrder;
    cache.nearWidth = options.pmmmNearWidth;
    cache.sourceCount = fmmMultipoleCoefficientCount(cache.order);
    cache.localCount = fmmLocalHarmonicCoefficientCount(cache.order);
    int const n = cache.meshSize;
    uint64_t const limit = std::min(uint64_t(std::numeric_limits<ptrdiff_t>::max()),
                               uint64_t(std::numeric_limits<size_t>::max() / sizeof(double))) /
        cache.localCount / 2;
    if (uint64_t(n) > limit / n / n) {
        throw std::runtime_error("PMMM coefficient mesh exceeds indexing/allocation limits");
    }
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
                    MPI_Comm_dup(MPI_COMM_WORLD, &cache.communicator), "communicator duplication");
                mpiCheck(MPI_Comm_set_errhandler(cache.communicator, MPI_ERRORS_RETURN),
                    "error handler");
                int ranks = 0;
                mpiCheck(MPI_Comm_size(cache.communicator, &ranks), "communicator size");
                collectiveCheck(
                    ranks == localityCount, "PMMM requires one HPX locality per MPI process");
                bool allocated = true;
                try {
                    localityIds.resize(ranks);
                    slabs.resize(2 * ranks);
                } catch (...) {
                    allocated = false;
                }
                collectiveCheck(allocated, "Unable to allocate PMMM rank mapping");
                mpiCheck(MPI_Allgather(&locality, 1, MPI_INT, localityIds.data(), 1, MPI_INT,
                             cache.communicator),
                    "locality mapping");
                ptrdiff_t const dimensions[3] = {n, n, n / 2 + 1};
                ptrdiff_t otherSize = 0, otherBegin = 0, otherSpectralSize = 0,
                          otherSpectralBegin = 0;
                {
                    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
                    fftwf_mpi_init();
                    cache.sourceAllocation = fftwf_mpi_local_size_many_transposed(3, dimensions,
                        cache.sourceCount, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                        cache.communicator, &cache.localSize, &cache.localBegin,
                        &cache.spectralSize, &cache.spectralBegin);
                    cache.localAllocation =
                        fftwf_mpi_local_size_many_transposed(3, dimensions, cache.localCount,
                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, cache.communicator,
                            &otherSize, &otherBegin, &otherSpectralSize, &otherSpectralBegin);
                }
                collectiveCheck(otherSize == cache.localSize && otherBegin == cache.localBegin &&
                        otherSpectralSize == cache.spectralSize &&
                        otherSpectralBegin == cache.spectralBegin,
                    "PMMM coefficient batches have incompatible FFTW slab layouts");
                auto const validAllocation = [](ptrdiff_t count) {
                    return count >= 0 &&
                        uint64_t(count) <= std::numeric_limits<size_t>::max() / (2 * sizeof(float));
                };
                collectiveCheck(validAllocation(cache.sourceAllocation) &&
                        validAllocation(cache.localAllocation),
                    "PMMM FFT allocation exceeds addressable memory");
                double const localBytes =
                    sizeof(float) * (4.0 * cache.sourceAllocation + 6.0 * cache.localAllocation) +
                    sizeof(double) * cache.cellCount() * cache.sourceCount;
                double totalBytes = 0;
                mpiCheck(MPI_Allreduce(
                             &localBytes, &totalBytes, 1, MPI_DOUBLE, MPI_SUM, cache.communicator),
                    "workspace size estimate");
                if (locality == 0) {
                    terminal::detailPrint(
                        "PMMM persistent FFT/moment workspace: %.3f GiB across %i localities\n",
                        totalBytes / (1024.0 * 1024 * 1024), ranks);
                }
                {
                    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
                    size_t const sourceSize = std::max(ptrdiff_t(1), cache.sourceAllocation);
                    size_t const localSize = std::max(ptrdiff_t(1), cache.localAllocation);
                    cache.sourceReal = fftwf_alloc_real(2 * sourceSize);
                    cache.localReal = fftwf_alloc_real(2 * localSize);
                    cache.sourceSpectrum = fftwf_alloc_complex(sourceSize);
                    cache.localSpectrum = fftwf_alloc_complex(localSize);
                    cache.greenSpectrum = fftwf_alloc_complex(localSize);
                }
                collectiveCheck(cache.sourceReal && cache.localReal && cache.sourceSpectrum &&
                        cache.localSpectrum && cache.greenSpectrum,
                    "Unable to allocate PMMM FFT buffers; reduce the mesh or expansion order");
                ptrdiff_t const realDimensions[3] = {n, n, n};
                {
                    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
                    fftwf_plan_with_nthreads(fftThreads);
                    cache.forward = fftwf_mpi_plan_many_dft_r2c(3, realDimensions,
                        cache.sourceCount, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                        cache.sourceReal, cache.sourceSpectrum, cache.communicator,
                        FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
                    cache.greenForward = fftwf_mpi_plan_many_dft_r2c(3, realDimensions,
                        cache.localCount, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                        cache.localReal, cache.greenSpectrum, cache.communicator,
                        FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
                    cache.inverse = fftwf_mpi_plan_many_dft_c2r(3, realDimensions, cache.localCount,
                        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, cache.localSpectrum,
                        cache.localReal, cache.communicator,
                        FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
                    // Plans retain their thread limit. Cosmology already parallelizes
                    // independent 1D transforms with HPX, so keep its future plans serial.
                    fftwf_plan_with_nthreads(1);
                }
                collectiveCheck(cache.forward && cache.greenForward && cache.inverse,
                    "Unable to create the cached PMMM FFT plans");
                bool counted = true;
                try {
                    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
                    cache.forwardFlops = fftPlanFlops(cache.forward);
                    cache.greenForwardFlops = fftPlanFlops(cache.greenForward);
                    cache.inverseFlops = fftPlanFlops(cache.inverse);
                    // FFTW's MPI cost hook already sums all ranks. Every rank must
                    // query, but the distributed application counter must submit once.
                    if (hpxRank() != 0) {
                        cache.forwardFlops = cache.greenForwardFlops = cache.inverseFlops = 0;
                    }
                } catch (...) {
                    counted = false;
                }
                collectiveCheck(counted, "PMMM FFT operation count exceeds the supported range");
                ++state.diagnostics.planBuilds;
                int64_t const slab[2] = {cache.localBegin, cache.localSize};
                mpiCheck(MPI_Allgather(slab, 2, MPI_INT64_T, slabs.data(), 2, MPI_INT64_T,
                             cache.communicator),
                    "slab mapping");
            },
            "gravity.pmmm.initializeFft"))
        .get();
    prepareCollectively([&]() {
        cache.owners.resize(localityCount);
        cache.slabOwner.assign(n, -1);
        for (int rank = 0; rank < localityCount; ++rank) {
            auto const owner = std::find_if(
                hpxLocalities().begin(), hpxLocalities().end(), [&](hpx::id_type const& id) {
                    return hpx::naming::get_locality_id_from_id(id) == unsigned(localityIds[rank]);
                });
            if (owner == hpxLocalities().end()) {
                throw std::runtime_error("Cannot map PMMM MPI rank to HPX locality");
            }
            cache.owners[rank] = *owner;
            for (int64_t x = slabs[2 * rank]; x < slabs[2 * rank] + slabs[2 * rank + 1]; ++x) {
                cache.slabOwner.at(x) = rank;
            }
        }
        if (std::find(cache.slabOwner.begin(), cache.slabOwner.end(), -1) !=
            cache.slabOwner.end()) {
            throw std::runtime_error("PMMM FFTW slabs do not cover the mesh");
        }
        cache.terms = contractionTerms(cache.order);
    });
}

void initializeGreen() {
    if (cache.greenReady) {
        return;
    }
    int const n = cache.meshSize;
    double const h = 1.0 / n;
    addCpuFlops(1);
    prepareCollectively([&]() {
        std::fill_n(cache.localReal, 2 * cache.localAllocation, 0.0f);
        parallelFor(size_t(cache.localSize) * n, "gravity.pmmm.buildGreen", [&](size_t row) {
            FlopCount flops = 0;
            int const x = signedIndex(int(row / n + cache.localBegin));
            int const y = signedIndex(int(row % n));
            std::vector<double> coefficients(cache.localCount);
            for (int z = 0; z < n; ++z) {
                int const dz = signedIndex(z);
                bool const near =
                    std::max({std::abs(x), std::abs(y), std::abs(dz)}) <= cache.nearWidth;
                flops +=
                    3 + green(cache.order, coefficients.data(), {x * h, y * h, dz * h}, h, near);
                size_t const begin = (row * (2 * (n / 2 + 1)) + z) * cache.localCount;
                for (int component = 0; component < cache.localCount; ++component) {
                    if (!std::isfinite(coefficients[component]) ||
                        std::abs(coefficients[component]) > std::numeric_limits<float>::max()) {
                        throw std::runtime_error("PMMM Green coefficient is not representable");
                    }
                    cache.localReal[begin + component] = coefficients[component];
                }
            }
            addCpuFlops(flops);
        });
    });
    hpx::run_as_os_thread(hpx::annotated_function([]() { fftwf_execute(cache.greenForward); },
                              "gravity.pmmm.transformGreen"))
        .get();
    addCpuFlops(cache.greenForwardFlops);
    cache.greenReady = true;
    ++state.diagnostics.greenBuilds;
}

void transform() {
    int const n = cache.meshSize;
    std::array<double, 5> background{};
    prepareCollectively([&]() {
        std::fill_n(cache.sourceReal, 2 * cache.sourceAllocation, 0.0f);
        for (size_t cell = 0; cell < cache.cellCount(); ++cell) {
            double const* moments = state.moments.data() + cell * cache.sourceCount;
            background[0] += moments[0];
            background[1] += moments[3];
            background[2] += moments[1];
            background[3] += moments[2];
            size_t const padded = ((cell / n) * (2 * (n / 2 + 1)) + cell % n) * cache.sourceCount;
            for (int component = 0; component < cache.sourceCount; ++component) {
                if (!std::isfinite(moments[component]) ||
                    std::abs(moments[component]) > std::numeric_limits<float>::max()) {
                    throw std::runtime_error("PMMM source coefficient is not representable");
                }
                cache.sourceReal[padded + component] = moments[component];
            }
        }
        background[4] = state.radialTrace;
        state.diagnostics.depositedMass = background[0];
        addCpuFlops(4 * FlopCount(cache.cellCount()));
    });
    hpx::run_as_os_thread(hpx::annotated_function(
                              [&]() {
                                  mpiCheck(MPI_Allreduce(background.data(), state.background.data(),
                                               5, MPI_DOUBLE, MPI_SUM, cache.communicator),
                                      "periodic background reduction");
                                  fftwf_execute(cache.forward);
                              },
                              "gravity.pmmm.transformSources"))
        .get();
    addCpuFlops(cache.forwardFlops);
    // Count the logical five-component sum once across all localities. MPI's
    // internal transport/redundant reduction implementation is not measured.
    if (hpxRank() == 0) {
        addCpuFlops(5 * FlopCount(hpxSize() - 1));
    }
    prepareCollectively([&]() {
        size_t const modes = size_t(cache.spectralSize) * n * (n / 2 + 1);
        parallelFor(modes, "gravity.pmmm.contractFourier", [&](size_t mode) {
            // Complexify the real spherical bilinear map. No Fourier conjugation.
            std::array<double, kFmmMaxLocalCoefficients> real{};
            std::array<double, kFmmMaxLocalCoefficients> imaginary{};
            for (auto const& term : cache.terms) {
                auto const& source = cache.sourceSpectrum[mode * cache.sourceCount + term.source];
                auto const& kernel = cache.greenSpectrum[mode * cache.localCount + term.green];
                accumulateFourierTerm(real[term.local], imaginary[term.local], term.weight,
                    double(source[0]), double(source[1]), double(kernel[0]), double(kernel[1]));
            }
            for (int component = 0; component < cache.localCount; ++component) {
                if (!std::isfinite(real[component]) || !std::isfinite(imaginary[component]) ||
                    std::max(std::abs(real[component]), std::abs(imaginary[component])) >
                        std::numeric_limits<float>::max()) {
                    throw std::runtime_error("PMMM Fourier contraction is not representable");
                }
                auto& value = cache.localSpectrum[mode * cache.localCount + component];
                value[0] = real[component];
                value[1] = imaginary[component];
            }
        });
        addCpuFlops(kFourierTermFlops * FlopCount(modes) * cache.terms.size());
    });
    hpx::run_as_os_thread(hpx::annotated_function([]() { fftwf_execute(cache.inverse); },
                              "gravity.pmmm.transformLocalExpansions"))
        .get();
    addCpuFlops(cache.inverseFlops);
    state.farPromise->set_value();
}

void destroy() {
#ifdef USE_CUDA
    // Every solve has drained its tasks before this quiescent teardown. Release
    // retained PP buffers even when only the standalone near-field API ran.
    clearPmmmNearCudaWorkspaces();
#endif
    if (cache.communicator == MPI_COMM_NULL) {
        return;
    }
    hpx::run_as_os_thread(
        hpx::annotated_function(
            []() {
                int const present[3] = {cache.forward != nullptr, cache.inverse != nullptr,
                    cache.greenForward != nullptr};
                int allPresent[3]{};
                mpiCheck(
                    MPI_Allreduce(present, allPresent, 3, MPI_INT, MPI_MIN, cache.communicator),
                    "plan destruction agreement");
                {
                    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
                    if (allPresent[0]) {
                        fftwf_destroy_plan(cache.forward);
                    }
                    if (allPresent[1]) {
                        fftwf_destroy_plan(cache.inverse);
                    }
                    if (allPresent[2]) {
                        fftwf_destroy_plan(cache.greenForward);
                    }
                    cache.forward = nullptr;
                    cache.inverse = nullptr;
                    cache.greenForward = nullptr;
                    fftwf_free(cache.sourceReal);
                    fftwf_free(cache.localReal);
                    fftwf_free(cache.sourceSpectrum);
                    fftwf_free(cache.localSpectrum);
                    fftwf_free(cache.greenSpectrum);
                    cache.sourceReal = nullptr;
                    cache.localReal = nullptr;
                    cache.sourceSpectrum = nullptr;
                    cache.localSpectrum = nullptr;
                    cache.greenSpectrum = nullptr;
                }
                mpiCheck(MPI_Comm_free(&cache.communicator), "communicator destruction");
            },
            "gravity.pmmm.destroyFft"))
        .get();
    cache.greenReady = false;
    cache.owners.clear();
    cache.slabOwner.clear();
    state.cells.clear();
    state.moments.clear();
    state.sources.clear();
    state.sourceCells.clear();
    state.sourceCellsByKey.clear();
    state.sourceOwners.clear();
    state.tasks.clear();
    cache.terms.clear();
    state.acceleration.clear();
    state.potential.clear();
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmTransformAction, PmmmTransformAction);
HPX_REGISTER_ACTION(PmmmDestroyAction, PmmmDestroyAction);
