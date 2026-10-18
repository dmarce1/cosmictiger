/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

// One expansion per 32-thread block, matching the production tree walk. Build
// three variants: the current public wrapper; COSMICTIGER_STF_L2L_LANE0; and
// COSMICTIGER_STF_BASELINE_HEADER="/path/to/legacy/fmm_kernels.hpp".
// Correctness checks always precede timing. --check runs only the checks.
// Legacy timing reports reference failures and continues for historical timing
// comparisons; --check and both STF variants always fail on reference errors.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifdef COSMICTIGER_STF_BASELINE_HEADER
#include COSMICTIGER_STF_BASELINE_HEADER
#else
#include <cosmictiger/fmm_kernels.hpp>
#endif

namespace detail {

#ifdef COSMICTIGER_STF_BASELINE_HEADER
constexpr char const* kImplementation = "legacy_warp";
#elif defined(COSMICTIGER_STF_L2L_LANE0)
constexpr char const* kImplementation = "stf_lane0";
#else
constexpr char const* kImplementation = "stf_warp";
#endif
constexpr int kLanes = 32;
constexpr int kValidationSamples = 16;
using Canonical = std::array<double, 50>;
using Position = std::array<double, 3>;

struct Fixture {
    Expansion<float> source;
    Array<float, NDIM> displacement;
};

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

int canonicalIndex(int x, int y, int z) {
    int const rank = x + y + z;
    return rank * rank + (z == 0 ? x : rank + 1 + x);
}

double component(Canonical const& source, int x, int y, int z) {
    if (z <= 1) {
        return source[canonicalIndex(x, y, z)];
    }
    double value = -component(source, x + 2, y, z - 2) - component(source, x, y + 2, z - 2);
    if (x == 0 && y == 0 && z == 2) {
        value += source[49];
    }
    return value;
}

Canonical canonical(Expansion<float> const& source) {
    Canonical result{};
    for (int rank = 0; rank <= 6; ++rank) {
        for (int z = 0; z <= 1 && z <= rank; ++z) {
            for (int x = 0; x <= rank - z; ++x) {
                int const y = rank - x - z;
                result[canonicalIndex(x, y, z)] = source(x, y, z);
            }
        }
    }
#ifdef COSMICTIGER_STF_BASELINE_HEADER
    result[49] = double(source(2, 0, 0)) + double(source(0, 2, 0)) + double(source(0, 0, 2));
#else
    result[49] = source[49];
#endif
    return result;
}

Expansion<float> stored(Canonical const& source) {
    Expansion<float> result{};
    for (int rank = 0; rank <= 6; ++rank) {
        for (int z = 0; z <= 1 && z <= rank; ++z) {
            for (int x = 0; x <= rank - z; ++x) {
                int const y = rank - x - z;
                result(x, y, z) = float(source[canonicalIndex(x, y, z)]);
            }
        }
    }
#ifdef COSMICTIGER_STF_BASELINE_HEADER
    result(0, 0, 2) =
        float(source[49] - source[canonicalIndex(2, 0, 0)] - source[canonicalIndex(0, 2, 0)]);
#else
    result[49] = float(source[49]);
#endif
    return result;
}

Fixture fixture(int sample) {
    Fixture result{};
    Canonical source{};
    int const kind = sample % 4;
    if (kind < 2) {
        for (int i = 0; i < 49; ++i) {
            source[i] = float(.025 * std::sin(.31 * (i + 1) * (sample + 2)));
        }
        source[49] = kind == 1 ? -.7 : 0.;
    } else if (kind == 3) {
        // A uniform-density root polynomial, including its nonzero gradient
        // when the expansion center is away from the density's reference point.
        source[49] = -.83;
        source[canonicalIndex(2, 0, 0)] = source[49] / 3;
        source[canonicalIndex(0, 2, 0)] = source[49] / 3;
    }
    source[0] = .75;
    source[canonicalIndex(1, 0, 0)] = .12;
    source[canonicalIndex(0, 1, 0)] = -.23;
    source[canonicalIndex(0, 0, 1)] = .31;
    result.source = stored(source);
    for (int d = 0; d < 3; ++d) {
        result.displacement[d] =
            sample % 7 == 0 ? 0.f : float(.00017 * std::sin(.71 * (sample + 1) + .37 * d));
    }
    return result;
}

Canonical translateReference(Canonical const& source, Position const& displacement, bool doPhi) {
    // Full Cartesian finite Taylor translation, independent of either generated
    // DAG and of the old CUDA term tables/reduction layout.
    double powers[3][7]{};
    for (int d = 0; d < 3; ++d) {
        powers[d][0] = 1;
        for (int n = 1; n <= 6; ++n) {
            powers[d][n] = powers[d][n - 1] * displacement[d] / n;
        }
    }
    Canonical result{};
    for (int rank = 0; rank <= 6; ++rank) {
        for (int z = 0; z <= 1 && z <= rank; ++z) {
            for (int x = 0; x <= rank - z; ++x) {
                int const y = rank - x - z;
                double value = 0;
                for (int bx = 0; bx <= 6 - rank; ++bx) {
                    for (int by = 0; by <= 6 - rank - bx; ++by) {
                        for (int bz = 0; bz <= 6 - rank - bx - by; ++bz) {
                            value += component(source, x + bx, y + by, z + bz) * powers[0][bx] *
                                powers[1][by] * powers[2][bz];
                        }
                    }
                }
                result[canonicalIndex(x, y, z)] = value;
            }
        }
    }
    result[49] = source[49];
    if (!doPhi) {
        result[0] = source[0];
    }
    return result;
}

__device__ Expansion<float> warpTranslate(
    Expansion<float> const& source, Array<float, NDIM> position, bool doPhi) {
#ifdef COSMICTIGER_STF_L2L_LANE0
    Expansion<float> result{};
    if (threadIdx.x == 0) {
        result = l2L(source, position, doPhi);
    }
    return result;
#else
    return l2LCuda(source, position, doPhi);
#endif
}

__global__ void validateWarp(Fixture const* input, Expansion<float>* workspace,
    Expansion<float>* returned, float* observed, bool doPhi, int steps) {
    int const sample = blockIdx.x * (blockDim.x / kLanes) + threadIdx.x / kLanes;
    int const lane = threadIdx.x & (kLanes - 1);
    if (sample >= kValidationSamples) {
        return;
    }
    for (int i = lane; i < EXPANSION_SIZE; i += kLanes) {
        workspace[sample][i] = input[sample].source[i];
    }
    __syncwarp();
    for (int step = 0; step < steps; ++step) {
        auto displacement = input[sample].displacement;
        for (int d = 0; d < NDIM; ++d) {
            displacement[d] *= step % 2 ? -1.f : 1.f;
        }
        auto const result = warpTranslate(workspace[sample], displacement, doPhi);
        if (step == steps - 1) {
            returned[sample * kLanes + lane] = result;
        }
        if (lane == 0) {
            workspace[sample] = result;
        }
        // The production call site consumes only lane zero's return, writes it
        // to the shared tree-walk expansion, then synchronizes before reuse.
        __syncwarp();
        float checksum = 0;
        float const volatile* coefficients = &workspace[sample][0];
        for (int i = 0; i < EXPANSION_SIZE; ++i) {
            checksum += (i + 1) * coefficients[i];
        }
        observed[sample * kLanes + lane] = checksum;
        __syncwarp();
    }
}

struct ComparisonError {
    std::string label;
    double relativeError = 0;
    double actual = 0;
    double expected = 0;
    int sample = 0;
    int lane = 0;
    int coefficient = 0;
    bool doPhi = false;
    int steps = 0;
};

std::vector<ComparisonError> comparisonErrors;
bool validationDoPhi;
int validationSteps;

void close(double actual, double expected, char const* label, int sample, int lane, int index) {
    auto found = std::find_if(comparisonErrors.begin(), comparisonErrors.end(),
        [=](auto const& record) { return record.label == label; });
    if (found == comparisonErrors.end()) {
        comparisonErrors.push_back(ComparisonError{label});
        found = comparisonErrors.end() - 1;
    }
    double const error = std::abs(actual - expected) / std::max(1., std::abs(expected));
    if (!std::isfinite(actual) || error > found->relativeError) {
        *found = {
            label, error, actual, expected, sample, lane, index, validationDoPhi, validationSteps};
    }
}

void validate(Fixture const* deviceInput, std::vector<Fixture> const& input,
    bool allowReferenceErrors, int warps = 1) {
    comparisonErrors.clear();
    Expansion<float>* workspace = nullptr;
    Expansion<float>* returned = nullptr;
    float* observed = nullptr;
    checkCuda(cudaMalloc(&workspace, kValidationSamples * sizeof(*workspace)));
    checkCuda(cudaMalloc(&returned, kValidationSamples * kLanes * sizeof(*returned)));
    checkCuda(cudaMalloc(&observed, kValidationSamples * kLanes * sizeof(*observed)));
    std::vector<Expansion<float>> output(kValidationSamples * kLanes);
    std::vector<float> checksums(kValidationSamples * kLanes);
    for (bool doPhi : {false, true}) {
        for (int steps : {1, 17}) {
            validationDoPhi = doPhi;
            validationSteps = steps;
            std::fprintf(stderr, "%s: checking doPhi=%d, steps=%d, warps per block=%d\n",
                kImplementation, int(doPhi), steps, warps);
            validateWarp<<<(kValidationSamples + warps - 1) / warps, kLanes * warps>>>(
                deviceInput, workspace, returned, observed, doPhi, steps);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaMemcpy(output.data(), returned, output.size() * sizeof(*returned),
                cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(checksums.data(), observed, checksums.size() * sizeof(*observed),
                cudaMemcpyDeviceToHost));
            for (int sample = 0; sample < kValidationSamples; ++sample) {
                auto expected = canonical(input[sample].source);
                auto scalar = input[sample].source;
                for (int step = 0; step < steps; ++step) {
                    auto shift = input[sample].displacement;
                    Position normalized{};
                    for (int d = 0; d < 3; ++d) {
                        shift[d] *= step % 2 ? -1.f : 1.f;
                        normalized[d] = double(shift[d] * SCALE_FACTOR);
                    }
                    expected = translateReference(expected, normalized, doPhi);
                    scalar = l2L(scalar, shift, doPhi);
                }
                auto const scalarCanonical = canonical(scalar);
                double checksum = 0;
                for (int i = 0; i < 50; ++i) {
                    close(scalarCanonical[i], expected[i], "host scalar/Cartesian parity", sample,
                        0, i);
                    // Visibility is checked against lane zero's actual return;
                    // the independent polynomial comparison above/below checks
                    // arithmetic accuracy without amplifying it by these weights.
                    checksum += (i + 1) * double(output[sample * kLanes][i]);
                }
                for (int lane = 0; lane < kLanes; ++lane) {
                    auto const actual = canonical(output[sample * kLanes + lane]);
                    for (int i = 0; i < 50; ++i) {
#ifdef COSMICTIGER_STF_L2L_LANE0
                        double const laneExpected = lane == 0 ? expected[i] : 0.;
#else
                        double const laneExpected = expected[i];
#endif
                        close(actual[i], laneExpected, "CUDA return/Cartesian parity", sample, lane,
                            i);
                        if (lane == 0) {
                            close(actual[i], scalarCanonical[i], "CUDA/own scalar parity", sample,
                                lane, i);
                        }
                    }
                    close(checksums[sample * kLanes + lane], checksum,
                        "post-sync expansion visibility", sample, lane, -1);
                }
            }
        }
    }
    checkCuda(cudaFree(observed));
    checkCuda(cudaFree(returned));
    checkCuda(cudaFree(workspace));
    bool passed = true;
    for (auto const& error : comparisonErrors) {
        std::fprintf(stderr,
            "%s: %s worst normalized error %.9g, doPhi=%d steps=%d sample=%d lane=%d "
            "coefficient=%d "
            "actual=%.17g expected=%.17g\n",
            kImplementation, error.label.c_str(), error.relativeError, int(error.doPhi),
            error.steps, error.sample, error.lane, error.coefficient, error.actual, error.expected);
        passed = passed && std::isfinite(error.actual) && error.relativeError <= 2e-5;
    }
    if (!passed) {
        if (allowReferenceErrors) {
            std::fprintf(stderr,
                "%s: REFERENCE CHECKS FAILED; retaining this legacy implementation "
                "only for historical timing comparison.\n",
                kImplementation);
            return;
        }
        throw std::runtime_error("CUDA L2L reference checks failed");
    }
    std::fprintf(stderr,
        "%s: Cartesian, own scalar, lane return, synchronized visibility, trace, "
        "linear field and doPhi checks passed.\n",
        kImplementation);
}

#ifndef COSMICTIGER_STF_L2L_VALIDATE_ONLY
__global__ void benchmarkWarp(Fixture const* input, Expansion<float>* workspace,
    Expansion<float>* output, int count, bool doPhi) {
    int const lane = threadIdx.x;
    for (int sample = blockIdx.x; sample < count; sample += gridDim.x) {
        for (int i = lane; i < EXPANSION_SIZE; i += kLanes) {
            workspace[blockIdx.x][i] = input[sample].source[i];
        }
        __syncwarp();
        auto const result = warpTranslate(workspace[blockIdx.x], input[sample].displacement, doPhi);
        if (lane == 0) {
            workspace[blockIdx.x] = result;
        }
        __syncwarp();
        for (int i = lane; i < EXPANSION_SIZE; i += kLanes) {
            output[sample][i] = workspace[blockIdx.x][i];
        }
        __syncwarp();
    }
}

void benchmark(
    Fixture const* input, int samples, int rounds, int repetitions, int blocks, bool doPhi) {
    Expansion<float>* workspace = nullptr;
    Expansion<float>* output = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    checkCuda(cudaMalloc(&workspace, blocks * sizeof(*workspace)));
    checkCuda(cudaMalloc(&output, samples * sizeof(*output)));
    checkCuda(cudaEventCreate(&begin));
    checkCuda(cudaEventCreate(&end));
    auto const launch = [&]() {
        benchmarkWarp<<<blocks, kLanes>>>(input, workspace, output, samples, doPhi);
        checkCuda(cudaGetLastError());
    };
    launch();
    launch();
    checkCuda(cudaDeviceSynchronize());
    std::vector<double> timings;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        checkCuda(cudaEventRecord(begin));
        for (int round = 0; round < rounds; ++round) {
            launch();
        }
        checkCuda(cudaEventRecord(end));
        checkCuda(cudaEventSynchronize(end));
        float elapsed;
        checkCuda(cudaEventElapsedTime(&elapsed, begin, end));
        timings.push_back(double(elapsed) * 1e-3);
    }
    std::vector<Expansion<float>> results(samples);
    checkCuda(
        cudaMemcpy(results.data(), output, samples * sizeof(*output), cudaMemcpyDeviceToHost));
    double checksum = 0;
    for (auto const& result : results) {
        auto const coefficients = canonical(result);
        for (int i = 0; i < 50; ++i) {
            checksum += (i + 1) * coefficients[i];
        }
    }
    std::sort(timings.begin(), timings.end());
    double const seconds = timings[timings.size() / 2];
    std::printf("%s,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.17g\n", kImplementation, int(doPhi), kLanes,
        blocks, samples, rounds, repetitions, seconds, seconds * 1e9 / (double(samples) * rounds),
        checksum);
    checkCuda(cudaEventDestroy(end));
    checkCuda(cudaEventDestroy(begin));
    checkCuda(cudaFree(output));
    checkCuda(cudaFree(workspace));
}

int positiveArgument(char const* argument, int maximum) {
    char* end = nullptr;
    long const value = std::strtol(argument, &end, 10);
    if (end == argument || *end != '\0' || value < 1 || value > maximum) {
        throw std::invalid_argument("Benchmark arguments must be bounded positive integers");
    }
    return int(value);
}
#endif

}    // namespace detail

int main(int argc, char** argv) {
    int deviceCount = 0;
    auto const deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus == cudaErrorNoDevice || deviceStatus == cudaErrorInsufficientDriver ||
        (deviceStatus == cudaSuccess && deviceCount == 0)) {
        std::puts("No CUDA device is available; cooperative L2L checks skipped.");
        return 77;
    }
    try {
        detail::checkCuda(deviceStatus);
        bool checkOnly = true;
        int samples = detail::kValidationSamples, rounds = 0, repetitions = 0;
#ifndef COSMICTIGER_STF_L2L_VALIDATE_ONLY
        checkOnly = argc == 2 && std::string(argv[1]) == "--check";
        if (argc > 4) {
            throw std::invalid_argument(
                "Usage: stf_l2l_cuda [--check | samples rounds repetitions]");
        }
        if (!checkOnly) {
            samples = argc > 1 ? detail::positiveArgument(argv[1], 1000000) : 32768;
            rounds = argc > 2 ? detail::positiveArgument(argv[2], 100000) : 20;
            repetitions = argc > 3 ? detail::positiveArgument(argv[3], 100) : 7;
        }
#endif
        std::vector<detail::Fixture> inputs(std::max(detail::kValidationSamples, samples));
        for (int sample = 0; sample < int(inputs.size()); ++sample) {
            inputs[sample] = detail::fixture(sample);
        }
        detail::Fixture* deviceInput = nullptr;
        detail::checkCuda(cudaMalloc(&deviceInput, inputs.size() * sizeof(*deviceInput)));
        detail::checkCuda(cudaMemcpy(deviceInput, inputs.data(),
            inputs.size() * sizeof(*deviceInput), cudaMemcpyHostToDevice));
#ifdef COSMICTIGER_STF_BASELINE_HEADER
        detail::validate(deviceInput, inputs, !checkOnly);
#else
        detail::validate(deviceInput, inputs, false);
#ifndef COSMICTIGER_STF_L2L_LANE0
        // Only the new implementation promises warp-relative lane indexing.
        // The timing launch remains exactly one warp per block, as in production.
        detail::validate(deviceInput, inputs, false, 3);
#endif
#endif
#ifndef COSMICTIGER_STF_L2L_VALIDATE_ONLY
        if (!checkOnly) {
            cudaDeviceProp properties{};
            detail::checkCuda(cudaGetDeviceProperties(&properties, 0));
            int const blocks = std::min(32, std::max(1, (properties.multiProcessorCount - 1) / 2));
            std::fprintf(stderr, "%s; %d one-warp blocks, one expansion per warp.\n",
                properties.name, blocks);
            std::puts(
                "implementation,do_phi,threads,blocks,samples,rounds,repetitions,median_seconds,"
                "ns_per_translation,checksum");
            detail::benchmark(deviceInput, samples, rounds, repetitions, blocks, false);
            detail::benchmark(deviceInput, samples, rounds, repetitions, blocks, true);
        }
#endif
        detail::checkCuda(cudaFree(deviceInput));
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "CUDA L2L check/benchmark failed: %s\n", error.what());
        return 1;
    }
}
