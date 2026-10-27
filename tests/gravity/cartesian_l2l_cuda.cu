/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

// Validate the production cooperative L2L entry points at every configured FMM order.
// Independent Taylor references cover all lanes, trace terms, potential modes,
// repeated translations, and one or several warps per block.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <tests/cuda_test.hpp>

namespace detail {

template <int order>
using LocalExpansion = cosmictiger::gravity::cartesian::Expansion<float, order>;
constexpr int kLanes = 32;
constexpr int kMaxWarpsPerBlock = 4;
constexpr int kValidationSamples = 16;
using Position = std::array<double, 3>;
template <int order>
using Canonical = std::array<double, FmmOrder<order>::kLocalCoefficients_>;

// Visit every configured order without the application's runtime dispatcher.
template <class F, int... offsets>
void forEachOrderImpl(F&& f, std::integer_sequence<int, offsets...>) {
    (f(std::integral_constant<int, kFmmMinOrder + offsets>{}), ...);
}

template <class F>
void forEachOrder(F&& f) {
    forEachOrderImpl(
        std::forward<F>(f), std::make_integer_sequence<int, kFmmMaxOrder - kFmmMinOrder + 1>{});
}

// The two production cooperative translation entry points.
enum class Variant { Returned, SharedTarget };

char const* variantName(Variant variant) {
    return variant == Variant::Returned ? "translate" : "translateInto";
}

template <int order>
struct Fixture {
    LocalExpansion<order> source;
    Array<float, kDimensions> displacement;
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

template <int order>
double component(Canonical<order> const& source, int x, int y, int z) {
    if (z <= 1) {
        return source[canonicalIndex(x, y, z)];
    }
    double value =
        -component<order>(source, x + 2, y, z - 2) - component<order>(source, x, y + 2, z - 2);
    if (x == 0 && y == 0 && z == 2) {
        value += source[FmmOrder<order>::kLocalTraceIndex_];
    }
    return value;
}

template <int order>
Canonical<order> canonical(LocalExpansion<order> const& source) {
    constexpr int kDegree = FmmOrder<order>::kLocalDegree_;
    constexpr int kTraceIndex = FmmOrder<order>::kLocalTraceIndex_;
    Canonical<order> result{};
    for (int rank = 0; rank <= kDegree; ++rank) {
        for (int z = 0; z <= 1 && z <= rank; ++z) {
            for (int x = 0; x <= rank - z; ++x) {
                int const y = rank - x - z;
                result[canonicalIndex(x, y, z)] = source(x, y, z);
            }
        }
    }
    result[kTraceIndex] = source[kTraceIndex];
    return result;
}

template <int order>
LocalExpansion<order> stored(Canonical<order> const& source) {
    constexpr int kDegree = FmmOrder<order>::kLocalDegree_;
    constexpr int kTraceIndex = FmmOrder<order>::kLocalTraceIndex_;
    LocalExpansion<order> result{};
    for (int rank = 0; rank <= kDegree; ++rank) {
        for (int z = 0; z <= 1 && z <= rank; ++z) {
            for (int x = 0; x <= rank - z; ++x) {
                int const y = rank - x - z;
                result(x, y, z) = float(source[canonicalIndex(x, y, z)]);
            }
        }
    }
    result[kTraceIndex] = float(source[kTraceIndex]);
    return result;
}

template <int order>
Fixture<order> fixture(int sample) {
    constexpr int kHarmonicCoefficients = FmmOrder<order>::kLocalHarmonicCoefficients_;
    constexpr int kTraceIndex = FmmOrder<order>::kLocalTraceIndex_;
    Fixture<order> result{};
    Canonical<order> source{};
    int const kind = sample % 4;
    if (kind < 2) {
        for (int i = 0; i < kHarmonicCoefficients; ++i) {
            source[i] = float(.025 * std::sin(.31 * (i + 1) * (sample + 2)));
        }
        source[kTraceIndex] = kind == 1 ? -.7 : 0.;
    } else if (kind == 3) {
        // A uniform-density root polynomial, including its nonzero gradient
        // when the expansion center is away from the density's reference point.
        source[kTraceIndex] = -.83;
        source[canonicalIndex(2, 0, 0)] = source[kTraceIndex] / 3;
        source[canonicalIndex(0, 2, 0)] = source[kTraceIndex] / 3;
    }
    source[0] = .75;
    source[canonicalIndex(1, 0, 0)] = .12;
    source[canonicalIndex(0, 1, 0)] = -.23;
    source[canonicalIndex(0, 0, 1)] = .31;
    result.source = stored<order>(source);
    for (int d = 0; d < 3; ++d) {
        result.displacement[d] =
            sample % 7 == 0 ? 0.f : float(.00017 * std::sin(.71 * (sample + 1) + .37 * d));
    }
    return result;
}

template <int order>
Canonical<order> translateReference(
    Canonical<order> const& source, Position const& displacement, bool doPhi) {
    constexpr int kDegree = FmmOrder<order>::kLocalDegree_;
    constexpr int kTraceIndex = FmmOrder<order>::kLocalTraceIndex_;
    // Full Cartesian finite Taylor translation, independent of the generated
    // arithmetic and cooperative CUDA schedule.
    double powers[3][kDegree + 1]{};
    for (int d = 0; d < 3; ++d) {
        powers[d][0] = 1;
        for (int n = 1; n <= kDegree; ++n) {
            powers[d][n] = powers[d][n - 1] * displacement[d] / n;
        }
    }
    Canonical<order> result{};
    for (int rank = 0; rank <= kDegree; ++rank) {
        for (int z = 0; z <= 1 && z <= rank; ++z) {
            for (int x = 0; x <= rank - z; ++x) {
                int const y = rank - x - z;
                double value = 0;
                for (int bx = 0; bx <= kDegree - rank; ++bx) {
                    for (int by = 0; by <= kDegree - rank - bx; ++by) {
                        for (int bz = 0; bz <= kDegree - rank - bx - by; ++bz) {
                            value += component<order>(source, x + bx, y + by, z + bz) *
                                powers[0][bx] * powers[1][by] * powers[2][bz];
                        }
                    }
                }
                result[canonicalIndex(x, y, z)] = value;
            }
        }
    }
    result[kTraceIndex] = source[kTraceIndex];
    if (!doPhi) {
        result[0] = source[0];
    }
    return result;
}

template <int order>
__device__ LocalExpansion<order> warpTranslate(
    LocalExpansion<order> const& source, Array<float, kDimensions> position, bool doPhi) {
    LocalExpansion<order> result{};
    auto const scaled = fmmScaleDisplacement(position);
    cosmictiger::gravity::cartesian::detail::detail::L2LWarp<order - 1, true>::translate(
        result, source, scaled[0], scaled[1], scaled[2], doPhi);
    return result;
}

template <int order>
__global__ void validateWarp(Fixture<order> const* input, LocalExpansion<order>* workspace,
    LocalExpansion<order>* returned, float* observed, bool doPhi, int steps) {
    constexpr int kCoefficients = FmmOrder<order>::kLocalCoefficients_;
    int const sample = blockIdx.x * (blockDim.x / kLanes) + threadIdx.x / kLanes;
    int const lane = threadIdx.x & (kLanes - 1);
    if (sample >= kValidationSamples) {
        return;
    }
    for (int i = lane; i < kCoefficients; i += kLanes) {
        workspace[sample][i] = input[sample].source[i];
    }
    __syncwarp();
    for (int step = 0; step < steps; ++step) {
        auto displacement = input[sample].displacement;
        for (int d = 0; d < kDimensions; ++d) {
            displacement[d] *= step % 2 ? -1.f : 1.f;
        }
        auto const result = warpTranslate<order>(workspace[sample], displacement, doPhi);
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
        for (int i = 0; i < kCoefficients; ++i) {
            checksum += (i + 1) * coefficients[i];
        }
        observed[sample * kLanes + lane] = checksum;
        __syncwarp();
    }
}

// Translate one shared expansion into another, as a tree walk does when it
// keeps its local expansion stack in shared memory. The target never aliases
// the source; the production wrapper ends with a warp barrier, so every lane
// may read the completed target immediately after the call.
template <int order>
__device__ void warpTranslateInto(LocalExpansion<order>& target,
    LocalExpansion<order> const& source, Array<float, kDimensions> const& position, bool doPhi) {
    cosmictiger::gravity::cartesian::L2LWarpInto(
        &target[0], source, fmmScaleDisplacement(position), doPhi);
}

template <int order>
__global__ void validateWarpInto(Fixture<order> const* input, LocalExpansion<order>* returned,
    float* observed, bool doPhi, int steps) {
    constexpr int kCoefficients = FmmOrder<order>::kLocalCoefficients_;
    __shared__ LocalExpansion<order> buffers[kMaxWarpsPerBlock][2];
    int const warp = threadIdx.x / kLanes;
    int const sample = blockIdx.x * (blockDim.x / kLanes) + warp;
    int const lane = threadIdx.x & (kLanes - 1);
    if (sample >= kValidationSamples) {
        return;
    }
    for (int i = lane; i < kCoefficients; i += kLanes) {
        buffers[warp][0][i] = input[sample].source[i];
    }
    __syncwarp();
    int current = 0;
    for (int step = 0; step < steps; ++step) {
        auto displacement = input[sample].displacement;
        for (int d = 0; d < kDimensions; ++d) {
            displacement[d] *= step % 2 ? -1.f : 1.f;
        }
        // Ping-pong: the previous source becomes the next target. Its last
        // reads completed inside the previous call's closing warp barrier.
        warpTranslateInto<order>(
            buffers[warp][1 - current], buffers[warp][current], displacement, doPhi);
        current = 1 - current;
        float checksum = 0;
        for (int i = 0; i < kCoefficients; ++i) {
            checksum += (i + 1) * buffers[warp][current][i];
        }
        observed[sample * kLanes + lane] = checksum;
        if (step == steps - 1) {
            for (int i = 0; i < kCoefficients; ++i) {
                returned[sample * kLanes + lane][i] = buffers[warp][current][i];
            }
        }
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

void close(
    double actual, double expected, std::string const& label, int sample, int lane, int index) {
    auto found = std::find_if(comparisonErrors.begin(), comparisonErrors.end(),
        [&](auto const& record) { return record.label == label; });
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

template <int order>
void validate(Fixture<order> const* deviceInput, std::vector<Fixture<order>> const& input,
    int warps, Variant variant) {
    using Local = LocalExpansion<order>;
    constexpr int kDegree = FmmOrder<order>::kLocalDegree_;
    constexpr int kCoefficients = FmmOrder<order>::kLocalCoefficients_;
    if (warps < 1 || warps > kMaxWarpsPerBlock) {
        throw std::invalid_argument("Unsupported number of warps per block");
    }
    comparisonErrors.clear();
    std::string const device = variant == Variant::Returned ? "CUDA return" : "CUDA shared target";
    Local* workspace = nullptr;
    Local* returned = nullptr;
    float* observed = nullptr;
    checkCuda(cudaMalloc(&workspace, kValidationSamples * sizeof(*workspace)));
    checkCuda(cudaMalloc(&returned, kValidationSamples * kLanes * sizeof(*returned)));
    checkCuda(cudaMalloc(&observed, kValidationSamples * kLanes * sizeof(*observed)));
    std::vector<Local> output(kValidationSamples * kLanes);
    std::vector<float> checksums(kValidationSamples * kLanes);
    for (bool doPhi : {false, true}) {
        for (int steps : {1, 17}) {
            validationDoPhi = doPhi;
            validationSteps = steps;
            std::fprintf(stderr,
                "Cartesian %s: checking local degree %d, doPhi=%d, steps=%d, warps per block=%d\n",
                variantName(variant), kDegree, int(doPhi), steps, warps);
            int const blocks = (kValidationSamples + warps - 1) / warps;
            if (variant == Variant::Returned) {
                validateWarp<order><<<blocks, kLanes * warps>>>(
                    deviceInput, workspace, returned, observed, doPhi, steps);
            } else {
                validateWarpInto<order>
                    <<<blocks, kLanes * warps>>>(deviceInput, returned, observed, doPhi, steps);
            }
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaMemcpy(output.data(), returned, output.size() * sizeof(*returned),
                cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(checksums.data(), observed, checksums.size() * sizeof(*observed),
                cudaMemcpyDeviceToHost));
            for (int sample = 0; sample < kValidationSamples; ++sample) {
                auto expected = canonical<order>(input[sample].source);
                auto scalar = input[sample].source;
                for (int step = 0; step < steps; ++step) {
                    auto shift = input[sample].displacement;
                    Position normalized{};
                    for (int d = 0; d < 3; ++d) {
                        shift[d] *= step % 2 ? -1.f : 1.f;
                        normalized[d] = double(shift[d] * kScaleFactor);
                    }
                    expected = translateReference<order>(expected, normalized, doPhi);
                    LocalExpansion<order> translated{};
                    if (!doPhi) {
                        translated[0] = scalar[0];
                    }
                    cosmictiger::gravity::cartesian::L2L(
                        translated, scalar, fmmScaleDisplacement(shift), doPhi);
                    scalar = translated;
                }
                auto const scalarCanonical = canonical<order>(scalar);
                double checksum = 0;
                for (int i = 0; i < kCoefficients; ++i) {
                    close(scalarCanonical[i], expected[i], "host scalar/Cartesian parity", sample,
                        0, i);
                    // Visibility is checked against lane zero's actual return;
                    // the independent polynomial comparison above/below checks
                    // arithmetic accuracy without amplifying it by these weights.
                    checksum += (i + 1) * double(output[sample * kLanes][i]);
                }
                for (int lane = 0; lane < kLanes; ++lane) {
                    auto const actual = canonical<order>(output[sample * kLanes + lane]);
                    for (int i = 0; i < kCoefficients; ++i) {
                        double const laneExpected = expected[i];
                        close(
                            actual[i], laneExpected, device + "/Cartesian parity", sample, lane, i);
                        if (lane == 0) {
                            close(actual[i], scalarCanonical[i], device + "/own scalar parity",
                                sample, lane, i);
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
            "Cartesian %s: local degree %d: %s worst normalized error %.9g, doPhi=%d steps=%d "
            "sample=%d "
            "lane=%d coefficient=%d actual=%.17g expected=%.17g\n",
            variantName(variant), kDegree, error.label.c_str(), error.relativeError,
            int(error.doPhi), error.steps, error.sample, error.lane, error.coefficient,
            error.actual, error.expected);
        passed = passed && std::isfinite(error.actual) && error.relativeError <= 2e-5;
    }
    if (!passed) {
        throw std::runtime_error("CUDA L2L reference checks failed");
    }
    std::fprintf(stderr,
        "Cartesian %s: Cartesian, own scalar, every lane, synchronized visibility, trace, "
        "linear field and doPhi checks passed at local degree %d.\n",
        variantName(variant), kDegree);
}

template <int order>
void runOrder() {
    std::vector<Fixture<order>> inputs(kValidationSamples);
    for (int sample = 0; sample < kValidationSamples; ++sample) {
        inputs[sample] = fixture<order>(sample);
    }
    Fixture<order>* deviceInput = nullptr;
    checkCuda(cudaMalloc(&deviceInput, inputs.size() * sizeof(*deviceInput)));
    try {
        checkCuda(cudaMemcpy(deviceInput, inputs.data(), inputs.size() * sizeof(*deviceInput),
            cudaMemcpyHostToDevice));
        for (int warps : {1, 3}) {
            validate<order>(deviceInput, inputs, warps, Variant::Returned);
            validate<order>(deviceInput, inputs, warps, Variant::SharedTarget);
        }
    } catch (...) {
        cudaFree(deviceInput);
        throw;
    }
    checkCuda(cudaFree(deviceInput));
}

}    // namespace detail

using CartesianL2lCuda = detail::CudaTest;

TEST_F(CartesianL2lCuda, CooperativeTranslationMatchesIndependentReference) {
    detail::forEachOrder([](auto order) { detail::runOrder<order.value>(); });
}
