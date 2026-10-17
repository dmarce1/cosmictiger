/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/fmm_acceptance.hpp>
#include <cosmictiger/range.hpp>

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

__device__ void cosmictigerCudaAssert(
    char const* condition, bool value, char const* file, int line) {
    if (!value) {
        printf("%s:%d: %s\n", file, line, condition);
        asm("trap;");
    }
}

namespace detail {

struct AcceptanceInput {
    float centerDistanceSquared;
    float sinkRadius;
    float sourceRadius;
    float sinkBoxDistanceSquared;
    float sourceBoxDistanceSquared;
    float sinkSupport;
    float sourceSupport;
    float theta;
    long long sinkCount;
    long long sourceCount;
    bool bothLeaves;
};

struct AcceptanceResult {
    FmmAcceptance<float> acceptance;
    FmmInteraction selected;
    float inverseTheta;
};

struct BoxInput {
    Fixed32Range box;
    Array<Fixed32, NDIM> point;
};

struct SelectionInput {
    bool cellCell;
    bool particleCell;
    bool cellParticle;
    bool bothLeaves;
    long long sinkCount;
    long long sourceCount;
    float sinkRadius;
    float sourceRadius;
};

void require(bool condition, std::string const& label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

CUDA_EXPORT AcceptanceResult evaluate(AcceptanceInput const& input) {
    float const inverseTheta = fmmInverseOpeningAngle(input.theta);
    auto const accepted = computeFmmAcceptance(input.centerDistanceSquared, input.sinkRadius,
        input.sourceRadius, input.sinkBoxDistanceSquared, input.sourceBoxDistanceSquared,
        fmmPairSupportRadius(0.0f, input.sinkSupport, input.sourceSupport), inverseTheta);
    auto const selected = selectFmmInteraction(accepted.cellCell != 0.0f,
        accepted.particleCell != 0.0f, accepted.cellParticle != 0.0f, input.bothLeaves,
        input.sinkCount, input.sourceCount, input.sinkRadius, input.sourceRadius);
    return {accepted, selected, inverseTheta};
}

__global__ void evaluateDevice(
    AcceptanceInput const* inputs, AcceptanceResult* results, int count) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        results[index] = evaluate(inputs[index]);
    }
}

__global__ void evaluateBoxes(BoxInput const* inputs, float* results, int count) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        results[index] = boxDistanceSquared(inputs[index].box, inputs[index].point);
    }
}

__global__ void evaluateSelections(
    SelectionInput const* inputs, FmmInteraction* results, int count) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        auto const& input = inputs[index];
        results[index] = selectFmmInteraction(input.cellCell, input.particleCell,
            input.cellParticle, input.bothLeaves, input.sinkCount, input.sourceCount,
            input.sinkRadius, input.sourceRadius);
    }
}

AcceptanceInput reversed(AcceptanceInput input) {
    std::swap(input.sinkRadius, input.sourceRadius);
    std::swap(input.sinkBoxDistanceSquared, input.sourceBoxDistanceSquared);
    std::swap(input.sinkSupport, input.sourceSupport);
    std::swap(input.sinkCount, input.sourceCount);
    return input;
}

FmmInteraction transposed(FmmInteraction interaction) {
    if (interaction == FmmInteraction::ParticleCell) {
        return FmmInteraction::CellParticle;
    }
    if (interaction == FmmInteraction::CellParticle) {
        return FmmInteraction::ParticleCell;
    }
    return interaction;
}

template <class Input, class Output, class Launch>
std::vector<Output> runDevice(std::vector<Input> const& inputs, Launch launch) {
    Input* deviceInputs = nullptr;
    Output* deviceResults = nullptr;
    std::vector<Output> results(inputs.size());
    try {
        checkCuda(cudaMalloc(&deviceInputs, sizeof(Input) * inputs.size()), "Allocate inputs");
        checkCuda(cudaMalloc(&deviceResults, sizeof(Output) * inputs.size()), "Allocate results");
        checkCuda(cudaMemcpy(deviceInputs, inputs.data(), sizeof(Input) * inputs.size(),
                      cudaMemcpyHostToDevice),
            "Upload inputs");
        launch(deviceInputs, deviceResults, static_cast<int>(inputs.size()));
        checkCuda(cudaGetLastError(), "Launch acceptance probe");
        checkCuda(cudaMemcpy(results.data(), deviceResults, sizeof(Output) * results.size(),
                      cudaMemcpyDeviceToHost),
            "Read results");
    } catch (...) {
        cudaFree(deviceResults);
        cudaFree(deviceInputs);
        throw;
    }
    checkCuda(cudaFree(deviceResults), "Free results");
    checkCuda(cudaFree(deviceInputs), "Free inputs");
    return results;
}

void runAcceptance() {
    std::vector<AcceptanceInput> inputs;
    auto const add = [&](AcceptanceInput const& input) {
        inputs.push_back(input);
        inputs.push_back(reversed(input));
    };
    // Far cells, overlapping supports, and each one-sided geometric opening.
    add({0.25f, .03125f, .0625f, 0.0f, 0.0f, .015625f, .03125f, .8f, 38, 80, true});
    add({.000625f, .001f, .01f, .000576f, .000225f, .02f, .01f, .8f, 38, 80, true});
    add({.015625f, .0625f, .015625f, .00390625f, 0.0f, .00390625f, .0078125f, .5f, 38, 80, true});
    // Exactly representable support boundary, then its adjacent float values.
    float const boundary = .015625f;
    for (float distanceSquared : {std::nextafter(boundary, 0.0f), boundary,
             std::nextafter(boundary, std::numeric_limits<float>::infinity())}) {
        add({distanceSquared, .03125f, .03125f, 0.0f, 0.0f, .0625f, .015625f, .8f, 38, 80, true});
    }
    // Boundary probes isolate center and box alternatives, including reciprocal
    // inequalities whose fused multiply-add must agree under CUDA fast math.
    for (float theta : {.4f, .8f, .9375f, 1.0f}) {
        for (float sinkRadius : {.001f, .03125f, .08f}) {
            for (float sourceRadius : {.003f, .0625f}) {
                for (float support : {0.0f, .0001f, .02f, .125f}) {
                    float const inverseTheta = fmmInverseOpeningAngle(theta);
                    float const distances[] = {(sinkRadius + sourceRadius) * inverseTheta,
                        std::fma(inverseTheta, sourceRadius, sinkRadius),
                        std::fma(inverseTheta, sinkRadius, sourceRadius),
                        sinkRadius + sourceRadius + support};
                    for (float distance : distances) {
                        float const distanceSquared = distance * distance;
                        for (float probe : {std::nextafter(distanceSquared, 0.0f), distanceSquared,
                                 std::nextafter(
                                     distanceSquared, std::numeric_limits<float>::infinity())}) {
                            add({probe, sinkRadius, sourceRadius, 0.0f, 0.0f, support,
                                .5f * support, theta, 38, 39, true});
                        }
                    }
                    float const boxBoundary = sourceRadius + support;
                    float const boxBoundarySquared = boxBoundary * boxBoundary;
                    for (float probe :
                        {std::nextafter(boxBoundarySquared, 0.0f), boxBoundarySquared,
                            std::nextafter(
                                boxBoundarySquared, std::numeric_limits<float>::infinity())}) {
                        add({0.0f, sinkRadius, sourceRadius, probe, 0.0f, .5f * support, support,
                            theta, 38, 39, true});
                    }
                }
            }
        }
    }
    auto const results = runDevice<AcceptanceInput, AcceptanceResult>(inputs,
        [](AcceptanceInput const* deviceInputs, AcceptanceResult* deviceResults, int count) {
            evaluateDevice<<<(count + 31) / 32, 32>>>(deviceInputs, deviceResults, count);
        });
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto const expected = evaluate(inputs[i]);
        auto const& actual = results[i];
        auto const label = "Acceptance case " + std::to_string(i);
        require(actual.inverseTheta == expected.inverseTheta, label + " inverse theta");
        require(actual.acceptance.cellCell == expected.acceptance.cellCell &&
                actual.acceptance.particleCell == expected.acceptance.particleCell &&
                actual.acceptance.cellParticle == expected.acceptance.cellParticle &&
                actual.acceptance.supportSeparated == expected.acceptance.supportSeparated &&
                actual.selected == expected.selected,
            label + " host/device mismatch");
        if (i % 2 == 0) {
            auto const& reverse = results[i + 1];
            require(actual.acceptance.cellCell == reverse.acceptance.cellCell &&
                    actual.acceptance.particleCell == reverse.acceptance.cellParticle &&
                    actual.acceptance.cellParticle == reverse.acceptance.particleCell &&
                    actual.acceptance.supportSeparated == reverse.acceptance.supportSeparated &&
                    actual.selected == transposed(reverse.selected),
                label + " reciprocal mismatch");
        }
    }
    require(results[2].acceptance.supportSeparated == 0.0f,
        "Overlapping support must reject the former CUDA box shortcut");
    require(results[6].acceptance.supportSeparated == 0.0f &&
            results[8].acceptance.supportSeparated == 0.0f &&
            results[10].acceptance.supportSeparated == 1.0f,
        "Support separation must be strictly greater than the support boundary");
}

void runSelections() {
    std::vector<SelectionInput> inputs;
    long long const counts[] = {0, 37, 38, 39, 16777216, 16777217};
    float const radii[] = {.03125f, std::nextafter(.03125f, 0.0f),
        std::nextafter(.03125f, std::numeric_limits<float>::infinity()), .0625f};
    for (int masks = 0; masks < 16; ++masks) {
        for (long long sinkCount : counts) {
            for (long long sourceCount : counts) {
                for (float sinkRadius : radii) {
                    for (float sourceRadius : radii) {
                        inputs.push_back({bool(masks & 1), bool(masks & 2), bool(masks & 4),
                            bool(masks & 8), sinkCount, sourceCount, sinkRadius, sourceRadius});
                        inputs.push_back({bool(masks & 1), bool(masks & 4), bool(masks & 2),
                            bool(masks & 8), sourceCount, sinkCount, sourceRadius, sinkRadius});
                    }
                }
            }
        }
    }
    auto const results = runDevice<SelectionInput, FmmInteraction>(
        inputs, [](SelectionInput const* deviceInputs, FmmInteraction* deviceResults, int count) {
            evaluateSelections<<<(count + 31) / 32, 32>>>(deviceInputs, deviceResults, count);
        });
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto const& input = inputs[i];
        auto const expected = selectFmmInteraction(input.cellCell, input.particleCell,
            input.cellParticle, input.bothLeaves, input.sinkCount, input.sourceCount,
            input.sinkRadius, input.sourceRadius);
        auto const label = "Selection case " + std::to_string(i);
        require(results[i] == expected, label + " host/device mismatch");
        if (i % 2 == 0) {
            require(results[i] == transposed(results[i + 1]), label + " reciprocal mismatch");
        }
        bool const particleCell = input.particleCell && input.sinkCount >= 38;
        bool const cellParticle = input.cellParticle && input.sourceCount >= 38;
        if (input.cellCell) {
            require(results[i] == FmmInteraction::CellCell, label + " CC priority");
        } else if (!input.bothLeaves || (!particleCell && !cellParticle)) {
            require(results[i] == FmmInteraction::ParticleParticle, label + " PP fallback");
        } else if (particleCell && !cellParticle) {
            require(results[i] == FmmInteraction::ParticleCell,
                label + " sole eligible PC ignores count and radius preference");
        } else if (cellParticle && !particleCell) {
            require(results[i] == FmmInteraction::CellParticle,
                label + " sole eligible CP ignores count and radius preference");
        } else if (input.sinkCount != input.sourceCount) {
            require(results[i] ==
                    (input.sinkCount < input.sourceCount ? FmmInteraction::ParticleCell :
                                                           FmmInteraction::CellParticle),
                label + " explicit count ordering takes priority over radius");
        } else if (input.sinkRadius != input.sourceRadius) {
            require(results[i] ==
                    (input.sourceRadius < input.sinkRadius ? FmmInteraction::ParticleCell :
                                                             FmmInteraction::CellParticle),
                label + " equal counts prefer the smaller cell radius");
        } else {
            require(results[i] == FmmInteraction::ParticleParticle,
                label + " equal counts and radii retain PP fallback");
        }
    }
}

BoxInput makeBox(double low, double high, double x, double y, double z) {
    BoxInput input;
    double const coordinates[] = {x, y, z};
    for (int dim = 0; dim < NDIM; ++dim) {
        input.box.begin[dim] = low;
        input.box.end[dim] = high;
        input.point[dim] = coordinates[dim];
    }
    return input;
}

void runBoxes() {
    constexpr double kTick = Fixed32::kToFloatFactor_;
    std::vector<BoxInput> inputs;
    inputs.push_back(makeBox(.25, .375, .3125, .3125, .3125));
    inputs.push_back(makeBox(.25, .375, .375, .3125, .3125));
    inputs.push_back(makeBox(.25, .375, .375 + kTick, .3125, .3125));
    inputs.push_back(makeBox(.25, .375, .25 - kTick, .3125, .3125));
    inputs.push_back(makeBox(.25, .375, .375 + kTick, .375 + 3 * kTick, .25 - 5 * kTick));
    inputs.push_back(makeBox(.75, .875, .875 + kTick, .8125, .8125));
    inputs.push_back(makeBox(.125, .25, .75, .5, .375));
    inputs.push_back(makeBox(0.0, .125, .9375, .0625, .0625));
    inputs.push_back(makeBox(.875, 1.0, .0625, .9375, .9375));
    inputs.push_back(makeBox(0.0, .125, 1.0 - kTick, .0625, .0625));
    inputs.push_back(makeBox(-.0625, .125, .9375, .0625, .0625));
    inputs.push_back(makeBox(.875, 1.0625, .0625, .9375, .9375));
    auto const results = runDevice<BoxInput, float>(
        inputs, [](BoxInput const* deviceInputs, float* deviceResults, int count) {
            evaluateBoxes<<<(count + 31) / 32, 32>>>(deviceInputs, deviceResults, count);
        });
    for (size_t i = 0; i < inputs.size(); ++i) {
        float const expected = boxDistanceSquared(inputs[i].box, inputs[i].point);
        require(std::isfinite(results[i]) && results[i] == expected,
            "Box distance case " + std::to_string(i) + " host/device mismatch");
    }
    float const tickSquared = static_cast<float>(kTick * kTick);
    require(results[0] == 0.0f && results[1] == 0.0f, "Box interior and boundary distances");
    require(results[2] == tickSquared && results[3] == tickSquared && results[5] == tickSquared,
        "Full 32-bit point coordinates must not be truncated to box precision");
    require(results[4] == 35.0f * tickSquared, "Three-axis sub-box-tick distances");
    require(results[7] == .00390625f && results[8] == .00390625f,
        "Periodic box distance across either side of the domain");
}

}    // namespace detail

int main() {
    int deviceCount = 0;
    cudaError_t const status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && deviceCount == 0)) {
        std::puts("No CUDA device is available; FMM acceptance device checks skipped.");
        return 77;
    }
    try {
        detail::checkCuda(status, "Find CUDA device");
        detail::runAcceptance();
        detail::runSelections();
        detail::runBoxes();
        std::puts("FMM acceptance CUDA parity passed.");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
