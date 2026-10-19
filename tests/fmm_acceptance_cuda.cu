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

#include <cosmictiger/fmm/fmm_acceptance.hpp>
#include <cosmictiger/math/range.hpp>

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
    bool sinkLeaf;
    bool sourceLeaf;
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
    bool sinkLeaf;
    bool sourceLeaf;
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
        accepted.particleCell != 0.0f, accepted.cellParticle != 0.0f, input.sinkLeaf,
        input.sourceLeaf, input.sinkRadius, input.sourceRadius);
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
        results[index] =
            selectFmmInteraction(input.cellCell, input.particleCell, input.cellParticle,
                input.sinkLeaf, input.sourceLeaf, input.sinkRadius, input.sourceRadius);
    }
}

AcceptanceInput reversed(AcceptanceInput input) {
    std::swap(input.sinkRadius, input.sourceRadius);
    std::swap(input.sinkBoxDistanceSquared, input.sourceBoxDistanceSquared);
    std::swap(input.sinkSupport, input.sourceSupport);
    std::swap(input.sinkLeaf, input.sourceLeaf);
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
    add({0.25f, .03125f, .0625f, 0.0f, 0.0f, .015625f, .03125f, .8f, true, true});
    add({.000625f, .001f, .01f, .000576f, .000225f, .02f, .01f, .8f, true, true});
    add({.015625f, .0625f, .015625f, .00390625f, 0.0f, .00390625f, .0078125f, .5f, true, true});
    // Exactly representable support boundary, then its adjacent float values.
    float const boundary = .015625f;
    for (float distanceSquared : {std::nextafter(boundary, 0.0f), boundary,
             std::nextafter(boundary, std::numeric_limits<float>::infinity())}) {
        add({distanceSquared, .03125f, .03125f, 0.0f, 0.0f, .0625f, .015625f, .8f, true, true});
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
                                .5f * support, theta, true, true});
                        }
                    }
                    float const boxBoundary = sourceRadius + support;
                    float const boxBoundarySquared = boxBoundary * boxBoundary;
                    for (float probe :
                        {std::nextafter(boxBoundarySquared, 0.0f), boxBoundarySquared,
                            std::nextafter(
                                boxBoundarySquared, std::numeric_limits<float>::infinity())}) {
                        add({0.0f, sinkRadius, sourceRadius, probe, 0.0f, .5f * support, support,
                            theta, true, true});
                    }
                }
            }
        }
    }
    // Exercise accepted and rejected geometry at each tree-level orientation.
    // Reversing each input also exchanges the explicit leaf and internal cloud.
    for (bool sinkLeaf : {false, true}) {
        for (bool sourceLeaf : {false, true}) {
            add({0.25f, .03125f, .0625f, 0.0f, 0.0f, .015625f, .03125f, .8f, sinkLeaf, sourceLeaf});
            add({.015625f, .0625f, .015625f, .00390625f, 0.0f, .00390625f, .0078125f, .5f, sinkLeaf,
                sourceLeaf});
            add({1.69f, .5f, .5f, 1.21f, 1.21f, .1f, .1f, .5f, sinkLeaf, sourceLeaf});
            add({.000625f, .001f, .01f, .000576f, .000225f, .02f, .01f, .8f, sinkLeaf, sourceLeaf});
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
    struct SelectionCase {
        char const* label;
        SelectionInput input;
        FmmInteraction expected;
    };
    std::vector<SelectionCase> const cases = {
        {"CC takes priority over both hybrids", {true, true, true, true, true, .25f, .5f},
            FmmInteraction::CellCell},
        {"CC can join two internal cells", {true, false, false, false, false, .25f, .5f},
            FmmInteraction::CellCell},
        {"PC joins a sink leaf to an internal source", {false, true, false, true, false, .25f, .5f},
            FmmInteraction::ParticleCell},
        {"CP joins an internal sink to a source leaf", {false, false, true, false, true, .5f, .25f},
            FmmInteraction::CellParticle},
        {"PC requires an explicit sink leaf", {false, true, false, false, true, .25f, .5f},
            FmmInteraction::ParticleParticle},
        {"CP requires an explicit source leaf", {false, false, true, true, false, .5f, .25f},
            FmmInteraction::ParticleParticle},
        {"two internal cells must descend after CC fails",
            {false, true, true, false, false, .25f, .5f}, FmmInteraction::ParticleParticle},
        {"leaf eligibility takes priority over cloud radius",
            {false, true, true, true, false, .25f, .5f}, FmmInteraction::ParticleCell},
        {"the sole accepted geometry takes priority over cloud radius",
            {false, true, false, true, true, .25f, .5f}, FmmInteraction::ParticleCell},
        {"two eligible hybrids expand the smaller source cloud",
            {false, true, true, true, true, .5f, .25f}, FmmInteraction::ParticleCell},
        {"two eligible hybrids expand the smaller sink cloud",
            {false, true, true, true, true, .25f, .5f}, FmmInteraction::CellParticle},
        {"identical leaves retain reciprocal PP fallback",
            {false, true, true, true, true, .5f, .5f}, FmmInteraction::ParticleParticle},
        {"failed geometry retains PP fallback", {false, false, false, true, true, .25f, .5f},
            FmmInteraction::ParticleParticle},
    };
    std::vector<SelectionInput> inputs;
    auto const add = [&](SelectionInput const& input) {
        inputs.push_back(input);
        inputs.push_back({input.cellCell, input.cellParticle, input.particleCell, input.sourceLeaf,
            input.sinkLeaf, input.sourceRadius, input.sinkRadius});
    };
    for (auto const& value : cases) {
        add(value.input);
    }
    float const radii[] = {.03125f, std::nextafter(.03125f, 0.0f),
        std::nextafter(.03125f, std::numeric_limits<float>::infinity()), .0625f};
    for (int masks = 0; masks < 32; ++masks) {
        for (float sinkRadius : radii) {
            for (float sourceRadius : radii) {
                add({bool(masks & 1), bool(masks & 2), bool(masks & 4), bool(masks & 8),
                    bool(masks & 16), sinkRadius, sourceRadius});
            }
        }
    }
    auto const results = runDevice<SelectionInput, FmmInteraction>(
        inputs, [](SelectionInput const* deviceInputs, FmmInteraction* deviceResults, int count) {
            evaluateSelections<<<(count + 31) / 32, 32>>>(deviceInputs, deviceResults, count);
        });
    for (size_t i = 0; i < cases.size(); ++i) {
        require(results[2 * i] == cases[i].expected &&
                results[2 * i + 1] == transposed(cases[i].expected),
            cases[i].label);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto const& input = inputs[i];
        auto const expected =
            selectFmmInteraction(input.cellCell, input.particleCell, input.cellParticle,
                input.sinkLeaf, input.sourceLeaf, input.sinkRadius, input.sourceRadius);
        auto const label = "Selection case " + std::to_string(i);
        require(results[i] == expected, label + " host/device mismatch");
        if (i % 2 == 0) {
            require(results[i] == transposed(results[i + 1]), label + " reciprocal mismatch");
        }
        if (input.cellCell) {
            require(results[i] == FmmInteraction::CellCell, label + " CC priority");
        }
        require(input.sinkLeaf || results[i] != FmmInteraction::ParticleCell,
            label + " PC must expose particles only from a leaf");
        require(input.sourceLeaf || results[i] != FmmInteraction::CellParticle,
            label + " CP must expose particles only from a leaf");
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
