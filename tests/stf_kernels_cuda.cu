/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#include "stf_kernel_test_helpers.hpp"

namespace detail {

constexpr int kSampleCount = 32;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename Real, int order>
__global__ void evaluate(
    KernelInput<Real, order> const* inputs, KernelResult<Real, order>* results) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kSampleCount) {
        evaluateKernels(inputs[index], results[index]);
    }
}

// One launch per order and precision: the device results of every generated
// kernel of that order must match the host evaluation of the same inputs.
template <typename Real, int order>
double testDeviceParity() {
    using Input = KernelInput<Real, order>;
    using Result = KernelResult<Real, order>;
    std::array<Input, kSampleCount> inputs{};
    std::array<Result, kSampleCount> results{};
    for (int sample = 0; sample < kSampleCount; ++sample) {
        inputs[sample] = makeKernelInput<Real, order>(sample);
    }
    Input* deviceInputs = nullptr;
    Result* deviceResults = nullptr;
    double worst = 0;
    check(cudaMalloc(&deviceInputs, sizeof(inputs)));
    try {
        check(cudaMalloc(&deviceResults, sizeof(results)));
        check(cudaMemcpy(deviceInputs, inputs.data(), sizeof(inputs), cudaMemcpyHostToDevice));
        check(cudaMemset(deviceResults, 0, sizeof(results)));
        // One block is sufficient to validate the generated device implementations.
        evaluate<Real, order><<<1, kSampleCount>>>(deviceInputs, deviceResults);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
        double const tolerance = sizeof(Real) == sizeof(float) ? 2e-5 : 3e-12;
        for (int sample = 0; sample < kSampleCount; ++sample) {
            Result expected{};
            evaluateKernels(inputs[sample], expected);
            for (int i = 0; i < KernelLayout<order>::kResultCount_; ++i) {
                double const actualValue = results[sample][i];
                double const expectedValue = expected[i];
                double const normalized =
                    std::abs(actualValue - expectedValue) / std::max(1., std::abs(expectedValue));
                worst = std::isfinite(actualValue) ? std::max(worst, normalized) : INFINITY;
                if (!std::isfinite(actualValue) || normalized > tolerance) {
                    std::fprintf(stderr,
                        "CUDA %s STF parity at order %d: sample %d, component %d: %.17g != %.17g\n",
                        sizeof(Real) == sizeof(float) ? "float" : "double", order, sample, i,
                        actualValue, expectedValue);
                    throw std::runtime_error("CUDA STF coefficient mismatch");
                }
            }
        }
    } catch (...) {
        cudaFree(deviceResults);
        cudaFree(deviceInputs);
        throw;
    }
    check(cudaFree(deviceResults));
    check(cudaFree(deviceInputs));
    return worst;
}

template <int order>
void testOrder() {
    double const worstFloat = testDeviceParity<float, order>();
    double const worstDouble = testDeviceParity<double, order>();
    std::printf("STF CUDA order %d (degree %d/%d) float/double kernel parity passed; worst "
                "normalized discrepancy: float %.1e, double %.1e.\n",
        order, KernelLayout<order>::kMultipoleDegree_, KernelLayout<order>::kLocalDegree_,
        worstFloat, worstDouble);
}

}    // namespace detail

int main() {
    int deviceCount = 0;
    auto const status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && deviceCount == 0)) {
        std::puts("No CUDA device is available; STF device checks skipped.");
        return 77;
    }
    try {
        detail::check(status);
        detail::forEachFmmOrder([](auto order) { detail::testOrder<order.value>(); });
        std::printf("STF CUDA float/double kernel parity checks passed for FMM orders %d to %d.\n",
            kFmmMinOrder, kFmmMaxOrder);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "STF CUDA test failed: %s\n", error.what());
        return 1;
    }
}
