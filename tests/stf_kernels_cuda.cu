/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
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

template <typename Real>
__global__ void evaluate(KernelInput<Real> const* inputs, KernelResult<Real>* results) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kSampleCount) {
        evaluateKernels(inputs[index], results[index]);
    }
}

template <typename Real>
void testDeviceParity() {
    std::array<KernelInput<Real>, kSampleCount> inputs{};
    std::array<KernelResult<Real>, kSampleCount> results{};
    for (int sample = 0; sample < kSampleCount; ++sample) {
        inputs[sample] = makeKernelInput<Real>(sample);
    }
    KernelInput<Real>* deviceInputs = nullptr;
    KernelResult<Real>* deviceResults = nullptr;
    check(cudaMalloc(&deviceInputs, sizeof(inputs)));
    try {
        check(cudaMalloc(&deviceResults, sizeof(results)));
        check(cudaMemcpy(deviceInputs, inputs.data(), sizeof(inputs), cudaMemcpyHostToDevice));
        check(cudaMemset(deviceResults, 0, sizeof(results)));
        // One block is sufficient to validate the generated device implementations.
        evaluate<<<1, kSampleCount>>>(deviceInputs, deviceResults);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
        double const tolerance = sizeof(Real) == sizeof(float) ? 2e-5 : 3e-12;
        for (int sample = 0; sample < kSampleCount; ++sample) {
            KernelResult<Real> expected{};
            evaluateKernels(inputs[sample], expected);
            for (int i = 0; i < kResultCount; ++i) {
                double const actualValue = results[sample][i];
                double const expectedValue = expected[i];
                if (!std::isfinite(actualValue) ||
                    std::abs(actualValue - expectedValue) >
                        tolerance * std::max(1., std::abs(expectedValue))) {
                    std::fprintf(stderr,
                        "CUDA %s STF parity: sample %d, component %d: %.17g != %.17g\n",
                        sizeof(Real) == sizeof(float) ? "float" : "double", sample, i, actualValue,
                        expectedValue);
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
        detail::testDeviceParity<float>();
        detail::testDeviceParity<double>();
        std::puts("STF CUDA float/double kernel parity checks passed.");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "STF CUDA test failed: %s\n", error.what());
        return 1;
    }
}
