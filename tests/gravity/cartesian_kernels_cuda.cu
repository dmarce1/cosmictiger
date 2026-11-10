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
#include <string>

#include <cuda_runtime.h>

#include <tests/cuda_test.hpp>
#include "cartesian_kernel_test_helpers.hpp"

namespace detail {

constexpr int kSampleCount = 32;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <int order>
__global__ void evaluate(
    KernelInput<float, order> const* inputs, KernelResult<float, order>* results) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kSampleCount) {
        evaluateKernels(inputs[index], results[index]);
    }
}

// One float launch per order: the device results of every generated
// kernel of that order must match the host evaluation of the same inputs.
template <int order>
double testDeviceParity() {
    using Input = KernelInput<float, order>;
    using Result = KernelResult<float, order>;
    std::array<Input, kSampleCount> inputs{};
    std::array<Result, kSampleCount> results{};
    for (int sample = 0; sample < kSampleCount; ++sample) {
        inputs[sample] = makeKernelInput<float, order>(sample);
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
        evaluate<order><<<1, kSampleCount>>>(deviceInputs, deviceResults);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
        double const tolerance = 2e-5;
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
                        "CUDA float Cartesian parity at order %d: sample %d, "
                        "component %d: %.17g != %.17g\n",
                        order, sample, i, actualValue, expectedValue);
                    throw std::runtime_error("CUDA Cartesian coefficient mismatch");
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
    double const worstFloat = testDeviceParity<order>();
    std::printf("Cartesian CUDA order %d (degree %d/%d) float kernel "
                "parity passed; worst "
                "normalized discrepancy: float %.1e.\n",
        order, KernelLayout<order>::kMultipoleDegree_, KernelLayout<order>::kLocalDegree_,
        worstFloat);
}

}    // namespace detail

class CartesianKernelsCuda : public detail::CudaTest, public ::testing::WithParamInterface<int> {};

TEST_P(CartesianKernelsCuda, FloatMatchesHost) {
    detail::forEachFmmOrder([&](auto order) {
        if (order.value == GetParam()) {
            detail::testOrder<order.value>();
        }
    });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, CartesianKernelsCuda,
    ::testing::Range(kFmmMinOrder, kFmmMaxOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
