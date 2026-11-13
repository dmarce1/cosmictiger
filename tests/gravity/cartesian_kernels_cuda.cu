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

template <int order>
double resultScale(KernelResult<double, order> const& reference, int component) {
    using Layout = KernelLayout<order>;
    struct CoefficientBlock {
        int offset;
        int degree;
        int count;
    };
    constexpr std::array<CoefficientBlock, 6> kBlocks{{
        {Layout::kP2mOffset_, Layout::kMultipoleDegree_, Layout::kMultipoleCount_},
        {Layout::kM2mOffset_, Layout::kMultipoleDegree_, Layout::kMultipoleCount_},
        {Layout::kP2lOffset_, Layout::kLocalDegree_, Layout::kLocalCount_},
        {Layout::kM2lOffset_, Layout::kLocalDegree_, Layout::kLocalCount_},
        {Layout::kEwaldM2lOffset_, Layout::kLocalDegree_, Layout::kLocalCount_},
        {Layout::kL2lOffset_, Layout::kLocalDegree_, Layout::kLocalCount_},
    }};
    for (auto const& block : kBlocks) {
        if (component >= block.offset && component < block.offset + block.count) {
            int const coefficient = component - block.offset;
            int const harmonicCount = (block.degree + 1) * (block.degree + 1);
            int const rank = coefficient == harmonicCount ? 2 : int(std::sqrt(double(coefficient)));
            double scale = 1;
            for (int i = rank * rank; i < (rank + 1) * (rank + 1); ++i) {
                scale = std::max(scale, std::abs(reference[block.offset + i]));
            }
            if (rank == 2 && block.count > harmonicCount) {
                scale = std::max(scale, std::abs(reference[block.offset + harmonicCount]));
            }
            return scale;
        }
    }
    int const offset = component < Layout::kL2pOffset_ ? Layout::kM2pOffset_ : Layout::kL2pOffset_;
    if (component == offset) {
        return std::max(1., std::abs(reference[component]));
    }
    double scale = 1;
    for (int i = 1; i < kFieldCount; ++i) {
        scale = std::max(scale, std::abs(reference[offset + i]));
    }
    return scale;
}

// One float launch per order: compare against host double arithmetic on the
// exact float inputs. Independent moment components can nearly cancel a local
// coefficient, so host float arithmetic is not a sufficiently accurate oracle.
// As in the independent CPU Cartesian oracle, errors are scaled by the largest
// coefficient of the same tensor rank, including the separately stored zz.
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
            KernelInput<double, order> preciseInput{};
            for (int i = 0; i < KernelLayout<order>::kMultipoleCount_; ++i) {
                preciseInput.multipole[i] = inputs[sample].multipole[i];
            }
            for (int i = 0; i < KernelLayout<order>::kLocalCount_; ++i) {
                preciseInput.local[i] = inputs[sample].local[i];
            }
            for (int i = 0; i < kGeometryCount; ++i) {
                preciseInput.geometry[i] = inputs[sample].geometry[i];
            }
            KernelResult<double, order> precise{};
            evaluateKernels(preciseInput, precise);
            for (int i = 0; i < KernelLayout<order>::kResultCount_; ++i) {
                double const actualValue = results[sample][i];
                double const expectedValue = precise[i];
                double const scale = resultScale<order>(precise, i);
                double const normalized = std::abs(actualValue - expectedValue) / scale;
                worst = std::isfinite(actualValue) ? std::max(worst, normalized) : INFINITY;
                if (!std::isfinite(actualValue) || normalized > tolerance) {
                    std::fprintf(stderr,
                        "CUDA float Cartesian parity at order %d: sample %d, "
                        "component %d: %.17g != %.17g; rank scale %.9g, normalized error %.3g\n",
                        order, sample, i, actualValue, expectedValue, scale, normalized);
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
    ::testing::Range(kFmmOrder, kFmmOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
