/* CosmicTiger - full Cartesian CPU/CUDA operator parity; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include <tests/cuda_test.hpp>
#include "cartesian_symmetric_test_helpers.hpp"

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace detail {

constexpr int kSymmetricSamples = 8;

void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <class Real, int order>
__global__ void evaluateSymmetricDevice(SymmetricResult<Real, order>* results) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kSymmetricSamples) {
        evaluateSymmetricOperators(Real(index), results[index]);
    }
}

template <class Real, int order>
void checkSymmetricDevice() {
    using Result = SymmetricResult<Real, order>;
    std::array<Result, kSymmetricSamples> results{};
    Result* deviceResults = nullptr;
    checkCudaStatus(cudaMalloc(&deviceResults, sizeof(results)));
    try {
        // One small block per order/precision bounds validation resource use.
        evaluateSymmetricDevice<Real, order><<<1, kSymmetricSamples>>>(deviceResults);
        checkCudaStatus(cudaGetLastError());
        checkCudaStatus(cudaDeviceSynchronize());
        checkCudaStatus(
            cudaMemcpy(results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
        constexpr double kTolerance = std::is_same_v<Real, float> ? 2.0e-5 : 3.0e-12;
        for (int sample = 0; sample < kSymmetricSamples; ++sample) {
            SCOPED_TRACE("order=" + std::to_string(order) + " sample=" + std::to_string(sample) +
                (std::is_same_v<Real, float> ? " float" : " double"));
            SymmetricResult<double, order> reference{};
            evaluateSymmetricOperators(double(sample), reference);
            compareSymmetricResults(
                results[sample], reference, [&](Real actual, double expected, double scale) {
                    EXPECT_NEAR(
                        double(actual), expected, kTolerance * std::max(scale, std::abs(expected)));
                });
            EXPECT_EQ(results[sample].noPotentialField.potential(), Real(0.053));

            // The quadratic Green function terminates exactly. This independent
            // oracle catches missing trace and dipole terms in device M2P.
            std::array<double, 3> const position{0.037 + sample * 0.001, -0.029, 0.021};
            std::array<double, 3> const separation{
                0.8 + sample * 0.011, -0.4, 0.9 - sample * 0.003};
            double const mass = 1.37 + sample * 0.007;
            double potential = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                double const displacement = separation[axis] - position[axis];
                potential += mass * displacement * displacement;
                if constexpr (order == 3) {
                    // Order three retains source ranks zero and one only.
                    potential -= mass * position[axis] * position[axis];
                }
                EXPECT_NEAR(double(results[sample].quadraticField[axis + 1]),
                    -2 * mass * displacement, kTolerance * 4);
            }
            EXPECT_NEAR(
                double(results[sample].quadraticField.potential()), potential, kTolerance * 4);
        }
    } catch (...) {
        cudaFree(deviceResults);
        throw;
    }
    checkCudaStatus(cudaFree(deviceResults));
}

}    // namespace detail

class CartesianSymmetricCuda : public detail::CudaTest,
                               public ::testing::WithParamInterface<int> {};

TEST_P(CartesianSymmetricCuda, AllOperatorsMatchDoubleHostAndQuadraticOracle) {
    fmmDispatch(GetParam(), [](auto order) {
        detail::checkSymmetricDevice<float, order.value>();
        detail::checkSymmetricDevice<double, order.value>();
    });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, CartesianSymmetricCuda,
    ::testing::Range(kFmmMinOrder, kFmmMaxOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
