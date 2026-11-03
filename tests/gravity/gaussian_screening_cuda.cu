/* CosmicTiger - CUDA Gaussian derivatives against independent quadrature; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include <tests/cuda_test.hpp>
#include "gaussian_screening_test_helpers.hpp"

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace detail {

constexpr int kGaussianSamples = 16;

template <class Real, int order>
struct GaussianDeviceResult {
    GaussianResult<Real, order> ordinary;
    cosmictiger::gravity::cartesian::Green<Real, order> nearOrigin;
    cosmictiger::gravity::Field<Real> shortField;
};

template <class Real, int order>
__global__ void evaluateGaussianDevice(
    GaussianInput<Real> const* inputs, GaussianDeviceResult<Real, order>* outputs) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kGaussianSamples) {
        using cosmictiger::gravity::GaussianComponent;
        evaluateGaussianFixture(inputs[index], outputs[index].ordinary);
        cosmictiger::gravity::GaussianScreening<Real> const screening(inputs[index].alpha);
        Real const small = index % 2 == 0 ? Real(0) : Real(1.0e-20);
        Vector3<Real> const position{small, Real(-0.5) * small, Real(0.25) * small};
        cosmictiger::gravity::cartesian::evaluateGreen<GaussianComponent::Long>(
            outputs[index].nearOrigin, position, screening);
        cosmictiger::gravity::cartesian::Multipole<Real, order> point{};
        point = Real(0);
        P2M(point, Real(1.75), Vector3<Real>{});
        for (int component = 0; component < 4; ++component) {
            outputs[index].shortField[component] = Real(0);
        }
        M2P(outputs[index].shortField, point, outputs[index].ordinary.shortRange);
    }
}

void requireGaussianCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <class Real, int order>
void expectGaussianDeviceOracle(cosmictiger::gravity::cartesian::Green<Real, order> const& actual,
    Vector3<Real> const& position, Real alpha, bool longRange) {
    auto const expected =
        gaussianDerivativeOracle<order>({position[0], position[1], position[2]}, alpha, longRange);
    constexpr double kTolerance = std::is_same_v<Real, float> ? 6.0e-5 : 4.0e-11;
    int index = 0;
    for (int rank = 0; rank < order; ++rank) {
        int const count = (rank + 1) * (rank + 2) / 2;
        long double scale = 0;
        for (int component = 0; component < count; ++component) {
            scale = std::max(scale, std::abs(expected[index + component]));
        }
        for (int component = 0; component < count; ++component) {
            EXPECT_NEAR(double(actual[index]), double(expected[index]),
                kTolerance * double(scale) +
                    gaussianDerivativeUnderflowUnits(rank) *
                        double(std::numeric_limits<Real>::denorm_min()))
                << "rank=" << rank << " coefficient=" << index << " long=" << longRange;
            ++index;
        }
    }
}

template <class Real, int order>
void checkGaussianDevice() {
    using Input = GaussianInput<Real>;
    using Result = GaussianDeviceResult<Real, order>;
    std::array<Input, kGaussianSamples> inputs{};
    std::array<Result, kGaussianSamples> results{};
    Input* deviceInputs = nullptr;
    Result* deviceResults = nullptr;
    requireGaussianCuda(cudaMalloc(&deviceInputs, sizeof(inputs)));
    try {
        requireGaussianCuda(cudaMalloc(&deviceResults, sizeof(results)));
        constexpr int kBatches = (kGaussianFixtureCount + kGaussianSamples - 1) / kGaussianSamples;
        for (int batch = 0; batch < kBatches; ++batch) {
            for (int sample = 0; sample < kGaussianSamples; ++sample) {
                inputs[sample] = gaussianFixture<Real>(batch * kGaussianSamples + sample);
            }
            requireGaussianCuda(
                cudaMemcpy(deviceInputs, inputs.data(), sizeof(inputs), cudaMemcpyHostToDevice));
            evaluateGaussianDevice<Real, order>
                <<<1, kGaussianSamples>>>(deviceInputs, deviceResults);
            requireGaussianCuda(cudaGetLastError());
            requireGaussianCuda(cudaDeviceSynchronize());
            requireGaussianCuda(
                cudaMemcpy(results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
            for (int sample = 0; sample < kGaussianSamples; ++sample) {
                SCOPED_TRACE("order=" + std::to_string(order) +
                    " sample=" + std::to_string(batch * kGaussianSamples + sample) +
                    (std::is_same_v<Real, float> ? " float" : " double"));
                auto const& input = inputs[sample];
                auto const& result = results[sample];
                expectGaussianDeviceOracle(
                    result.ordinary.longRange, input.position, input.alpha, true);
                expectGaussianDeviceOracle(
                    result.ordinary.shortRange, input.position, input.alpha, false);
                Real const small = sample % 2 == 0 ? Real(0) : Real(1.0e-20);
                expectGaussianDeviceOracle(result.nearOrigin,
                    Vector3<Real>{small, Real(-0.5) * small, Real(0.25) * small}, input.alpha,
                    true);
                auto const oracle = gaussianDerivativeOracle<order>(
                    {input.position[0], input.position[1], input.position[2]}, input.alpha, false);
                auto const& field = result.shortField;
                double const tolerance = std::is_same_v<Real, float> ? 6.0e-5 : 4.0e-11;
                auto const pairOracle = gaussianPairOracle(
                    {input.position[0], input.position[1], input.position[2]}, input.alpha);
                for (int component = 0; component < 4; ++component) {
                    EXPECT_NEAR(double(result.ordinary.shortPair[component]),
                        double(pairOracle[component]),
                        tolerance * std::abs(double(pairOracle[component])) +
                            32 * double(std::numeric_limits<Real>::denorm_min()))
                        << "PP adapter component=" << component;
                }
                EXPECT_NEAR(double(field.potential()), 1.75 * double(oracle[0]),
                    tolerance * std::abs(1.75 * double(oracle[0])));
                for (int axis = 0; axis < 3; ++axis) {
                    std::array<int, 3> derivative{};
                    derivative[axis] = 1;
                    int const index = cosmictiger::gravity::cartesian::index(
                        derivative[0], derivative[1], derivative[2]);
                    double const expected = -1.75 * double(oracle[index]);
                    EXPECT_NEAR(double(field[axis + 1]), expected,
                        tolerance * std::abs(expected) +
                            32 * double(std::numeric_limits<Real>::denorm_min()));
                }
            }
        }
    } catch (...) {
        cudaFree(deviceResults);
        cudaFree(deviceInputs);
        throw;
    }
    requireGaussianCuda(cudaFree(deviceResults));
    requireGaussianCuda(cudaFree(deviceInputs));
}

}    // namespace detail

class GaussianScreeningCuda : public detail::CudaTest, public ::testing::WithParamInterface<int> {};

TEST_P(GaussianScreeningCuda, DerivativesAndParticleFieldsMatchIndependentQuadrature) {
    fmmDispatch(GetParam(), [](auto order) {
        detail::checkGaussianDevice<float, order.value>();
        detail::checkGaussianDevice<double, order.value>();
    });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, GaussianScreeningCuda,
    ::testing::Range(kFmmMinOrder, kFmmMaxOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
