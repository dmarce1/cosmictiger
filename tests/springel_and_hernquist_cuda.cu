#include <cmath>
#include <limits>

#include <cuda_runtime.h>

#include <cosmictiger/sph/springel_and_hernquist/equations.hpp>
#include "cuda_test.hpp"

namespace {
namespace sh = springel_and_hernquist;
inline constexpr int kSampleCount = 16;
inline constexpr int kValueCount = 16;

struct Sample {
    bool valid{};
    double values[kValueCount]{};
};

CUDA_EXPORT Sample sample(int index) {
    sh::Parameters parameters;
    parameters.coldEnergy = 1.0e11;
    parameters.warmEnergy = 1.0e12;
    parameters.supernovaEnergy = 2.0e16;
    sh::Model model;
    sh::Equilibrium state;
    double const coefficient = 1.0e25;
    Sample result;
    result.valid = sh::calibrate(parameters, coefficient, model) &&
        sh::equilibrium(
            model, model.thresholdDensity * ::pow(10.0, 0.5 * index), coefficient, state);
    if (!result.valid) {
        return result;
    }
    double const ratios[kSampleCount] = {0.0, 1.0e-300, 1.0e-30, 1.0e-12, 1.0e-6, 0.01, 0.1, 0.9,
        1.0, 1.1, 10.0, 1.0e6, 1.0e12, 1.0e30, 1.0e100, 1.0e300};
    double const timestep = parameters.starFormationTime * ::pow(10.0, 0.5 * index - 6.0);
    result.values[0] = model.thresholdDensity;
    result.values[1] = state.starFormationTime;
    result.values[2] = state.evaporationFactor;
    result.values[3] = state.hotEnergy;
    result.values[4] = state.coldFraction;
    result.values[5] = state.hotFraction;
    result.values[6] = state.effectiveEnergy;
    result.values[7] = state.pressure;
    result.values[8] = state.relaxationTime;
    result.values[9] = state.specificStarFormationRate;
    result.values[10] = sh::coldFraction(ratios[index]);
    result.values[11] = sh::hotFraction(ratios[index]);
    result.values[12] = sh::starFormationProbability(state, timestep);
    result.values[13] = sh::enrichmentIncrement(state, timestep, 0.02);
    result.values[14] = sh::windProbability(state, timestep, 2.0);
    result.values[15] = sh::windSpeed(parameters, 2.0, 0.25);
    return result;
}

__global__ void sampleKernel(Sample* output) {
    int const index = int(threadIdx.x);
    if (index < kSampleCount) {
        output[index] = sample(index);
    }
}
}    // namespace

using SpringelAndHernquistCuda = detail::CudaTest;

TEST_F(SpringelAndHernquistCuda, ScalarEquationsMatchCpu) {
    Sample* device = nullptr;
    Sample actual[kSampleCount];
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device), sizeof(actual));
    if (status == cudaSuccess) {
        sampleKernel<<<1, kSampleCount>>>(device);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    }
    cudaError_t const release = cudaFree(device);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_EQ(release, cudaSuccess) << cudaGetErrorString(release);
    for (int index = 0; index != kSampleCount; ++index) {
        Sample const expected = sample(index);
        ASSERT_TRUE(expected.valid);
        ASSERT_TRUE(actual[index].valid) << index;
        for (int value = 0; value != kValueCount; ++value) {
            double const scale = std::abs(expected.values[value]);
            ASSERT_TRUE(std::isfinite(actual[index].values[value])) << index << ":" << value;
            EXPECT_LE(std::abs(actual[index].values[value] - expected.values[value]),
                5.0e-12 * scale + std::numeric_limits<double>::denorm_min())
                << index << ":" << value;
        }
    }
}
