#include <cmath>
#include <limits>

#include <cuda_runtime.h>

#include <cosmictiger/sph/springel_and_hernquist/equations.hpp>
#include "cuda_test.hpp"

namespace {
namespace sh = springel_and_hernquist;
inline constexpr int kSampleCount = 16;
inline constexpr int kValueCount = 12;

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
        sh::equilibrium(model, model.thresholdDensity * ::pow(10.0, 0.5 * index), 1.e13, state);
    if (!result.valid) {
        return result;
    }
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
    result.values[10] = sh::starFormationProbability(state, timestep);
    result.valid = sh::clampSourceTime(parameters, 10.0 * state.starFormationTime, state);
    result.values[11] = state.specificStarFormationRate;
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
