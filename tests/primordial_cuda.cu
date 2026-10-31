/* CosmicTiger - scalar Anninos CPU/CUDA parity; BSL-1.0. */
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>
#include "cuda_test.hpp"

namespace {
using namespace primordial;
using PrimordialCuda = ::detail::CudaTest;

template <class Real>
struct ScalarInput {
    StateT<Real> state{};
    EnvironmentT<Real> environment{};
    Real interval{};
};

template <class Real>
ScalarInput<Real> fixture(unsigned index) {
    ScalarInput<Real> input;
    input.environment.cmbTemperature = Real(0);
    input.environment.network = index % 2 == 0 ? Network::Atomic6 : Network::Molecular9;
    Real const temperatures[] = {Real(10), Real(100), Real(1000), Real(30000), Real(1.e5)};
    Real const fractions[6] = {Real(0.2), Real(0.01), Real(0.001), Real(0),
        index % 2 == 0 ? Real(0) : Real(0.002), Real(0)};
    EXPECT_EQ(initializeState(input.state, input.environment, fractions), Status::Success);
    EXPECT_EQ(thermalEnergyAtTemperature(input.state, input.environment, temperatures[index % 5],
                  input.state.thermalEnergy),
        Status::Success);
    input.environment.nHydrogen = index < 10 ? Real(0.01) : Real(0);
    input.environment.radiation.hydrogenIonization = Real(1.e-12);
    input.interval = Real(1.e7);
    if (index % 5 == 0) {
        // The lower thermal endpoint is stationary: ionization alone would
        // lower its temperature by increasing particle count at fixed energy.
        input.environment.nHydrogen = Real(0);
        input.environment.radiation.hydrogenIonization = Real(0);
    }
    if (index == 10) {
        // Directly stored neutral trace, too small to survive a 1-x encoding.
        input.state.abundances[0] = Real(1.e-30);
        input.state.abundances[1] = Real(1);
        input.state.abundances[5] =
            Real(1) + input.state.abundances[3] + Real(2) * input.state.abundances[4];
        input.environment.radiation.hydrogenIonization = Real(1.e-12);
        EXPECT_EQ(thermalEnergyAtTemperature(
                      input.state, input.environment, Real(10000), input.state.thermalEnergy),
            Status::Success);
    } else if (index == 11) {
        input.interval = Real(0);
    } else if (index == 12) {
        input.state.abundances[0] = Real(-1);
    }
    return input;
}

template <class Real>
__global__ void scalarKernel(ScalarInput<Real> const* inputs, StepResultT<Real>* outputs,
    unsigned count, SolverOptionsT<Real> options) {
    unsigned const index = threadIdx.x;
    if (index < count) {
        auto const& input = inputs[index];
        outputs[index] = advance(input.state, input.environment, input.interval, options);
    }
}

template <class Real>
void compare(StepResultT<Real> const& cpu, StepResultT<Real> const& gpu) {
    ASSERT_EQ(gpu.status, cpu.status);
    EXPECT_EQ(gpu.acceptedTime, cpu.acceptedTime);
    EXPECT_EQ(gpu.acceptedSubsteps, cpu.acceptedSubsteps);
    // Native CPU/CUDA elementary functions may differ by a few ulps. This
    // tolerance covers their accumulated effect, without an abundance floor.
    Real const relative = Real(128) * std::numeric_limits<Real>::epsilon();
    auto check = [&](Real left, Real right) {
        EXPECT_TRUE(finite(left));
        EXPECT_TRUE(finite(right));
        Real const scale = std::max(std::abs(left), std::abs(right));
        EXPECT_NEAR(
            left, right, relative * scale + Real(8) * std::numeric_limits<Real>::denorm_min());
    };
    for (int species = 0; species < kAbundanceCount; ++species) {
        check(cpu.state.abundances[species], gpu.state.abundances[species]);
    }
    check(cpu.state.thermalEnergy, gpu.state.thermalEnergy);
}

template <class Real>
void checkScalar() {
    constexpr unsigned count = 13;
    std::vector<ScalarInput<Real>> inputs;
    for (unsigned index = 0; index < count; ++index) {
        inputs.push_back(fixture<Real>(index));
    }
    ScalarInput<Real>* deviceInputs = nullptr;
    StepResultT<Real>* deviceResults = nullptr;
    ASSERT_EQ(
        cudaMalloc(reinterpret_cast<void**>(&deviceInputs), sizeof(ScalarInput<Real>) * count),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(reinterpret_cast<void**>(&deviceResults), sizeof(StepResultT<Real>) * count),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(deviceInputs, inputs.data(), sizeof(ScalarInput<Real>) * count,
                  cudaMemcpyHostToDevice),
        cudaSuccess);
    std::vector<StepResultT<Real>> results(count);
    for (unsigned budget : {4096U, 1U}) {
        SolverOptionsT<Real> options;
        options.maxSubsteps = budget;
        // One block only; direct float/double tests run sequentially.
        scalarKernel<<<1, 32>>>(deviceInputs, deviceResults, count, options);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(results.data(), deviceResults, sizeof(StepResultT<Real>) * count,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        for (unsigned index = 0; index < count; ++index) {
            SCOPED_TRACE(index);
            auto const cpu = advance(
                inputs[index].state, inputs[index].environment, inputs[index].interval, options);
            if (index < 12 && budget == 4096) {
                ASSERT_EQ(cpu.status, Status::Success);
            }
            compare(cpu, results[index]);
        }
    }
    EXPECT_EQ(cudaFree(deviceResults), cudaSuccess);
    EXPECT_EQ(cudaFree(deviceInputs), cudaSuccess);
}

TEST_F(PrimordialCuda, ScalarFloatCpuParity) {
    checkScalar<float>();
}
TEST_F(PrimordialCuda, ScalarDoubleCpuParity) {
    checkScalar<double>();
}

TEST_F(PrimordialCuda, ProductionBatchesAndInvalidLanes) {
    // 65 lanes exercises a tail block; production uses 64-thread blocks.
    // Maximum launch in this entire suite is therefore two blocks.
    std::vector<BatchInput> inputs(65);
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto const input = fixture<float>(unsigned(index % 13));
        inputs[index] = {input.state, input.environment, input.interval};
    }
    std::vector<StepResult> cpu(inputs.size()), gpu(inputs.size());
    for (unsigned budget : {4096U, 1U}) {
        SolverOptions options;
        options.maxSubsteps = budget;
        ASSERT_TRUE(integrateBatchCpu(inputs.data(), cpu.data(), inputs.size(), options, 1));
        std::string error;
        ASSERT_TRUE(integrateBatchCuda(inputs.data(), gpu.data(), inputs.size(), options, &error))
            << error;
        EXPECT_TRUE(error.empty());
        for (size_t index = 0; index < inputs.size(); ++index) {
            SCOPED_TRACE(index);
            compare(cpu[index], gpu[index]);
        }
    }
    EXPECT_TRUE(integrateBatchCuda(nullptr, nullptr, 0));
    EXPECT_FALSE(integrateBatchCuda(nullptr, gpu.data(), 1));
}
}    // namespace
