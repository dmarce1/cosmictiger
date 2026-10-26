#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include "cuda_test.hpp"

namespace {
bool close(float left, float right, float scale, float absolute = 1.e-30f) {
    return primordial::finite(left) && primordial::finite(right) &&
        ::fabsf(left - right) <= 2.e-4f * scale + absolute;
}
struct QuadraticTraceLoss {
    float coefficient{};
    bool upper{};
    CUDA_EXPORT primordial::Status operator()(primordial::WorkingState const& state,
        primordial::Environment const& environment, primordial::Derivative& derivative) const {
        auto const status = primordial::validateComposition(state, environment);
        if (status != primordial::Status::Success) {
            return status;
        }
        FloatFloat const minority =
            upper ? FloatFloat(1.0f) - state.abundances[0] : state.abundances[0];
        FloatFloat const loss = FloatFloat(coefficient) * minority * minority;
        derivative = {};
        derivative.abundances[0] = upper ? loss : -loss;
        return primordial::Status::Success;
    }
};
CUDA_EXPORT primordial::StepResult quadraticCase(unsigned index) {
    using namespace primordial;
    float const minority = index < 2 ? 1.e-13f : 1.e-20f;
    bool const upper = index % 2 != 0;
    State initial;
    initial.abundances[0] = SignedAbundance::fromMinority(minority, upper);
    initial.thermalEnergy = 1.0f;
    Environment environment;
    environment.heliumToHydrogenNumberRatio = 0.0f;
    SolverOptions options;
    options.abundanceAbsoluteTolerance[0] = minority * 1.e-11f;
    return advanceWithModel(initial, environment, 1.e10f, options,
        QuadraticTraceLoss{index < 2 ? 1.e6f : 1.e13f, upper});
}
__global__ void quadraticTraceKernel(primordial::StepResult* results) {
    if (threadIdx.x < 4) {
        results[threadIdx.x] = quadraticCase(threadIdx.x);
    }
}
struct ColdSample {
    primordial::Status status{};
    float cooling[5]{}, lte{}, heating{}, eosTemperature{};
};
CUDA_EXPORT ColdSample coldSample(unsigned index) {
    using namespace primordial;
    float const temperatures[] = {10.0f, 20.0f, 30.0f, 32.0f, 35.0f, 40.0f, 49.0f, 65.0f, 79.99f,
        80.0f, 90.0f, 99.99f, 100.0f};
    float const temperature = temperatures[index];
    ColdSample result;
    result.status = molecularCoolingCoefficients(temperature, result.cooling, result.lte);
    if (result.status != Status::Success) {
        return result;
    }
    Environment environment;
    environment.network = Network::Molecular9;
    environment.nHydrogen = index % 2 == 0 ? 1.e4f : 0.0f;
    environment.cmbTemperature = 136.275f;
    State state;
    state.abundances[0] = SignedAbundance::fromFraction(2.e-4f);
    state.abundances[4] = SignedAbundance::fromFraction(0.2f);
    state.abundances[5] = SignedAbundance::fromFraction(1.e-6f);
    result.status =
        thermalEnergyAtTemperature(state, environment, temperature, state.thermalEnergy);
    if (result.status != Status::Success) {
        return result;
    }
    Eos eos;
    result.status = evaluateEos(state, environment, eos);
    if (result.status != Status::Success) {
        return result;
    }
    result.eosTemperature = eos.temperature;
    Derivative derivative;
    result.status =
        evaluateRhsAtTemperature(workingState(state), environment, temperature, derivative);
    result.heating = derivative.thermalEnergy.toFloat();
    return result;
}
__global__ void coldSampleKernel(ColdSample* results) {
    if (threadIdx.x < 13) {
        results[threadIdx.x] = coldSample(threadIdx.x);
    }
}
}    // namespace
using PrimordialCuda = detail::CudaTest;

TEST_F(PrimordialCuda, ColdMolecularRhsAndEosMatchCpu) {
    using namespace primordial;
    // Include subnormal H2+ cooling, zero collider density, and both matching
    // endpoints. This one-block check exercises the production device formulas.
    ColdSample* coldDevice = nullptr;
    ColdSample coldResults[13];
    cudaError_t coldStatus = cudaMalloc(reinterpret_cast<void**>(&coldDevice), sizeof(coldResults));
    if (coldStatus == cudaSuccess) {
        coldSampleKernel<<<1, 13>>>(coldDevice);
        coldStatus = cudaGetLastError();
    }
    if (coldStatus == cudaSuccess) {
        coldStatus =
            cudaMemcpy(coldResults, coldDevice, sizeof(coldResults), cudaMemcpyDeviceToHost);
    }
    cudaError_t const coldRelease = cudaFree(coldDevice);
    if (coldStatus != cudaSuccess || coldRelease != cudaSuccess) {
        std::fprintf(stderr, "Cold molecular CUDA kernel failed\n");
        FAIL() << "Primordial CUDA regression failed; see diagnostic above";
    }
    for (unsigned i = 0; i != 13; ++i) {
        ColdSample const cpu = coldSample(i);
        ColdSample const& gpu = coldResults[i];
        if (cpu.status != Status::Success || gpu.status != Status::Success ||
            !close(cpu.heating, gpu.heating, ::fabsf(cpu.heating), 1.e-40f) ||
            !close(cpu.lte, gpu.lte, cpu.lte, 1.e-44f) ||
            !close(cpu.eosTemperature, gpu.eosTemperature, cpu.eosTemperature)) {
            std::fprintf(stderr, "Cold molecular CUDA RHS/EOS mismatch at case%u\n", i);
            FAIL() << "Primordial CUDA regression failed; see diagnostic above";
        }
        for (int partner = 0; partner != 5; ++partner) {
            if (!close(cpu.cooling[partner], gpu.cooling[partner], cpu.cooling[partner], 1.e-44f)) {
                std::fprintf(
                    stderr, "Cold molecular CUDA coefficient mismatch at case%u/%d\n", i, partner);
                FAIL() << "Primordial CUDA regression failed; see diagnostic above";
            }
        }
    }
}

TEST_F(PrimordialCuda, QuadraticTraceMatchesExactSolution) {
    using namespace primordial;
    // Device regression for the former false-convergence failure. The exact
    // solution is independent of both CPU and device implementations.
    StepResult* quadraticDevice = nullptr;
    StepResult quadraticResults[4];
    cudaError_t quadraticStatus =
        cudaMalloc(reinterpret_cast<void**>(&quadraticDevice), sizeof(quadraticResults));
    if (quadraticStatus == cudaSuccess) {
        quadraticTraceKernel<<<1, 4>>>(quadraticDevice);
        quadraticStatus = cudaGetLastError();
    }
    if (quadraticStatus == cudaSuccess) {
        quadraticStatus = cudaMemcpy(
            quadraticResults, quadraticDevice, sizeof(quadraticResults), cudaMemcpyDeviceToHost);
    }
    cudaError_t const quadraticRelease = cudaFree(quadraticDevice);
    if (quadraticStatus != cudaSuccess || quadraticRelease != cudaSuccess) {
        std::fprintf(stderr, "Quadratic trace CUDA kernel failed\n");
        FAIL() << "Primordial CUDA regression failed; see diagnostic above";
    }
    for (unsigned i = 0; i != 4; ++i) {
        auto const cpu = quadraticCase(i);
        auto const& gpu = quadraticResults[i];
        bool const upper = i % 2 != 0;
        double const initialMinority = static_cast<double>(i < 2 ? 1.e-13f : 1.e-20f);
        double const coefficient = static_cast<double>(i < 2 ? 1.e6f : 1.e13f);
        double const exact = initialMinority / (1.0 + coefficient * initialMinority * 1.e10);
        float const cpuMinority = upper ? cpu.state.abundances[0].complementFraction() :
                                          cpu.state.abundances[0].fraction();
        float const gpuMinority = upper ? gpu.state.abundances[0].complementFraction() :
                                          gpu.state.abundances[0].fraction();
        if (cpu.status != Status::Success || gpu.status != Status::Success ||
            cpu.acceptedTime != 1.e10f || gpu.acceptedTime != 1.e10f ||
            std::fabs(static_cast<double>(gpuMinority) / exact - 1.0) >= 0.015 ||
            !close(cpuMinority, gpuMinority, cpuMinority)) {
            std::fprintf(stderr, "Quadratic trace CUDA regression failed in case%u\n", i);
            FAIL() << "Primordial CUDA regression failed; see diagnostic above";
        }
    }
}

TEST_F(PrimordialCuda, StiffNetworksTailsTracesAndFailureLanesMatchCpu) {
    using namespace primordial;
    SolverOptions options;
    options.maxAttempts = 512;
    options.maxSubsteps = 256;
    std::string error;
    if (!integrateBatchCuda(nullptr, nullptr, 0, options, &error)) {
        std::fprintf(stderr, "Empty CUDA batch failed: %s\n", error.c_str());
        FAIL() << "Primordial CUDA regression failed; see diagnostic above";
    }
    unsigned randomState = 1234567U;
    for (size_t count : {size_t(1), size_t(31), size_t(33), size_t(65), size_t(257)}) {
        std::vector<BatchInput> inputs(count);
        std::vector<StepResult> cpu(count), gpu(count);
        for (size_t i = 0; i < count; ++i) {
            randomState = 1664525U * randomState + 1013904223U;
            float const uniform = static_cast<float>((randomState >> 8) & 0xFFFFU) / 65535.0f;
            auto& input = inputs[i];
            input.environment.nHydrogen = 0.01f + uniform;
            input.state.abundances[0] = SignedAbundance::fromFraction(0.01f + 0.9f * uniform);
            input.state.abundances[1] = SignedAbundance::fromFraction(0.05f);
            input.state.abundances[2] = SignedAbundance::fromFraction(0.01f);
            bool const molecular = i % 5 == 0;
            if (molecular) {
                input.environment.network = Network::Molecular9;
                input.state.abundances[4] = SignedAbundance::fromFraction(2.e-6f);
            }
            if (thermalEnergyAtTemperature(input.state, input.environment,
                    molecular ? 500.0f + 2000.0f * uniform : 1.e4f + 4.e4f * uniform,
                    input.state.thermalEnergy) != Status::Success) {
                FAIL() << "Primordial CUDA regression failed; see diagnostic above";
            }
            input.interval = 1.e6f;
            if (i == 0 || i == 4) {
                // Nontrivial atomic and molecular transients, including fast
                // intermediate relaxation from zero, span many source steps.
                input.environment.network = i == 4 ? Network::Molecular9 : Network::Atomic6;
                input.environment.nHydrogen = 1.0f;
                input.environment.cmbTemperature = 30.0f;
                input.state = State{};
                input.state.abundances[0] = SignedAbundance::fromFraction(1.e-4f);
                input.state.abundances[4] = SignedAbundance::fromFraction(i == 4 ? 2.e-6f : 0.0f);
                thermalEnergyAtTemperature(input.state, input.environment,
                    i == 4 ? 1000.0f : 20000.0f, input.state.thermalEnergy);
                input.interval = 1.e12f;
            } else if (i == 5 || i == 6) {
                // Both signed endpoints and a minority far below epsilon32
                // pass through actual accepted steps without decoding loss.
                input.environment = Environment{};
                input.environment.nHydrogen = 0.0f;
                input.environment.cmbTemperature = 0.0f;
                input.state = State{};
                input.state.abundances[0] =
                    SignedAbundance::fromMinority(i == 5 ? 0.0f : 1.e-20f, true);
                thermalEnergyAtTemperature(
                    input.state, input.environment, 20000.0f, input.state.thermalEnergy);
            }
            switch (i % 11) {
            case 7:
                input.interval = 0.0f;
                break;
            case 1:
                input.interval = -1.0f;
                break;
            case 2:
                input.environment.nHydrogen = 1.e5f;
                break;
            case 3:
                input.state.thermalEnergy = std::numeric_limits<float>::quiet_NaN();
                break;
            default:
                break;
            }
        }
        if (!integrateBatchCpu(inputs.data(), cpu.data(), count, options, 1) ||
            !integrateBatchCuda(inputs.data(), gpu.data(), count, options, &error)) {
            std::fprintf(stderr, "CUDA dispatch failed: %s\n", error.c_str());
            FAIL() << "Primordial CUDA regression failed; see diagnostic above";
        }
        if (cpu[0].status != Status::Success || gpu[0].status != Status::Success ||
            cpu[0].newtonIterations <= 3 || gpu[0].newtonIterations <= 3 ||
            (count > 4 &&
                (cpu[4].status != Status::Success || gpu[4].status != Status::Success ||
                    cpu[4].newtonIterations <= 3 || gpu[4].newtonIterations <= 3))) {
            std::fprintf(stderr, "Long atomic/molecular CUDA trajectories did not solve\n");
            FAIL() << "Primordial CUDA regression failed; see diagnostic above";
        }
        for (size_t i = 0; i < count; ++i) {
            if (cpu[i].status != gpu[i].status || cpu[i].acceptedTime != gpu[i].acceptedTime) {
                std::fprintf(stderr, "CUDA status/time mismatch at %zu/%zu\n", i, count);
                FAIL() << "Primordial CUDA regression failed; see diagnostic above";
            }
            if (cpu[i].status != Status::Success) {
                continue;
            }
            if (!close(cpu[i].state.thermalEnergy, gpu[i].state.thermalEnergy,
                    cpu[i].state.thermalEnergy, 1.e-7f)) {
                std::fprintf(stderr, "CUDA thermal mismatch at %zu/%zu\n", i, count);
                FAIL() << "Primordial CUDA regression failed; see diagnostic above";
            }
            for (int j = 0; j < kAbundanceCount; ++j) {
                if (cpu[i].state.abundances[j].upper() != gpu[i].state.abundances[j].upper() ||
                    !close(cpu[i].state.abundances[j].raw(), gpu[i].state.abundances[j].raw(),
                        ::fabsf(cpu[i].state.abundances[j].raw()))) {
                    std::fprintf(
                        stderr, "CUDA species mismatch at %zu/%zu species%d\n", i, count, j);
                    FAIL() << "Primordial CUDA regression failed; see diagnostic above";
                }
            }
        }
    }
}

TEST_F(PrimordialCuda, CoupledMolecularCoolingMatchesCpu) {
    using namespace primordial;
    std::string error;
    // Significant molecular cooling couples species and energy across a large
    // temperature change; use the full production work budget for this case.
    BatchInput coolingInput;
    coolingInput.environment.network = Network::Molecular9;
    coolingInput.environment.nHydrogen = 1.0f;
    coolingInput.environment.cmbTemperature = 30.0f;
    coolingInput.state.abundances[0] = SignedAbundance::fromFraction(1.e-4f);
    coolingInput.state.abundances[4] = SignedAbundance::fromFraction(0.2f);
    coolingInput.interval = 1.e12f;
    thermalEnergyAtTemperature(
        coolingInput.state, coolingInput.environment, 2000.0f, coolingInput.state.thermalEnergy);
    StepResult coolingCpu, coolingGpu;
    SolverOptions const coolingOptions;
    if (!integrateBatchCpu(&coolingInput, &coolingCpu, 1, coolingOptions) ||
        !integrateBatchCuda(&coolingInput, &coolingGpu, 1, coolingOptions, &error) ||
        coolingCpu.status != Status::Success || coolingGpu.status != Status::Success ||
        coolingCpu.acceptedTime != coolingInput.interval ||
        coolingGpu.acceptedTime != coolingInput.interval ||
        coolingCpu.state.thermalEnergy > 0.7f * coolingInput.state.thermalEnergy ||
        !close(coolingCpu.state.thermalEnergy, coolingGpu.state.thermalEnergy,
            coolingCpu.state.thermalEnergy, 1.e-7f)) {
        std::fprintf(stderr, "Coupled molecular cooling CUDA parity failed: %s\n", error.c_str());
        FAIL() << "Primordial CUDA regression failed; see diagnostic above";
    }
    for (int j = 0; j < kAbundanceCount; ++j) {
        if (coolingCpu.state.abundances[j].upper() != coolingGpu.state.abundances[j].upper() ||
            !close(coolingCpu.state.abundances[j].raw(), coolingGpu.state.abundances[j].raw(),
                ::fabsf(coolingCpu.state.abundances[j].raw()))) {
            std::fprintf(stderr, "Molecular cooling CUDA species mismatch at%d\n", j);
            FAIL() << "Primordial CUDA regression failed; see diagnostic above";
        }
    }
}

TEST_F(PrimordialCuda, RadiationHeatingMatchesExactEnergyIncrement) {
    using namespace primordial;
    SolverOptions const coolingOptions;
    std::string error;
    // Radiation-only heating has an independent exact energy increment. It
    // crosses both the cold matching interval and the established 100 K domain.
    BatchInput coldHeating;
    coldHeating.environment.network = Network::Molecular9;
    coldHeating.environment.nHydrogen = 0.0f;
    coldHeating.environment.radiation.heatingPerHydrogen = 2.e-14f;
    coldHeating.state.abundances[4] = SignedAbundance::fromFraction(0.2f);
    coldHeating.interval = 1.e12f;
    thermalEnergyAtTemperature(
        coldHeating.state, coldHeating.environment, 20.0f, coldHeating.state.thermalEnergy);
    StepResult coldCpu, coldGpu;
    Eos heatedEos;
    float const expectedEnergy = coldHeating.state.thermalEnergy +
        0.8f * coldHeating.environment.radiation.heatingPerHydrogen * coldHeating.interval;
    if (!integrateBatchCpu(&coldHeating, &coldCpu, 1, coolingOptions, 1) ||
        !integrateBatchCuda(&coldHeating, &coldGpu, 1, coolingOptions, &error) ||
        coldCpu.status != Status::Success || coldGpu.status != Status::Success ||
        coldCpu.acceptedTime != coldHeating.interval ||
        coldGpu.acceptedTime != coldHeating.interval ||
        !close(coldCpu.state.thermalEnergy, coldGpu.state.thermalEnergy, expectedEnergy, 1.e-9f) ||
        !close(expectedEnergy, coldGpu.state.thermalEnergy, expectedEnergy, 1.e-9f) ||
        evaluateEos(coldGpu.state, coldHeating.environment, heatedEos) != Status::Success ||
        heatedEos.temperature <= 100.0f) {
        std::fprintf(stderr, "Cold molecular CUDA heating trajectory failed: %s\n", error.c_str());
        FAIL() << "Primordial CUDA regression failed; see diagnostic above";
    }
}

TEST_F(PrimordialCuda, WorkBoundsPreservePartialStateAndAcceptedTime) {
    using namespace primordial;
    SolverOptions options;
    options.maxAttempts = 512;
    options.maxSubsteps = 256;
    std::string error;
    // Work bounds must preserve the same accepted partial interval and state.
    BatchInput limitedInput;
    limitedInput.environment.nHydrogen = 1.0f;
    limitedInput.state.abundances[0] = SignedAbundance::fromFraction(1.e-4f);
    limitedInput.interval = 1.e12f;
    thermalEnergyAtTemperature(
        limitedInput.state, limitedInput.environment, 20000.0f, limitedInput.state.thermalEnergy);
    SolverOptions limitedOptions = options;
    limitedOptions.maxSubsteps = 1;
    limitedOptions.initialStepFraction = 1.e-4f;
    StepResult limitedCpu, limitedGpu;
    if (!integrateBatchCpu(&limitedInput, &limitedCpu, 1, limitedOptions) ||
        !integrateBatchCuda(&limitedInput, &limitedGpu, 1, limitedOptions, &error) ||
        limitedCpu.status != Status::StepLimit || limitedGpu.status != Status::StepLimit ||
        limitedCpu.acceptedTime <= 0.0f || limitedGpu.acceptedTime <= 0.0f ||
        !close(limitedCpu.acceptedTime, limitedGpu.acceptedTime, limitedCpu.acceptedTime) ||
        !close(limitedCpu.state.thermalEnergy, limitedGpu.state.thermalEnergy,
            limitedCpu.state.thermalEnergy, 1.e-7f)) {
        std::fprintf(stderr, "CUDA bounded-work parity failed: %s\n", error.c_str());
        FAIL() << "Primordial CUDA regression failed; see diagnostic above";
    }
    limitedOptions.maxAttempts = 1;
    limitedOptions.maxNewtonIterations = 1;
    limitedOptions.initialStepFraction = 1.0f;
    if (!integrateBatchCpu(&limitedInput, &limitedCpu, 1, limitedOptions) ||
        !integrateBatchCuda(&limitedInput, &limitedGpu, 1, limitedOptions, &error) ||
        limitedCpu.status != Status::StepLimit || limitedGpu.status != Status::StepLimit ||
        limitedCpu.acceptedTime != 0.0f || limitedGpu.acceptedTime != 0.0f) {
        std::fprintf(stderr, "CUDA rejected-attempt bound parity failed\n");
        FAIL() << "Primordial CUDA regression failed; see diagnostic above";
    }
}
