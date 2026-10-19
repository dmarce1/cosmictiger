#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>

namespace {
bool close(float left, float right, float scale, float absolute = 1.e-30f) {
    return primordial::finite(left) && primordial::finite(right) &&
        ::fabsf(left - right) <= 2.e-4f * scale + absolute;
}
struct QuadraticTraceLoss {
    float coefficient{};
    bool upper{};
    PRIMORDIAL_HD primordial::Status operator()(primordial::WorkingState const& state,
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
PRIMORDIAL_HD primordial::StepResult quadraticCase(unsigned index) {
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
}    // namespace
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: CUDA device unavailable");
        return 77;
    }
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
        return 1;
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
            return 1;
        }
    }
    SolverOptions options;
    options.maxAttempts = 512;
    options.maxSubsteps = 256;
    std::string error;
    if (!integrateBatchCuda(nullptr, nullptr, 0, options, &error)) {
        std::fprintf(stderr, "Empty CUDA batch failed: %s\n", error.c_str());
        return 1;
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
                return 1;
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
            return 1;
        }
        if (cpu[0].status != Status::Success || gpu[0].status != Status::Success ||
            cpu[0].newtonIterations <= 3 || gpu[0].newtonIterations <= 3 ||
            (count > 4 &&
                (cpu[4].status != Status::Success || gpu[4].status != Status::Success ||
                    cpu[4].newtonIterations <= 3 || gpu[4].newtonIterations <= 3))) {
            std::fprintf(stderr, "Long atomic/molecular CUDA trajectories did not solve\n");
            return 1;
        }
        for (size_t i = 0; i < count; ++i) {
            if (cpu[i].status != gpu[i].status || cpu[i].acceptedTime != gpu[i].acceptedTime) {
                std::fprintf(stderr, "CUDA status/time mismatch at %zu/%zu\n", i, count);
                return 1;
            }
            if (cpu[i].status != Status::Success) {
                continue;
            }
            if (!close(cpu[i].state.thermalEnergy, gpu[i].state.thermalEnergy,
                    cpu[i].state.thermalEnergy, 1.e-7f)) {
                std::fprintf(stderr, "CUDA thermal mismatch at %zu/%zu\n", i, count);
                return 1;
            }
            for (int j = 0; j < kAbundanceCount; ++j) {
                if (cpu[i].state.abundances[j].upper() != gpu[i].state.abundances[j].upper() ||
                    !close(cpu[i].state.abundances[j].raw(), gpu[i].state.abundances[j].raw(),
                        ::fabsf(cpu[i].state.abundances[j].raw()))) {
                    std::fprintf(
                        stderr, "CUDA species mismatch at %zu/%zu species%d\n", i, count, j);
                    return 1;
                }
            }
        }
    }
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
        return 1;
    }
    for (int j = 0; j < kAbundanceCount; ++j) {
        if (coolingCpu.state.abundances[j].upper() != coolingGpu.state.abundances[j].upper() ||
            !close(coolingCpu.state.abundances[j].raw(), coolingGpu.state.abundances[j].raw(),
                ::fabsf(coolingCpu.state.abundances[j].raw()))) {
            std::fprintf(stderr, "Molecular cooling CUDA species mismatch at%d\n", j);
            return 1;
        }
    }

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
        return 1;
    }
    limitedOptions.maxAttempts = 1;
    limitedOptions.maxNewtonIterations = 1;
    limitedOptions.initialStepFraction = 1.0f;
    if (!integrateBatchCpu(&limitedInput, &limitedCpu, 1, limitedOptions) ||
        !integrateBatchCuda(&limitedInput, &limitedGpu, 1, limitedOptions, &error) ||
        limitedCpu.status != Status::StepLimit || limitedGpu.status != Status::StepLimit ||
        limitedCpu.acceptedTime != 0.0f || limitedGpu.acceptedTime != 0.0f) {
        std::fprintf(stderr, "CUDA rejected-attempt bound parity failed\n");
        return 1;
    }
    std::puts("Primordial CPU/CUDA parity passed, including stiff networks, tails, traces and "
              "failure lanes");
    return 0;
}
