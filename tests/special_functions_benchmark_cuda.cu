#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/math/special_functions.hpp>
#include "special_functions_reference.hpp"

// Direct libdevice entry points preserve their accurate implementations when
// --use_fast_math rewrites ordinary device math calls to approximate variants.
extern "C" {
__device__ float __nv_sinf(float);
__device__ float __nv_cosf(float);
__device__ void __nv_sincosf(float, float*, float*);
__device__ float __nv_expf(float);
__device__ float __nv_exp2f(float);
__device__ float __nv_logf(float);
__device__ float __nv_log2f(float);
}

namespace detail {

enum class Operation { Sin, Cos, Sincos, Exp, Exp2, Log, Log2, Erf, Erfc, Erfcexp, Sqrt, Rsqrt };

enum class Path { Custom, Legacy, Library, LibraryFma, LibraryAccurate, LibraryDouble };

char const* operationName(Operation operation) {
    switch (operation) {
    case Operation::Sin:
        return "sin";
    case Operation::Cos:
        return "cos";
    case Operation::Sincos:
        return "sincos";
    case Operation::Exp:
        return "exp";
    case Operation::Exp2:
        return "exp2";
    case Operation::Log:
        return "log";
    case Operation::Log2:
        return "log2";
    case Operation::Erf:
        return "erf";
    case Operation::Erfc:
        return "erfc";
    case Operation::Erfcexp:
        return "erfcexp";
    case Operation::Sqrt:
        return "sqrt";
    case Operation::Rsqrt:
        return "rsqrt";
    }
    return "unknown";
}

char const* pathName(Path path) {
    switch (path) {
    case Path::Custom:
        return "custom";
    case Path::Legacy:
        return "legacy";
    case Path::Library:
        return "library";
    case Path::LibraryFma:
        return "library_fma";
    case Path::LibraryAccurate:
        return "libdevice_accurate";
    case Path::LibraryDouble:
        return "library_double";
    }
    return "unknown";
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <Operation operation, Path path>
__device__ float evaluate(float input, float* second) {
    if constexpr (path == Path::Custom) {
        if constexpr (operation == Operation::Sin) {
            return special_functions::sin(input);
        } else if constexpr (operation == Operation::Cos) {
            return special_functions::cos(input);
        } else if constexpr (operation == Operation::Sincos) {
            float sine;
            special_functions::sincos(input, &sine, second);
            return sine;
        } else if constexpr (operation == Operation::Exp) {
            return special_functions::exp(input);
        } else if constexpr (operation == Operation::Exp2) {
            return special_functions::exp2(input);
        } else if constexpr (operation == Operation::Log) {
            return special_functions::log(input);
        } else if constexpr (operation == Operation::Log2) {
            return special_functions::log2(input);
        } else if constexpr (operation == Operation::Erf) {
            return special_functions::erf(input);
        } else if constexpr (operation == Operation::Erfc) {
            return special_functions::erfc(input);
        } else if constexpr (operation == Operation::Erfcexp) {
            float complement;
            special_functions::erfcexp(input, &complement, second);
            return complement;
        } else if constexpr (operation == Operation::Sqrt) {
            return special_functions::sqrt(input);
        } else {
            return special_functions::rsqrt(input);
        }
    } else if constexpr (path == Path::Legacy) {
        static_assert(operation == Operation::Erfcexp);
        // Reproduce cosmictiger/math/math.hpp's CUDA A&S five-term path.
        float const p = 0.3275911f;
        float const a1 = 0.254829592f;
        float const a2 = -0.284496736f;
        float const a3 = 1.421413741f;
        float const a4 = -1.453152027f;
        float const a5 = 1.061405429f;
        float const t1 = 1.0f / ::fma(p, input, 1.0f);
        float const t2 = t1 * t1;
        float const t3 = t2 * t1;
        float const t4 = t2 * t2;
        float const t5 = t2 * t3;
        *second = ::expf(-input * input);
        return ::fmaf(a1, t1, ::fmaf(a2, t2, ::fmaf(a3, t3, ::fmaf(a4, t4, a5 * t5)))) * *second;
    } else if constexpr (path == Path::LibraryAccurate) {
        if constexpr (operation == Operation::Sin) {
            return __nv_sinf(input);
        } else if constexpr (operation == Operation::Cos) {
            return __nv_cosf(input);
        } else if constexpr (operation == Operation::Sincos) {
            float sine;
            __nv_sincosf(input, &sine, second);
            return sine;
        } else if constexpr (operation == Operation::Exp) {
            return __nv_expf(input);
        } else if constexpr (operation == Operation::Exp2) {
            return __nv_exp2f(input);
        } else if constexpr (operation == Operation::Log) {
            return __nv_logf(input);
        } else if constexpr (operation == Operation::Log2) {
            return __nv_log2f(input);
        } else if constexpr (operation == Operation::Erf) {
            return ::erff(input);
        } else if constexpr (operation == Operation::Erfc) {
            return ::erfcf(input);
        } else if constexpr (operation == Operation::Erfcexp) {
            float const square = input * input;
            float const gaussian = __nv_expf(-square);
            *second = ::fmaf(::fmaf(-input, input, square), gaussian, gaussian);
            return ::erfcf(input);
        } else if constexpr (operation == Operation::Sqrt) {
            return ::__fsqrt_rn(input);
        } else {
            return ::__frsqrt_rn(input);
        }
    } else {
        if constexpr (operation == Operation::Sin) {
            return ::sinf(input);
        } else if constexpr (operation == Operation::Cos) {
            return ::cosf(input);
        } else if constexpr (operation == Operation::Sincos) {
            float sine;
            ::sincosf(input, &sine, second);
            return sine;
        } else if constexpr (operation == Operation::Exp) {
            return ::expf(input);
        } else if constexpr (operation == Operation::Exp2) {
            return ::exp2f(input);
        } else if constexpr (operation == Operation::Log) {
            return ::logf(input);
        } else if constexpr (operation == Operation::Log2) {
            return ::log2f(input);
        } else if constexpr (operation == Operation::Erf) {
            return ::erff(input);
        } else if constexpr (operation == Operation::Erfc) {
            return ::erfcf(input);
        } else if constexpr (operation == Operation::Erfcexp) {
            float const square = input * input;
            float const gaussian = path == Path::LibraryDouble ?
                float(::exp(-double(input) * double(input))) :
                ::expf(-square);
            *second = path == Path::LibraryFma ?
                ::fmaf(::fmaf(-input, input, square), gaussian, gaussian) :
                gaussian;
            return ::erfcf(input);
        } else if constexpr (operation == Operation::Sqrt) {
            return ::sqrtf(input);
        } else {
            return ::rsqrtf(input);
        }
    }
}

template <Operation operation, Path path>
__global__ void evaluateKernel(float const* input, float* first, float* second, size_t count) {
    size_t const stride = size_t(blockDim.x) * gridDim.x;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
        index += stride) {
        float extra;
        first[index] = evaluate<operation, path>(input[index], &extra);
        if constexpr (operation == Operation::Sincos || operation == Operation::Erfcexp) {
            second[index] = extra;
        }
    }
}

template <Operation operation, Path path>
__global__ void evaluateRegisterKernel(float const* seeds, float* firstOutput, float* secondOutput,
    unsigned long long* elapsedCycles, size_t iterations) {
    size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    float const seed = seeds[index];
    float value = seed;
    float first = 0.0f;
    float second = 0.0f;
    unsigned long long begin;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin) : "f"(value) : "memory");
#pragma unroll 1
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        first = evaluate<operation, path>(value, &second);
        uint32_t bits = __float_as_uint(first);
        if constexpr (operation == Operation::Sincos || operation == Operation::Erfcexp) {
            bits ^= __float_as_uint(second);
        }
        float const relative = 1.0f - float(bits & 1023U) * 0x1p-20f;
        value = seed * relative;
    }
    unsigned long long end;
    asm volatile("mov.u64 %0, %%clock64;"
        : "=l"(end)
        : "f"(value), "f"(first), "f"(second)
        : "memory");
    elapsedCycles[index] = end - begin;
    firstOutput[index] = first;
    if constexpr (operation == Operation::Sincos || operation == Operation::Erfcexp) {
        secondOutput[index] = second;
    }
}

template <Operation operation>
std::vector<float> makeInputs(size_t count) {
    std::vector<float> inputs(count);
    std::mt19937 generator(0x5f3759dfU + unsigned(operation));
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::uniform_int_distribution<int> exponent(-120, 120);
    constexpr float kPi = 3.14159265358979323846f;
    for (float& input : inputs) {
        float const unit = uniform(generator);
        if constexpr (operation == Operation::Sin || operation == Operation::Cos ||
            operation == Operation::Sincos) {
            input = (2.0f * unit - 1.0f) * (4.5f * kPi);
        } else if constexpr (operation == Operation::Exp) {
            input = -75.0f + unit * 75.0f;
        } else if constexpr (operation == Operation::Exp2) {
            input = -110.0f + unit * 235.0f;
        } else if constexpr (operation == Operation::Erf) {
            input = (2.0f * unit - 1.0f) * 8.660254f;
        } else if constexpr (operation == Operation::Erfc || operation == Operation::Erfcexp) {
            input = unit * 8.660254f;
        } else {
            input = std::ldexp(0.5f + 0.5f * unit, exponent(generator));
        }
    }
    if (count >= 32) {
        float const errorBoundaries[] = {0.0f, 0.75f, 1.0f, 2.0f, 3.0f, 5.2f, 8.0f, 8.660254f,
            -0.75f, -1.0f, -2.0f, -5.2f, -8.660254f};
        float const phaseBoundaries[] = {0.0f, kPi / 2.0f, kPi, 1.5f * kPi, 2.0f * kPi, 4.0f * kPi,
            -kPi / 2.0f, -kPi, -4.0f * kPi};
        float const exponentialBoundaries[] = {-75.0f, -27.04f, -1.0f, 0.0f};
        float const baseTwoBoundaries[] = {-110.0f, -75.0f, -1.0f, 0.0f, 1.0f, 100.0f, 125.0f};
        if constexpr (operation == Operation::Sin || operation == Operation::Cos ||
            operation == Operation::Sincos) {
            for (size_t index = 0; index < std::size(phaseBoundaries); ++index) {
                inputs[index] = phaseBoundaries[index];
            }
        } else if constexpr (operation == Operation::Exp) {
            for (size_t index = 0; index < std::size(exponentialBoundaries); ++index) {
                inputs[index] = exponentialBoundaries[index];
            }
        } else if constexpr (operation == Operation::Exp2) {
            for (size_t index = 0; index < std::size(baseTwoBoundaries); ++index) {
                inputs[index] = baseTwoBoundaries[index];
            }
        } else if constexpr (operation == Operation::Erf) {
            for (size_t index = 0; index < std::size(errorBoundaries); ++index) {
                inputs[index] = errorBoundaries[index];
            }
        } else if constexpr (operation == Operation::Erfc || operation == Operation::Erfcexp) {
            for (size_t index = 0; index < 8; ++index) {
                inputs[index] = errorBoundaries[index];
            }
        } else {
            inputs[0] = std::numeric_limits<float>::min();
            inputs[1] = 0.5f;
            inputs[2] = 1.0f;
            inputs[3] = 2.0f;
            inputs[4] = std::numeric_limits<float>::max();
        }
    }
    return inputs;
}

struct Accuracy {
    uint64_t maximumUlps;
    size_t failures;
};

template <Operation operation>
Accuracy assessAccuracy(std::vector<float> const& inputs, std::vector<float> const& first,
    std::vector<float> const& second) {
    using special_functions_test::UnaryFunction;
    Accuracy accuracy{};
    auto observe = [&](float actual, long double reference) {
        float const expected = static_cast<float>(reference);
        uint64_t const distance = special_functions_test::ulpDistance(actual, expected);
        accuracy.maximumUlps = std::max(accuracy.maximumUlps, distance);
        accuracy.failures += distance > special_functions_test::kMaximumUlps;
    };
    auto checkIndex = [&](size_t index) {
        float const input = inputs[index];
        if constexpr (operation == Operation::Sin || operation == Operation::Sincos) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Sin, input));
        } else if constexpr (operation == Operation::Cos) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Cos, input));
        } else if constexpr (operation == Operation::Exp) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Exp, input));
        } else if constexpr (operation == Operation::Exp2) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Exp2, input));
        } else if constexpr (operation == Operation::Log) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Log, input));
        } else if constexpr (operation == Operation::Log2) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Log2, input));
        } else if constexpr (operation == Operation::Erf) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Erf, input));
        } else if constexpr (operation == Operation::Erfc || operation == Operation::Erfcexp) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Erfc, input));
        } else if constexpr (operation == Operation::Sqrt) {
            observe(first[index], special_functions_test::reference(UnaryFunction::Sqrt, input));
        } else {
            observe(first[index], special_functions_test::reference(UnaryFunction::Rsqrt, input));
        }
        if constexpr (operation == Operation::Sincos) {
            observe(second[index], special_functions_test::reference(UnaryFunction::Cos, input));
        } else if constexpr (operation == Operation::Erfcexp) {
            observe(second[index], special_functions_test::exponentialSquareReference(input));
        }
    };
    for (size_t index = 0; index < std::min(size_t(32), inputs.size()); ++index) {
        checkIndex(index);
    }
    size_t const stride = std::max(size_t(1), inputs.size() / 1024);
    for (size_t index = 32; index < inputs.size(); index += stride) {
        checkIndex(index);
    }
    return accuracy;
}

double outputChecksum(
    std::vector<float> const& first, std::vector<float> const& second, bool twoOutputs) {
    double checksum = 0.0;
    for (size_t index = 0; index < first.size(); ++index) {
        checksum += double(first[index]);
        if (twoOutputs) {
            checksum += double(second[index]);
        }
    }
    return checksum;
}

struct Measurement {
    double nanoseconds;
    double checksum;
};

template <Operation operation, Path path>
Measurement run(float const* deviceInput, float* deviceFirst, float* deviceSecond,
    std::vector<float>& first, std::vector<float>& second, cudaEvent_t begin, cudaEvent_t end,
    size_t count, int blockCount, size_t launches) {
    cudaCheck(cudaEventRecord(begin));
    for (size_t launch = 0; launch < launches; ++launch) {
        evaluateKernel<operation, path>
            <<<blockCount, 128>>>(deviceInput, deviceFirst, deviceSecond, count);
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end));
    float milliseconds;
    cudaCheck(cudaEventElapsedTime(&milliseconds, begin, end));
    cudaCheck(cudaMemcpy(first.data(), deviceFirst, count * sizeof(float), cudaMemcpyDeviceToHost));
    if constexpr (operation == Operation::Sincos || operation == Operation::Erfcexp) {
        cudaCheck(
            cudaMemcpy(second.data(), deviceSecond, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return {double(milliseconds) * 1.0e6 / (double(count) * double(launches)),
        outputChecksum(
            first, second, operation == Operation::Sincos || operation == Operation::Erfcexp)};
}

template <Operation operation>
Measurement runPath(Path path, float const* deviceInput, float* deviceFirst, float* deviceSecond,
    std::vector<float>& first, std::vector<float>& second, cudaEvent_t begin, cudaEvent_t end,
    size_t count, int blockCount, size_t launches) {
    switch (path) {
    case Path::Custom:
        return run<operation, Path::Custom>(deviceInput, deviceFirst, deviceSecond, first, second,
            begin, end, count, blockCount, launches);
    case Path::Legacy:
        if constexpr (operation == Operation::Erfcexp) {
            return run<operation, Path::Legacy>(deviceInput, deviceFirst, deviceSecond, first,
                second, begin, end, count, blockCount, launches);
        } else {
            throw std::runtime_error("legacy path is only defined for erfcexp");
        }
    case Path::Library:
        return run<operation, Path::Library>(deviceInput, deviceFirst, deviceSecond, first, second,
            begin, end, count, blockCount, launches);
    case Path::LibraryFma:
        return run<operation, Path::LibraryFma>(deviceInput, deviceFirst, deviceSecond, first,
            second, begin, end, count, blockCount, launches);
    case Path::LibraryAccurate:
        return run<operation, Path::LibraryAccurate>(deviceInput, deviceFirst, deviceSecond, first,
            second, begin, end, count, blockCount, launches);
    case Path::LibraryDouble:
        return run<operation, Path::LibraryDouble>(deviceInput, deviceFirst, deviceSecond, first,
            second, begin, end, count, blockCount, launches);
    }
    throw std::runtime_error("invalid path");
}

struct TimingSpread {
    double minimum;
    double median;
    double maximum;
};

TimingSpread timingSpread(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {samples.front(), samples[samples.size() / 2], samples.back()};
}

template <Operation operation>
void report(size_t count, size_t repetitions, int blockCount, cudaEvent_t begin, cudaEvent_t end,
    size_t launches) {
    auto const inputs = makeInputs<operation>(count);
    std::vector<float> first(count);
    std::vector<float> second(count);
    float* deviceInput;
    float* deviceFirst;
    float* deviceSecond;
    cudaCheck(cudaMalloc(&deviceInput, count * sizeof(float)));
    cudaCheck(cudaMalloc(&deviceFirst, count * sizeof(float)));
    cudaCheck(cudaMalloc(&deviceSecond, count * sizeof(float)));
    cudaCheck(
        cudaMemcpy(deviceInput, inputs.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    constexpr size_t kPathCount = operation == Operation::Erfcexp ? 6 : 3;
    Path const paths[] = {Path::Custom, Path::Library, Path::LibraryAccurate, Path::Legacy,
        Path::LibraryFma, Path::LibraryDouble};
    std::vector<double> samples[kPathCount];
    double checksums[kPathCount]{};
    for (size_t index = 0; index < kPathCount; ++index) {
        (void) runPath<operation>(paths[index], deviceInput, deviceFirst, deviceSecond, first,
            second, begin, end, count, blockCount, launches);
    }
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (size_t offset = 0; offset < kPathCount; ++offset) {
            size_t const index = (offset + repetition) % kPathCount;
            Measurement const measurement = runPath<operation>(paths[index], deviceInput,
                deviceFirst, deviceSecond, first, second, begin, end, count, blockCount, launches);
            samples[index].push_back(measurement.nanoseconds);
            checksums[index] = measurement.checksum;
        }
    }
    for (size_t index = 0; index < kPathCount; ++index) {
        (void) runPath<operation>(paths[index], deviceInput, deviceFirst, deviceSecond, first,
            second, begin, end, count, blockCount, launches);
        Accuracy const accuracy = assessAccuracy<operation>(inputs, first, second);
        TimingSpread const timing = timingSpread(samples[index]);
        std::printf("cuda\t%s\t%s\t%.3f\t%.3f\t%.3f\t%llu\t%zu\t%.9g\n", operationName(operation),
            pathName(paths[index]), timing.minimum, timing.median, timing.maximum,
            static_cast<unsigned long long>(accuracy.maximumUlps), accuracy.failures,
            checksums[index]);
    }
    cudaCheck(cudaFree(deviceSecond));
    cudaCheck(cudaFree(deviceFirst));
    cudaCheck(cudaFree(deviceInput));
}

struct RegisterMeasurement {
    double cyclesPerEvaluation;
    double checksum;
};

template <Operation operation, Path path>
RegisterMeasurement runRegister(float const* deviceSeeds, float* deviceFirst, float* deviceSecond,
    unsigned long long* deviceCycles, std::vector<float>& first, std::vector<float>& second,
    std::vector<unsigned long long>& cycles, int blockCount, size_t iterations) {
    evaluateRegisterKernel<operation, path>
        <<<blockCount, 128>>>(deviceSeeds, deviceFirst, deviceSecond, deviceCycles, iterations);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    size_t const count = first.size();
    cudaCheck(cudaMemcpy(
        cycles.data(), deviceCycles, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(first.data(), deviceFirst, count * sizeof(float), cudaMemcpyDeviceToHost));
    if constexpr (operation == Operation::Sincos || operation == Operation::Erfcexp) {
        cudaCheck(
            cudaMemcpy(second.data(), deviceSecond, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    // A warp completes when its slowest lane does. The fastest warp reduces
    // distortion from delayed blocks or preemption in this low-occupancy run.
    unsigned long long minimumWarpMaximum = std::numeric_limits<unsigned long long>::max();
    for (size_t warp = 0; warp < count; warp += 32) {
        unsigned long long warpMaximum = 0;
        for (size_t lane = 0; lane < 32; ++lane) {
            warpMaximum = std::max(warpMaximum, cycles[warp + lane]);
        }
        minimumWarpMaximum = std::min(minimumWarpMaximum, warpMaximum);
    }
    return {double(minimumWarpMaximum) / double(iterations),
        outputChecksum(
            first, second, operation == Operation::Sincos || operation == Operation::Erfcexp)};
}

template <Operation operation>
RegisterMeasurement runRegisterPath(Path path, float const* deviceSeeds, float* deviceFirst,
    float* deviceSecond, unsigned long long* deviceCycles, std::vector<float>& first,
    std::vector<float>& second, std::vector<unsigned long long>& cycles, int blockCount,
    size_t iterations) {
    switch (path) {
    case Path::Custom:
        return runRegister<operation, Path::Custom>(deviceSeeds, deviceFirst, deviceSecond,
            deviceCycles, first, second, cycles, blockCount, iterations);
    case Path::Legacy:
        if constexpr (operation == Operation::Erfcexp) {
            return runRegister<operation, Path::Legacy>(deviceSeeds, deviceFirst, deviceSecond,
                deviceCycles, first, second, cycles, blockCount, iterations);
        } else {
            throw std::runtime_error("legacy path is only defined for erfcexp");
        }
    case Path::Library:
        return runRegister<operation, Path::Library>(deviceSeeds, deviceFirst, deviceSecond,
            deviceCycles, first, second, cycles, blockCount, iterations);
    case Path::LibraryFma:
        return runRegister<operation, Path::LibraryFma>(deviceSeeds, deviceFirst, deviceSecond,
            deviceCycles, first, second, cycles, blockCount, iterations);
    case Path::LibraryAccurate:
        return runRegister<operation, Path::LibraryAccurate>(deviceSeeds, deviceFirst, deviceSecond,
            deviceCycles, first, second, cycles, blockCount, iterations);
    case Path::LibraryDouble:
        return runRegister<operation, Path::LibraryDouble>(deviceSeeds, deviceFirst, deviceSecond,
            deviceCycles, first, second, cycles, blockCount, iterations);
    }
    throw std::runtime_error("invalid path");
}

template <Operation operation>
void reportRegister(size_t repetitions, int blockCount, size_t iterations) {
    size_t const count = size_t(blockCount) * 128;
    auto seeds = makeInputs<operation>(count);
    if constexpr (operation == Operation::Log || operation == Operation::Log2 ||
        operation == Operation::Sqrt || operation == Operation::Rsqrt) {
        float const smallestSeed = 2.0f * std::numeric_limits<float>::min();
        for (float& seed : seeds) {
            seed = std::max(seed, smallestSeed);
        }
    }
    std::vector<float> first(count);
    std::vector<float> second(count);
    std::vector<unsigned long long> cycles(count);
    float* deviceSeeds;
    float* deviceFirst;
    float* deviceSecond;
    unsigned long long* deviceCycles;
    cudaCheck(cudaMalloc(&deviceSeeds, count * sizeof(float)));
    cudaCheck(cudaMalloc(&deviceFirst, count * sizeof(float)));
    cudaCheck(cudaMalloc(&deviceSecond, count * sizeof(float)));
    cudaCheck(cudaMalloc(&deviceCycles, count * sizeof(unsigned long long)));
    cudaCheck(cudaMemcpy(deviceSeeds, seeds.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    constexpr size_t kPathCount = operation == Operation::Erfcexp ? 6 : 3;
    Path const paths[] = {Path::Custom, Path::Library, Path::LibraryAccurate, Path::Legacy,
        Path::LibraryFma, Path::LibraryDouble};
    std::vector<double> samples[kPathCount];
    double checksums[kPathCount]{};
    for (size_t index = 0; index < kPathCount; ++index) {
        (void) runRegisterPath<operation>(paths[index], deviceSeeds, deviceFirst, deviceSecond,
            deviceCycles, first, second, cycles, blockCount, iterations);
    }
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (size_t offset = 0; offset < kPathCount; ++offset) {
            size_t const index = (offset + repetition) % kPathCount;
            RegisterMeasurement const measurement =
                runRegisterPath<operation>(paths[index], deviceSeeds, deviceFirst, deviceSecond,
                    deviceCycles, first, second, cycles, blockCount, iterations);
            samples[index].push_back(measurement.cyclesPerEvaluation);
            checksums[index] = measurement.checksum;
        }
    }
    for (size_t index = 0; index < kPathCount; ++index) {
        TimingSpread const timing = timingSpread(samples[index]);
        std::printf("cuda_register_cycles\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.9g\n",
            operationName(operation), pathName(paths[index]), timing.minimum, timing.median,
            timing.maximum, checksums[index]);
    }
    cudaCheck(cudaFree(deviceCycles));
    cudaCheck(cudaFree(deviceSecond));
    cudaCheck(cudaFree(deviceFirst));
    cudaCheck(cudaFree(deviceSeeds));
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        int deviceCount = 0;
        cudaError_t const status = cudaGetDeviceCount(&deviceCount);
        if (status != cudaSuccess || deviceCount == 0) {
            std::fprintf(stderr, "CUDA device unavailable: %s\n", cudaGetErrorString(status));
            return 77;
        }
        cudaDeviceProp properties{};
        detail::cudaCheck(cudaGetDeviceProperties(&properties, 0));
        size_t const count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1U << 18;
        size_t const repetitions = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 7;
        int const requestedBlocks = argc > 3 ? std::atoi(argv[3]) : 1;
        size_t const launches = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 32;
        size_t const registerIterations = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 0;
        int const maximumBlocks = std::max(1, properties.multiProcessorCount / 4);
        int const blockCount = std::clamp(requestedBlocks, 1, maximumBlocks);
        if (count == 0 || repetitions == 0 || launches == 0) {
            throw std::runtime_error("count, repetitions, and launches must be positive");
        }
        if (registerIterations > 0) {
            std::printf("# CUDA register probe on %s, %d blocks of 128 threads, %zu "
                        "iterations/thread, %zu trials\n",
                properties.name, blockCount, registerIterations, repetitions);
            std::printf("# min/median/max across trials of each trial's minimum warp maximum "
                        "clock64 delta; cycles per dependent evaluation\n");
            std::printf("# first 32 seeds include domain boundaries; positive roots/log seeds stay "
                        "normal; no whole-kernel throughput claim\n");
            std::printf("backend\toperation\tpath\tmin_cycles_per_eval\tmedian_cycles_per_"
                        "eval\tmax_cycles_per_eval\tchecksum\n");
            detail::reportRegister<detail::Operation::Sin>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Cos>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Sincos>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Exp>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Exp2>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Log>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Log2>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Erf>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Erfc>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Erfcexp>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Sqrt>(
                repetitions, blockCount, registerIterations);
            detail::reportRegister<detail::Operation::Rsqrt>(
                repetitions, blockCount, registerIterations);
            return 0;
        }
        cudaEvent_t begin;
        cudaEvent_t end;
        detail::cudaCheck(cudaEventCreate(&begin));
        detail::cudaCheck(cudaEventCreate(&end));
        std::printf("# CUDA device %s, %d SMs, %d blocks of 128 threads (cap SM/4)\n",
            properties.name, properties.multiProcessorCount, blockCount);
        std::printf("# %zu values, %zu samples, %zu launches/sample, event ns/value; limited "
                    "occupancy with few blocks\n",
            count, repetitions, launches);
        std::printf("# phase [-4.5pi,4.5pi]; exp [-75,0]; erf signed; erfc/erfcexp [0,sqrt(75)]\n");
        std::printf("# erfcexp library is naive expf(-x*x); library_fma corrects square rounding; "
                    "library_double uses double exp\n");
        std::printf("backend\toperation\tpath\tmin_ns_per_value\tmedian_ns_per_value\tmax_ns_per_"
                    "value\tmax_ulps_sample\tfailures_8ulp_"
                    "sample\tchecksum\n");
        detail::report<detail::Operation::Sin>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Cos>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Sincos>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Exp>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Exp2>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Log>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Log2>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Erf>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Erfc>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Erfcexp>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Sqrt>(
            count, repetitions, blockCount, begin, end, launches);
        detail::report<detail::Operation::Rsqrt>(
            count, repetitions, blockCount, begin, end, launches);
        detail::cudaCheck(cudaEventDestroy(end));
        detail::cudaCheck(cudaEventDestroy(begin));
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "special-functions CUDA benchmark: %s\n", error.what());
        return 1;
    }
}
