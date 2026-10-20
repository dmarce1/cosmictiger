#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/math/special_functions.hpp>
#include "special_functions_reference.hpp"

namespace detail {

enum class Operation : uint8_t {
    Sin,
    Cos,
    Sincos,
    Exp,
    Exp2,
    Log,
    Log2,
    Erf,
    Erfc,
    Erfcexp,
    Pow,
    Sqrt,
    Rsqrt,
    Fma
};

struct TestCase {
    Operation operation;
    float first;
    float second;
    float third;
};

struct TestResult {
    float first;
    float second;
};

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void evaluateCases(TestCase const* cases, TestResult* results, size_t count) {
    for (size_t index = threadIdx.x; index < count; index += blockDim.x) {
        TestCase const test = cases[index];
        TestResult result{};
        switch (test.operation) {
        case Operation::Sin:
            result.first = special_functions::sin(test.first);
            break;
        case Operation::Cos:
            result.first = special_functions::cos(test.first);
            break;
        case Operation::Sincos:
            special_functions::sincos(test.first, &result.first, &result.second);
            break;
        case Operation::Exp:
            result.first = special_functions::exp(test.first);
            break;
        case Operation::Exp2:
            result.first = special_functions::exp2(test.first);
            break;
        case Operation::Log:
            result.first = special_functions::log(test.first);
            break;
        case Operation::Log2:
            result.first = special_functions::log2(test.first);
            break;
        case Operation::Erf:
            result.first = special_functions::erf(test.first);
            break;
        case Operation::Erfc:
            result.first = special_functions::erfc(test.first);
            break;
        case Operation::Erfcexp:
            special_functions::erfcexp(test.first, &result.first, &result.second);
            break;
        case Operation::Pow:
            result.first = special_functions::pow(test.first, test.second);
            break;
        case Operation::Sqrt:
            result.first = special_functions::sqrt(test.first);
            break;
        case Operation::Rsqrt:
            result.first = special_functions::rsqrt(test.first);
            break;
        case Operation::Fma:
            result.first = special_functions::fma(test.first, test.second, test.third);
            break;
        }
        results[index] = result;
    }
}

void appendUnaryCases(
    std::vector<TestCase>& cases, Operation operation, std::vector<float> const& inputs) {
    for (float const input : inputs) {
        cases.push_back({operation, input, 0.0f, 0.0f});
    }
}

std::vector<TestCase> makeCases() {
    std::vector<TestCase> cases;
    auto const phases = special_functions_test::phaseInputs();
    auto const exponentials = special_functions_test::exponentialInputs(false);
    auto const powersOfTwo = special_functions_test::exponentialInputs(true);
    auto const logarithms = special_functions_test::logarithmInputs();
    auto const errors = special_functions_test::errorInputs();
    auto const squareRoots = special_functions_test::squareRootInputs();
    auto const powers = special_functions_test::powerInputs();
    auto const fusedMultiplyAdds = special_functions_test::fusedMultiplyAddInputs();
    cases.reserve(3 * phases.size() + exponentials.size() + powersOfTwo.size() +
        2 * logarithms.size() + 3 * errors.size() + 2 * squareRoots.size() + powers.size() +
        fusedMultiplyAdds.size());
    appendUnaryCases(cases, Operation::Sin, phases);
    appendUnaryCases(cases, Operation::Cos, phases);
    appendUnaryCases(cases, Operation::Sincos, phases);
    appendUnaryCases(cases, Operation::Exp, exponentials);
    appendUnaryCases(cases, Operation::Exp2, powersOfTwo);
    appendUnaryCases(cases, Operation::Log, logarithms);
    appendUnaryCases(cases, Operation::Log2, logarithms);
    appendUnaryCases(cases, Operation::Erf, errors);
    appendUnaryCases(cases, Operation::Erfc, errors);
    appendUnaryCases(cases, Operation::Erfcexp, errors);
    appendUnaryCases(cases, Operation::Sqrt, squareRoots);
    appendUnaryCases(cases, Operation::Rsqrt, squareRoots);
    for (auto const& power : powers) {
        cases.push_back({Operation::Pow, power.base, power.exponent, 0.0f});
    }
    for (auto const& fma : fusedMultiplyAdds) {
        cases.push_back({Operation::Fma, fma.first, fma.second, fma.third});
    }
    return cases;
}

size_t checkResults(std::vector<TestCase> const& cases, std::vector<TestResult> const& results) {
    size_t checks = 0;
    for (size_t index = 0; index < cases.size(); ++index) {
        TestCase const& test = cases[index];
        TestResult const& actual = results[index];
        using special_functions_test::UnaryFunction;
        switch (test.operation) {
        case Operation::Sin:
            special_functions_test::requireUlps("CUDA sin", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Sin, test.first));
            break;
        case Operation::Cos:
            special_functions_test::requireUlps("CUDA cos", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Cos, test.first));
            break;
        case Operation::Sincos:
            special_functions_test::requireUlps("CUDA sincos sine", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Sin, test.first));
            special_functions_test::requireUlps("CUDA sincos cosine", test.first, actual.second,
                special_functions_test::reference(UnaryFunction::Cos, test.first));
            ++checks;
            break;
        case Operation::Exp:
            special_functions_test::requireUlps("CUDA exp", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Exp, test.first));
            break;
        case Operation::Exp2:
            special_functions_test::requireUlps("CUDA exp2", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Exp2, test.first));
            break;
        case Operation::Log:
            special_functions_test::requireUlps("CUDA log", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Log, test.first));
            break;
        case Operation::Log2:
            special_functions_test::requireUlps("CUDA log2", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Log2, test.first));
            break;
        case Operation::Erf:
            special_functions_test::requireUlps("CUDA erf", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Erf, test.first));
            break;
        case Operation::Erfc:
            special_functions_test::requireUlps("CUDA erfc", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Erfc, test.first));
            break;
        case Operation::Erfcexp:
            special_functions_test::requireUlps("CUDA erfcexp erfc", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Erfc, test.first));
            special_functions_test::requireUlps("CUDA erfcexp exp(-x*x)", test.first, actual.second,
                special_functions_test::exponentialSquareReference(test.first));
            ++checks;
            break;
        case Operation::Pow:
            special_functions_test::requireUlps("CUDA pow", test.first, test.second, actual.first,
                special_functions_test::powerReference(test.first, test.second));
            break;
        case Operation::Sqrt:
            special_functions_test::requireUlps("CUDA sqrt", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Sqrt, test.first));
            if (std::bit_cast<uint32_t>(test.first) == 0x80000000U) {
                special_functions_test::requireBits(
                    "CUDA sqrt negative zero", test.first, actual.first, -0.0f);
            }
            break;
        case Operation::Rsqrt:
            special_functions_test::requireUlps("CUDA rsqrt", test.first, actual.first,
                special_functions_test::reference(UnaryFunction::Rsqrt, test.first));
            if (std::bit_cast<uint32_t>(test.first) == 0x80000000U) {
                special_functions_test::requireBits("CUDA rsqrt negative zero", test.first,
                    actual.first, -std::numeric_limits<float>::infinity());
            }
            break;
        case Operation::Fma:
            special_functions_test::requireUlps("CUDA fma", test.first, test.second, actual.first,
                special_functions_test::fmaReference(test.first, test.second, test.third));
            break;
        }
        ++checks;
    }
    return checks;
}

}    // namespace detail

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::puts("CUDA device unavailable; skipping special functions reference test");
        return 77;
    }
    detail::TestCase* deviceCases = nullptr;
    detail::TestResult* deviceResults = nullptr;
    try {
        auto const cases = detail::makeCases();
        std::vector<detail::TestResult> results(cases.size());
        detail::cudaCheck(cudaMalloc(&deviceCases, cases.size() * sizeof(detail::TestCase)));
        detail::cudaCheck(cudaMalloc(&deviceResults, results.size() * sizeof(detail::TestResult)));
        detail::cudaCheck(cudaMemcpy(deviceCases, cases.data(),
            cases.size() * sizeof(detail::TestCase), cudaMemcpyHostToDevice));
        detail::evaluateCases<<<1, 128>>>(deviceCases, deviceResults, cases.size());
        detail::cudaCheck(cudaGetLastError());
        detail::cudaCheck(cudaDeviceSynchronize());
        detail::cudaCheck(cudaMemcpy(results.data(), deviceResults,
            results.size() * sizeof(detail::TestResult), cudaMemcpyDeviceToHost));
        detail::cudaCheck(cudaFree(deviceResults));
        deviceResults = nullptr;
        detail::cudaCheck(cudaFree(deviceCases));
        deviceCases = nullptr;
        size_t const checks = detail::checkResults(cases, results);
        std::printf("CUDA special function reference checks passed: %zu values\n", checks);
        return 0;
    } catch (std::exception const& error) {
        if (deviceResults) {
            cudaFree(deviceResults);
        }
        if (deviceCases) {
            cudaFree(deviceCases);
        }
        std::fprintf(stderr, "CUDA special function reference test failed: %s\n", error.what());
        return 1;
    }
}
