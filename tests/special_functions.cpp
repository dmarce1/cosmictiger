#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <vector>

#include <cosmictiger/math/simd_backend.hpp>
#include <cosmictiger/math/special_functions.hpp>
#include "special_functions_reference.hpp"

namespace detail {

using special_functions_test::FmaInput;
using special_functions_test::PowInput;
using special_functions_test::UnaryFunction;

void checkSignedZero(char const* name, UnaryFunction function, float input, float actual) {
    if (std::bit_cast<uint32_t>(input) == 0x80000000U) {
        if (function == UnaryFunction::Sqrt) {
            special_functions_test::requireBits(
                name, input, actual, std::bit_cast<float>(uint32_t(0x80000000U)));
        } else if (function == UnaryFunction::Rsqrt) {
            special_functions_test::requireBits(
                name, input, actual, -std::numeric_limits<float>::infinity());
        }
    }
}

template <class Function>
size_t checkUnary(char const* name, UnaryFunction referenceFunction,
    std::vector<float> const& inputs, Function function) {
    for (float const input : inputs) {
        float const actual = function(input);
        special_functions_test::requireUlps(
            name, input, actual, special_functions_test::reference(referenceFunction, input));
        checkSignedZero(name, referenceFunction, input, actual);
    }
    for (size_t first = 0; first < inputs.size(); first += SimdFloat::size()) {
        SimdFloat const input(
            [&](auto lane) { return inputs[std::min(first + size_t(lane), inputs.size() - 1)]; });
        auto const actual = function(input);
        size_t const active = std::min(size_t(SimdFloat::size()), inputs.size() - first);
        for (size_t lane = 0; lane < active; ++lane) {
            float const value = inputs[first + lane];
            special_functions_test::requireUlps(name, value, actual[lane],
                special_functions_test::reference(referenceFunction, value));
            checkSignedZero(name, referenceFunction, value, actual[lane]);
        }
    }
    return 2 * inputs.size();
}

size_t checkSincos(std::vector<float> const& inputs) {
    for (float const input : inputs) {
        float sine;
        float cosine;
        special_functions::sincos(input, &sine, &cosine);
        special_functions_test::requireUlps("sincos sine", input, sine,
            special_functions_test::reference(UnaryFunction::Sin, input));
        special_functions_test::requireUlps("sincos cosine", input, cosine,
            special_functions_test::reference(UnaryFunction::Cos, input));
    }
    for (size_t first = 0; first < inputs.size(); first += SimdFloat::size()) {
        SimdFloat const input(
            [&](auto lane) { return inputs[std::min(first + size_t(lane), inputs.size() - 1)]; });
        SimdFloat sine;
        SimdFloat cosine;
        special_functions::sincos(input, &sine, &cosine);
        size_t const active = std::min(size_t(SimdFloat::size()), inputs.size() - first);
        for (size_t lane = 0; lane < active; ++lane) {
            float const value = inputs[first + lane];
            special_functions_test::requireUlps("SIMD sincos sine", value, sine[lane],
                special_functions_test::reference(UnaryFunction::Sin, value));
            special_functions_test::requireUlps("SIMD sincos cosine", value, cosine[lane],
                special_functions_test::reference(UnaryFunction::Cos, value));
        }
    }
    return 4 * inputs.size();
}

size_t checkErfcexp(std::vector<float> const& inputs) {
    for (float const input : inputs) {
        float complement;
        float exponential;
        special_functions::erfcexp(input, &complement, &exponential);
        special_functions_test::requireUlps("erfcexp erfc", input, complement,
            special_functions_test::reference(UnaryFunction::Erfc, input));
        special_functions_test::requireUlps("erfcexp exp(-x*x)", input, exponential,
            special_functions_test::exponentialSquareReference(input));
    }
    for (size_t first = 0; first < inputs.size(); first += SimdFloat::size()) {
        SimdFloat const input(
            [&](auto lane) { return inputs[std::min(first + size_t(lane), inputs.size() - 1)]; });
        SimdFloat complement;
        SimdFloat exponential;
        special_functions::erfcexp(input, &complement, &exponential);
        size_t const active = std::min(size_t(SimdFloat::size()), inputs.size() - first);
        for (size_t lane = 0; lane < active; ++lane) {
            float const value = inputs[first + lane];
            special_functions_test::requireUlps("SIMD erfcexp erfc", value, complement[lane],
                special_functions_test::reference(UnaryFunction::Erfc, value));
            special_functions_test::requireUlps("SIMD erfcexp exp(-x*x)", value, exponential[lane],
                special_functions_test::exponentialSquareReference(value));
        }
    }
    return 4 * inputs.size();
}

size_t checkPow(std::vector<PowInput> const& inputs) {
    for (PowInput const& input : inputs) {
        long double const reference =
            special_functions_test::powerReference(input.base, input.exponent);
        special_functions_test::requireUlps("pow", input.base, input.exponent,
            special_functions::pow(input.base, input.exponent), reference);
    }
    for (size_t first = 0; first < inputs.size(); first += SimdFloat::size()) {
        SimdFloat const base([&](auto lane) {
            return inputs[std::min(first + size_t(lane), inputs.size() - 1)].base;
        });
        SimdFloat const exponent([&](auto lane) {
            return inputs[std::min(first + size_t(lane), inputs.size() - 1)].exponent;
        });
        auto const actual = special_functions::pow(base, exponent);
        size_t const active = std::min(size_t(SimdFloat::size()), inputs.size() - first);
        for (size_t lane = 0; lane < active; ++lane) {
            PowInput const& input = inputs[first + lane];
            long double const reference =
                special_functions_test::powerReference(input.base, input.exponent);
            special_functions_test::requireUlps(
                "SIMD pow", input.base, input.exponent, actual[lane], reference);
        }
    }
    return 2 * inputs.size();
}

size_t checkFma(std::vector<FmaInput> const& inputs) {
    for (FmaInput const& input : inputs) {
        special_functions_test::requireUlps("fma", input.first, input.second,
            special_functions::fma(input.first, input.second, input.third),
            special_functions_test::fmaReference(input.first, input.second, input.third));
    }
    for (size_t first = 0; first < inputs.size(); first += SimdFloat::size()) {
        SimdFloat const a([&](auto lane) {
            return inputs[std::min(first + size_t(lane), inputs.size() - 1)].first;
        });
        SimdFloat const b([&](auto lane) {
            return inputs[std::min(first + size_t(lane), inputs.size() - 1)].second;
        });
        SimdFloat const c([&](auto lane) {
            return inputs[std::min(first + size_t(lane), inputs.size() - 1)].third;
        });
        auto const actual = special_functions::fma(a, b, c);
        size_t const active = std::min(size_t(SimdFloat::size()), inputs.size() - first);
        for (size_t lane = 0; lane < active; ++lane) {
            FmaInput const& input = inputs[first + lane];
            special_functions_test::requireUlps("SIMD fma", input.first, input.second, actual[lane],
                special_functions_test::fmaReference(input.first, input.second, input.third));
        }
    }
    return 2 * inputs.size();
}

}    // namespace detail

int main() {
    try {
        auto const phaseInputs = special_functions_test::phaseInputs();
        auto const exponentialInputs = special_functions_test::exponentialInputs(false);
        auto const exponentialBaseTwoInputs = special_functions_test::exponentialInputs(true);
        auto const logarithmInputs = special_functions_test::logarithmInputs();
        auto const errorInputs = special_functions_test::errorInputs();
        auto const squareRootInputs = special_functions_test::squareRootInputs();
        auto const powerInputs = special_functions_test::powerInputs();
        auto const fusedMultiplyAddInputs = special_functions_test::fusedMultiplyAddInputs();

        size_t checks = 0;
        checks += detail::checkUnary("sin", special_functions_test::UnaryFunction::Sin, phaseInputs,
            [](auto input) { return special_functions::sin(input); });
        checks += detail::checkUnary("cos", special_functions_test::UnaryFunction::Cos, phaseInputs,
            [](auto input) { return special_functions::cos(input); });
        checks += detail::checkSincos(phaseInputs);
        checks += detail::checkUnary("exp", special_functions_test::UnaryFunction::Exp,
            exponentialInputs, [](auto input) { return special_functions::exp(input); });
        checks += detail::checkUnary("exp2", special_functions_test::UnaryFunction::Exp2,
            exponentialBaseTwoInputs, [](auto input) { return special_functions::exp2(input); });
        checks += detail::checkUnary("log", special_functions_test::UnaryFunction::Log,
            logarithmInputs, [](auto input) { return special_functions::log(input); });
        checks += detail::checkUnary("log2", special_functions_test::UnaryFunction::Log2,
            logarithmInputs, [](auto input) { return special_functions::log2(input); });
        checks += detail::checkUnary("erf", special_functions_test::UnaryFunction::Erf, errorInputs,
            [](auto input) { return special_functions::erf(input); });
        checks += detail::checkUnary("erfc", special_functions_test::UnaryFunction::Erfc,
            errorInputs, [](auto input) { return special_functions::erfc(input); });
        checks += detail::checkErfcexp(errorInputs);
        checks += detail::checkPow(powerInputs);
        checks += detail::checkUnary("sqrt", special_functions_test::UnaryFunction::Sqrt,
            squareRootInputs, [](auto input) { return special_functions::sqrt(input); });
        checks += detail::checkUnary("rsqrt", special_functions_test::UnaryFunction::Rsqrt,
            squareRootInputs, [](auto input) { return special_functions::rsqrt(input); });
        checks += detail::checkFma(fusedMultiplyAddInputs);

        std::printf(
            "special function scalar and SIMD reference checks passed: %zu values\n", checks);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "special function reference test failed: %s\n", error.what());
        return 1;
    }
}
