#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace special_functions_test {

inline constexpr long double kPi = 3.141592653589793238462643383279502884L;
inline constexpr uint64_t kMaximumUlps = 8;
inline constexpr size_t kRandomSamples = 65536;

enum class UnaryFunction { Sin, Cos, Exp, Exp2, Log, Log2, Erf, Erfc, Sqrt, Rsqrt };

struct PowInput {
    float base;
    float exponent;
};

struct FmaInput {
    float first;
    float second;
    float third;
};

inline uint32_t orderedBits(float value) {
    uint32_t const bits = std::bit_cast<uint32_t>(value);
    return (bits & 0x80000000U) ? ~bits : bits ^ 0x80000000U;
}

inline uint64_t ulpDistance(float left, float right) {
    uint32_t const a = orderedBits(left);
    uint32_t const b = orderedBits(right);
    return a > b ? uint64_t(a - b) : uint64_t(b - a);
}

inline void requireUlps(char const* name, float input, float actual, long double reference,
    uint64_t maximumUlps = kMaximumUlps) {
    float const expected = static_cast<float>(reference);
    uint32_t const actualBits = std::bit_cast<uint32_t>(actual);
    uint32_t const expectedBits = std::bit_cast<uint32_t>(expected);
    // Inspect bits so this guard remains effective under -ffast-math.
    bool const actualSpecial = (actualBits & 0x7f800000U) == 0x7f800000U;
    bool const expectedSpecial = (expectedBits & 0x7f800000U) == 0x7f800000U;
    if (expectedSpecial) {
        bool const expectedNan = (expectedBits & 0x007fffffU) != 0;
        bool const actualNan = (actualBits & 0x007fffffU) != 0;
        if ((expectedNan && actualSpecial && actualNan) ||
            (!expectedNan && actualBits == expectedBits)) {
            return;
        }
    }
    uint64_t const distance = ulpDistance(actual, expected);
    if (actualSpecial || expectedSpecial || distance > maximumUlps) {
        std::fprintf(stderr,
            "%s input=%a actual=%a expected=%a reference=%.21Lg ulps=%llu limit=%llu\n", name,
            double(input), double(actual), double(expected), reference,
            static_cast<unsigned long long>(distance),
            static_cast<unsigned long long>(maximumUlps));
        throw std::runtime_error(name);
    }
}

inline void requireUlps(char const* name, float first, float second, float actual,
    long double reference, uint64_t maximumUlps = kMaximumUlps) {
    try {
        requireUlps(name, first, actual, reference, maximumUlps);
    } catch (std::runtime_error const&) {
        std::fprintf(stderr, "second input=%a\n", double(second));
        throw;
    }
}

inline void requireBits(char const* name, float input, float actual, float expected) {
    if (std::bit_cast<uint32_t>(actual) != std::bit_cast<uint32_t>(expected)) {
        std::fprintf(stderr, "%s input=%a actual=%a expected=%a (exact bits required)\n", name,
            double(input), double(actual), double(expected));
        throw std::runtime_error(name);
    }
}

long double reference(UnaryFunction function, float input);
long double exponentialSquareReference(float input);
long double powerReference(float base, float exponent);
long double fmaReference(float first, float second, float third);

inline void appendBoundary(std::vector<float>& values, float value) {
    values.push_back(std::nextafter(value, -std::numeric_limits<float>::infinity()));
    values.push_back(value);
    values.push_back(std::nextafter(value, std::numeric_limits<float>::infinity()));
}

inline std::vector<float> phaseInputs() {
    std::vector<float> values;
    values.reserve(kRandomSamples + 1024);
    appendBoundary(values, 0.0f);
    for (int power = -32; power <= -8; power += 4) {
        float const small = std::ldexp(1.0f, power);
        appendBoundary(values, small);
        appendBoundary(values, -small);
    }
    for (int halfPi = -9; halfPi <= 9; ++halfPi) {
        appendBoundary(values, static_cast<float>(halfPi * kPi / 2));
    }
    std::mt19937 generator(0x54726967U);
    std::uniform_real_distribution<double> phase(-4.5 * double(kPi), 4.5 * double(kPi));
    for (size_t sample = 0; sample < kRandomSamples; ++sample) {
        values.push_back(static_cast<float>(phase(generator)));
    }
    return values;
}

inline std::vector<float> exponentialInputs(bool baseTwo) {
    std::vector<float> values;
    values.reserve(kRandomSamples + 1024);
    if (baseTwo) {
        for (int exponent : {-110, -108, -100, -75, -32, -1, 0, 1, 32, 100, 120, 125}) {
            appendBoundary(values, float(exponent));
        }
    } else {
        for (float exponent : {-75.0f, -27.04f, -20.0f, -3.0f, -1.0f, 0.0f, 1.0f, 20.0f, 85.0f}) {
            appendBoundary(values, exponent);
        }
    }
    std::mt19937 generator(0x45617078U);
    std::uniform_real_distribution<double> exponent(
        baseTwo ? -110.0 : -75.0, baseTwo ? 125.0 : 85.0);
    for (size_t sample = 0; sample < kRandomSamples; ++sample) {
        values.push_back(static_cast<float>(exponent(generator)));
    }
    return values;
}

inline std::vector<float> logarithmInputs() {
    std::vector<float> values;
    values.reserve(kRandomSamples + 1024);
    for (int exponent = -120; exponent <= 120; exponent += 8) {
        appendBoundary(values, std::ldexp(1.0f, exponent));
    }
    values.push_back(std::numeric_limits<float>::min());
    values.push_back(std::numeric_limits<float>::max());
    appendBoundary(values, 1.0f);
    std::mt19937 generator(0x4c6f6732U);
    std::uniform_int_distribution<int> exponent(-120, 120);
    std::uniform_real_distribution<double> mantissa(0.5, 1.0);
    for (size_t sample = 0; sample < kRandomSamples; ++sample) {
        values.push_back(static_cast<float>(std::ldexp(mantissa(generator), exponent(generator))));
    }
    return values;
}

inline std::vector<float> squareRootInputs() {
    auto values = logarithmInputs();
    values.push_back(0.0f);
    values.push_back(std::bit_cast<float>(uint32_t(0x80000000U)));
    values.push_back(std::numeric_limits<float>::infinity());
    values.push_back(-std::numeric_limits<float>::infinity());
    return values;
}

inline std::vector<float> errorInputs() {
    std::vector<float> values;
    values.reserve(kRandomSamples + 1024);
    // Adjacent lanes include both sides of the scalar Ewald cutoff. The
    // generated SIMD Ewald branch evaluates every lane when any lane qualifies.
    float const maximumEwaldArgument = static_cast<float>(std::sqrt(75.0L));
    for (float input : {1.0f, maximumEwaldArgument, 2.0f, 8.0f, 5.2f, 0.0f, 3.0f,
             -maximumEwaldArgument, 1.5f, 7.0f, 4.0f, 6.0f, 0.5f, 5.5f, -1.0f, 8.5f}) {
        values.push_back(input);
    }
    for (float input :
        {0.0f, 0.01f, 0.125f, 0.5f, 1.0f, 2.0f, 3.0f, 5.2f, 8.0f, maximumEwaldArgument}) {
        appendBoundary(values, input);
        appendBoundary(values, -input);
    }
    for (int power = -32; power <= -8; power += 4) {
        float const small = std::ldexp(1.0f, power);
        appendBoundary(values, small);
        appendBoundary(values, -small);
    }
    std::mt19937 generator(0x45726643U);
    std::uniform_real_distribution<double> input(-std::sqrt(75.0), std::sqrt(75.0));
    for (size_t sample = 0; sample < kRandomSamples; ++sample) {
        values.push_back(static_cast<float>(input(generator)));
    }
    return values;
}

inline std::vector<PowInput> powerInputs() {
    std::vector<PowInput> values;
    values.reserve(kRandomSamples + 1024);
    for (float base :
        {0.125f, 0.5f, std::nextafter(1.0f, 0.0f), 1.0f, std::nextafter(1.0f, 2.0f), 2.0f, 8.0f}) {
        for (float exponent : {-6.0f, -3.5f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f}) {
            values.push_back({base, exponent});
        }
    }
    std::mt19937 generator(0x506f7732U);
    std::uniform_real_distribution<double> logBase(-8.0, 8.0);
    std::uniform_real_distribution<double> exponent(-6.0, 6.0);
    for (size_t sample = 0; sample < kRandomSamples; ++sample) {
        values.push_back({static_cast<float>(std::exp2(logBase(generator))),
            static_cast<float>(exponent(generator))});
    }
    return values;
}

inline std::vector<FmaInput> fusedMultiplyAddInputs() {
    std::vector<FmaInput> values;
    values.reserve(kRandomSamples + 1024);
    values.push_back({0.0f, 0.0f, 0.0f});
    values.push_back({1.0f + 0x1p-12f, 1.0f - 0x1p-12f, -1.0f});
    values.push_back({1.0f - 0x1p-12f, 1.0f + 0x1p-12f, -1.0f});
    values.push_back(
        {std::numeric_limits<float>::max(), 0.5f, -std::numeric_limits<float>::max() * 0.5f});
    std::mt19937 generator(0x466d6143U);
    std::uniform_int_distribution<int> exponent(-32, 32);
    std::uniform_real_distribution<double> mantissa(0.5, 1.0);
    std::uniform_int_distribution<int> offset(-1, 1);
    for (size_t sample = 0; sample < kRandomSamples; ++sample) {
        float const first =
            static_cast<float>(std::ldexp(mantissa(generator), exponent(generator)));
        float const second =
            static_cast<float>(std::ldexp(mantissa(generator), exponent(generator)));
        float const roundedProduct = static_cast<float>(double(first) * double(second));
        float third = -roundedProduct;
        int const adjustment = offset(generator);
        if (adjustment > 0) {
            third = std::nextafter(third, std::numeric_limits<float>::infinity());
        } else if (adjustment < 0) {
            third = std::nextafter(third, -std::numeric_limits<float>::infinity());
        }
        values.push_back({first, second, third});
    }
    return values;
}

}    // namespace special_functions_test
