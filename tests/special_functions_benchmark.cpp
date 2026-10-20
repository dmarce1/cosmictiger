#include <algorithm>
#include <array>
#include <chrono>
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
#include <type_traits>
#include <vector>

#include <cosmictiger/math/simd.hpp>
#include <cosmictiger/math/simd_backend.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/special_functions.hpp>
#include "special_functions_reference.hpp"

namespace detail {

enum class Operation { Sin, Cos, Sincos, Exp, Exp2, Log, Log2, Erf, Erfc, Erfcexp, Sqrt, Rsqrt };

enum class Path { Custom, Legacy, Library, LibraryFma, LibraryDouble };

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
    case Path::LibraryDouble:
        return "library_double";
    }
    return "unknown";
}

template <class Value, class Function>
Value mapLanes(Value input, Function function) {
    if constexpr (std::is_same_v<Value, float>) {
        return function(input);
    } else {
        return Value([&](auto lane) { return function(float(input[lane])); });
    }
}

template <Operation operation, Path path, class Value>
[[gnu::always_inline]] inline Value evaluate(Value input, Value* second) {
    if constexpr (path == Path::Custom) {
        if constexpr (operation == Operation::Sin) {
            return special_functions::sin(input);
        } else if constexpr (operation == Operation::Cos) {
            return special_functions::cos(input);
        } else if constexpr (operation == Operation::Sincos) {
            Value first;
            special_functions::sincos(input, &first, second);
            return first;
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
            Value first;
            special_functions::erfcexp(input, &first, second);
            return first;
        } else if constexpr (operation == Operation::Sqrt) {
            return special_functions::sqrt(input);
        } else {
            return special_functions::rsqrt(input);
        }
    } else if constexpr (path == Path::Legacy) {
        static_assert(std::is_same_v<Value, SimdFloat>);
        if constexpr (operation == Operation::Sin) {
            return ::sin(input);
        } else if constexpr (operation == Operation::Cos) {
            return ::cos(input);
        } else if constexpr (operation == Operation::Sincos) {
            Value first;
            ::sincos(input, &first, second);
            return first;
        } else if constexpr (operation == Operation::Exp) {
            return ::exp(input);
        } else if constexpr (operation == Operation::Exp2) {
            return ::twoPow(input);
        } else if constexpr (operation == Operation::Log) {
            return ::log(input);
        } else if constexpr (operation == Operation::Log2) {
            return ::log2(input);
        } else if constexpr (operation == Operation::Erf) {
            return ::erf(input);
        } else if constexpr (operation == Operation::Erfc) {
            return ::erfc(input);
        } else if constexpr (operation == Operation::Erfcexp) {
            Value first;
            ::erfcexp(input, &first, second);
            return first;
        } else if constexpr (operation == Operation::Sqrt) {
            return ::simdSqrt(input);
        } else {
            return ::simdRsqrt(input);
        }
    } else if constexpr (std::is_same_v<Value, float>) {
        if constexpr (operation == Operation::Sin) {
            return std::sin(input);
        } else if constexpr (operation == Operation::Cos) {
            return std::cos(input);
        } else if constexpr (operation == Operation::Sincos) {
            float sine;
            float cosine;
            ::sincosf(input, &sine, &cosine);
            *second = cosine;
            return sine;
        } else if constexpr (operation == Operation::Exp) {
            return std::exp(input);
        } else if constexpr (operation == Operation::Exp2) {
            return std::exp2(input);
        } else if constexpr (operation == Operation::Log) {
            return std::log(input);
        } else if constexpr (operation == Operation::Log2) {
            return std::log2(input);
        } else if constexpr (operation == Operation::Erf) {
            return std::erf(input);
        } else if constexpr (operation == Operation::Erfc) {
            return std::erfc(input);
        } else if constexpr (operation == Operation::Erfcexp) {
            float const square = input * input;
            float const gaussian = path == Path::LibraryDouble ?
                float(std::exp(-double(input) * double(input))) :
                std::exp(-square);
            *second = path == Path::LibraryFma ?
                std::fma(std::fma(-input, input, square), gaussian, gaussian) :
                gaussian;
            return std::erfc(input);
        } else if constexpr (operation == Operation::Sqrt) {
            return std::sqrt(input);
        } else {
            return 1.0f / std::sqrt(input);
        }
    } else {
#if COSMICTIGER_SIMD_STANDARD
        // Early C++26 SIMD implementations omit optional vector math functions.
        if constexpr (operation == Operation::Sin) {
            return mapLanes(input, [](float value) { return std::sin(value); });
        } else if constexpr (operation == Operation::Cos) {
            return mapLanes(input, [](float value) { return std::cos(value); });
        } else if constexpr (operation == Operation::Sincos) {
            *second = mapLanes(input, [](float value) { return std::cos(value); });
            return mapLanes(input, [](float value) { return std::sin(value); });
        } else if constexpr (operation == Operation::Exp) {
            return mapLanes(input, [](float value) { return std::exp(value); });
        } else if constexpr (operation == Operation::Exp2) {
            return mapLanes(input, [](float value) { return std::exp2(value); });
        } else if constexpr (operation == Operation::Log) {
            return mapLanes(input, [](float value) { return std::log(value); });
        } else if constexpr (operation == Operation::Log2) {
            return mapLanes(input, [](float value) { return std::log2(value); });
        } else if constexpr (operation == Operation::Erf) {
            return mapLanes(input, [](float value) { return std::erf(value); });
        } else if constexpr (operation == Operation::Erfc) {
            return mapLanes(input, [](float value) { return std::erfc(value); });
        } else if constexpr (operation == Operation::Erfcexp) {
            Value const square = input * input;
            *second = mapLanes(
                input, [](float value) { return float(std::exp(-double(value) * double(value))); });
            if constexpr (path != Path::LibraryDouble) {
                *second = mapLanes(input, [&](float value) {
                    float const product = value * value;
                    float const gaussian = std::exp(-product);
                    return path == Path::LibraryFma ?
                        std::fma(std::fma(-value, value, product), gaussian, gaussian) :
                        gaussian;
                });
            }
            return mapLanes(input, [](float value) { return std::erfc(value); });
        } else if constexpr (operation == Operation::Sqrt) {
            return mapLanes(input, [](float value) { return std::sqrt(value); });
        } else {
            return Value(1.0f) / mapLanes(input, [](float value) { return std::sqrt(value); });
        }
#else
        if constexpr (operation == Operation::Sin) {
            return std::experimental::sin(input);
        } else if constexpr (operation == Operation::Cos) {
            return std::experimental::cos(input);
        } else if constexpr (operation == Operation::Sincos) {
            *second = std::experimental::cos(input);
            return std::experimental::sin(input);
        } else if constexpr (operation == Operation::Exp) {
            return std::experimental::exp(input);
        } else if constexpr (operation == Operation::Exp2) {
            return std::experimental::exp2(input);
        } else if constexpr (operation == Operation::Log) {
            return std::experimental::log(input);
        } else if constexpr (operation == Operation::Log2) {
            return std::experimental::log2(input);
        } else if constexpr (operation == Operation::Erf) {
            return std::experimental::erf(input);
        } else if constexpr (operation == Operation::Erfc) {
            return std::experimental::erfc(input);
        } else if constexpr (operation == Operation::Erfcexp) {
            Value const square = input * input;
            Value gaussian = path == Path::LibraryDouble ?
                mapLanes(input,
                    [](float value) { return float(std::exp(-double(value) * double(value))); }) :
                std::experimental::exp(-square);
            if constexpr (path == Path::LibraryFma) {
                gaussian = simdFma(simdFma(-input, input, square), gaussian, gaussian);
            }
            *second = gaussian;
            return std::experimental::erfc(input);
        } else if constexpr (operation == Operation::Sqrt) {
            return std::experimental::sqrt(input);
        } else {
            return Value(1.0f) / std::experimental::sqrt(input);
        }
#endif
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
        float const boundaries[] = {0.0f, 0.75f, 1.0f, 2.0f, 3.0f, 5.2f, 8.0f, 8.660254f, -0.75f,
            -1.0f, -2.0f, -5.2f, -8.660254f};
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
            for (size_t index = 0; index < std::size(boundaries); ++index) {
                inputs[index] = boundaries[index];
            }
        } else if constexpr (operation == Operation::Erfc || operation == Operation::Erfcexp) {
            for (size_t index = 0; index < 8; ++index) {
                inputs[index] = boundaries[index];
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

struct Measurement {
    double nanoseconds;
    double checksum;
};

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

template <Operation operation, Path path, bool packed>
double run(
    std::vector<float> const& inputs, std::vector<float>& first, std::vector<float>& second) {
    auto const begin = std::chrono::steady_clock::now();
    if constexpr (packed) {
        for (size_t index = 0; index < inputs.size(); index += SimdFloat::size()) {
            SimdFloat const input([&](auto lane) { return inputs[index + size_t(lane)]; });
            SimdFloat extra;
            SimdFloat const output = evaluate<operation, path>(input, &extra);
            for (size_t lane = 0; lane < SimdFloat::size(); ++lane) {
                first[index + lane] = output[lane];
                if constexpr (operation == Operation::Sincos || operation == Operation::Erfcexp) {
                    second[index + lane] = extra[lane];
                }
            }
        }
    } else {
        for (size_t index = 0; index < inputs.size(); ++index) {
            first[index] = evaluate<operation, path>(inputs[index], &second[index]);
        }
    }
    auto const end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(end - begin).count() / double(inputs.size());
}

template <Operation operation, bool packed>
Measurement runPath(Path path, std::vector<float> const& inputs, std::vector<float>& first,
    std::vector<float>& second) {
    double nanoseconds;
    switch (path) {
    case Path::Custom:
        nanoseconds = run<operation, Path::Custom, packed>(inputs, first, second);
        break;
    case Path::Legacy:
        if constexpr (packed) {
            nanoseconds = run<operation, Path::Legacy, packed>(inputs, first, second);
        } else {
            throw std::runtime_error("legacy path is SIMD-only");
        }
        break;
    case Path::Library:
        nanoseconds = run<operation, Path::Library, packed>(inputs, first, second);
        break;
    case Path::LibraryFma:
        nanoseconds = run<operation, Path::LibraryFma, packed>(inputs, first, second);
        break;
    case Path::LibraryDouble:
        nanoseconds = run<operation, Path::LibraryDouble, packed>(inputs, first, second);
        break;
    }
    return {nanoseconds,
        outputChecksum(
            first, second, operation == Operation::Sincos || operation == Operation::Erfcexp)};
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <Operation operation, bool packed>
void report(size_t count, size_t repetitions) {
    auto const inputs = makeInputs<operation>(count);
    std::vector<float> first(count);
    std::vector<float> second(count);
    constexpr size_t kPathCount =
        2 + size_t(packed) + (operation == Operation::Erfcexp ? size_t(2) : size_t(0));
    std::array<Path, kPathCount> paths;
    size_t nextPath = 0;
    paths[nextPath++] = Path::Custom;
    if constexpr (packed) {
        paths[nextPath++] = Path::Legacy;
    }
    paths[nextPath++] = Path::Library;
    if constexpr (operation == Operation::Erfcexp) {
        paths[nextPath++] = Path::LibraryFma;
        paths[nextPath++] = Path::LibraryDouble;
    }
    std::vector<double> samples[kPathCount];
    double checksums[kPathCount]{};
    for (size_t index = 0; index < kPathCount; ++index) {
        (void) runPath<operation, packed>(paths[index], inputs, first, second);
    }
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (size_t offset = 0; offset < kPathCount; ++offset) {
            size_t const index = (offset + repetition) % kPathCount;
            Measurement const measurement =
                runPath<operation, packed>(paths[index], inputs, first, second);
            samples[index].push_back(measurement.nanoseconds);
            checksums[index] = measurement.checksum;
        }
    }
    for (size_t index = 0; index < kPathCount; ++index) {
        (void) runPath<operation, packed>(paths[index], inputs, first, second);
        Accuracy const accuracy = assessAccuracy<operation>(inputs, first, second);
        std::printf("%s\t%s\t%s\t%.3f\t%llu\t%zu\t%.9g\n", packed ? "simd8" : "scalar",
            operationName(operation), pathName(paths[index]), median(samples[index]),
            static_cast<unsigned long long>(accuracy.maximumUlps), accuracy.failures,
            checksums[index]);
    }
}

template <Operation operation>
void reportBoth(size_t count, size_t repetitions) {
    report<operation, false>(count, repetitions);
    report<operation, true>(count, repetitions);
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        size_t const count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1U << 17;
        size_t const repetitions = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 7;
        if (count == 0 || count % SimdFloat::size() != 0 || repetitions == 0) {
            throw std::runtime_error(
                "count must be a positive multiple of eight; repeats must be positive");
        }
        std::printf("# CPU independent arrays, %zu values, %zu samples, median ns/value\n", count,
            repetitions);
        std::printf("# phase [-4.5pi,4.5pi]; exp [-75,0]; erf signed; erfc/erfcexp [0,sqrt(75)]\n");
        std::printf("# scalar_source loops may auto-vectorize, especially with fast-math\n");
        std::printf("# custom = new candidate; legacy = original project SIMD implementation\n");
#if COSMICTIGER_SIMD_STANDARD
        std::printf("# simd8 library path uses lane-wise standard math\n");
#else
        std::printf("# simd8 library path uses std::experimental SIMD math; sincos uses sin+cos\n");
#endif
        std::printf("# erfcexp library is naive expf(-x*x); library_fma corrects square rounding; "
                    "library_double uses double exp\n");
        std::printf("backend\toperation\tpath\tns_per_value\tmax_ulps_sample\tfailures_8ulp_"
                    "sample\tchecksum\n");
        detail::reportBoth<detail::Operation::Sin>(count, repetitions);
        detail::reportBoth<detail::Operation::Cos>(count, repetitions);
        detail::reportBoth<detail::Operation::Sincos>(count, repetitions);
        detail::reportBoth<detail::Operation::Exp>(count, repetitions);
        detail::reportBoth<detail::Operation::Exp2>(count, repetitions);
        detail::reportBoth<detail::Operation::Log>(count, repetitions);
        detail::reportBoth<detail::Operation::Log2>(count, repetitions);
        detail::reportBoth<detail::Operation::Erf>(count, repetitions);
        detail::reportBoth<detail::Operation::Erfc>(count, repetitions);
        detail::reportBoth<detail::Operation::Erfcexp>(count, repetitions);
        detail::reportBoth<detail::Operation::Sqrt>(count, repetitions);
        detail::reportBoth<detail::Operation::Rsqrt>(count, repetitions);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "special-functions CPU benchmark: %s\n", error.what());
        return 1;
    }
}
