/* CosmicTiger - small fixed matrix algorithm benchmark; BSL-1.0. */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>

#include <cosmictiger/math/matrix.hpp>

namespace detail {
enum class Operation { Inverse, Determinant, Solve };

// Keep the complete result observable without timing a checksum reduction.
template <class T>
inline void consume(T const& value) {
    asm volatile("" : : "g"(&value) : "memory");
}

template <Operation operation, bool explicitAlgorithm, class A>
[[gnu::noinline]] void evaluate(
    A const& a, math::Vector<typename A::ValueType, A::rows()> const& b) {
    using T = typename A::ValueType;
    math::MatrixStatus status;
    if constexpr (operation == Operation::Inverse) {
        A inverse;
        if constexpr (explicitAlgorithm) {
            status = math::tryInverseExplicit(a, inverse);
        } else {
            status = math::tryInversePivoted(a, inverse);
        }
        consume(inverse);
    } else if constexpr (operation == Operation::Determinant) {
        T determinant{};
        if constexpr (explicitAlgorithm) {
            status = math::tryDeterminantExplicit(a, determinant);
        } else {
            status = math::tryDeterminantPivoted(a, determinant);
        }
        consume(determinant);
    } else {
        math::Vector<T, A::rows()> x{};
        if constexpr (explicitAlgorithm) {
            status = math::trySolveExplicit(a, b, x);
        } else {
            status = math::trySolvePivoted(a, b, x);
        }
        consume(x);
    }
    if (status != math::MatrixStatus::Success) {
        std::fprintf(stderr, "Benchmark input rejected\n");
        std::abort();
    }
}

template <class T, int dimensions, bool symmetric, Operation operation>
void measure(int iterations) {
    using A = std::conditional_t<symmetric, math::SymmetricMatrix<T, dimensions>,
        math::Matrix<T, dimensions>>;
    constexpr int kSamples = 256;
    std::array<A, kSamples> samples{};
    std::array<math::Vector<T, dimensions>, kSamples> rhs{};
    std::mt19937_64 generator(20261102 + dimensions);
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    for (int sample = 0; sample < kSamples; ++sample) {
        for (int i = 0; i < dimensions; ++i) {
            rhs[sample][i] = T(distribution(generator));
            for (int j = 0; j < (symmetric ? i + 1 : dimensions); ++j) {
                samples[sample](i, j) = T(distribution(generator) + (i == j ? dimensions : 0));
            }
        }
    }
    auto run = [&](bool explicitAlgorithm, int count) {
        auto const begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < count; ++iteration) {
            int const sample = iteration % kSamples;
            if (explicitAlgorithm) {
                evaluate<operation, true>(samples[sample], rhs[sample]);
            } else {
                evaluate<operation, false>(samples[sample], rhs[sample]);
            }
        }
        auto const elapsed = std::chrono::steady_clock::now() - begin;
        return std::chrono::duration<double, std::nano>(elapsed).count() / count;
    };
    run(true, 1024);
    run(false, 1024);
    std::array<double, 7> explicitTimes{}, pivotTimes{};
    for (int repeat = 0; repeat < 7; ++repeat) {
        if (repeat % 2 == 0) {
            explicitTimes[repeat] = run(true, iterations);
            pivotTimes[repeat] = run(false, iterations);
        } else {
            pivotTimes[repeat] = run(false, iterations);
            explicitTimes[repeat] = run(true, iterations);
        }
    }
    std::sort(explicitTimes.begin(), explicitTimes.end());
    std::sort(pivotTimes.begin(), pivotTimes.end());
    char const* name = operation == Operation::Inverse ? "inverse" :
        operation == Operation::Determinant            ? "determinant" :
                                                         "solve";
    std::printf("%s,%s,%d,%s,%.3f,%.3f,%.4f\n", std::is_same_v<T, double> ? "double" : "float",
        symmetric ? "symmetric" : "general", dimensions, name, explicitTimes[3], pivotTimes[3],
        explicitTimes[3] / pivotTimes[3]);
    std::fflush(stdout);
}

template <class T, int dimensions>
void measureSize(int iterations) {
    measure<T, dimensions, false, Operation::Inverse>(iterations);
    measure<T, dimensions, true, Operation::Inverse>(iterations);
    measure<T, dimensions, false, Operation::Determinant>(iterations);
    measure<T, dimensions, true, Operation::Determinant>(iterations);
    measure<T, dimensions, false, Operation::Solve>(iterations);
    measure<T, dimensions, true, Operation::Solve>(iterations);
}
template <class T>
void measureType(int iterations) {
    measureSize<T, 1>(iterations);
    measureSize<T, 2>(iterations);
    measureSize<T, 3>(iterations);
    measureSize<T, 4>(iterations);
    measureSize<T, 5>(iterations);
    measureSize<T, 6>(iterations);
}
}    // namespace detail

int main(int argc, char** argv) {
    int const iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 32768;
    std::puts("scalar,structure,size,operation,explicit_ns,pivoted_ns,ratio");
    detail::measureType<double>(iterations);
    detail::measureType<float>(iterations);
}
