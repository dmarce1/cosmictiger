/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

// Opt-in microbenchmark; this is deliberately not a CTest. The CUDA driver
// includes this source to share exactly the same fixtures and operator calls.
// Define COSMICTIGER_CARTESIAN_BASELINE_HEADER to an external legacy
// fmm_kernels.hpp snapshot. Do not copy that snapshot into the repository. CPU type: 0=float,
// 1=double, 2=SimdFloat, 3=SimdDouble, 4=SimdDouble8.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef COSMICTIGER_CARTESIAN_BASELINE_HEADER
#if ORDER != 7
#error "The legacy baseline header implements only multipole degree five (ORDER 7)"
#endif
#include COSMICTIGER_CARTESIAN_BASELINE_HEADER
#else
#include <cosmictiger/gravity/fmm_kernels.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#endif

#ifndef COSMICTIGER_CARTESIAN_BENCHMARK_TYPE
#define COSMICTIGER_CARTESIAN_BENCHMARK_TYPE 2
#endif

namespace detail {

using std::size_t;

// The microbenchmark times the build's default FMM order; every order is
// compiled into the application, but the legacy header knows only order seven.
#ifdef COSMICTIGER_CARTESIAN_BASELINE_HEADER
constexpr char const* kImplementation = "legacy";
template <class Real>
using BenchmarkMultipole = Multipole<Real>;
template <class Real>
using BenchmarkExpansion = Expansion<Real>;
constexpr int kMultipoleCount = MULTIPOLE_SIZE;
constexpr int kExpansionCount = EXPANSION_SIZE;

// Legacy P2M mutates its argument; the copy keeps the caller's position intact.
template <class Real>
CUDA_EXPORT BenchmarkMultipole<Real> benchmarkP2m(Array<Real, NDIM> position) {
    return p2M(position);
}
#else
constexpr char const* kImplementation = "cartesian";
constexpr int kBenchmarkOrder = kFmmDefaultOrder;
template <class Real>
using BenchmarkMultipole = FmmMultipole<Real, kBenchmarkOrder>;
template <class Real>
using BenchmarkExpansion = FmmExpansion<Real, kBenchmarkOrder>;
constexpr int kMultipoleCount = FmmOrder<kBenchmarkOrder>::kMultipoleCoefficients_;
constexpr int kExpansionCount = FmmOrder<kBenchmarkOrder>::kLocalCoefficients_;

template <class Real>
CUDA_EXPORT BenchmarkMultipole<Real> benchmarkP2m(Array<Real, NDIM> const& position) {
    return p2M<kBenchmarkOrder>(position);
}
#endif

enum class BenchmarkOperator { P2m, M2m, CcDirect, PcDirect, L2l, L2p };

template <class Real>
struct BenchmarkInput {
    Array<Real, NDIM> displacement;
    Array<Real, NDIM> separation;
    BenchmarkMultipole<Real> multipole;
    BenchmarkExpansion<Real> local;
};

template <BenchmarkOperator operation>
CUDA_EXPORT constexpr int resultCount() {
    if constexpr (operation == BenchmarkOperator::P2m || operation == BenchmarkOperator::M2m) {
        return kMultipoleCount;
    } else if constexpr (operation == BenchmarkOperator::PcDirect ||
        operation == BenchmarkOperator::L2p) {
        return 4;
    } else {
        return kExpansionCount;
    }
}

template <BenchmarkOperator operation, class Real>
using BenchmarkResult = Array<Real, resultCount<operation>()>;

template <class Real>
constexpr int laneCount() {
    if constexpr (std::is_arithmetic_v<Real>) {
        return 1;
    } else {
        return int(Real::size());
    }
}

template <class Real>
Real sampleValue(int sample, int axis, bool separation) {
    auto const scalar = [=](int lane) {
        int const seed = sample * laneCount<Real>() + lane;
        double const variation = double((seed * 37 + axis * 173 + 19) % 997) / 997.;
        return separation ? (axis == 1 ? -1. : 1.) * (0.004 + 0.001 * variation) :
                            0.00012 * (variation - 0.5);
    };
    if constexpr (std::is_arithmetic_v<Real>) {
        return Real(scalar(0));
    } else {
        Real result{};
        for (int lane = 0; lane < laneCount<Real>(); ++lane) {
            result[lane] = static_cast<typename Real::value_type>(scalar(lane));
        }
        return result;
    }
}

template <class Real>
BenchmarkInput<Real> makeBenchmarkInput(int sample) {
    BenchmarkInput<Real> input{};
    Array<Real, NDIM> particlePosition;
    for (int axis = 0; axis < NDIM; ++axis) {
        input.displacement[axis] = sampleValue<Real>(sample, axis, false);
        input.separation[axis] = sampleValue<Real>(sample, axis, true);
        particlePosition[axis] = sampleValue<Real>(sample + 31, axis, false);
    }
    // Legacy P2M mutates its argument and stores odd source ranks with the
    // opposite sign. Each implementation prepares its own representation of
    // the same physical point mass before the timed region.
    input.multipole = benchmarkP2m<Real>(particlePosition);
    greensFunction(input.local, input.separation);
    return input;
}

template <BenchmarkOperator operation, bool doPhi, class Real>
CUDA_EXPORT void evaluateBenchmark(
    BenchmarkInput<Real> const& input, BenchmarkResult<operation, Real>& output) {
    if constexpr (operation == BenchmarkOperator::P2m) {
        auto const result = benchmarkP2m<Real>(input.displacement);
        for (int i = 0; i < resultCount<operation>(); ++i) {
            output[i] = result[i];
        }
    } else if constexpr (operation == BenchmarkOperator::M2m) {
        auto displacement = input.displacement;
        auto const result = m2M(input.multipole, displacement);
        for (int i = 0; i < resultCount<operation>(); ++i) {
            output[i] = result[i];
        }
    } else if constexpr (operation == BenchmarkOperator::CcDirect) {
        BenchmarkExpansion<Real> result{};
        result = simdBroadcast<Real>(0);
#ifdef COSMICTIGER_CARTESIAN_BASELINE_HEADER
        BenchmarkExpansion<Real> derivatives;
        greensFunction(derivatives, input.separation);
        m2L(result, input.multipole, derivatives, doPhi);
#else
        m2LDirect(result, input.multipole, input.separation, doPhi);
#endif
        for (int i = 0; i < resultCount<operation>(); ++i) {
            output[i] = result[i];
        }
    } else if constexpr (operation == BenchmarkOperator::PcDirect) {
        Expansion2<Real> result{};
        result = simdBroadcast<Real>(0);
#ifdef COSMICTIGER_CARTESIAN_BASELINE_HEADER
        BenchmarkExpansion<Real> derivatives;
        greensFunction(derivatives, input.separation);
        m2L(result, input.multipole, derivatives, doPhi);
#else
        m2P(result, input.multipole, input.separation, doPhi);
#endif
        output[0] = result[0];
        output[1] = result(1, 0, 0);
        output[2] = result(0, 1, 0);
        output[3] = result(0, 0, 1);
    } else if constexpr (operation == BenchmarkOperator::L2l) {
        auto const result = l2L(input.local, input.displacement, doPhi);
        for (int i = 0; i < resultCount<operation>(); ++i) {
            output[i] = result[i];
        }
    } else {
        auto const result = l2P(input.local, input.displacement, doPhi);
        // Legacy Expansion2 includes an unused fifth entry; never read it.
        output[0] = doPhi ? result[0] : simdBroadcast<Real>(0);
        output[1] = result(1, 0, 0);
        output[2] = result(0, 1, 0);
        output[3] = result(0, 0, 1);
    }
}

int positiveArgument(char const* text, char const* name, int maximum) {
    char* end = nullptr;
    long const value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > maximum) {
        throw std::invalid_argument(std::string(name) + " is outside its supported positive range");
    }
    return int(value);
}

template <class Real>
double sumLanes(Real const& value) {
    if constexpr (std::is_arithmetic_v<Real>) {
        return double(value);
    } else {
        double sum = 0;
        for (int lane = 0; lane < laneCount<Real>(); ++lane) {
            sum += value[lane];
        }
        return sum;
    }
}

template <BenchmarkOperator operation, class Real>
double checksum(std::vector<BenchmarkResult<operation, Real>> const& output) {
    double sum = 0;
    for (auto const& result : output) {
        for (int i = 0; i < resultCount<operation>(); ++i) {
            sum += double(i + 1) * sumLanes(result[i]);
        }
    }
    if (!std::isfinite(sum)) {
        throw std::runtime_error("Nonfinite benchmark checksum");
    }
    return sum;
}

#ifndef COSMICTIGER_CARTESIAN_BENCHMARK_CUDA

template <class Result>
inline void escapeResults(Result const* results) {
    // Every sweep writes externally observable output; repeated sweeps cannot
    // be removed or hoisted, and all output coefficients remain live.
    asm volatile("" : : "g"(results) : "memory");
}

template <BenchmarkOperator operation, bool doPhi, class Real>
void runCpuCase(std::vector<BenchmarkInput<Real>> const& input, char const* typeName,
    char const* operationName, int rounds, int repetitions) {
    std::vector<BenchmarkResult<operation, Real>> output(input.size());
    auto const sweep = [&]() {
        for (size_t sample = 0; sample < input.size(); ++sample) {
            evaluateBenchmark<operation, doPhi>(input[sample], output[sample]);
        }
        escapeResults(output.data());
    };
    sweep();
    sweep();
    std::vector<double> times;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        auto const begin = std::chrono::steady_clock::now();
        for (int round = 0; round < rounds; ++round) {
            sweep();
        }
        auto const end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(end - begin).count());
    }
    std::sort(times.begin(), times.end());
    double const seconds = times[times.size() / 2];
    double const interactions = double(rounds) * input.size() * laneCount<Real>();
    std::printf("%s,cpu,%s,%s,%d,%d,%zu,%d,%d,%.9g,%.9g,%.17g\n", kImplementation, typeName,
        operationName, int(doPhi), laneCount<Real>(), input.size(), rounds, repetitions, seconds,
        seconds * 1e9 / interactions, checksum<operation, Real>(output));
}

template <class Real>
void runCpu(char const* typeName, int samples, int rounds, int repetitions) {
    std::vector<BenchmarkInput<Real>> input;
    input.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        input.push_back(makeBenchmarkInput<Real>(sample));
    }
    runCpuCase<BenchmarkOperator::P2m, false>(input, typeName, "p2m", rounds, repetitions);
    runCpuCase<BenchmarkOperator::M2m, false>(input, typeName, "m2m", rounds, repetitions);
    runCpuCase<BenchmarkOperator::CcDirect, false>(
        input, typeName, "cc_direct", rounds, repetitions);
    runCpuCase<BenchmarkOperator::CcDirect, true>(
        input, typeName, "cc_direct", rounds, repetitions);
    runCpuCase<BenchmarkOperator::PcDirect, false>(
        input, typeName, "pc_direct", rounds, repetitions);
    runCpuCase<BenchmarkOperator::PcDirect, true>(
        input, typeName, "pc_direct", rounds, repetitions);
    runCpuCase<BenchmarkOperator::L2l, false>(input, typeName, "l2l", rounds, repetitions);
    runCpuCase<BenchmarkOperator::L2l, true>(input, typeName, "l2l", rounds, repetitions);
    runCpuCase<BenchmarkOperator::L2p, false>(input, typeName, "l2p", rounds, repetitions);
    runCpuCase<BenchmarkOperator::L2p, true>(input, typeName, "l2p", rounds, repetitions);
}

#endif

}    // namespace detail

#ifndef COSMICTIGER_CARTESIAN_BENCHMARK_CUDA
int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            throw std::invalid_argument("Usage: cartesian_kernels "
                                        "[samples=128] [rounds=1000] [repetitions=7]");
        }
        int const samples = argc > 1 ? detail::positiveArgument(argv[1], "samples", 1000000) : 128;
        int const rounds = argc > 2 ? detail::positiveArgument(argv[2], "rounds", 1000000) : 1000;
        int const repetitions =
            argc > 3 ? detail::positiveArgument(argv[3], "repetitions", 1000) : 7;
        std::puts("implementation,device,type,operator,do_phi,lanes,samples,rounds,repetitions,"
                  "median_seconds,ns_per_interaction,checksum");
#if COSMICTIGER_CARTESIAN_BENCHMARK_TYPE == 0
        detail::runCpu<float>("float", samples, rounds, repetitions);
#elif COSMICTIGER_CARTESIAN_BENCHMARK_TYPE == 1
        detail::runCpu<double>("double", samples, rounds, repetitions);
#elif COSMICTIGER_CARTESIAN_BENCHMARK_TYPE == 2
        detail::runCpu<SimdFloat>("simd_float8", samples, rounds, repetitions);
#elif COSMICTIGER_CARTESIAN_BENCHMARK_TYPE == 3
        detail::runCpu<SimdDouble>("simd_double2", samples, rounds, repetitions);
#elif COSMICTIGER_CARTESIAN_BENCHMARK_TYPE == 4
        detail::runCpu<SimdDouble8>("simd_double8", samples, rounds, repetitions);
#else
#error "Unsupported COSMICTIGER_CARTESIAN_BENCHMARK_TYPE"
#endif
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Cartesian benchmark failed: %s\n", error.what());
        return 1;
    }
}
#endif
