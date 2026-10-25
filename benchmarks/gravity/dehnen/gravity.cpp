/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hpx/runtime_local/runtime_local_fwd.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/dehnen/distributed_tree.hpp>
#include <cosmictiger/gravity/dehnen/periodic.hpp>
#include <cosmictiger/gravity/fmm_kernels.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particles.hpp>

// This source is injected into each comparison snapshot by the benchmark runner.
// It deliberately uses only scalar/public interfaces common to both snapshots.
// It is not a CTest and must only be dispatched by an explicit benchmark run.
namespace detail {

using std::size_t;

constexpr int kBenchmarkSinkParticles = 32;
constexpr float kBenchmarkSupport = 0.0625f;
constexpr float kBenchmarkTheta = 0.8f;

enum class BenchmarkOperator { CcDirect, CcEwald, Cp, Pc, Pp };
enum class BenchmarkGeometry { Far, Near, Mixed };

struct BenchmarkShape {
    char const* name;
    int sourceCells;
    int particlesPerCell;
};

struct BenchmarkCase {
    std::string name;
    BenchmarkOperator operation;
    BenchmarkGeometry geometry;
    BenchmarkShape shape;
    bool doPhi;
};

struct BenchmarkSettings {
    int repetitions;
    int batches;
    int warmup;
    std::string selectedCase;
};

int benchmarkInteger(char const* name, int fallback, int minimum, int maximum) {
    char const* text = std::getenv(name);
    if (text == nullptr) {
        return fallback;
    }
    std::string_view const value(text);
    int result = 0;
    auto const parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        result < minimum || result > maximum) {
        throw std::invalid_argument(std::string(name) + " must be an integer in [" +
            std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
    }
    return result;
}

BenchmarkSettings benchmarkSettings() {
    BenchmarkSettings settings{benchmarkInteger("COSMICTIGER_BENCH_REPETITIONS", 8, 1, 256),
        benchmarkInteger("COSMICTIGER_BENCH_BATCHES", 5, 1, 1000),
        benchmarkInteger("COSMICTIGER_BENCH_WARMUP", 2, 0, 256), {}};
    if (char const* selected = std::getenv("COSMICTIGER_BENCH_CASE")) {
        settings.selectedCase = selected;
        if (settings.selectedCase.empty()) {
            throw std::invalid_argument("COSMICTIGER_BENCH_CASE cannot be empty when set");
        }
    }
    return settings;
}

std::vector<BenchmarkCase> benchmarkCases() {
    // Width eight is a workload property, not an inferred hardware width. Both
    // snapshots therefore receive the same particles, even on narrower targets.
    std::array<BenchmarkShape, 5> const shapes{{{"packet_full", 8, 8}, {"packet_tail", 9, 7},
        {"chunk_31", 31, 7}, {"chunk_32", 32, 8}, {"chunk_33", 33, 7}}};
    struct OperatorCase {
        char const* name;
        BenchmarkOperator operation;
        BenchmarkGeometry geometry;
    };
    std::array<OperatorCase, 7> const operators{{
        {"cc_direct", BenchmarkOperator::CcDirect, BenchmarkGeometry::Far},
        {"cc_ewald", BenchmarkOperator::CcEwald, BenchmarkGeometry::Far},
        {"cp", BenchmarkOperator::Cp, BenchmarkGeometry::Far},
        {"pc", BenchmarkOperator::Pc, BenchmarkGeometry::Far},
        {"pp_far", BenchmarkOperator::Pp, BenchmarkGeometry::Far},
        {"pp_near", BenchmarkOperator::Pp, BenchmarkGeometry::Near},
        {"pp_mixed", BenchmarkOperator::Pp, BenchmarkGeometry::Mixed},
    }};
    std::vector<BenchmarkCase> cases;
    for (auto const& operation : operators) {
        for (auto const& shape : shapes) {
            for (bool doPhi : {false, true}) {
                cases.push_back({std::string(operation.name) + "_" + shape.name +
                        (doPhi ? "_potential" : "_force"),
                    operation.operation, operation.geometry, shape, doPhi});
            }
        }
    }
    return cases;
}

int benchmarkOffset(int index, int dimension) {
    uint32_t value = uint32_t(index + 1) * 2654435761u + uint32_t(dimension + 1) * 2246822519u;
    value ^= value >> 13;
    return int(value % 63u) - 31;
}

double benchmarkWrap(double value) {
    return value - std::floor(value);
}

Array<Position, NDIM> benchmarkCenter(int cell, bool near) {
    // Source x coordinates wrap through zero while sinks remain near one. This
    // exercises the signed interpretation of periodic fixed-point subtraction.
    std::array<double, NDIM> const origin{1023.0 / 1024, 371.0 / 1024, 433.0 / 1024};
    std::array<double, NDIM> const separation = near ?
        std::array<double, NDIM>{1.0 / 64, 1.0 / 128, -1.0 / 256} :
        std::array<double, NDIM>{3.0 / 16, 1.0 / 32, -1.0 / 64};
    Array<Position, NDIM> position;
    for (int dimension = 0; dimension < NDIM; ++dimension) {
        position[dimension] = benchmarkWrap(origin[dimension] +
            (cell < 0 ? 0 :
                        separation[dimension] + benchmarkOffset(cell + 101, dimension) / 16384.0));
    }
    return position;
}

Array<Position, NDIM> benchmarkParticle(Array<Position, NDIM> const& center, int index) {
    Array<Position, NDIM> position;
    for (int dimension = 0; dimension < NDIM; ++dimension) {
        position[dimension] = benchmarkWrap(
            center[dimension].toDouble() + benchmarkOffset(index, dimension) / 32768.0);
    }
    return position;
}

void benchmarkSetPositions(std::vector<Array<Position, NDIM>> const& positions) {
    for (size_t index = 0; index < positions.size(); ++index) {
        for (int dimension = 0; dimension < NDIM; ++dimension) {
            particlesPos(dimension, index) = positions[index][dimension];
            particlesVel(dimension, index) = 0.0f;
        }
        particlesRung(index) = 1;
    }
}

TreeId benchmarkAddNode(
    Array<Position, NDIM> const& center, Pair<ParticleInteger> range, bool sink) {
    TreeNode node{};
    node.pos = center;
    node.particleRange = range;
    node.processRange = Pair<int>(hpxRank(), hpxRank() + 1);
    node.depth = 4;
    node.localRoot = sink;
    node.leaf = true;
    node.children[LEFT] = node.children[RIGHT] = TreeId{-1, -1};
    node.maxGravitySupportRadius = kBenchmarkSupport;
    MultipoleBuffer multipole;
    multipole.zero();
    double radiusSquared = 0;
    fmmDispatch([&](auto order) {
        auto& moments = multipole.multipole<order.value>();
        for (ParticleInteger index = range.first; index < range.second; ++index) {
            Array<float, NDIM> displacement;
            double particleRadiusSquared = 0;
            for (int dimension = 0; dimension < NDIM; ++dimension) {
                displacement[dimension] =
                    distance(particlesPos(dimension, index), center[dimension]);
                particleRadiusSquared += double(displacement[dimension]) * displacement[dimension];
            }
            radiusSquared = std::max(radiusSquared, particleRadiusSquared);
            auto const moment = p2M<order.value>(displacement);
            for (int coefficient = 0; coefficient < FmmOrder<order.value>::kMultipoleCoefficients_;
                ++coefficient) {
                moments[coefficient] += particlesMass(index) * moment[coefficient];
            }
        }
    });
    node.radius = float(std::sqrt(radiusSquared));
    node.mpos = treeAddMultipole(center, multipole);
    // The direct operator entry points do not use opening boxes. A complete
    // periodic-domain box remains conservative for cells crossing the seam.
    for (int dimension = 0; dimension < NDIM; ++dimension) {
        node.box.begin[dimension] = 0.0;
        node.box.end[dimension] = 1.0;
    }
    return TreeId{hpxRank(), int(treeAddRemote(node))};
}

class BenchmarkArena {
public:
    TreeId sink_;
    Vector<TreeId> sources_;

    explicit BenchmarkArena(BenchmarkCase const& testCase) {
        int const sourceParticles = testCase.shape.sourceCells * testCase.shape.particlesPerCell;
        std::vector<Array<Position, NDIM>> positions(kBenchmarkSinkParticles + sourceParticles);
        auto const sinkCenter = benchmarkCenter(-1, false);
        for (int index = 0; index < kBenchmarkSinkParticles; ++index) {
            positions[index] = benchmarkParticle(sinkCenter, index);
        }
        for (int cell = 0; cell < testCase.shape.sourceCells; ++cell) {
            for (int particle = 0; particle < testCase.shape.particlesPerCell; ++particle) {
                int const sourceIndex = cell * testCase.shape.particlesPerCell + particle;
                bool const near = testCase.geometry == BenchmarkGeometry::Near ||
                    (testCase.geometry == BenchmarkGeometry::Mixed && sourceIndex % 8 < 4);
                positions[kBenchmarkSinkParticles + sourceIndex] =
                    benchmarkParticle(benchmarkCenter(cell, near), sourceIndex + 1009);
            }
        }
        particlesResize(positions.size());
        benchmarkSetPositions(positions);
        particlesSortByRung(0);
        treeCreate(TreeCreateParameters(0, kBenchmarkTheta, kBenchmarkSupport));
        treeReady_ = true;
        // treeCreate initializes arenas and reorders particles. Restore the
        // prescribed ranges before constructing the synthetic moments.
        benchmarkSetPositions(positions);
        sink_ =
            benchmarkAddNode(sinkCenter, Pair<ParticleInteger>(0, kBenchmarkSinkParticles), true);
        for (int cell = 0; cell < testCase.shape.sourceCells; ++cell) {
            ParticleInteger const first =
                kBenchmarkSinkParticles + cell * testCase.shape.particlesPerCell;
            sources_.push_back(benchmarkAddNode(
                benchmarkCenter(cell, testCase.geometry == BenchmarkGeometry::Near),
                Pair<ParticleInteger>(first, first + testCase.shape.particlesPerCell), false));
        }
    }

    BenchmarkArena(BenchmarkArena const&) = delete;
    BenchmarkArena& operator=(BenchmarkArena const&) = delete;

    ~BenchmarkArena() {
        if (treeReady_) {
            treeDestroy();
        }
    }

private:
    bool treeReady_ = false;
};

struct BenchmarkOutput {
    ExpansionBuffer expansion;
    ForceVectors forces{kBenchmarkSinkParticles};

    void clear() {
        expansion.zero();
        for (int index = 0; index < kBenchmarkSinkParticles; ++index) {
            forces.gx[index] = forces.gy[index] = forces.gz[index] = forces.phi[index] = 0.0f;
        }
    }
};

bool benchmarkHasExpansion(BenchmarkOperator operation) {
    return operation == BenchmarkOperator::CcDirect || operation == BenchmarkOperator::CcEwald ||
        operation == BenchmarkOperator::Cp;
}

void benchmarkInvoke(
    BenchmarkCase const& testCase, BenchmarkArena const& arena, BenchmarkOutput& output) {
    // The runtime FMM order selects the instantiated operator outside the timed
    // call, as the production walk does.
    fmmDispatch([&](auto order) {
        auto& expansion = output.expansion.expansion<order.value>();
        switch (testCase.operation) {
        case BenchmarkOperator::CcDirect:
            cpuGravityCc<order.value>(expansion, arena.sources_, arena.sink_, testCase.doPhi);
            break;
        case BenchmarkOperator::CcEwald:
            cpuGravityEwald<order.value>(expansion, arena.sources_, arena.sink_, testCase.doPhi);
            break;
        case BenchmarkOperator::Cp:
            // This production entry point currently computes all coefficients even
            // when doPhi is false. Keep both modes to expose that actual behavior.
            cpuGravityCp<order.value>(expansion, arena.sources_, arena.sink_, testCase.doPhi);
            break;
        case BenchmarkOperator::Pc:
            cpuGravityPc<order.value>(output.forces, testCase.doPhi, arena.sink_, arena.sources_);
            break;
        case BenchmarkOperator::Pp:
            cpuGravityPp(
                output.forces, testCase.doPhi, arena.sink_, arena.sources_, kBenchmarkSupport);
            break;
        }
    });
}

std::vector<double> benchmarkValues(
    BenchmarkCase const& testCase, BenchmarkArena const& arena, BenchmarkOutput const& output) {
    std::vector<double> values;
    if (benchmarkHasExpansion(testCase.operation)) {
        int const coefficientCount = fmmLocalCoefficientCount(fmmOrder());
        for (int coefficient = 0; coefficient < coefficientCount; ++coefficient) {
            values.push_back(output.expansion[coefficient]);
        }
        // Preserve all expansion coefficients and also record their physical
        // acceleration/potential at every sink; scaling must not hide errors.
        auto const center = treeGetNode(arena.sink_)->pos;
        for (int index = 0; index < kBenchmarkSinkParticles; ++index) {
            Array<float, NDIM> displacement;
            for (int dimension = 0; dimension < NDIM; ++dimension) {
                displacement[dimension] =
                    distance(particlesPos(dimension, index), center[dimension]);
            }
            auto const force = fmmDispatch([&](auto order) {
                return l2P(output.expansion.expansion<order.value>(), displacement, testCase.doPhi);
            });
            values.push_back(-SCALE_FACTOR2 * force(1, 0, 0));
            values.push_back(-SCALE_FACTOR2 * force(0, 1, 0));
            values.push_back(-SCALE_FACTOR2 * force(0, 0, 1));
            values.push_back(testCase.doPhi ? SCALE_FACTOR1 * force(0, 0, 0) : 0.0f);
        }
    } else {
        for (int index = 0; index < kBenchmarkSinkParticles; ++index) {
            values.push_back(output.forces.gx[index]);
            values.push_back(output.forces.gy[index]);
            values.push_back(output.forces.gz[index]);
            values.push_back(output.forces.phi[index]);
        }
    }
    for (double value : values) {
        // A bit test remains effective when the application uses -ffast-math.
        if ((std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) {
            throw std::runtime_error(testCase.name + ": nonfinite benchmark output");
        }
    }
    return values;
}

void benchmarkRunCase(BenchmarkCase const& testCase, BenchmarkSettings const& settings,
    std::ofstream& timings, std::ofstream& savedValues) {
    BenchmarkArena arena(testCase);
    std::vector<BenchmarkOutput> outputs(settings.repetitions);
    BenchmarkOutput reference;
    for (int iteration = 0; iteration < settings.warmup; ++iteration) {
        reference.clear();
        benchmarkInvoke(testCase, arena, reference);
    }
    reference.clear();
    benchmarkInvoke(testCase, arena, reference);
    auto const expected = benchmarkValues(testCase, arena, reference);
    for (size_t index = 0; index < expected.size(); ++index) {
        savedValues << testCase.name << ',' << index << ',' << expected[index] << '\n';
    }
    // Unit counts are cell-cell for CC/Ewald, cell-particle for CP,
    // particle-cell for PC, and particle-particle for PP. Padding is excluded.
    int const sinks = benchmarkHasExpansion(testCase.operation) ? 1 : kBenchmarkSinkParticles;
    int const sources = testCase.shape.sourceCells *
        ((testCase.operation == BenchmarkOperator::Cp ||
             testCase.operation == BenchmarkOperator::Pp) ?
                testCase.shape.particlesPerCell :
                1);
    uint64_t const interactions = uint64_t(settings.repetitions) * sinks * sources;
    for (int sample = 0; sample < settings.batches; ++sample) {
        // Each timed call writes its own zeroed slot exactly once. Allocation,
        // resets, verification, checksums and file output are outside timing.
        for (auto& output : outputs) {
            output.clear();
        }
        auto const start = std::chrono::steady_clock::now();
        for (auto& output : outputs) {
            benchmarkInvoke(testCase, arena, output);
        }
        auto const stop = std::chrono::steady_clock::now();
        double const seconds = std::chrono::duration<double>(stop - start).count();
        double checksum = 0;
        for (auto const& output : outputs) {
            auto const values = benchmarkValues(testCase, arena, output);
            if (values != expected) {
                throw std::runtime_error(testCase.name + ": repeated call changed output");
            }
            for (size_t index = 0; index < values.size(); ++index) {
                checksum += values[index] * double(1 + index % 13);
            }
        }
        timings << testCase.name << ',' << sample << ',' << seconds << ',' << settings.repetitions
                << ',' << interactions << ',' << sinks << ',' << sources << ',' << testCase.doPhi
                << ',' << checksum << '\n';
        timings.flush();
    }
    savedValues.flush();
}

}    // namespace detail

void simdBenchmark() {
    if (hpxSize() != 1 || hpx::get_num_worker_threads() != 1) {
        throw std::invalid_argument("SIMD gravity benchmark requires one rank and one HPX thread");
    }
    if (getOptions().compute.cuda) {
        throw std::invalid_argument("SIMD gravity benchmark requires --compute.cuda=false");
    }
    auto const settings = detail::benchmarkSettings();
    auto const cases = detail::benchmarkCases();
    bool selectedCaseExists = settings.selectedCase.empty();
    for (auto const& testCase : cases) {
        selectedCaseExists = selectedCaseExists || testCase.name == settings.selectedCase;
    }
    if (!selectedCaseExists) {
        throw std::invalid_argument("Unknown COSMICTIGER_BENCH_CASE: " + settings.selectedCase);
    }
    if (std::filesystem::exists("bench_timings.csv") ||
        std::filesystem::exists("bench_values.csv")) {
        throw std::invalid_argument(
            "Benchmark output files already exist; use an empty run directory");
    }
    std::ofstream timings("bench_timings.csv");
    std::ofstream values("bench_values.csv");
    timings.exceptions(std::ios::badbit | std::ios::failbit);
    values.exceptions(std::ios::badbit | std::ios::failbit);
    timings << std::setprecision(std::numeric_limits<double>::max_digits10);
    values << std::setprecision(std::numeric_limits<double>::max_digits10);
    timings << "case,sample,seconds,calls,interactions,sinks,sources,do_phi,checksum\n";
    values << "case,index,value\n";
    auto const originalOptions = getOptions();
    auto const originalGravity = selectedGravityKernel().spec;
    auto const originalHydro = selectedHydroKernel().spec;
    auto options = originalOptions;
    options.hydro.enabled = false;
    options.gravity.enabled = true;
    options.analysis.saveForce = false;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.compute.bucketSize = 64;
    options.gravity.supportRadius = detail::kBenchmarkSupport;
    options.initial.numberOfParticles = 32 + 33 * 8;
    setOptions(options);
    setKernelSelections(originalGravity, originalHydro, false);
    try {
        for (auto const& testCase : cases) {
            if (settings.selectedCase.empty() || settings.selectedCase == testCase.name) {
                detail::benchmarkRunCase(testCase, settings, timings, values);
            }
        }
    } catch (...) {
        setKernelSelections(originalGravity, originalHydro, false);
        setOptions(originalOptions);
        throw;
    }
    setKernelSelections(originalGravity, originalHydro, false);
    setOptions(originalOptions);
}
