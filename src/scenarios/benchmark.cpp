/* CosmicTiger - static particle FMM benchmark; BSL-1.0. */
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/scenarios/benchmark.hpp>

namespace scenarios {
namespace detail {
    using Clock = std::chrono::steady_clock;

    bool benchmarkFinite(double value) {
        return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    }

    std::vector<std::string> benchmarkList(std::string const& text, char const* option) {
        std::vector<std::string> result;
        size_t start = 0;
        do {
            size_t const end = text.find(',', start);
            auto item = text.substr(start, end == std::string::npos ? end : end - start);
            size_t const first = item.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) {
                throw std::invalid_argument(std::string(option) + " contains an empty list entry");
            }
            item = item.substr(first, item.find_last_not_of(" \t\r\n") - first + 1);
            result.push_back(std::move(item));
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        } while (true);
        return result;
    }

    Vector<Vector3<double>> readBenchmarkParticles(std::string const& filename) {
        std::ifstream input(filename);
        if (!input) {
            throw std::runtime_error("Cannot open benchmark particle file: " + filename);
        }
        Vector<Vector3<double>> positions;
        std::string line;
        size_t lineNumber = 0;
        while (std::getline(input, line)) {
            ++lineNumber;
            if (auto const comment = line.find('#'); comment != std::string::npos) {
                line.resize(comment);
            }
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }
            std::istringstream row(line);
            Vector3<double> position;
            bool valid = bool(row >> position[0] >> position[1] >> position[2]);
            std::string extra;
            valid = valid && !(row >> extra);
            for (int dim = 0; valid && dim < kDimensions; ++dim) {
                valid =
                    benchmarkFinite(position[dim]) && position[dim] >= 0.0 && position[dim] < 1.0;
            }
            if (!valid) {
                throw std::invalid_argument(
                    std::format("{}:{}: expected exactly three finite coordinates in [0,1)",
                        filename, lineNumber));
            }
            positions.push_back(position);
        }
        if (input.bad() || positions.empty()) {
            throw std::runtime_error("Empty or unreadable benchmark particle file: " + filename);
        }
        return positions;
    }

    size_t benchmarkParticleCount(size_t baseCount, int side) {
        size_t count = baseCount;
        size_t const limit = size_t(std::numeric_limits<ptrdiff_t>::max());
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (count > limit / size_t(side)) {
                throw std::overflow_error("benchmark.size produces too many particles");
            }
            count *= size_t(side);
        }
        // Domain decomposition may move all particles onto one rank for extreme inputs.
        if (count > size_t(std::numeric_limits<ParticleInteger>::max())) {
            throw std::overflow_error("Benchmark population exceeds the particle index range");
        }
        return count;
    }

    void initializeBenchmarkParticles(
        Vector<Vector3<double>> const& positions, int side, size_t count) {
        size_t const ranks = size_t(hpxSize());
        size_t const rank = size_t(hpxRank());
        size_t const begin = count / ranks * rank + std::min(count % ranks, rank);
        size_t const end = begin + count / ranks + (rank < count % ranks);
        particlesResize(ParticleInteger(end - begin));
        for (size_t index = begin; index < end; ++index) {
            auto const& position = positions[index % positions.size()];
            size_t tile = index / positions.size();
            ParticleInteger const local = ParticleInteger(index - begin);
            // Derived unit-box tiling: each copy occupies one cell of an n^3 grid.
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const coordinate = (position[dim] + double(tile % size_t(side))) / side;
                particlesPos(dim, local) = coordinate;
                particlesVel(dim, local) = 0.0f;
                tile /= size_t(side);
            }
            particlesRung(local) = 0;
        }
    }

    // Validate finite results, unchanged velocities and all-active rungs outside
    // the timed repetitions.
    void checkBenchmarkParticles() {
        for (ParticleInteger i = 0; i < particlesOwnedSize(); ++i) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                if (particlesVel(dim, i) != 0.0f ||
                    !benchmarkFinite(particlesGravityAcceleration(dim, i))) {
                    throw std::runtime_error(
                        "Benchmark solve changed velocity or produced a nonfinite force");
                }
            }
            if (particlesRung(i) != 0 || !benchmarkFinite(particlesPot(i))) {
                throw std::runtime_error(
                    "Benchmark solve changed rung or produced a nonfinite potential");
            }
        }
    }

    double seconds(Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double>(end - begin).count();
    }

    std::string appendBenchmarkResult(size_t caseNumber, std::string const& row) {
        auto const filename = outputPath(std::format("bench.{}.dat", caseNumber));
        std::ofstream output(filename, std::ios::app | std::ios::ate);
        if (!output) {
            throw std::ios_base::failure("Cannot open benchmark results file: " + filename);
        }
        if (output.tellp() == std::streampos(0)) {
            output << "# particles size theta bucket_size fmm_order localities threads backend "
                      "repeats tree_nodes leaves particles_per_leaf softening tree_mean_s "
                      "force_mean_s cleanup_mean_s min_s median_s mean_s max_s flops gflops "
                      "load_s domain_s\n";
        }
        output << row << '\n';
        output.close();
        if (!output) {
            throw std::ios_base::failure("Cannot append benchmark results file: " + filename);
        }
        return filename;
    }
}    // namespace detail
}    // namespace scenarios

HPX_PLAIN_ACTION(
    scenarios::detail::initializeBenchmarkParticles, InitializeBenchmarkParticlesAction);
HPX_PLAIN_ACTION(scenarios::detail::checkBenchmarkParticles, CheckBenchmarkParticlesAction);

namespace scenarios {
std::vector<BenchmarkCase> parseBenchmarkCases(BenchmarkScenario const& options) {
    if (options.size < 1 || options.repeats < 1) {
        throw std::invalid_argument("benchmark.size and benchmark.repeats must be at least 1");
    }
    auto const files = detail::benchmarkList(options.testFiles, "benchmark.testFiles");
    auto const angles = detail::benchmarkList(options.openingAngles, "benchmark.openingAngles");
    if (files.size() != angles.size()) {
        throw std::invalid_argument(
            "benchmark.openingAngles must contain exactly one angle per test file");
    }
    std::vector<BenchmarkCase> result;
    for (size_t i = 0; i < files.size(); ++i) {
        size_t consumed = 0;
        double angle;
        try {
            angle = std::stod(angles[i], &consumed);
        } catch (std::exception const&) {
            throw std::invalid_argument("Invalid benchmark.openingAngles entry: " + angles[i]);
        }
        if (consumed != angles[i].size() || !detail::benchmarkFinite(angle) ||
            angle < std::numeric_limits<float>::min() || angle > 1.0) {
            throw std::invalid_argument(
                "benchmark.openingAngles entries must be finite, float-representable and in (0,1]");
        }
        result.push_back({files[i], angle});
    }
    return result;
}

void runBenchmark() {
    auto const original = getOptions();
    auto const& benchmark = original.scenario.benchmark;
    auto const cases = parseBenchmarkCases(benchmark);
    std::cout
        << "Benchmark: static periodic gravity; loading and initial decomposition excluded.\n"
           "Each total includes tree build, completed force solve and cleanup; no warmup is "
           "omitted.\n"
           "FLOPs are distributed source-operation estimates (FMA = 2), including CPU and GPU.\n"
        << std::flush;
    for (size_t test = 0; test < cases.size(); ++test) {
        profile::checkForInterrupt();
        auto const loadStart = detail::Clock::now();
        auto const positions = detail::readBenchmarkParticles(cases[test].filename);
        size_t const count = detail::benchmarkParticleCount(positions.size(), benchmark.size);
        auto options = original;
        options.initial.numberOfParticles = count;
        options.run.darkMatterParticleCount = count;
        options.run.gasParticleCount = 0;
        options.gravity.openingAngle = cases[test].openingAngle;
        options.gravity.gm = 1.0 / double(count);
        // processOptions expresses softening in units of the initial mean spacing.
        options.gravity.smoothingLength = original.gravity.smoothingLength *
            std::cbrt(double(original.initial.numberOfParticles) / double(count));
        if (!detail::benchmarkFinite(options.gravity.smoothingLength) ||
            options.gravity.smoothingLength <= 0.0 || options.gravity.smoothingLength >= 0.5) {
            throw std::invalid_argument(
                "Benchmark normalized gravity smoothing length must be in (0,0.5)");
        }
        options.analysis.saveForce = true;
        options.analysis.groups = options.analysis.tracers = false;
        options.analysis.power = false;
        options.lightcone.enabled = false;
        setOptions(options);
        Vector<hpx::future<void>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<InitializeBenchmarkParticlesAction>(
                locality, positions, benchmark.size, count));
        }
        hpx::wait_all(futures.begin(), futures.end());
        for (auto& future : futures) {
            future.get();
        }
        particlesSortByRung(0);
        auto const loadEnd = detail::Clock::now();
        domainsRebound();
        domainsBegin(0);
        domainsEnd();
        double const domainSeconds = detail::seconds(loadEnd, detail::Clock::now());
        std::cout << std::format("\nBenchmark case {} / {}\n"
                                 "{}\n"
                                 "  File:       {}\n"
                                 "  Particles:  {} ({} base particles x {}^3 copies)\n"
                                 "  Solver:     {} | Localities: {} | Repeats: {}\n"
                                 "  Theta:      {:.6g}\n"
                                 "  Softening:  {:.6e}\n"
                                 "  Setup (s):  load {:.6f} | decomposition {:.6f}\n\n"
                                 "  {:>6} {:>13} {:>13} {:>13} {:>13} {:>12}\n"
                                 "  {}\n",
                         test + 1, cases.size(), std::string(82, '='), cases[test].filename, count,
                         positions.size(), benchmark.size, options.gravity.cuda ? "cuda" : "cpu",
                         hpxSize(), benchmark.repeats, options.gravity.openingAngle,
                         options.gravity.smoothingLength, detail::seconds(loadStart, loadEnd),
                         domainSeconds, "Repeat", "Tree (s)", "Force (s)", "Cleanup (s)",
                         "Total (s)", "GFLOP/s", std::string(75, '-'))
                  << std::flush;
        std::vector<double> times;
        double treeTotal = 0.0;
        double forceTotal = 0.0;
        double cleanupTotal = 0.0;
        FlopCount solveFlops = 0;
        size_t treeNodes = 0;
        size_t treeLeaves = 0;
        FlopCounts totalCounts{};
        for (int repeat = 0; repeat < benchmark.repeats; ++repeat) {
            profile::checkForInterrupt();
            resetFlops();
            auto const start = detail::Clock::now();
            double treeSeconds, forceSeconds;
            try {
                auto const built =
                    treeCreate(TreeCreateParameters(0, float(options.gravity.openingAngle)));
                auto const treeEnd = detail::Clock::now();
                treeNodes = built.nodeCount;
                treeLeaves = built.leafCount;
                KickParameters parameters;
                parameters.htime = true;
                parameters.theta = options.gravity.openingAngle;
                parameters.minRung = 0;
                parameters.t0 = 1.0f;
                parameters.gm = options.gravity.gm;
                parameters.eta = options.gravity.timestepEta;
                parameters.gravitySmoothingLength = options.gravity.smoothingLength;
                parameters.saveForce = true;
                parameters.firstCall = true;
                parameters.gpu = options.gravity.cuda;
                parameters.nodeLoad = 1.0f;
                parameters.ascending = parameters.descending = false;
                parameters.top = true;
                parameters.doPhi = true;
                ExpansionBuffer local;
                local.zero();
                Vector3<Position> center{};
                Vector<TreeId> checks{built.id};
                // kick joins remote tasks and local/GPU completion before returning.
                kick(parameters, local, center, built.id, {}, checks, nullptr);
                auto const forceEnd = detail::Clock::now();
                treeSeconds = detail::seconds(start, treeEnd);
                forceSeconds = detail::seconds(treeEnd, forceEnd);
            } catch (...) {
                treeDestroy(true);
                particlesCacheFree();
                throw;
            }
            auto const cleanupStart = detail::Clock::now();
            treeDestroy(true);
            particlesCacheFree();
            auto const end = detail::Clock::now();
            double const elapsed = detail::seconds(start, end);
            auto const counts = readFlopCounts();
            totalCounts += counts;
            if (repeat == 0) {
                solveFlops = counts.total();
            }
            double const cleanupSeconds = detail::seconds(cleanupStart, end);
            treeTotal += treeSeconds;
            forceTotal += forceSeconds;
            cleanupTotal += cleanupSeconds;
            times.push_back(elapsed);
            std::cout << std::format("  {:>6} {:>13.6f} {:>13.6f} {:>13.3e} {:>13.6f} {:>12.2f}\n",
                             repeat + 1, treeSeconds, forceSeconds, cleanupSeconds, elapsed,
                             double(counts.total()) / elapsed * 1.0e-9)
                      << std::flush;
        }
        futures.clear();
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<CheckBenchmarkParticlesAction>(locality));
        }
        hpx::wait_all(futures.begin(), futures.end());
        for (auto& future : futures) {
            future.get();
        }
        double sum = 0.0;
        for (double const value : times) {
            sum += value;
        }
        std::sort(times.begin(), times.end());
        double const median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) * 0.5;
        std::cout << std::format("  {}\n"
                                 "  {:>6} {:>13.6f} {:>13.6f} {:>13.3e} {:>13.6f} {:>12.2f}\n"
                                 "\n  Timing summary\n"
                                 "  {:>13} {:>13} {:>13} {:>13}\n"
                                 "  {}\n"
                                 "  {:>13.6f} {:>13.6f} {:>13.6f} {:>13.6f}\n"
                                 "\n  Tree and work per solve\n"
                                 "  {:>12} {:>12} {:>12} {:>16} {:>18}\n"
                                 "  {}\n"
                                 "  {:>12} {:>12} {:>12} {:>16.2f} {:>18}\n\n",
                         std::string(75, '-'), "Mean", treeTotal / times.size(),
                         forceTotal / times.size(), cleanupTotal / times.size(), sum / times.size(),
                         double(totalCounts.total()) / sum * 1.0e-9, "Minimum (s)", "Median (s)",
                         "Mean (s)", "Maximum (s)", std::string(55, '-'), times.front(), median,
                         sum / times.size(), times.back(), "Particles", "Tree nodes", "Leaves",
                         "Particles/leaf", "FLOPs", std::string(74, '-'), count, treeNodes,
                         treeLeaves, double(count) / double(treeLeaves), solveFlops)
                  << std::flush;
        auto const resultsFile = detail::appendBenchmarkResult(test + 1,
            std::format("{} {} {:.9g} {} {} {} {} {} {} {} {} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} "
                        "{:.9e} {:.9e} {:.9e} {:.9e} {} {:.9e} {:.9e} {:.9e}",
                count, benchmark.size, options.gravity.openingAngle, options.compute.bucketSize,
                kFmmOrder, hpxSize(), hpxHardwareConcurrency(),
                options.gravity.cuda ? "cuda" : "cpu", benchmark.repeats, treeNodes, treeLeaves,
                double(count) / double(treeLeaves), options.gravity.smoothingLength,
                treeTotal / times.size(), forceTotal / times.size(), cleanupTotal / times.size(),
                times.front(), median, sum / times.size(), times.back(), solveFlops,
                double(totalCounts.total()) / sum * 1.0e-9, detail::seconds(loadStart, loadEnd),
                domainSeconds));
        std::cout << "  Results appended to: " << resultsFile << "\n\n" << std::flush;
    }
}
}    // namespace scenarios
