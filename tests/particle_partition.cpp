/* CosmicTiger - SoA particle partition correctness and CPU timing; BSL-1.0. */

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define PARTICLES_CPP
#include <cosmictiger/particles/particle_partition.hpp>
#include <cosmictiger/particles/particles.hpp>
#undef PARTICLES_CPP
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>

ParticleInteger* sphParticlesParticleIndices = nullptr;
#endif

namespace detail {

Options options;
ParticleInteger boundSize = 0;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Value>
bool sameBits(Value const& first, Value const& second) {
    return std::memcmp(&first, &second, sizeof(Value)) == 0;
}

float payload(size_t index) {
    constexpr std::array<uint32_t, 8> kPatterns = {0x00000000, 0x80000000, 0x00000001, 0x007fffff,
        0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345};
    return std::bit_cast<float>(kPatterns[index % kPatterns.size()]);
}

struct Fixture {
    std::array<std::vector<Position>, kSpatialDimensions> positions;
    std::vector<ParticleVector<float>> velocities;
    std::vector<char> rungs;
    std::array<std::vector<float>, kSpatialDimensions> accelerations;
    std::vector<float> potentials;
    std::vector<GroupInteger> localGroups;
    std::vector<GroupInteger> atomicGroups;
    std::vector<char> tracers;
    std::vector<char> types;
    std::vector<ParticleInteger> gasIndices;
    std::vector<ParticleInteger> inverseGasIndices;
    std::vector<ParticleInteger> samples;
    std::vector<StarFormationState> stars;

    explicit Fixture(size_t count)
      : velocities(count)
      , rungs(count)
      , potentials(count)
      , localGroups(count)
      , atomicGroups(count)
      , tracers(count)
      , types(count)
      , gasIndices(count)
      , inverseGasIndices(count)
      , samples(count)
      , stars(count) {
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            positions[dim].resize(count);
            accelerations[dim].resize(count);
        }
        for (size_t index = 0; index < count; ++index) {
            for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                positions[dim][index].setInteger(PositionInteger(index * 2654435761ULL + dim));
                velocities[index][dim] = payload(index + dim);
                accelerations[dim][index] = payload(index + dim + 2);
            }
            // An exact bit identity, including subnormal representations, travels in vx.
            velocities[index][0] = std::bit_cast<float>(uint32_t(index + 1));
            rungs[index] = char(index % 16);
            potentials[index] = payload(index + 5);
            localGroups[index] = GroupInteger(0x1234567800000000ULL + index);
            atomicGroups[index] = GroupInteger(0x2345678900000000ULL + index);
            tracers[index] = char(index % 113);
            types[index] = index % 2 == 0 ? kDarkMatterType : kSphType;
            gasIndices[index] = ParticleInteger(index);
            inverseGasIndices[index] = ParticleInteger(index);
            samples[index] = ParticleInteger(index * 3);
            stars[index].id = index + 1;
            stars[index].eventCounter = 0x123456789abcdef0ULL ^ index;
            stars[index].formationTime = double(index) * 0.25;
            stars[index].specificStarFormationRate = double(index) * 0.5;
            stars[index].elapsedPhysicalTime = double(index) * 0.75;
        }
    }
};

struct Binding {
    Fixture& fixture;
    std::unique_ptr<std::atomic<GroupInteger>[]> groups;

    explicit Binding(Fixture& source, bool atomicGroups = false)
      : fixture(source) {
        boundSize = ParticleInteger(source.rungs.size());
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            particlesX[dim] = source.positions[dim].data();
            particlesGravityAccelerations[dim] = saveParticleAcceleration(options.analysis) ?
                source.accelerations[dim].data() :
                nullptr;
        }
        particlesV = source.velocities.data();
        particlesR = source.rungs.data();
        particlesP = saveParticlePotential(options.analysis) ? source.potentials.data() : nullptr;
        particlesLgrp = options.analysis.groups ? source.localGroups.data() : nullptr;
        particlesTr = options.analysis.tracers ? source.tracers.data() : nullptr;
        particlesTy = options.hydro.enabled ? source.types.data() : nullptr;
        particlesSph = options.hydro.enabled ? source.gasIndices.data() : nullptr;
        particlesStarFormationStates =
            options.starFormation.enabled ? source.stars.data() : nullptr;
        particlesGravitySampleIndices =
            options.gravity.errorInterval > 0 ? source.samples.data() : nullptr;
        particlesGrp = nullptr;
        if (atomicGroups) {
            groups = std::make_unique<std::atomic<GroupInteger>[]>(source.rungs.size());
            for (size_t index = 0; index < source.rungs.size(); ++index) {
                groups[index].store(source.atomicGroups[index], std::memory_order_relaxed);
            }
            particlesGrp = groups.get();
        }
#if COSMICTIGER_WITH_SPH
        sphParticlesParticleIndices = source.inverseGasIndices.data();
#endif
    }

    ~Binding() {
        if (groups) {
            for (size_t index = 0; index < fixture.rungs.size(); ++index) {
                fixture.atomicGroups[index] = groups[index].load(std::memory_order_relaxed);
            }
        }
        particlesGrp = nullptr;
    }
};

// The original HEAD implementation. Keep the call boundary comparable to production.
[[gnu::noinline]] ParticleInteger originalPartition(
    Pair<ParticleInteger> range, Position split, int dimension) {
    ParticleInteger lo = range.first;
    ParticleInteger hi = range.second;
    auto& positions = particlesX[dimension];
    while (lo < hi) {
        if (positions[lo] >= split) {
            while (lo != hi) {
                --hi;
                if (positions[hi] < split) {
                    particlesSwap(hi, lo);
                    break;
                }
            }
        }
        ++lo;
    }
    return hi;
}

void verifyPermutation(Fixture const& result, Fixture const& original, Pair<ParticleInteger> range,
    bool atomicGroups) {
    std::vector<bool> seen(original.rungs.size(), false);
    for (size_t index = 0; index < original.rungs.size(); ++index) {
        size_t const identity = std::bit_cast<uint32_t>(result.velocities[index][0]) - 1;
        require(identity < original.rungs.size(), "invalid particle identity");
        require(!seen[identity], "duplicated particle identity");
        seen[identity] = true;
        if (index < size_t(range.first) || index >= size_t(range.second)) {
            require(identity == index, "partition changed a particle outside its subrange");
        } else {
            require(identity >= size_t(range.first) && identity < size_t(range.second),
                "partition brought an outside particle into its subrange");
        }
        require(sameBits(result.velocities[index], original.velocities[identity]),
            "velocity representations detached from particle");
        require(result.rungs[index] == original.rungs[identity], "rung detached from particle");
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            require(sameBits(result.positions[dim][index], original.positions[dim][identity]),
                "position detached from particle");
            size_t const source = saveParticleAcceleration(options.analysis) ? identity : index;
            require(sameBits(result.accelerations[dim][index], original.accelerations[dim][source]),
                "acceleration representations changed");
        }
        size_t source = saveParticlePotential(options.analysis) ? identity : index;
        require(sameBits(result.potentials[index], original.potentials[source]),
            "potential representations changed");
        source = options.analysis.groups ? identity : index;
        require(result.localGroups[index] == original.localGroups[source], "local group changed");
        source = atomicGroups ? identity : index;
        require(
            result.atomicGroups[index] == original.atomicGroups[source], "atomic group changed");
        source = options.analysis.tracers ? identity : index;
        require(result.tracers[index] == original.tracers[source], "tracer changed");
        source = options.hydro.enabled ? identity : index;
        require(result.gasIndices[index] == original.gasIndices[source], "gas index changed");
        require(result.types[index] == original.types[source], "particle type changed");
        source = options.gravity.errorInterval > 0 ? identity : index;
        require(result.samples[index] == original.samples[source], "gravity sample index changed");
        source = options.starFormation.enabled ? identity : index;
        require(sameBits(result.stars[index], original.stars[source]), "star state changed");
#if COSMICTIGER_WITH_SPH
        if (options.hydro.enabled && result.types[index] == kSphType) {
            require(result.inverseGasIndices[result.gasIndices[index]] == ParticleInteger(index),
                "inverse gas mapping detached from particle");
        }
#endif
    }
}

void checkCase(Fixture const& original, Pair<ParticleInteger> range, Position split, int dimension,
    bool atomicGroups) {
    Fixture actual = original;
    ParticleInteger boundary;
    {
        Binding binding(actual, atomicGroups);
        boundary = particlesPartition(range, split, dimension);
    }
    ParticleInteger expected = range.first;
    for (ParticleInteger index = range.first; index < range.second; ++index) {
        expected += original.positions[dimension][index] < split;
    }
    require(boundary == expected, "incorrect partition boundary");
    for (ParticleInteger index = range.first; index < range.second; ++index) {
        require((actual.positions[dimension][index] < split) == (index < boundary),
            "incorrect partition membership");
    }
    verifyPermutation(actual, original, range, atomicGroups);
}

size_t checkCorrectness() {
    size_t cases = 0;
    Position const split(0.5);
    for (unsigned fields = 0; fields < 5; ++fields) {
        for (int fallback = 0; fallback < 8; ++fallback) {
            options = {};
            options.analysis.saveAcceleration = (fields & 1) != 0;
            options.analysis.savePotential = (fields & 2) != 0;
            options.analysis.saveForce = fields == 4;
            options.hydro.enabled = fallback == 1 || fallback == 7;
            options.starFormation.enabled = fallback == 2 || fallback == 7;
            options.analysis.groups = fallback == 3 || fallback == 7;
            options.analysis.tracers = fallback == 4 || fallback == 7;
            options.gravity.errorInterval = (fallback == 5 || fallback == 7) ? 1 : 0;
            bool const atomicGroups = fallback == 6 || fallback == 7;
            for (int dimension = 0; dimension < kSpatialDimensions; ++dimension) {
                // Exhaust every binary arrangement through 10 particles for the fast path;
                // fallback gates use the same edge coverage through six particles.
                int const maximum = fallback == 0 ? 10 : 6;
                for (int count = 0; count <= maximum; ++count) {
                    for (unsigned pattern = 0; pattern < (1u << count); ++pattern) {
                        for (ParticleInteger offset : {0, 3}) {
                            Fixture fixture(size_t(count + offset + (offset == 0 ? 0 : 3)));
                            for (int index = 0; index < count; ++index) {
                                bool const high = (pattern & (1u << index)) != 0;
                                fixture.positions[dimension][size_t(offset + index)] = high ?
                                    (index % 2 == 0 ? split : Position(0.75)) :
                                    Position(0.25);
                            }
                            checkCase(
                                fixture, {offset, offset + count}, split, dimension, atomicGroups);
                            ++cases;
                        }
                    }
                }
            }
        }
    }
    // An empty allocation has null backing arrays, unlike the guarded empty subranges.
    options = {};
    Fixture empty(0);
    checkCase(empty, {0, 0}, split, 0, false);
    ++cases;
    // Test long ranges and raw fixed-point ticks immediately either side of the split.
    options.analysis.saveForce = true;
    std::mt19937_64 random(0x541282eb);
    for (int dimension = 0; dimension < kSpatialDimensions; ++dimension) {
        for (int count : {127, 128, 255, 256, 257, 4096}) {
            Fixture fixture(size_t(count + 8));
            for (int index = 3; index < count + 3; ++index) {
                PositionInteger const raw = split.raw() + PositionInteger(int(random() % 3) - 1);
                fixture.positions[dimension][index].setInteger(raw);
            }
            checkCase(fixture, {3, count + 3}, split, dimension, false);
            ++cases;
        }
    }
    return cases;
}

using Partition = ParticleInteger (*)(Pair<ParticleInteger>, Position, int);

uint64_t partitionTree(Partition partition, Pair<ParticleInteger> range,
    std::array<double, kSpatialDimensions> lower, std::array<double, kSpatialDimensions> upper,
    int depth = 0) {
    if (range.second - range.first <= 256 || depth == 64) {
        return uint64_t(range.second - range.first);
    }
    int const dimension = depth % kSpatialDimensions;
    double const middle = (lower[dimension] + upper[dimension]) * 0.5;
    ParticleInteger const boundary = partition(range, Position(middle), dimension);
    auto leftUpper = upper;
    auto rightLower = lower;
    leftUpper[dimension] = middle;
    rightLower[dimension] = middle;
    return uint64_t(boundary) +
        partitionTree(partition, {range.first, boundary}, lower, leftUpper, depth + 1) +
        partitionTree(partition, {boundary, range.second}, rightLower, upper, depth + 1);
}

uint64_t runWorkload(Partition partition, ParticleInteger count, bool tree) {
    if (!tree) {
        return uint64_t(partition({0, count}, Position(0.5), 0));
    }
    std::array<double, kSpatialDimensions> lower{};
    std::array<double, kSpatialDimensions> upper;
    upper.fill(1.0);
    return partitionTree(partition, {0, count}, lower, upper);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t const middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
}

void benchmark(size_t count, int repeats) {
    require(count > 0 && count < size_t(std::numeric_limits<ParticleInteger>::max()),
        "benchmark particle count is out of range");
    require(count < 0x7f800000u && repeats >= 2,
        "benchmark requires unique IDs and at least two repeats");
    options = {};
    options.analysis.saveForce = true;
    std::printf(
        "# single CPU thread, no GPU; restoration and binding excluded; force fields enabled\n");
    std::printf("# particles distribution workload repeats original_median_s production_median_s "
                "ratio original_min_s production_min_s\n");
    for (int distribution = 0; distribution < 3; ++distribution) {
        Fixture input(count);
        std::mt19937_64 random(0x651423a);
        for (size_t index = 0; index < count; ++index) {
            for (int dimension = 0; dimension < kSpatialDimensions; ++dimension) {
                double coordinate = double(random() >> 11) * 0x1p-53;
                if (distribution == 1) {
                    coordinate = (index % 2 == 0 ? 0.125 : 0.875) + (coordinate - 0.5) * 0.1;
                }
                input.positions[dimension][index] = Position(coordinate);
            }
        }
        if (distribution == 2) {
            Binding binding(input);
            runWorkload(originalPartition, ParticleInteger(count), true);
        }
        for (bool tree : {false, true}) {
            Fixture work = input;
            std::array<std::vector<double>, 2> elapsed;
            std::array<uint64_t, 2> sums{};
            for (int repeat = -1; repeat < repeats; ++repeat) {
                for (int order = 0; order < 2; ++order) {
                    int const implementation = (order + repeat + 1) % 2;
                    work = input;
                    Binding binding(work);
                    Partition const partition =
                        implementation == 0 ? originalPartition : particlesPartition;
                    auto const start = std::chrono::steady_clock::now();
                    uint64_t const sum = runWorkload(partition, ParticleInteger(count), tree);
                    auto const stop = std::chrono::steady_clock::now();
                    sums[implementation] = sum;
                    if (repeat >= 0) {
                        elapsed[implementation].push_back(
                            std::chrono::duration<double>(stop - start).count());
                    }
                }
                require(sums[0] == sums[1], "benchmark partition tree boundaries differ");
            }
            double const original = median(elapsed[0]);
            double const production = median(elapsed[1]);
            char const* name = distribution == 0 ? "random" :
                distribution == 1                ? "clustered" :
                                                   "prepartitioned";
            std::printf("%zu %s %s %d %.9f %.9f %.5f %.9f %.9f\n", count, name,
                tree ? "tree" : "single", repeats, original, production, production / original,
                *std::min_element(elapsed[0].begin(), elapsed[0].end()),
                *std::min_element(elapsed[1].begin(), elapsed[1].end()));
            std::fflush(stdout);
        }
    }
}

}    // namespace detail

Options const& getOptions() {
    return detail::options;
}

ParticleInteger particlesSize() {
    return detail::boundSize;
}

#if COSMICTIGER_WITH_SPH
ParticleInteger sphParticlesSize() {
    return detail::boundSize;
}
#endif

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--benchmark") {
            size_t const count = argc > 2 ? std::stoull(argv[2]) : 1048576;
            int const repeats = argc > 3 ? std::stoi(argv[3]) : 7;
            detail::benchmark(count, repeats);
        } else {
            detail::require(
                argc == 1, "usage: particle_partition_test [--benchmark [count [repeats]]]");
            size_t const cases = detail::checkCorrectness();
            std::printf(
                "particle partition: %zu cases passed (dimensions=%d, position_bits=%d, sph=%d)\n",
                cases, kSpatialDimensions, kPositionBits, COSMICTIGER_WITH_SPH);
        }
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "particle partition: %s\n", error.what());
        return 1;
    }
}
