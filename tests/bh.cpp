#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/bh.hpp>
#include <cosmictiger/gravity/gravity.hpp>

namespace detail {
using BhPositions = DeviceVector<Array<float, NDIM>>;

Array<float, NDIM> bhPosition(float x, float y, float z) {
    Array<float, NDIM> position;
    position[0] = x;
    position[1] = y;
    position[2] = z;
    return position;
}

void requireBhClose(char const* label, double actual, double expected, double tolerance) {
    auto const& options = getOptions();
    double const roundingScale = options.gravity.gm / options.gravity.supportRadius;
    // The application enables fast-math, so inspect exponent bits rather than
    // allowing the compiler to assume std::isfinite is always true.
    constexpr std::uint64_t kExponentMask = 0x7ff0000000000000ULL;
    bool const finite = (std::bit_cast<std::uint64_t>(actual) & kExponentMask) != kExponentMask;
    if (!finite ||
        std::abs(actual - expected) > tolerance * std::abs(expected) + 1.e-6 * roundingScale) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

double directBhPotential(Array<float, NDIM> const& sink, BhPositions const& sources, int selfIndex,
    bool skipCoincident) {
    auto const& options = getOptions();
    double const support = options.gravity.supportRadius;
    double potential = 0.0;
    for (int j = 0; j < sources.size(); ++j) {
        if (j == selfIndex) {
            continue;
        }
        double distanceSquared = 0.0;
        for (int dim = 0; dim < NDIM; ++dim) {
            double const delta = double(sink[dim]) - sources[j][dim];
            distanceSquared += delta * delta;
        }
        if (skipCoincident && distanceSquared == 0.0) {
            continue;
        }
        potential -=
            radial_kernel::potential(selectedGravityKernel(), std::sqrt(distanceSquared) / support);
    }
    return potential * options.gravity.gm / support;
}

void compareBhSelf(
    char const* label, BhPositions const& positions, bool requestGpu, double tolerance) {
    auto sortedPositions = positions;
    auto const potentials = bhEvaluatePotential(sortedPositions, requestGpu);
    for (int i = 0; i < positions.size(); ++i) {
        requireBhClose(
            label, potentials[i], directBhPotential(positions[i], positions, i, false), tolerance);
    }
}

BhPositions makeBhCloud(int count, float width) {
    std::mt19937 generator(176119);
    std::uniform_real_distribution<float> distribution(-width, width);
    BhPositions positions(count);
    for (int i = 0; i < count; ++i) {
        for (int dim = 0; dim < NDIM; ++dim) {
            positions[i][dim] = distribution(generator);
        }
    }
    return positions;
}

void compareBhPoints(char const* label, BhPositions const& sources, BhPositions const& sinks,
    bool requestGpu, double tolerance) {
    auto sortedSources = sources;
    auto sinkCopy = sinks;
    auto const potentials = bhEvaluatePoints(sinkCopy, sortedSources, requestGpu);
    for (int i = 0; i < sinks.size(); ++i) {
        requireBhClose(
            label, potentials[i], directBhPotential(sinks[i], sources, -1, true), tolerance);
    }
}

#ifdef USE_CUDA
void verifyBhCudaWorkspaceReuse() {
    float const support = getOptions().gravity.supportRadius;
    BhPositions positions(2 * BH_BUCKET_SIZE);
    for (int i = 0; i < positions.size(); ++i) {
        positions[i] = bhPosition(0.1f * support * i, 0.0f, 0.0f);
    }
    // Use the production kernel with two valid leaf buckets, and bound its
    // worker count so both buckets necessarily reuse the same workspace.
    DeviceVector<BhTreeNode> nodes(3);
    nodes[0].children[LEFT] = 1;
    nodes[0].children[RIGHT] = 2;
    for (int nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        auto& node = nodes[nodeIndex];
        int const begin = nodeIndex == 2 ? BH_BUCKET_SIZE : 0;
        int const end = nodeIndex == 1 ? BH_BUCKET_SIZE : positions.size();
        node.parts = {begin, end};
        node.mass = end - begin;
        node.pos = bhPosition(0.05f * support * (begin + end - 1), 0.0f, 0.0f);
        node.radius = 0.05f * support * (end - begin - 1) * 1.00001f;
        node.meanSquaredRadius = 0.0f;
        for (int i = begin; i < end; ++i) {
            float const delta = positions[i][0] - node.pos[0];
            node.meanSquaredRadius += delta * delta / node.mass;
        }
    }
    DeviceVector<int> sinkBuckets(2);
    sinkBuckets[0] = 1;
    sinkBuckets[1] = 2;
    auto const potentials = bhEvaluatePotentialGpu(nodes, positions, sinkBuckets,
        kBhMaximumRelativePotentialError, support, getOptions().gravity.gm, 1);
    for (int i = 0; i < positions.size(); ++i) {
        requireBhClose("CUDA workspace reused across self-potential buckets", potentials[i],
            directBhPotential(positions[i], positions, i, false), 3.e-5);
    }
    // The production points launcher creates at most sinkCount/64 blocks;
    // every worker must therefore evaluate many sinks using its workspace.
    auto sinks = makeBhCloud(8192, 4.0f * support);
    compareBhPoints("CUDA workspace reused across point sinks", positions, sinks, true, 0.04);
}
#endif
}    // namespace detail

void bhPotentialTest() {
    auto const& options = getOptions();
    float const support = options.gravity.supportRadius;
    detail::BhPositions empty;
    if (bhEvaluatePotential(empty, true).size() != 0) {
        throw std::runtime_error("Empty halo potential must be empty");
    }
    detail::BhPositions singleton(1);
    singleton[0] = detail::bhPosition(0.0f, 0.0f, 0.0f);
    detail::compareBhSelf("Singleton has no self potential", singleton, true, 0.0);
    detail::compareBhPoints("Empty source potential", empty, singleton, true, 0.0);
    detail::BhPositions isolatedSink(1);
    isolatedSink[0] = detail::bhPosition(0.3f * support, 0.0f, 0.0f);
    detail::compareBhPoints("Single softened source", singleton, isolatedSink, true, 3.e-5);

    auto softened = detail::makeBhCloud(192, 0.25f * support);
    detail::compareBhSelf("Overlapping support must be evaluated directly", softened, false, 3.e-5);
    auto extended = detail::makeBhCloud(640, 32.0f * support);
    detail::compareBhSelf("BSP potential satisfies paper error budget", extended, false, 0.04);
    auto filament = extended;
    for (auto& position : filament) {
        position[1] *= 0.0001f;
        position[2] *= 0.0001f;
    }
    detail::compareBhSelf("Anisotropic BSP potential error", filament, false, 0.04);
    auto probes = detail::makeBhCloud(128, 50.0f * support);
    detail::compareBhPoints("Point potential error", extended, probes, false, 0.04);

    detail::BhPositions coincident(192);
    for (auto& position : coincident) {
        position = detail::bhPosition(support, support, support);
    }
    detail::compareBhSelf(
        "Coincident particles terminate and retain mutual potential", coincident, false, 3.e-5);
    // Requesting CUDA while compute.cuda=false must still run the CPU route.
    detail::compareBhSelf("Runtime backend selection", extended, true, 0.04);
#ifdef USE_CUDA
    if (options.compute.cuda) {
        detail::compareBhSelf("CUDA softened potential", softened, true, 3.e-5);
        detail::compareBhSelf("CUDA degenerate tree", coincident, true, 3.e-5);
        detail::verifyBhCudaWorkspaceReuse();
    }
#endif
    std::printf("Barnes-Hut production potential tests passed (%s).\n",
        options.compute.cuda ? "CPU and CUDA" : "CPU");
}
