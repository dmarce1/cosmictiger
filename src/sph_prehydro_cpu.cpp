/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/sph_cpu.hpp>

namespace detail {

bool solveSphCpuSupportRadius(SphRunParameters const& parameters, SphRunData const& data,
    SphTreeNode const& targetNode, std::vector<int> const& neighbors, int particleIndex,
    float& supportRadius) {
    float const minimumSupportRadius = parameters.minimumSupportRadius;
    float const maximumSupportRadius = parameters.maximumSupportRadius;
    ALWAYS_ASSERT(minimumSupportRadius > 0.0f && maximumSupportRadius >= minimumSupportRadius &&
        data.targetNeighborCount > 0.0f);
    supportRadius = std::min(std::max(supportRadius, minimumSupportRadius), maximumSupportRadius);
    Array<Fixed32, NDIM> position;
    position[XDIM] = data.sourcePositionX[particleIndex];
    position[YDIM] = data.sourcePositionY[particleIndex];
    position[ZDIM] = data.sourcePositionZ[particleIndex];
    // Distances do not change during the bounded support-radius solve.
    std::vector<SimdFloat> radii;
    radii.reserve((neighbors.size() + SimdFloat::size() - 1) / SimdFloat::size());
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const separation = sphCpuSeparations(data, particleIndex, neighbors, first);
        radii.push_back(simdSqrt(sqr(separation[XDIM], separation[YDIM], separation[ZDIM])));
    }
    float lower = minimumSupportRadius;
    float upper = maximumSupportRadius;
    float const target = data.targetNeighborCount * float(3.0 / (4.0 * M_PI));
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < NDIM; ++dim) {
            if (targetNode.outerBox.end[dim] <
                    RangeFixed(position[dim]) + RangeFixed(supportRadius) + RangeFixed::min() ||
                RangeFixed(position[dim]) < targetNode.outerBox.begin[dim] +
                        RangeFixed(supportRadius) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const inverseSupportRadius(1.0f / supportRadius);
        SimdFloat sum(0.0f);
        SimdFloat derivative(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const normalizedRadius = radii[first / SimdFloat::size()] * inverseSupportRadius;
            auto const contributes =
                sphCpuValidLanes(neighbors, first) && normalizedRadius < SimdFloat(1.0f);
            sum += simdSelect(contributes, kernelW(normalizedRadius), SimdFloat(0.0f));
            derivative -= simdSelect(contributes,
                normalizedRadius * dkernelWDq(normalizedRadius) * inverseSupportRadius,
                SimdFloat(0.0f));
        }
        float const residual = simdSum(sum) - target;
        if (std::abs(residual) <= SPH_SUPPORT_RADIUS_TOLERANCE * target ||
            (supportRadius <= minimumSupportRadius && residual >= 0.0f) ||
            (supportRadius >= maximumSupportRadius && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = supportRadius;
        } else {
            lower = supportRadius;
        }
        float const derivativeSum = simdSum(derivative);
        float change = derivativeSum > 0.0f ? -residual / derivativeSum : 0.2f * supportRadius;
        change = std::min(std::max(change, -0.2f * supportRadius), 0.2f * supportRadius);
        float next =
            std::min(std::max(supportRadius + change, minimumSupportRadius), maximumSupportRadius);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
        }
        if (next == supportRadius) {
            return true;
        }
        supportRadius = next;
    }
    PRINT("SPH smoothing-length solve did not converge: supportRadius=%e bounds=[%e,%e] "
          "target=%e\n",
        supportRadius, minimumSupportRadius, maximumSupportRadius, data.targetNeighborCount);
    ALWAYS_ASSERT(false);
    return false;
}

bool sphCpuHasActiveNeighbor(SphRunParameters const& parameters, SphRunData const& data,
    std::vector<int> const& neighbors, int particleIndex, float supportRadius) {
    SimdFloat const squaredSupportRadius(supportRadius * supportRadius);
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const separation = sphCpuSeparations(data, particleIndex, neighbors, first);
        auto const squaredSeparation = sqr(separation[XDIM], separation[YDIM], separation[ZDIM]);
        auto const neighborSupport = sphCpuLoad(data.sourceSupportRadius, neighbors, first, 0.0f);
        SimdFloat const neighborRung([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
        });
        auto const active = neighborRung >= SimdFloat(float(parameters.minimumRung));
        auto const intersects =
            squaredSeparation < simdMax(squaredSupportRadius, sqr(neighborSupport));
        if (simdAnyOf(sphCpuValidLanes(neighbors, first) && active && intersects)) {
            return true;
        }
    }
    return false;
}

}    // namespace detail

SphRunReturn sphCpuPrehydro1(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, false);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] >= parameters.minimumRung;
        if (active && !data.sinkSmoothingConverged[sinkIndex]) {
            float& supportRadius = data.sinkThermodynamicStates[sinkIndex].supportRadius;
            bool const converged = detail::solveSphCpuSupportRadius(
                parameters, data, targetNode, neighbors, particleIndex, supportRadius);
            data.sinkSmoothingConverged[sinkIndex] = converged;
            if (converged) {
                result.minimumSupportRadius = std::min(result.minimumSupportRadius, supportRadius);
                result.maximumSupportRadius = std::max(result.maximumSupportRadius, supportRadius);
            } else {
                result.needsAnotherPass = true;
            }
        }
    }
    return result;
}

SphRunReturn sphCpuPrehydro2(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] >= parameters.minimumRung;
        float& supportRadius = data.sinkThermodynamicStates[sinkIndex].supportRadius;
        bool const semiactive = active ||
            detail::sphCpuHasActiveNeighbor(
                parameters, data, neighbors, particleIndex, supportRadius);
        data.sinkSemiactive[sinkIndex] = semiactive;
        if (!semiactive) {
            continue;
        }
        if (!active && !data.sinkSmoothingConverged[sinkIndex] &&
            !detail::solveSphCpuSupportRadius(
                parameters, data, targetNode, neighbors, particleIndex, supportRadius)) {
            data.sinkSmoothingConverged[sinkIndex] = false;
            result.needsAnotherPass = true;
            continue;
        }
        float const inverseSupportRadius = 1.0f / supportRadius;
        SimdFloat const normalization(
            data.gasMass * inverseSupportRadius * inverseSupportRadius * inverseSupportRadius);
        SimdFloat densitySum(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const separation =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            auto const normalizedRadius =
                simdSqrt(sqr(separation[XDIM], separation[YDIM], separation[ZDIM])) *
                SimdFloat(inverseSupportRadius);
            densitySum += simdSelect(detail::sphCpuValidLanes(neighbors, first),
                normalization * kernelW(normalizedRadius), SimdFloat(0.0f));
        }
        float const density = simdSum(densitySum);
        ALWAYS_ASSERT(supportRadius > 0.0f && density > 0.0f);
        data.sinkDensity[sinkIndex] = density;
        data.sinkSmoothingConverged[sinkIndex] = true;
        result.minimumSupportRadius = std::min(result.minimumSupportRadius, supportRadius);
        result.maximumSupportRadius = std::max(result.maximumSupportRadius, supportRadius);
    }
    return result;
}
