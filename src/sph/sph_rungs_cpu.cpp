/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cosmictiger/math/math.hpp>
#include <cosmictiger/sph/sph_cpu.hpp>

SphRunReturn sphCpuRungs(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors =
        detail::sphCpuNeighbors(data, targetNode, true, true, parameters.minimumRung);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkRung = data.sourceRungs[particleIndex];
        if (sinkRung < parameters.minimumRung) {
            continue;
        }
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        float const supportRadius = data.sinkThermodynamicStates[sinkIndex].supportRadius;
        SimdFloat const squaredSupportRadius(supportRadius * supportRadius);
        SimdFloat maximumNeighborRungs(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const separation =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            auto const squaredSeparation =
                sqr(separation[XDIM], separation[YDIM], separation[ZDIM]);
            auto const neighborSupport =
                detail::sphCpuLoad(data.sourceSupportRadius, neighbors, first, 0.0f);
            SimdFloat const neighborRungs([&](auto lane) {
                size_t const index = first + size_t(lane);
                return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
            });
            auto const intersects = detail::sphCpuValidLanes(neighbors, first) &&
                squaredSeparation < simdMax(squaredSupportRadius, sqr(neighborSupport));
            maximumNeighborRungs = simdMax(
                maximumNeighborRungs, simdSelect(intersects, neighborRungs, SimdFloat(0.0f)));
        }
        int const maximumNeighborRung = int(simdMaximum(maximumNeighborRungs));
        if (sinkRung < maximumNeighborRung - MAX_NEIGHBOR_RUNG_DIFFERENCE) {
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] =
                maximumNeighborRung - MAX_NEIGHBOR_RUNG_DIFFERENCE;
            result.needsAnotherPass = true;
        }
        result.floatingPointOperations += 9.0 * double(neighbors.size());
    }
    return result;
}
