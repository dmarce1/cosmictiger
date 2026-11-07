/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cosmictiger/math/math.hpp>
#include <cosmictiger/sph/cpu.hpp>

SphRunReturn sphCpuRungs(SphRunParameters const&, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    sph_flops::Counter flops;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkRung = data.sourceRungs[particleIndex];
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        float const smoothingLength = data.sinkSmoothingLength[sinkIndex];
        SimdFloat const squaredSmoothingLength(smoothingLength * smoothingLength);
        flops += 1;
        SimdFloat maximumNeighborRungs(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const separation =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            auto const squaredSeparation =
                sqr(separation[kXi], separation[kYi], separation[kZi]);
            auto const neighborSupport =
                detail::sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 0.0f);
            SimdFloat const neighborRungs([&](auto lane) {
                size_t const index = first + size_t(lane);
                return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
            });
            auto const intersects = detail::sphCpuValidLanes(neighbors, first) &&
                squaredSeparation <= simdMax(squaredSmoothingLength, sqr(neighborSupport));
            flops += 9 * SimdFloat::size();    // Includes arithmetic in padded SIMD lanes.
            maximumNeighborRungs = simdMax(
                maximumNeighborRungs, simdSelect(intersects, neighborRungs, SimdFloat(0.0f)));
        }
        int const maximumNeighborRung = int(simdMaximum(maximumNeighborRungs));
        // Locally derived monotone neighbor-graph closure, including inactive endpoints:
        //   rᵢ⁽ᵏ⁺¹⁾ = max(rᵢ⁽ᵏ⁾, max_{j: dᵢⱼ ≤ max(Hᵢ,Hⱼ)} rⱼ⁽ᵏ⁾ − 1).
        // rᵢ -> sinkRung; max rⱼ -> maximumNeighborRung; H -> smoothingLength;
        // dᵢⱼ² -> squaredSeparation. Each pass reads a frozen source snapshot.
        // Candidate storage defers live-rung changes until the driver schedules them.
        if (sinkRung < maximumNeighborRung - kMaxNeighborRungDifference) {
            auto* const candidates =
                data.sinkRungCandidates ? data.sinkRungCandidates : data.sinkParticleRungs;
            candidates[data.sinkParticleIndices[sinkIndex]] =
                maximumNeighborRung - kMaxNeighborRungDifference;
            result.needsAnotherPass = true;
        }
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}
