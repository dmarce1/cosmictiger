/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cosmictiger/math/math.hpp>
#include <cosmictiger/sph/cpu.hpp>

// Locally derived monotone neighbor-graph closure, including inactive endpoints:
//   L_a⁽ᵏ⁺¹⁾ = max(L_a⁽ᵏ⁾, max_{b: |r_ab| ≤ max(H_a, H_b)} L_b⁽ᵏ⁾ − 1).
// Symbol directory: L_a → sink rung; max L_b → maximum neighbor rung;
// H_a, H_b → sink, neighbor kernel smoothing length; r_ab → separation;
// r_ab² → squared separation. Code names: r2_ab = r_ab²; H2_a = H_a².
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
        float const H_a = data.sinkSmoothingLength[sinkIndex];
        SimdFloat const H2_a(H_a * H_a);
        flops += 1;
        SimdFloat maximumNeighborRungs(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const r_ab = detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            auto const r2_ab = r_ab.dot(r_ab);
            auto const H_b = detail::sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 0.0f);
            SimdFloat const neighborRungs([&](auto lane) {
                size_t const index = first + size_t(lane);
                return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
            });
            auto const intersects =
                detail::sphCpuValidLanes(neighbors, first) && r2_ab <= simdMax(H2_a, sqr(H_b));
            // Fixed scales, dot, H_b², including arithmetic in padded SIMD lanes.
            flops += (sph_flops::kVectorOperation + sph_flops::kDotProduct + 1) * SimdFloat::size();
            maximumNeighborRungs = simdMax(
                maximumNeighborRungs, simdSelect(intersects, neighborRungs, SimdFloat(0.0f)));
        }
        int const maximumNeighborRung = int(simdMaximum(maximumNeighborRungs));
        // Each pass of the closure above reads a frozen source snapshot.
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
