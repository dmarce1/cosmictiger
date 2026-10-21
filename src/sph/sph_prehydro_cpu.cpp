/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/sph_cpu.hpp>

// Monaghan (1992), ARA&A 30, 543; cosmictiger/monaghan_sph.hpp gives the full citation
// and the notation. Particle a is the sink and particle b a pack of staged neighbors.
// NOLINTBEGIN(readability-identifier-naming)
namespace detail {

// Section 6: the resolution length follows the local density, h_a ∝ ρ_a^(−1/ν) with
// ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2 or 6.7, the solve holds
//   (4π/3) ρ_a H_a³/m = N_target,   i.e.   ΣW = Σ_b w(r_ab/H_a) = 3 N_target/(4π)
// for the selected kernel W(r, H) = w(r/H)/H³, by a bounded Newton iteration on the
// support radius H_a with dΣW/dH_a = −Σ_b q_ab w'(q_ab)/H_a, q_ab = r_ab/H_a. The bounds
// limit the support radius without modifying the density sum.
bool solveSphCpuSupportRadius(SphRunParameters const& parameters, SphRunData const& data,
    SphTreeNode const& targetNode, std::vector<int> const& neighbors, int particleIndex,
    float& supportRadius, FlopCounter<>& flops) {
    float const minimumSupportRadius = parameters.minimumSupportRadius;
    float const maximumSupportRadius = parameters.maximumSupportRadius;
    ALWAYS_ASSERT(minimumSupportRadius > 0.0f && maximumSupportRadius >= minimumSupportRadius &&
        data.targetNeighborCount > 0.0f);
    float& H_a = supportRadius;
    H_a = std::min(std::max(H_a, minimumSupportRadius), maximumSupportRadius);
    Array<Fixed32, NDIM> position;
    position[XDIM] = data.sourcePositionX[particleIndex];
    position[YDIM] = data.sourcePositionY[particleIndex];
    position[ZDIM] = data.sourcePositionZ[particleIndex];
    // Distances do not change during the bounded support-radius solve.
    std::vector<SimdFloat> r_ab;    // |r_ab| for each pack of neighbors
    r_ab.reserve((neighbors.size() + SimdFloat::size() - 1) / SimdFloat::size());
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        Vector3<SimdFloat> const separation =
            sphCpuSeparations(data, particleIndex, neighbors, first);
        r_ab.push_back(simdSqrt(separation.dot(separation)));
        flops += 9 * SimdFloat::size();    // Three fixed scales, dot product, square root.
    }
    float lower = minimumSupportRadius;
    float upper = maximumSupportRadius;
    float const ΣW_target = data.targetNeighborCount * float(3.0 / (4.0 * M_PI));
    flops += 1;
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < NDIM; ++dim) {
            flops += 1;    // Float-to-fixed scaling; fixed-point arithmetic is integer.
            if (targetNode.outerBox.end[dim] <
                RangeFixed(position[dim]) + RangeFixed(H_a) + RangeFixed::min()) {
                return false;
            }
            flops += 1;
            if (RangeFixed(position[dim]) <
                targetNode.outerBox.begin[dim] + RangeFixed(H_a) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const one_over_H_a(1.0f / H_a);
        flops += 1;
        SimdFloat ΣW_pack(0.0f);
        SimdFloat dΣW_dH_a_pack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const q_ab = r_ab[first / SimdFloat::size()] * one_over_H_a;
            auto const contributes = sphCpuValidLanes(neighbors, first) && q_ab < SimdFloat(1.0f);
            ΣW_pack += simdSelect(contributes, kernelW(q_ab), SimdFloat(0.0f));
            dΣW_dH_a_pack -=
                simdSelect(contributes, q_ab * dkernelWDq(q_ab) * one_over_H_a, SimdFloat(0.0f));
            // q, two accumulations and the two derivative products, in every lane.
            flops += 5 * SimdFloat::size() +
                sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab) +
                sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab, true);
        }
        float const residual = simdSum(ΣW_pack) - ΣW_target;
        flops += SimdFloat::size() - 1 +
            2;    // Reduction, residual and tolerance; abs is a sign operation.
        if (std::abs(residual) <= SPH_SUPPORT_RADIUS_TOLERANCE * ΣW_target ||
            (H_a <= minimumSupportRadius && residual >= 0.0f) ||
            (H_a >= maximumSupportRadius && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = H_a;
        } else {
            lower = H_a;
        }
        float const dΣW_dH_a = simdSum(dΣW_dH_a_pack);
        flops += SimdFloat::size() - 1 + 4;    // Reduction, change, two bounds, next H.
        float change = dΣW_dH_a > 0.0f ? -residual / dΣW_dH_a : 0.2f * H_a;
        change = std::min(std::max(change, -0.2f * H_a), 0.2f * H_a);
        float next = std::min(std::max(H_a + change, minimumSupportRadius), maximumSupportRadius);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == H_a) {
            return true;
        }
        H_a = next;
    }
    PRINT("SPH smoothing-length solve did not converge: supportRadius=%e bounds=[%e,%e] "
          "target=%e\n",
        H_a, minimumSupportRadius, maximumSupportRadius, data.targetNeighborCount);
    ALWAYS_ASSERT(false);
    return false;
}

bool sphCpuHasActiveNeighbor(SphRunParameters const& parameters, SphRunData const& data,
    std::vector<int> const& neighbors, int particleIndex, float supportRadius,
    FlopCounter<>& flops) {
    SimdFloat const squaredSupportRadius(supportRadius * supportRadius);
    flops += 1;
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const separation = sphCpuSeparations(data, particleIndex, neighbors, first);
        auto const squaredSeparation = separation.dot(separation);
        auto const neighborSupport = sphCpuLoad(data.sourceSupportRadius, neighbors, first, 0.0f);
        SimdFloat const neighborRung([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
        });
        auto const active = neighborRung >= SimdFloat(float(parameters.minimumRung));
        auto const intersects =
            squaredSeparation < simdMax(squaredSupportRadius, sqr(neighborSupport));
        flops += 9 * SimdFloat::size();    // Three fixed scales, five dot ops, support square.
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
    FlopCounter<> flops;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, false, flops);
    for (int particleIndex = targetNode.particleRange.first;
         particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] >= parameters.minimumRung;
        if (active && !data.sinkSmoothingConverged[sinkIndex]) {
            float& supportRadius = data.sinkThermodynamicStates[sinkIndex].supportRadius;
            bool const converged = detail::solveSphCpuSupportRadius(
                parameters, data, targetNode, neighbors, particleIndex, supportRadius, flops);
            data.sinkSmoothingConverged[sinkIndex] = converged;
            if (converged) {
                result.minimumSupportRadius = std::min(result.minimumSupportRadius, supportRadius);
                result.maximumSupportRadius = std::max(result.maximumSupportRadius, supportRadius);
            } else {
                result.needsAnotherPass = true;
            }
        }
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops);
    return result;
}

SphRunReturn sphCpuPrehydro2(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    FlopCounter<> flops;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    for (int particleIndex = targetNode.particleRange.first;
         particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] >= parameters.minimumRung;
        float& supportRadius = data.sinkThermodynamicStates[sinkIndex].supportRadius;
        bool const semiactive = active ||
            detail::sphCpuHasActiveNeighbor(
                parameters, data, neighbors, particleIndex, supportRadius, flops);
        data.sinkSemiactive[sinkIndex] = semiactive;
        if (!semiactive) {
            continue;
        }
        if (!active && !data.sinkSmoothingConverged[sinkIndex] &&
            !detail::solveSphCpuSupportRadius(
                parameters, data, targetNode, neighbors, particleIndex, supportRadius, flops)) {
            data.sinkSmoothingConverged[sinkIndex] = false;
            result.needsAnotherPass = true;
            continue;
        }
        // Equation 3.8 (the interpolant 2.7 at r_a): ρ_a = Σ_b m_b W_ab, with the sink's own
        // support radius in W(r, H_a) = w(r/H_a)/H_a³, a gather form of equation 6.3 (which
        // uses h_b). Every gas particle has mass m_b = gasMass, and the sum includes b = a.
        float const H_a = supportRadius;
        float const one_over_H_a = 1.0f / H_a;
        SimdFloat const m_b_over_H_a3(data.gasMass * one_over_H_a * one_over_H_a * one_over_H_a);
        flops += 4;    // Reciprocal and three mass/support products.
        SimdFloat ρ_a_pack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            Vector3<SimdFloat> const r_ab =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const q_ab = simdSqrt(r_ab.dot(r_ab)) * SimdFloat(one_over_H_a);
            ρ_a_pack += simdSelect(detail::sphCpuValidLanes(neighbors, first),
                m_b_over_H_a3 * kernelW(q_ab), SimdFloat(0.0f));
            flops +=
                12 * SimdFloat::size() + sph_flops::densityKernelFlops(selectedHydroKernel(), q_ab);
        }
        float const ρ_a = simdSum(ρ_a_pack);
        flops += SimdFloat::size() - 1;
        ALWAYS_ASSERT(H_a > 0.0f && ρ_a > 0.0f);
        data.sinkDensity[sinkIndex] = ρ_a;
        data.sinkSmoothingConverged[sinkIndex] = true;
        result.minimumSupportRadius = std::min(result.minimumSupportRadius, H_a);
        result.maximumSupportRadius = std::max(result.maximumSupportRadius, H_a);
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops);
    return result;
}
// NOLINTEND(readability-identifier-naming)
