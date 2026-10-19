/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
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
    float& supportRadius) {
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
    }
    float lower = minimumSupportRadius;
    float upper = maximumSupportRadius;
    float const ΣW_target = data.targetNeighborCount * float(3.0 / (4.0 * M_PI));
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < NDIM; ++dim) {
            if (targetNode.outerBox.end[dim] <
                    RangeFixed(position[dim]) + RangeFixed(H_a) + RangeFixed::min() ||
                RangeFixed(position[dim]) <
                    targetNode.outerBox.begin[dim] + RangeFixed(H_a) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const one_over_H_a(1.0f / H_a);
        SimdFloat ΣW_pack(0.0f);
        SimdFloat dΣW_dH_a_pack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const q_ab = r_ab[first / SimdFloat::size()] * one_over_H_a;
            auto const contributes = sphCpuValidLanes(neighbors, first) && q_ab < SimdFloat(1.0f);
            ΣW_pack += simdSelect(contributes, kernelW(q_ab), SimdFloat(0.0f));
            dΣW_dH_a_pack -=
                simdSelect(contributes, q_ab * dkernelWDq(q_ab) * one_over_H_a, SimdFloat(0.0f));
        }
        float const residual = simdSum(ΣW_pack) - ΣW_target;
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
        float change = dΣW_dH_a > 0.0f ? -residual / dΣW_dH_a : 0.2f * H_a;
        change = std::min(std::max(change, -0.2f * H_a), 0.2f * H_a);
        float next = std::min(std::max(H_a + change, minimumSupportRadius), maximumSupportRadius);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
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
    std::vector<int> const& neighbors, int particleIndex, float supportRadius) {
    SimdFloat const squaredSupportRadius(supportRadius * supportRadius);
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
        // Equation 3.8 (the interpolant 2.7 at r_a): ρ_a = Σ_b m_b W_ab, with the sink's own
        // support radius in W(r, H_a) = w(r/H_a)/H_a³, a gather form of equation 6.3 (which
        // uses h_b). Every gas particle has mass m_b = gasMass, and the sum includes b = a.
        float const H_a = supportRadius;
        float const one_over_H_a = 1.0f / H_a;
        SimdFloat const m_b_over_H_a3(data.gasMass * one_over_H_a * one_over_H_a * one_over_H_a);
        SimdFloat ρ_a_pack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            Vector3<SimdFloat> const r_ab =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const q_ab = simdSqrt(r_ab.dot(r_ab)) * SimdFloat(one_over_H_a);
            ρ_a_pack += simdSelect(detail::sphCpuValidLanes(neighbors, first),
                m_b_over_H_a3 * kernelW(q_ab), SimdFloat(0.0f));
        }
        float const ρ_a = simdSum(ρ_a_pack);
        ALWAYS_ASSERT(H_a > 0.0f && ρ_a > 0.0f);
        data.sinkDensity[sinkIndex] = ρ_a;
        data.sinkSmoothingConverged[sinkIndex] = true;
        result.minimumSupportRadius = std::min(result.minimumSupportRadius, H_a);
        result.maximumSupportRadius = std::max(result.maximumSupportRadius, H_a);
    }
    return result;
}
// NOLINTEND(readability-identifier-naming)
