/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <vector>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/cpu.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/viscosity/morris_and_monaghan.hpp>

// Monaghan (1992), ARA&A 30, 543; cosmictiger/sph/viscosity/monaghan.hpp gives the full
// citation and the notation. Particle a is the sink and particle b a pack of staged neighbors.

using sph_geometry::SpatialVector;

namespace detail {

// clang-format off
/*
 * Section 6: the resolution length follows the local density, H_a ∝ ρ_a^(−1/ν) with
 * ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2 or 6.7, the solve holds
 *   (4π/3) ρ_a H_a³/m = N_target,   i.e.   ΣW = Σ_b w(|r_ab|/H_a) = 3 N_target/(4π)
 * for the selected kernel W(r, H) = w(r/H)/H³, by a bounded Newton iteration on the
 * support radius H_a with dΣW/dH_a = −Σ_b q_ab w'(q_ab)/H_a, q_ab = |r_ab|/H_a. The bounds
 * limit the support radius without modifying the density sum.
 *
 * Symbol directory (in the code below):
 *   q_ab → normalized radius
 *   r_ab → separation pack; |r_ab| → separation magnitude packs
 *   H_a, H_a⁻¹ → sink kernel smoothing length, its inverse; H_min, H_max → smoothing length bounds
 *   ρ_a → density
 *   ΣW, dΣW/dH_a → kernel sum, its support derivative
 *   N_target, ΣW_target → target neighbor count, target kernel sum
 *   paddedX_a → sink position as the three-axis record that the search-box bounds compare.
 *   Code names: abs_r_ab = |r_ab|; H_inv_a = H_a⁻¹; dΣW_dH_a = dΣW/dH_a.
 *   SIMD values append Pack.
 */
// clang-format on
bool solveSphCpuSmoothingLength(SphRunParameters const& parameters, SphRunData const& data,
    SphTreeNode const& targetNode, std::vector<int> const& neighbors, int particleIndex, float& H_a,
    sph_flops::Counter& flops) {
    float const H_min = parameters.minimumSmoothingLength;
    float const H_max = parameters.maximumSmoothingLength;
    ALWAYS_ASSERT(H_min > 0.0f && H_max >= H_min && data.targetNeighborCount > 0.0f);
    H_a = std::min(std::max(H_a, H_min), H_max);
    Vector3<Position> const paddedX_a = data.sourcePaddedPosition(particleIndex);
    // Distances do not change during the bounded support-radius solve.
    std::vector<SimdFloat> abs_r_ab;    // |r_ab| for each pack of neighbors
    abs_r_ab.reserve((neighbors.size() + SimdFloat::size() - 1) / SimdFloat::size());
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        SpatialVector<SimdFloat> const r_ab =
            sphCpuSeparations(data, particleIndex, neighbors, first);
        abs_r_ab.push_back(r_ab.norm());
        // Fixed scales, dot, root.
        flops += (sph_flops::kVectorOperation + sph_flops::kNorm) * SimdFloat::size();
    }
    float lower = H_min;
    float upper = H_max;
    float const ΣW_target = data.targetNeighborCount * float(1.0 / sph_geometry::kUnitBallVolume);
    flops += 1;
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            flops.singlePrecision += 1;    // Single-precision fixed-point scale.
            if (targetNode.outerBox.end[dim] <
                RangeFixed(paddedX_a[dim]) + RangeFixed(H_a) + RangeFixed::min()) {
                return false;
            }
            flops.singlePrecision += 1;
            if (RangeFixed(paddedX_a[dim]) <
                targetNode.outerBox.begin[dim] + RangeFixed(H_a) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const H_inv_aPack(1.0f / H_a);
        flops += 1;
        SimdFloat ΣWPack(0.0f);
        SimdFloat dΣW_dH_aPack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const q_ab = abs_r_ab[first / SimdFloat::size()] * H_inv_aPack;
            auto const contributes = sphCpuValidLanes(neighbors, first) && q_ab < SimdFloat(1.0f);
            ΣWPack += simdSelect(contributes, kernelW(q_ab), SimdFloat(0.0f));
            dΣW_dH_aPack -=
                simdSelect(contributes, q_ab * dkernelWDq(q_ab) * H_inv_aPack, SimdFloat(0.0f));
            // q, two accumulations and the two derivative products, in every lane.
            flops += 5 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab) +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
        }
        float const ΣW = simdSum(ΣWPack);
        auto const publishInitialDensity = [&]() {
            if (parameters.initializeDensity) {
                // Derived from Monaghan (1992), equation (3.8), cited above:
                // ρ_a = m H_a⁻³ Σ_b w(q_ab), including bounded supports.
                // Symbol directory: ρ_a → sink density; m → gas mass; H_a⁻¹ → inverse sink kernel
                // smoothing length; Σ_b w(q_ab) → kernel sum. Code names: H_inv_a = H_a⁻¹.
                // No target-count approximation is used.
                int const sinkIndex = targetNode.sinkParticleRange.first -
                    targetNode.particleRange.first + particleIndex;
                float const H_inv_a = 1.0f / H_a;
                data.sinkDensity[sinkIndex] =
                    data.gasMass * sph_geometry::inverseVolume(H_inv_a) * ΣW;
                flops += 3 + sph_flops::kInverseVolume;
            }
        };
        float const residual = ΣW - ΣW_target;
        flops += SimdFloat::size() - 1 +
            2;    // Reduction, residual and tolerance; abs is a sign operation.
        if (std::abs(residual) <= kSphSmoothingLengthTolerance * ΣW_target ||
            (H_a <= H_min && residual >= 0.0f) || (H_a >= H_max && residual <= 0.0f)) {
            publishInitialDensity();
            return true;
        }
        if (residual > 0.0f) {
            upper = H_a;
        } else {
            lower = H_a;
        }
        float const dΣW_dH_a = simdSum(dΣW_dH_aPack);
        flops += SimdFloat::size() - 1 + 4;    // Reduction, change, two bounds, next H.
        float change = dΣW_dH_a > 0.0f ? -residual / dΣW_dH_a : 0.2f * H_a;
        change = std::min(std::max(change, -0.2f * H_a), 0.2f * H_a);
        float next = std::min(std::max(H_a + change, H_min), H_max);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == H_a) {
            publishInitialDensity();
            return true;
        }
        H_a = next;
    }
    std::cout << std::format("SPH smoothing-length solve did not converge: smoothingLength={:e} "
                             "bounds=[{:e},{:e}] target={:e}\n",
                     H_a, H_min, H_max, data.targetNeighborCount)
              << std::flush;
    ALWAYS_ASSERT(false);
    return false;
}

// clang-format off
/*
 * A held support uses ζ_a = 0. A free support whose correction is non-finite or has
 * Ω_a ≤ 0 cannot be used and stops the run; a small positive Ω_a is counted.
 *
 * Symbol directory (in the code below):
 *   ζ_a → weighted support correction
 *   H_a, ρ_a → sink kernel smoothing length, density
 *   ΣW, Σ_b q_ab w′ → kernel sum, kernel derivative moment
 *   Code names: Σqwʹ = Σ_b q_ab w′.
 */
// clang-format on
void recordSphCpuGradH(SphGradHEvaluation const& gradH, float H_a, float ρ_a, float ΣW, float Σqwʹ,
    SphRunReturn& result) {
    if (gradH.constraint != sph_volume::SupportConstraint::Free) {
        result.heldSupports++;
        return;
    }
    if (!sph_volume::isUsable(gradH.correction)) {
        std::cout << std::format(
                         "SPH grad-H correction is unusable: smoothingLength={:e} density={:e} "
                         "\u03a3W={:e} \u03a3q dw/dq={:e} \u03a9={:e} \u03b6={:e} residual={:e}\n",
                         H_a, ρ_a, ΣW, Σqwʹ, gradH.correction.supportJacobian,
                         gradH.correction.weightedSupportCorrection, gradH.relativeResidual)
                  << std::flush;
        ALWAYS_ASSERT(false);
    }
    result.minimumGradHOmega = std::min(result.minimumGradHOmega, gradH.correction.supportJacobian);
    result.maximumSupportResidual =
        std::max(result.maximumSupportResidual, std::abs(gradH.relativeResidual));
    if (!sph_volume::isWellConditioned(gradH.correction)) {
        result.illConditionedSupports++;
    }
}

// Whether an active neighbor touches particle a: r_ab² < max(H_a², H_b²).
// Symbol directory: r_ab → separation pack; r_ab² → squared separation;
// H_a, H_b → sink, neighbor kernel smoothing length. Code names: r2_ab = r_ab²; H2_a = H_a².
bool sphCpuHasActiveNeighbor(SphRunParameters const& parameters, SphRunData const& data,
    std::vector<int> const& neighbors, int particleIndex, float H_a, sph_flops::Counter& flops) {
    SimdFloat const H2_a(H_a * H_a);
    flops += 1;
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const r_ab = sphCpuSeparations(data, particleIndex, neighbors, first);
        auto const r2_ab = r_ab.dot(r_ab);
        auto const H_b = sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 0.0f);
        SimdFloat const neighborRung([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
        });
        auto const active = (neighborRung >= SimdFloat(float(parameters.minimumRung))) &&
            (neighborRung <= SimdFloat(float(parameters.maximumRung)));
        auto const intersects = r2_ab < simdMax(H2_a, sqr(H_b));
        // Fixed scales, dot operations, support square.
        flops += (sph_flops::kVectorOperation + sph_flops::kDotProduct + 1) * SimdFloat::size();
        if (simdAnyOf(sphCpuValidLanes(neighbors, first) && active && intersects)) {
            return true;
        }
    }
    return false;
}

}    // namespace detail

// H_a → sink kernel smoothing length, solved in place.
SphRunReturn sphCpuSmoothingLengths(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    sph_flops::Counter flops;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, false, flops);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            parameters.isActiveRung(data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
        if (active && !data.sinkFlags[sinkIndex].converged()) {
            float& H_a = data.sinkSmoothingLength[sinkIndex];
            bool const converged = detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, H_a, flops);
            data.sinkFlags[sinkIndex].setConverged(converged);
            if (converged) {
                result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, H_a);
                result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, H_a);
            } else {
                result.needsAnotherPass = true;
            }
        }
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}

// clang-format off
/*
 * Density and velocity-derivative sums: Monaghan (1992), ARA&A 30, 543, equation (3.8),
 * doi:10.1146/annurev.aa.30.090192.002551, with the divergence and curl of GADGET-4
 * equation (72), cited in cosmictiger/sph/viscosity/balsara.hpp.
 *   ρ_a = Σ_b m_b W_ab(H_a), W_ab = H_a⁻³ w(q_ab), q_ab = |r_ab|/H_a,
 *   ρ_a ∇·v_a = −Σ_b m_b v_ab · ∇_a W_ab,
 *   ρ_a ∇×v_a = Σ_b m_b v_ab × ∇_a W_ab.
 * Symbol directory (Pack denotes SIMD partial sums):
 *   H_a, H_a⁻¹ → sink kernel smoothing length, its inverse; q_ab → normalized radius
 *   H_a,stored → stored smoothing length, updated by the solver
 *   w(q_ab) → kernel weight; m_b/H_aᴰ → mass over support volume
 *   ρ_a → density; ρ_a ∇·v_a → density-weighted divergence
 *   ρ_a ∇×v_a → density-weighted curl
 *   γ, γ − 1 → gas adiabatic index, P/(ρu) of all gas
 *   r_ab, |r_ab| → separation, distance
 *   v_ab → velocity difference (peculiar stored velocity)
 *   m_b F_a → mass-weighted kernel gradient factor; m_b ∇_a W_ab → mass-weighted kernel gradient
 *   Σ_b w → kernel sum; Σ_b q_ab w′ → kernel derivative moment
 *   Code names: H_inv_a = H_a⁻¹; H_a_stored = H_a,stored; m_b_over_Hᴰ_a = m_b/H_aᴰ;
 *     γ_minus_1 = γ − 1; abs_r_ab = |r_ab|; w_q_ab = w(q_ab); ρ_a_div_v_a = ρ_a ∇·v_a; ρ_a_curl_v_a = ρ_a ∇×v_a;
 *     m_b_F_a = m_b F_a; m_b_grad_a_W_ab = m_b ∇_a W_ab; ΣW = Σ_b w; Σqwʹ = Σ_b q_ab w′.
 * Peculiar velocities enter the divergence and the curl.
 * The curl stays a three-component record: a two-dimensional curl points along the inactive axis.
 */
// clang-format on
SphRunReturn sphCpuDensityAndGradients(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    bool const evolveMorrisMonaghan =
        parameters.variableViscosity && parameters.viscosityTimestep > 0.0;
    sph_flops::Counter flops = 1;    // γ − 1.
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every gas particle
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            parameters.isActiveRung(data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
        float& H_a_stored = data.sinkSmoothingLength[sinkIndex];
        bool const semiactive = active ||
            detail::sphCpuHasActiveNeighbor(
                parameters, data, neighbors, particleIndex, H_a_stored, flops);
        data.sinkFlags[sinkIndex].setSemiactive(semiactive);
        if (!semiactive) {
            continue;
        }
        bool const updateMorrisMonaghan =
            evolveMorrisMonaghan && active && !data.sinkFlags[sinkIndex].converged();
        bool const computeSphDivergence = parameters.balsaraSwitch || updateMorrisMonaghan;
        if (!active && !data.sinkFlags[sinkIndex].converged() &&
            !detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, H_a_stored, flops)) {
            data.sinkFlags[sinkIndex].setConverged(false);
            result.needsAnotherPass = true;
            continue;
        }
        // clang-format off
        /*
         * Equation 3.8 (the interpolant 2.7 at r_a): ρ_a = Σ_b m_b W_ab, with the sink's own
         * support radius in W(r, H_a) = w(r/H_a)/H_a³, a gather form of equation 6.3 (which
         * uses h_b). Every gas particle has mass m_b = gasMass, and the sum includes b = a.
         *
         * Symbol directory (in the code below):
         *   H_a → sink kernel smoothing length
         *   m_b → gas mass
         *   ρ_a → density
         */
        // clang-format on
        float const H_a = H_a_stored;
        float const H_inv_a = 1.0f / H_a;
        SimdFloat const m_b_over_Hᴰ_a(data.gasMass * sph_geometry::inverseVolume(H_inv_a));
        flops += 2 + sph_flops::kInverseVolume;    // Reciprocal and mass/support products.
        SimdFloat ρ_aPack(0.0f);
        SimdFloat ρ_a_div_v_aPack(0.0f);
        Vector3<SimdFloat> ρ_a_curl_v_aPack(SimdFloat(0.0f));    // padded: see the curl above
        // clang-format off
        /*
         * Variational operator: the unweighted sums ΣW = Σ_b w(q_ab) and Σ_b q_ab w'(q_ab)
         * of cosmictiger/sph/volume.hpp give Ω_a and ζ_a at this support radius.
         *
         * Symbol directory (in the code below):
         *   q_ab → normalized radius
         *   ζ_a → weighted support correction
         *   ΣW → kernel sum
         */
        // clang-format on
        SimdFloat ΣWPack(0.0f);
        SimdFloat ΣqwʹPack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            SpatialVector<SimdFloat> const r_ab =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const abs_r_ab = r_ab.norm();
            SimdFloat const q_ab = abs_r_ab * SimdFloat(H_inv_a);
            SimdFloat const w_q_ab = kernelW(q_ab);
            ρ_aPack += simdSelect(detail::sphCpuValidLanes(neighbors, first),
                m_b_over_Hᴰ_a * w_q_ab, SimdFloat(0.0f));
            // Fixed scales, norm, q_ab, density product and sum.
            flops += (sph_flops::kVectorOperation + sph_flops::kNorm + 3) * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab);
            auto const inside =
                detail::sphCpuValidLanes(neighbors, first) && q_ab < SimdFloat(1.0f);
            ΣWPack += simdSelect(inside, w_q_ab, SimdFloat(0.0f));
            ΣqwʹPack += simdSelect(inside, q_ab * dkernelWDq(q_ab), SimdFloat(0.0f));
            flops += 3 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
            if (computeSphDivergence) {
                // clang-format off
                /*
                 * GADGET-4 equation (72) uses −Σ_b m_b v_ab·gradW_ab and Σ_b m_b v_ab×gradW_ab.
                 *
                 * Symbol directory (in the code below):
                 *   v_a, v_b → sink, neighbor stored velocity; v_ab → velocity difference
                 *   m_b → gas mass
                 *   ρ_a → density
                 */
                // clang-format on
                auto const contributes = detail::sphCpuValidLanes(neighbors, first) &&
                    abs_r_ab > SimdFloat(0.0f) && q_ab < SimdFloat(1.0f);
                SimdFloat const m_b_F_a = simdSelect(contributes,
                    m_b_over_Hᴰ_a * SimdFloat(H_inv_a) * dkernelWDq(q_ab) /
                        simdSelect(contributes, abs_r_ab, SimdFloat(1.0f)),
                    SimdFloat(0.0f));
                SpatialVector<SimdFloat> const v_a(data.sourceVelocity(particleIndex));
                SpatialVector<SimdFloat> const v_b = detail::sphCpuLoad3(data.sourceVelocityX,
                    data.sourceVelocityY, data.sourceVelocityZ, neighbors, first, 0.0f);
                SpatialVector<SimdFloat> const v_ab =
                    simdSelect(contributes, v_a - v_b, SpatialVector<SimdFloat>(SimdFloat(0)));
                // Kernel-gradient coefficient (3) and relative velocity.
                flops += (3 + sph_flops::kVectorOperation) * SimdFloat::size() +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
                SpatialVector<SimdFloat> const m_b_grad_a_W_ab = m_b_F_a * r_ab;
                flops += sph_flops::kVectorOperation * SimdFloat::size();
                ρ_a_div_v_aPack -= v_ab.dot(m_b_grad_a_W_ab);
                flops += (sph_flops::kDotProduct + 1) * SimdFloat::size();
                if (parameters.balsaraSwitch) {
                    ρ_a_curl_v_aPack += sph_geometry::paddedVector(v_ab).cross(
                        sph_geometry::paddedVector(m_b_grad_a_W_ab));
                    flops += 12 * SimdFloat::size();
                }
            }
        }
        float const ρ_a = simdSum(ρ_aPack);
        flops += SimdFloat::size() - 1;
        ALWAYS_ASSERT(H_a > 0.0f && ρ_a > 0.0f);
        data.sinkDensity[sinkIndex] = ρ_a;
        float const ΣW = simdSum(ΣWPack);
        float const Σqwʹ = simdSum(ΣqwʹPack);
        auto const gradH = sphGradHEvaluation(H_a, parameters.minimumSmoothingLength,
            parameters.maximumSmoothingLength, data.targetNeighborCount, data.gasMass, ΣW, Σqwʹ);
        /*
         * Reductions, residual (2), and Ω, ζ, the weighted sums (6).
         */
        flops += 2 * (SimdFloat::size() - 1) + 8;
        detail::recordSphCpuGradH(gradH, H_a, ρ_a, ΣW, Σqwʹ, result);
        data.sinkGradHCorrection[sinkIndex] = gradH.correction.weightedSupportCorrection;
        if (parameters.balsaraSwitch && data.sinkBalsaraFactor) {
            Vector3<float> const ρ_a_curl_v_a(simdSum(ρ_a_curl_v_aPack[kXi]),
                simdSum(ρ_a_curl_v_aPack[kYi]), simdSum(ρ_a_curl_v_aPack[kZi]));
            data.sinkBalsaraFactor[sinkIndex] = balsara::encodeFactor(
                balsara::balsaraFactor(simdSum(ρ_a_div_v_aPack), ρ_a_curl_v_a.norm()));
            flops += 4 * (SimdFloat::size() - 1) + 8;
        } else if (!parameters.balsaraSwitch && data.sinkBalsaraFactor) {
            data.sinkBalsaraFactor[sinkIndex] = 255;
        }
        if (updateMorrisMonaghan) {
            /*
             * Morris & Monaghan (1997), equations (5), (6), (13), in conformal time; the update is
             * in cosmictiger/sph/viscosity/morris_and_monaghan.hpp. The sum/rho is a² div_phys(v);
             * conformal-time rates divide by a. τ_a⁻¹ = C₁ c_a/h_a, ∇·v_a = ρ_a ∇·v_a/(a ρ_a).
             * Symbol directory:
             *   ∇·v_a → divergence; c_a → sound speed; h_a → viscosity smoothing length
             *   τ_a⁻¹ → inverse decay time; C₁ → viscosity decay parameter; Δt → time step
             *   α_a → stored bulk viscosity coefficient.
             *   Code names: div_v_a = ∇·v_a; τ_inv_a = τ_a⁻¹.
             */
            float const div_v_a = simdSum(ρ_a_div_v_aPack) / (ρ_a * parameters.scaleFactor);
            float const c_a = std::sqrt(γ * γ_minus_1 * data.sourceInternalEnergy[particleIndex]);
            float const h_a = H_a / selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                float(monaghan::kViscosityWidthScale);
            float const τ_inv_a = parameters.viscosityDecay * c_a / (parameters.scaleFactor * h_a);
            int const rung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
            bool const actualInterval = parameters.globalTimestep;
            float const Δt = actualInterval ? parameters.viscosityTimestep :
                                              parameters.viscosityTimestep / (1 << rung);
            auto& α_a = data.sinkBulkViscosityCoefficient[sinkIndex];
            α_a = morris_and_monaghan::evolvedBulkViscosityCoefficient(
                float(α_a), parameters.minimumBulkViscosityCoefficient, τ_inv_a, div_v_a, Δt);
            flops += SimdFloat::size() - 1 + 19;
            flops.doublePrecision += actualInterval ? 1 : 2;
        }
        data.sinkFlags[sinkIndex].setConverged(true);
        result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, H_a);
        result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, H_a);
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}
