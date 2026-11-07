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
 * Section 6 of Monaghan (1992): the resolution length follows the local density,
 * H𞁑 ∝ ρ𞁑^(−1/ν) with ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2
 * or 6.7 of Monaghan (1992), the solve holds
 *   (4π/3) ρ𞁑 H𞁑³/m = Nₛₑₜ,   i.e.   ΣW = Σ𞁓 w(|r𞁑𞁓|/H𞁑) = 3 Nₛₑₜ/(4π)
 * for the selected kernel W(r, H) = w(r/H)/H³, by a bounded Newton iteration on the
 * support radius H𞁑 with dΣW/dH𞁑 = −Σ𞁓 q𞁑𞁓 w'(q𞁑𞁓)/H𞁑, q𞁑𞁓 = |r𞁑𞁓|/H𞁑. The bounds
 * limit the support radius without modifying the density sum.
 *
 * Symbol directory (in the code below):
 *   q𞁑𞁓 → normalized radius
 *   r𞁑𞁓 → separation pack; |r𞁑𞁓| → separation magnitude packs
 *   H𞁑, H𞁑⁻¹ → sink kernel smoothing length, its inverse; Hₘᵢₙ, Hₘ𞁑ₓ → smoothing length bounds
 *   ρ𞁑 → density
 *   ΣW, dΣW/dH𞁑 → kernel sum, its support derivative
 *   Nₛₑₜ, ΣWₛₑₜ → target neighbor count, target kernel sum
 *   paddedX_a → sink position as the three-axis record that the search-box bounds compare.
 *   Code names: abs_r𞁑𞁓 = |r𞁑𞁓|; iH𞁑 = H𞁑⁻¹; dΣW_dH𞁑 = dΣW/dH𞁑.
 *   SIMD values append Pack.
 */
// clang-format on
bool solveSphCpuSmoothingLength(SphRunParameters const& parameters, SphRunData const& data,
    SphTreeNode const& targetNode, std::vector<int> const& neighbors, int particleIndex, float& H𞁑,
    sph_flops::Counter& flops) {
    float const Hₘᵢₙ = parameters.minimumSmoothingLength;
    float const Hₘ𞁑ₓ = parameters.maximumSmoothingLength;
    ALWAYS_ASSERT(Hₘᵢₙ > 0.0f && Hₘ𞁑ₓ >= Hₘᵢₙ && data.targetNeighborCount > 0.0f);
    H𞁑 = std::min(std::max(H𞁑, Hₘᵢₙ), Hₘ𞁑ₓ);
    Vector3<Position> const paddedX_a = data.sourcePaddedPosition(particleIndex);
    // Distances do not change during the bounded support-radius solve.
    std::vector<SimdFloat> abs_r𞁑𞁓;    // |r𞁑𞁓| for each pack of neighbors
    abs_r𞁑𞁓.reserve((neighbors.size() + SimdFloat::size() - 1) / SimdFloat::size());
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        SpatialVector<SimdFloat> const r𞁑𞁓 =
            sphCpuSeparations(data, particleIndex, neighbors, first);
        abs_r𞁑𞁓.push_back(r𞁑𞁓.norm());
        // Fixed scales, dot, root.
        flops += (sph_flops::kVectorOperation + sph_flops::kNorm) * SimdFloat::size();
    }
    float lower = Hₘᵢₙ;
    float upper = Hₘ𞁑ₓ;
    float const ΣWₛₑₜ = data.targetNeighborCount * float(1.0 / sph_geometry::kUnitBallVolume);
    flops += 1;
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            flops.singlePrecision += 1;    // Single-precision fixed-point scale.
            if (targetNode.outerBox.end[dim] <
                RangeFixed(paddedX_a[dim]) + RangeFixed(H𞁑) + RangeFixed::min()) {
                return false;
            }
            flops.singlePrecision += 1;
            if (RangeFixed(paddedX_a[dim]) <
                targetNode.outerBox.begin[dim] + RangeFixed(H𞁑) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const iH𞁑Pack(1.0f / H𞁑);
        flops += 1;
        SimdFloat ΣWPack(0.0f);
        SimdFloat dΣW_dH𞁑Pack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const q𞁑𞁓 = abs_r𞁑𞁓[first / SimdFloat::size()] * iH𞁑Pack;
            auto const contributes = sphCpuValidLanes(neighbors, first) && q𞁑𞁓 < SimdFloat(1.0f);
            ΣWPack += simdSelect(contributes, kernelW(q𞁑𞁓), SimdFloat(0.0f));
            dΣW_dH𞁑Pack -=
                simdSelect(contributes, q𞁑𞁓 * dkernelWDq(q𞁑𞁓) * iH𞁑Pack, SimdFloat(0.0f));
            // q, two accumulations and the two derivative products, in every lane.
            flops += 5 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓) +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
        }
        float const ΣW = simdSum(ΣWPack);
        auto const publishInitialDensity = [&]() {
            if (parameters.initializeDensity) {
                // Derived from Monaghan (1992), equation (3.8):
                // ρ𞁑 = m H𞁑⁻³ Σ𞁓 w(q𞁑𞁓), including bounded supports.
                // Symbol directory: ρ𞁑 → sink density; m → gas mass; H𞁑⁻¹ → inverse sink kernel
                // smoothing length; Σ𞁓 w(q𞁑𞁓) → kernel sum. Code names: iH𞁑 = H𞁑⁻¹.
                // No target-count approximation is used.
                int const sinkIndex = targetNode.sinkParticleRange.first -
                    targetNode.particleRange.first + particleIndex;
                float const iH𞁑 = 1.0f / H𞁑;
                data.sinkDensity[sinkIndex] = data.gasMass * sph_geometry::inverseVolume(iH𞁑) * ΣW;
                flops += 3 + sph_flops::kInverseVolume;
            }
        };
        float const residual = ΣW - ΣWₛₑₜ;
        flops += SimdFloat::size() - 1 +
            2;    // Reduction, residual and tolerance; abs is a sign operation.
        if (std::abs(residual) <= kSphSmoothingLengthTolerance * ΣWₛₑₜ ||
            (H𞁑 <= Hₘᵢₙ && residual >= 0.0f) || (H𞁑 >= Hₘ𞁑ₓ && residual <= 0.0f)) {
            publishInitialDensity();
            return true;
        }
        if (residual > 0.0f) {
            upper = H𞁑;
        } else {
            lower = H𞁑;
        }
        float const dΣW_dH𞁑 = simdSum(dΣW_dH𞁑Pack);
        flops += SimdFloat::size() - 1 + 4;    // Reduction, change, two bounds, next H.
        float change = dΣW_dH𞁑 > 0.0f ? -residual / dΣW_dH𞁑 : 0.2f * H𞁑;
        change = std::min(std::max(change, -0.2f * H𞁑), 0.2f * H𞁑);
        float next = std::min(std::max(H𞁑 + change, Hₘᵢₙ), Hₘ𞁑ₓ);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == H𞁑) {
            publishInitialDensity();
            return true;
        }
        H𞁑 = next;
    }
    std::cout << std::format("SPH smoothing-length solve did not converge: smoothingLength={:e} "
                             "bounds=[{:e},{:e}] target={:e}\n",
                     H𞁑, Hₘᵢₙ, Hₘ𞁑ₓ, data.targetNeighborCount)
              << std::flush;
    ALWAYS_ASSERT(false);
    return false;
}

// clang-format off
/*
 * A held support uses ζ𞁑 = 0. A free support whose correction is non-finite or has
 * Ω𞁑 ≤ 0 cannot be used and stops the run; a small positive Ω𞁑 is counted.
 *
 * Symbol directory (in the code below):
 *   ζ𞁑 → weighted support correction
 *   H𞁑, ρ𞁑 → sink kernel smoothing length, density
 *   ΣW, Σ𞁓 q𞁑𞁓 w′ → kernel sum, kernel derivative moment
 *   Code names: Σqwʹ = Σ𞁓 q𞁑𞁓 w′.
 */
// clang-format on
void recordSphCpuGradH(SphGradHEvaluation const& gradH, float H𞁑, float ρ𞁑, float ΣW, float Σqwʹ,
    SphRunReturn& result) {
    if (gradH.constraint != sph_volume::SupportConstraint::Free) {
        result.heldSupports++;
        return;
    }
    if (!sph_volume::isUsable(gradH.correction)) {
        std::cout << std::format(
                         "SPH grad-H correction is unusable: smoothingLength={:e} density={:e} "
                         "\u03a3W={:e} \u03a3q dw/dq={:e} \u03a9={:e} \u03b6={:e} residual={:e}\n",
                         H𞁑, ρ𞁑, ΣW, Σqwʹ, gradH.correction.supportJacobian,
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

// Whether an active neighbor touches particle a: r𞁑𞁓² < max(H𞁑², H𞁓²).
// Symbol directory: r𞁑𞁓 → separation pack; r𞁑𞁓² → squared separation;
// H𞁑, H𞁓 → sink, neighbor kernel smoothing length. Code names: r2𞁑𞁓 = r𞁑𞁓²; H2𞁑 = H𞁑².
bool sphCpuHasActiveNeighbor(SphRunParameters const& parameters, SphRunData const& data,
    std::vector<int> const& neighbors, int particleIndex, float H𞁑, sph_flops::Counter& flops) {
    SimdFloat const H2𞁑(H𞁑 * H𞁑);
    flops += 1;
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const r𞁑𞁓 = sphCpuSeparations(data, particleIndex, neighbors, first);
        auto const r2𞁑𞁓 = r𞁑𞁓.dot(r𞁑𞁓);
        auto const H𞁓 = sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 0.0f);
        SimdFloat const neighborRung([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
        });
        auto const active = (neighborRung >= SimdFloat(float(parameters.minimumRung))) &&
            (neighborRung <= SimdFloat(float(parameters.maximumRung)));
        auto const intersects = r2𞁑𞁓 < simdMax(H2𞁑, sqr(H𞁓));
        // Fixed scales, dot operations, support square.
        flops += (sph_flops::kVectorOperation + sph_flops::kDotProduct + 1) * SimdFloat::size();
        if (simdAnyOf(sphCpuValidLanes(neighbors, first) && active && intersects)) {
            return true;
        }
    }
    return false;
}

}    // namespace detail

// H𞁑 → sink kernel smoothing length, solved in place.
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
            float& H𞁑 = data.sinkSmoothingLength[sinkIndex];
            bool const converged = detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, H𞁑, flops);
            data.sinkFlags[sinkIndex].setConverged(converged);
            if (converged) {
                result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, H𞁑);
                result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, H𞁑);
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
 * doi:10.1146/annurev.aa.30.090192.002551, with the divergence and curl of GADGET-4,
 * Springel et al. (2021), equation (72), cited in cosmictiger/sph/viscosity/balsara.hpp.
 *   ρ𞁑 = Σ𞁓 m𞁓 W𞁑𞁓(H𞁑), W𞁑𞁓 = H𞁑⁻³ w(q𞁑𞁓), q𞁑𞁓 = |r𞁑𞁓|/H𞁑,
 *   ρ𞁑 ∇·v𞁑 = −Σ𞁓 m𞁓 v𞁑𞁓 · ∇𞁑 W𞁑𞁓,
 *   ρ𞁑 ∇×v𞁑 = Σ𞁓 m𞁓 v𞁑𞁓 × ∇𞁑 W𞁑𞁓.
 * Symbol directory (Pack denotes SIMD partial sums):
 *   H𞁑, H𞁑⁻¹ → sink kernel smoothing length, its inverse; q𞁑𞁓 → normalized radius
 *   H𞁑,stored → stored smoothing length, updated by the solver
 *   w(q𞁑𞁓) → kernel weight; m𞁓/H𞁑ᴰ → mass over support volume
 *   ρ𞁑 → density; ρ𞁑 ∇·v𞁑 → density-weighted divergence
 *   ρ𞁑 ∇×v𞁑 → density-weighted curl
 *   γ, γ − 1 → gas adiabatic index, P/(ρu) of all gas
 *   r𞁑𞁓, |r𞁑𞁓| → separation, distance
 *   v𞁑𞁓 → velocity difference (peculiar stored velocity)
 *   m𞁓 F𞁑 → mass-weighted kernel gradient factor; m𞁓 ∇𞁑 W𞁑𞁓 → mass-weighted kernel gradient
 *   Σ𞁓 w → kernel sum; Σ𞁓 q𞁑𞁓 w′ → kernel derivative moment
 *   Code names: iH𞁑 = H𞁑⁻¹; H𞁑ₘₑₘ = H𞁑,stored; m𞁓_over_Hᴰ𞁑 = m𞁓/H𞁑ᴰ
 *     γ_minus_1 = γ − 1; abs_r𞁑𞁓 = |r𞁑𞁓|; w_q𞁑𞁓 = w(q𞁑𞁓); ρ𞁑_div_v𞁑 = ρ𞁑 ∇·v𞁑; ρ𞁑_curl_v𞁑 = ρ𞁑 ∇×v𞁑
 *     m𞁓F𞁑 = m𞁓 F𞁑; m𞁓grad𞁑W𞁑𞁓 = m𞁓 ∇𞁑 W𞁑𞁓; ΣW = Σ𞁓 w; Σqwʹ = Σ𞁓 q𞁑𞁓 w′.
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
        float& H𞁑ₘₑₘ = data.sinkSmoothingLength[sinkIndex];
        bool const semiactive = active ||
            detail::sphCpuHasActiveNeighbor(
                parameters, data, neighbors, particleIndex, H𞁑ₘₑₘ, flops);
        data.sinkFlags[sinkIndex].setSemiactive(semiactive);
        if (!semiactive) {
            continue;
        }
        bool const updateMorrisMonaghan =
            evolveMorrisMonaghan && active && !data.sinkFlags[sinkIndex].converged();
        bool const computeSphDivergence = parameters.balsaraSwitch || updateMorrisMonaghan;
        if (!active && !data.sinkFlags[sinkIndex].converged() &&
            !detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, H𞁑ₘₑₘ, flops)) {
            data.sinkFlags[sinkIndex].setConverged(false);
            result.needsAnotherPass = true;
            continue;
        }
        // clang-format off
        /*
         * Equation 3.8 of Monaghan (1992) (the interpolant 2.7 at r𞁑): ρ𞁑 = Σ𞁓 m𞁓 W𞁑𞁓,
         * with the sink's own support radius in W(r, H𞁑) = w(r/H𞁑)/H𞁑³, a gather form of
         * equation 6.3 (which uses h𞁓). Every gas particle has mass m𞁓 = gasMass, and the
         * sum includes b = a.
         *
         * Symbol directory (in the code below):
         *   H𞁑 → sink kernel smoothing length
         *   m𞁓 → gas mass
         *   ρ𞁑 → density
         */
        // clang-format on
        float const H𞁑 = H𞁑ₘₑₘ;
        float const iH𞁑 = 1.0f / H𞁑;
        SimdFloat const m𞁓_over_Hᴰ𞁑(data.gasMass * sph_geometry::inverseVolume(iH𞁑));
        flops += 2 + sph_flops::kInverseVolume;    // Reciprocal and mass/support products.
        SimdFloat ρ𞁑Pack(0.0f);
        SimdFloat ρ𞁑_div_v𞁑Pack(0.0f);
        Vector3<SimdFloat> ρ𞁑_curl_v𞁑Pack(SimdFloat(0.0f));    // padded: see the curl above
        // clang-format off
        /*
         * Variational operator: the unweighted sums ΣW = Σ𞁓 w(q𞁑𞁓) and Σ𞁓 q𞁑𞁓 w'(q𞁑𞁓)
         * of cosmictiger/sph/volume.hpp give Ω𞁑 and ζ𞁑 at this support radius.
         *
         * Symbol directory (in the code below):
         *   q𞁑𞁓 → normalized radius
         *   ζ𞁑 → weighted support correction
         *   ΣW → kernel sum
         */
        // clang-format on
        SimdFloat ΣWPack(0.0f);
        SimdFloat ΣqwʹPack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            SpatialVector<SimdFloat> const r𞁑𞁓 =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const abs_r𞁑𞁓 = r𞁑𞁓.norm();
            SimdFloat const q𞁑𞁓 = abs_r𞁑𞁓 * SimdFloat(iH𞁑);
            SimdFloat const w_q𞁑𞁓 = kernelW(q𞁑𞁓);
            ρ𞁑Pack += simdSelect(
                detail::sphCpuValidLanes(neighbors, first), m𞁓_over_Hᴰ𞁑 * w_q𞁑𞁓, SimdFloat(0.0f));
            // Fixed scales, norm, q𞁑𞁓, density product and sum.
            flops += (sph_flops::kVectorOperation + sph_flops::kNorm + 3) * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓);
            auto const inside = detail::sphCpuValidLanes(neighbors, first) && q𞁑𞁓 < SimdFloat(1.0f);
            ΣWPack += simdSelect(inside, w_q𞁑𞁓, SimdFloat(0.0f));
            ΣqwʹPack += simdSelect(inside, q𞁑𞁓 * dkernelWDq(q𞁑𞁓), SimdFloat(0.0f));
            flops += 3 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
            if (computeSphDivergence) {
                // clang-format off
                /*
                 * GADGET-4, Springel et al. (2021), equation (72) uses −Σ𞁓 m𞁓 v𞁑𞁓·gradW_ab
                 * and Σ𞁓 m𞁓 v𞁑𞁓×gradW_ab.
                 *
                 * Symbol directory (in the code below):
                 *   v𞁑, v𞁓 → sink, neighbor stored velocity; v𞁑𞁓 → velocity difference
                 *   m𞁓 → gas mass
                 *   ρ𞁑 → density
                 */
                // clang-format on
                auto const contributes = detail::sphCpuValidLanes(neighbors, first) &&
                    abs_r𞁑𞁓 > SimdFloat(0.0f) && q𞁑𞁓 < SimdFloat(1.0f);
                SimdFloat const m𞁓F𞁑 = simdSelect(contributes,
                    m𞁓_over_Hᴰ𞁑 * SimdFloat(iH𞁑) * dkernelWDq(q𞁑𞁓) /
                        simdSelect(contributes, abs_r𞁑𞁓, SimdFloat(1.0f)),
                    SimdFloat(0.0f));
                SpatialVector<SimdFloat> const v𞁑(data.sourceVelocity(particleIndex));
                SpatialVector<SimdFloat> const v𞁓 = detail::sphCpuLoad3(data.sourceVelocityX,
                    data.sourceVelocityY, data.sourceVelocityZ, neighbors, first, 0.0f);
                SpatialVector<SimdFloat> const v𞁑𞁓 =
                    simdSelect(contributes, v𞁑 - v𞁓, SpatialVector<SimdFloat>(SimdFloat(0)));
                // Kernel-gradient coefficient (3) and relative velocity.
                flops += (3 + sph_flops::kVectorOperation) * SimdFloat::size() +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
                SpatialVector<SimdFloat> const m𞁓grad𞁑W𞁑𞁓 = m𞁓F𞁑 * r𞁑𞁓;
                flops += sph_flops::kVectorOperation * SimdFloat::size();
                ρ𞁑_div_v𞁑Pack -= v𞁑𞁓.dot(m𞁓grad𞁑W𞁑𞁓);
                flops += (sph_flops::kDotProduct + 1) * SimdFloat::size();
                if (parameters.balsaraSwitch) {
                    ρ𞁑_curl_v𞁑Pack += sph_geometry::paddedVector(v𞁑𞁓).cross(
                        sph_geometry::paddedVector(m𞁓grad𞁑W𞁑𞁓));
                    flops += 12 * SimdFloat::size();
                }
            }
        }
        float const ρ𞁑 = simdSum(ρ𞁑Pack);
        flops += SimdFloat::size() - 1;
        ALWAYS_ASSERT(H𞁑 > 0.0f && ρ𞁑 > 0.0f);
        data.sinkDensity[sinkIndex] = ρ𞁑;
        float const ΣW = simdSum(ΣWPack);
        float const Σqwʹ = simdSum(ΣqwʹPack);
        auto const gradH = sphGradHEvaluation(H𞁑, parameters.minimumSmoothingLength,
            parameters.maximumSmoothingLength, data.targetNeighborCount, data.gasMass, ΣW, Σqwʹ);
        /*
         * Reductions, residual (2), and Ω, ζ, the weighted sums (6).
         */
        flops += 2 * (SimdFloat::size() - 1) + 8;
        detail::recordSphCpuGradH(gradH, H𞁑, ρ𞁑, ΣW, Σqwʹ, result);
        data.sinkGradHCorrection[sinkIndex] = gradH.correction.weightedSupportCorrection;
        if (parameters.balsaraSwitch && data.sinkBalsaraFactor) {
            Vector3<float> const ρ𞁑_curl_v𞁑(simdSum(ρ𞁑_curl_v𞁑Pack[kXi]),
                simdSum(ρ𞁑_curl_v𞁑Pack[kYi]), simdSum(ρ𞁑_curl_v𞁑Pack[kZi]));
            data.sinkBalsaraFactor[sinkIndex] = balsara::encodeFactor(
                balsara::balsaraFactor(simdSum(ρ𞁑_div_v𞁑Pack), ρ𞁑_curl_v𞁑.norm()));
            flops += 4 * (SimdFloat::size() - 1) + 8;
        } else if (!parameters.balsaraSwitch && data.sinkBalsaraFactor) {
            data.sinkBalsaraFactor[sinkIndex] = 255;
        }
        if (updateMorrisMonaghan) {
            /*
             * Morris & Monaghan (1997), equations (5), (6), (13), in conformal time; the update is
             * in cosmictiger/sph/viscosity/morris_and_monaghan.hpp. The sum/rho is a² div_phys(v);
             * conformal-time rates divide by a. τ𞁑⁻¹ = C₁ c𞁑/h𞁑, ∇·v𞁑 = ρ𞁑 ∇·v𞁑/(a ρ𞁑).
             * Symbol directory:
             *   ∇·v𞁑 → divergence; c𞁑 → sound speed; h𞁑 → viscosity smoothing length
             *   τ𞁑⁻¹ → inverse decay time; C₁ → viscosity decay parameter; Δt → time step
             *   α𞁑 → stored bulk viscosity coefficient.
             *   Code names: div_v_a = ∇·v𞁑; iτ𞁑 = τ𞁑⁻¹.
             */
            float const div_v_a = simdSum(ρ𞁑_div_v𞁑Pack) / (ρ𞁑 * parameters.scaleFactor);
            float const c𞁑 = std::sqrt(γ * γ_minus_1 * data.sourceInternalEnergy[particleIndex]);
            float const h𞁑 = H𞁑 / selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                float(monaghan::kViscosityWidthScale);
            float const iτ𞁑 = parameters.viscosityDecay * c𞁑 / (parameters.scaleFactor * h𞁑);
            int const rung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
            bool const actualInterval = parameters.globalTimestep;
            float const Δt = actualInterval ? parameters.viscosityTimestep :
                                              parameters.viscosityTimestep / (1 << rung);
            auto& α𞁑 = data.sinkBulkViscosityCoefficient[sinkIndex];
            α𞁑 = morris_and_monaghan::evolvedBulkViscosityCoefficient(
                float(α𞁑), parameters.minimumBulkViscosityCoefficient, iτ𞁑, div_v_a, Δt);
            flops += SimdFloat::size() - 1 + 19;
            flops.doublePrecision += actualInterval ? 1 : 2;
        }
        data.sinkFlags[sinkIndex].setConverged(true);
        result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, H𞁑);
        result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, H𞁑);
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}
