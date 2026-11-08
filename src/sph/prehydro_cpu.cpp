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
 * h𞁑 ∝ ρ𞁑^(−1/ν) with ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2
 * or 6.7 of Monaghan (1992), the solve holds
 *   (4π/3) ρ𞁑 h𞁑³/m = N(h𞁑),   i.e.   ΣW = Σ𞁓 w(|r𞁑𞁓|/h𞁑) = ΣWₛₑₜ(h𞁑) = 3 N(h𞁑)/(4π)
 * for the selected kernel W(r, h) = w(r/h)/h³, where N(h) = N₀ f(h) is the bounded target
 * of sph_volume::boundedTarget: N₀ between the layer junctions, infinite at hₘᵢₙ and zero
 * at hₘₐₓ, so the root always lies strictly inside the bounds. A bounded Newton
 * iteration on h𞁑 uses dΣW/dh𞁑 = −Σ𞁓 q𞁑𞁓 w'(q𞁑𞁓)/h𞁑, q𞁑𞁓 = |r𞁑𞁓|/h𞁑, and the target
 * slope dΣWₛₑₜ/dh𞁑 = 3 N₀ f′/(4π). Iterates stay a margin inside the bounds, where the
 * target is singular; an iterate pinned at the margin with a residual of the bound's
 * sign is accepted, as is a fixed support (hₘᵢₙ = hₘₐₓ).
 *
 * Symbol directory (in the code below):
 *   q𞁑𞁓 → normalized radius
 *   r𞁑𞁓 → separation; |r𞁑𞁓| → separation magnitude
 *   h𞁑, h𞁑⁻¹ → sink kernel smoothing length, its inverse; hₘᵢₙ, hₘₐₓ → smoothing length bounds
 *   ρ𞁑 → density
 *   ΣW, dΣW/dh𞁑 → kernel sum, its smoothing-length derivative
 *   N₀ → requested neighbor number; f, f′ → target factor and its slope
 *   ΣWₛₑₜ, dΣWₛₑₜ/dh𞁑 → target kernel sum at h𞁑, its slope; ΣW_0 → interior target 3N₀/(4π)
 *   dF/dh𞁑 → slope of the residual ΣW − ΣWₛₑₜ
 *   paddedX_a → sink position as the three-axis record that the search-box bounds compare.
 *   Code names: abs_r𞁑𞁓 = |r𞁑𞁓|; ih𞁑 = h𞁑⁻¹; dΣW_dh𞁑 = dΣW/dh𞁑; dΣWₛₑₜ_dh𞁑 = dΣWₛₑₜ/dh𞁑;
 *   dF_dh𞁑 = dF/dh𞁑.
 */
// clang-format on
bool solveSphCpuSmoothingLength(SphRunParameters const& parameters, SphRunData const& data,
    SphTreeNode const& targetNode, std::vector<int> const& neighbors, int particleIndex, float& h𞁑,
    sph_flops::Counter& flops) {
    float const hₘᵢₙ = parameters.minimumSmoothingLength;
    float const hₘₐₓ = parameters.maximumSmoothingLength;
    ALWAYS_ASSERT(hₘᵢₙ > 0.0f && hₘₐₓ >= hₘᵢₙ && data.targetNeighborCount > 0.0f);
    auto const layers = sph_volume::supportLayers(hₘᵢₙ, hₘₐₓ);
    float const lowerLimit =
        hₘᵢₙ + float(sph_volume::kSupportMargin) * (layers.lowerJunction - hₘᵢₙ);
    float const upperLimit =
        hₘₐₓ - float(sph_volume::kSupportMargin) * (hₘₐₓ - layers.upperJunction);
    h𞁑 = std::min(std::max(h𞁑, lowerLimit), upperLimit);
    Vector3<Position> const paddedX_a = data.sourcePaddedPosition(particleIndex);
    // Distances do not change during the bounded smoothing-length solve.
    std::vector<SimdFloat> abs_r𞁑𞁓;    // |r𞁑𞁓| for each pack of neighbors
    abs_r𞁑𞁓.reserve((neighbors.size() + SimdFloat::size() - 1) / SimdFloat::size());
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        SpatialVector<SimdFloat> const r𞁑𞁓 =
            sphCpuSeparations(data, particleIndex, neighbors, first);
        abs_r𞁑𞁓.push_back(r𞁑𞁓.norm());
        // Fixed scales, dot, root.
        flops += (sph_flops::kVectorOperation + sph_flops::kNorm) * SimdFloat::size();
    }
    float lower = lowerLimit;
    float upper = upperLimit;
    float const ΣW_0 = sphTargetKernelSum(data.targetNeighborCount);
    flops += 7;    // Interior target, junctions and limits.
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            flops.singlePrecision += 1;    // Single-precision fixed-point scale.
            if (targetNode.outerBox.end[dim] <
                RangeFixed(paddedX_a[dim]) + RangeFixed(h𞁑) + RangeFixed::min()) {
                return false;
            }
            flops.singlePrecision += 1;
            if (RangeFixed(paddedX_a[dim]) <
                targetNode.outerBox.begin[dim] + RangeFixed(h𞁑) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const ih𞁑Pack(1.0f / h𞁑);
        flops += 1;
        SimdFloat ΣWPack(0.0f);
        SimdFloat dΣW_dh𞁑Pack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const q𞁑𞁓 = abs_r𞁑𞁓[first / SimdFloat::size()] * ih𞁑Pack;
            auto const contributes = sphCpuValidLanes(neighbors, first) && q𞁑𞁓 < SimdFloat(1.0f);
            ΣWPack += simdSelect(contributes, kernelW(q𞁑𞁓), SimdFloat(0.0f));
            dΣW_dh𞁑Pack -=
                simdSelect(contributes, q𞁑𞁓 * dkernelWDq(q𞁑𞁓) * ih𞁑Pack, SimdFloat(0.0f));
            // q, two accumulations and the two derivative products, in every lane.
            flops += 5 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓) +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
        }
        float const ΣW = simdSum(ΣWPack);
        auto const publishInitialDensity = [&]() {
            if (parameters.initializeDensity) {
                // Derived from Monaghan (1992), equation (3.8):
                // ρ𞁑 = m h𞁑⁻³ Σ𞁓 w(q𞁑𞁓), including bounded supports.
                // Symbol directory: ρ𞁑 → sink density; m → gas mass; h𞁑⁻¹ → inverse sink kernel
                // smoothing length; Σ𞁓 w(q𞁑𞁓) → kernel sum. Code names: ih𞁑 = h𞁑⁻¹.
                // No target-count approximation is used.
                int const sinkIndex = targetNode.sinkParticleRange.first -
                    targetNode.particleRange.first + particleIndex;
                float const ih𞁑 = 1.0f / h𞁑;
                data.sinkDensity[sinkIndex] = data.gasMass * sph_geometry::inverseVolume(ih𞁑) * ΣW;
                flops += 3 + sph_flops::kInverseVolume;
            }
        };
        auto const target = sph_volume::boundedTarget(h𞁑, hₘᵢₙ, hₘₐₓ, layers);
        float const ΣWₛₑₜ = ΣW_0 * target.factor;
        float const dΣWₛₑₜ_dh𞁑 = ΣW_0 * target.slope;
        float const residual = ΣW - ΣWₛₑₜ;
        // Reduction, target (12), residual and tolerance; abs is a sign operation.
        flops += SimdFloat::size() - 1 + 14;
        if (std::abs(residual) <= kSphSmoothingLengthTolerance * ΣWₛₑₜ ||
            (h𞁑 <= lowerLimit && residual >= 0.0f) || (h𞁑 >= upperLimit && residual <= 0.0f)) {
            publishInitialDensity();
            return true;
        }
        if (residual > 0.0f) {
            upper = h𞁑;
        } else {
            lower = h𞁑;
        }
        float const dF_dh𞁑 = simdSum(dΣW_dh𞁑Pack) - dΣWₛₑₜ_dh𞁑;
        flops += SimdFloat::size() - 1 + 5;    // Reduction, slope, change, two bounds, next h.
        float change = dF_dh𞁑 > 0.0f ? -residual / dF_dh𞁑 : 0.2f * h𞁑;
        change = std::min(std::max(change, -0.2f * h𞁑), 0.2f * h𞁑);
        float next = std::min(std::max(h𞁑 + change, lowerLimit), upperLimit);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == h𞁑) {
            publishInitialDensity();
            return true;
        }
        h𞁑 = next;
    }
    std::cout << std::format("SPH smoothing-length solve did not converge: smoothingLength={:e} "
                             "bounds=[{:e},{:e}] target={:e}\n",
                     h𞁑, hₘᵢₙ, hₘₐₓ, data.targetNeighborCount)
              << std::flush;
    ALWAYS_ASSERT(false);
    return false;
}

// clang-format off
/*
 * A support inside a bound layer, or a fixed one, counts as held; a fixed support has
 * Ω𞁑 = 1 and ζ𞁑 = 0 and no constraint to meet, so it leaves the residual and Ω𞁑
 * statistics alone. Any other correction that is non-finite or has Ω𞁑 ≤ 0 cannot be
 * used and stops the run; a small positive Ω𞁑 is counted.
 *
 * Symbol directory (in the code below):
 *   ζ𞁑 → weighted support correction; ψ → support stiffness
 *   h𞁑, ρ𞁑 → sink kernel smoothing length, density
 *   ΣW, Σ𞁓 q𞁑𞁓 w′ → kernel sum, kernel derivative moment
 *   Code names: Σqwʹ = Σ𞁓 q𞁑𞁓 w′.
 */
// clang-format on
void recordSphCpuGradH(SphGradHEvaluation const& gradH, float h𞁑, float ρ𞁑, float ΣW, float Σqwʹ,
    SphRunReturn& result) {
    if (gradH.stiffness < 1.0f) {
        result.heldSupports++;
    }
    if (gradH.stiffness <= 0.0f) {
        return;
    }
    if (!sph_volume::isUsable(gradH.correction)) {
        std::cout << std::format(
                         "SPH grad-H correction is unusable: smoothingLength={:e} density={:e} "
                         "\u03a3W={:e} \u03a3q dw/dq={:e} \u03a9={:e} \u03b6={:e} \u03c8={:e} "
                         "residual={:e}\n",
                         h𞁑, ρ𞁑, ΣW, Σqwʹ, gradH.correction.supportJacobian,
                         gradH.correction.weightedSupportCorrection, gradH.stiffness,
                         gradH.relativeResidual)
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

// Whether an active neighbor touches particle a: r𞁑𞁓² < max(h𞁑², h𞁓²).
// Symbol directory: r𞁑𞁓 → separation; r𞁑𞁓² → squared separation;
// h𞁑, h𞁓 → sink, neighbor kernel smoothing length. Code names: r2𞁑𞁓 = r𞁑𞁓²; h2𞁑 = h𞁑².
bool sphCpuHasActiveNeighbor(SphRunParameters const& parameters, SphRunData const& data,
    std::vector<int> const& neighbors, int particleIndex, float h𞁑, sph_flops::Counter& flops) {
    SimdFloat const h2𞁑(h𞁑 * h𞁑);
    flops += 1;
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const r𞁑𞁓 = sphCpuSeparations(data, particleIndex, neighbors, first);
        auto const r2𞁑𞁓 = r𞁑𞁓.dot(r𞁑𞁓);
        auto const h𞁓 = sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 0.0f);
        SimdFloat const neighborRung([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
        });
        auto const active = (neighborRung >= SimdFloat(float(parameters.minimumRung))) &&
            (neighborRung <= SimdFloat(float(parameters.maximumRung)));
        auto const intersects = r2𞁑𞁓 < simdMax(h2𞁑, sqr(h𞁓));
        // Fixed scales, dot operations, support square.
        flops += (sph_flops::kVectorOperation + sph_flops::kDotProduct + 1) * SimdFloat::size();
        if (simdAnyOf(sphCpuValidLanes(neighbors, first) && active && intersects)) {
            return true;
        }
    }
    return false;
}

}    // namespace detail

// h𞁑 → sink kernel smoothing length, solved in place.
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
            float& h𞁑 = data.sinkSmoothingLength[sinkIndex];
            bool const converged = detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, h𞁑, flops);
            data.sinkFlags[sinkIndex].setConverged(converged);
            if (converged) {
                result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, h𞁑);
                result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, h𞁑);
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
 *   ρ𞁑 = Σ𞁓 m𞁓 W𞁑𞁓(h𞁑), W𞁑𞁓 = h𞁑⁻³ w(q𞁑𞁓), q𞁑𞁓 = |r𞁑𞁓|/h𞁑,
 *   ρ𞁑 ∇·v𞁑 = −Σ𞁓 m𞁓 v𞁑𞁓 · ∇𞁑 W𞁑𞁓,
 *   ρ𞁑 ∇×v𞁑 = Σ𞁓 m𞁓 v𞁑𞁓 × ∇𞁑 W𞁑𞁓.
 * Symbol directory:
 *   h𞁑, h𞁑⁻¹ → sink kernel smoothing length, its inverse; q𞁑𞁓 → normalized radius
 *   h𞁑,stored → stored smoothing length, updated by the solver
 *   w(q𞁑𞁓) → kernel weight; m𞁓/h𞁑ᴰ → mass over smoothing volume
 *   ρ𞁑 → density; ρ𞁑 ∇·v𞁑 → density-weighted divergence
 *   ρ𞁑 ∇×v𞁑 → density-weighted curl
 *   γ, γ − 1 → gas adiabatic index, P/(ρu) of all gas
 *   r𞁑𞁓, |r𞁑𞁓| → separation, distance
 *   v𞁑𞁓 → velocity difference (peculiar stored velocity)
 *   m𞁓 h𞁑⁻⁽ᴰ⁺¹⁾ → mass times the sink inverse gradient scale, m𞁓 F𞁑 = m𞁓 h𞁑⁻⁽ᴰ⁺¹⁾ w′(q𞁑𞁓)/|r𞁑𞁓|
 *   m𞁓 F𞁑 → mass-weighted kernel gradient factor; m𞁓 ∇𞁑 W𞁑𞁓 → mass-weighted kernel gradient
 *   h_M/h → Monaghan to stored smoothing-length ratio (monaghan::lengthPerSupport)
 *   Σ𞁓 w → kernel sum; Σ𞁓 q𞁑𞁓 w′ → kernel derivative moment
 *   Code names: ih𞁑 = h𞁑⁻¹; h𞁑ₘₑₘ = h𞁑,stored; m𞁓_over_hᴰ𞁑 = m𞁓/h𞁑ᴰ
 *     γ_minus_1 = γ − 1; abs_r𞁑𞁓 = |r𞁑𞁓|; w_q𞁑𞁓 = w(q𞁑𞁓); ρ𞁑_div_v𞁑 = ρ𞁑 ∇·v𞁑; ρ𞁑_curl_v𞁑 = ρ𞁑 ∇×v𞁑
 *     m𞁓F𞁑 = m𞁓 F𞁑; m𞁓grad𞁑W𞁑𞁓 = m𞁓 ∇𞁑 W𞁑𞁓; ΣW = Σ𞁓 w; Σqwʹ = Σ𞁓 q𞁑𞁓 w′
 *     m𞁓ih_D_plus_1𞁑 = m𞁓 h𞁑⁻⁽ᴰ⁺¹⁾; h_M_over_h = h_M/h.
 * Peculiar velocities enter the divergence and the curl.
 * The curl stays a three-component record: a two-dimensional curl points along the inactive axis.
 */
// clang-format on
SphRunReturn sphCpuDensityAndGradients(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    bool const evolveMorrisMonaghan = parameters.viscosityTimestep > 0.0;
    sph_flops::Counter flops = 2;    // γ − 1 and h_M/h.
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every gas particle
    float const h_M_over_h =
        monaghan::lengthPerSupport(selectedHydroKernel().smoothingLengthPerEffectiveRadius);
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            parameters.isActiveRung(data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
        float& h𞁑ₘₑₘ = data.sinkSmoothingLength[sinkIndex];
        bool const semiactive = active ||
            detail::sphCpuHasActiveNeighbor(
                parameters, data, neighbors, particleIndex, h𞁑ₘₑₘ, flops);
        data.sinkFlags[sinkIndex].setSemiactive(semiactive);
        if (!semiactive) {
            continue;
        }
        bool const updateMorrisMonaghan =
            evolveMorrisMonaghan && active && !data.sinkFlags[sinkIndex].converged();
        if (!active && !data.sinkFlags[sinkIndex].converged() &&
            !detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, h𞁑ₘₑₘ, flops)) {
            data.sinkFlags[sinkIndex].setConverged(false);
            result.needsAnotherPass = true;
            continue;
        }
        // clang-format off
        /*
         * Equation 3.8 of Monaghan (1992) (the interpolant 2.7 at r𞁑): ρ𞁑 = Σ𞁓 m𞁓 W𞁑𞁓,
         * with the sink's own smoothing length in W(r, h𞁑) = w(r/h𞁑)/h𞁑³, a gather form of
         * equation 6.3 (which uses h_M𞁓). Every gas particle has mass m𞁓 = gasMass, and the
         * sum includes b = a.
         *
         * Symbol directory (in the code below):
         *   h𞁑 → sink kernel smoothing length
         *   m𞁓 → gas mass
         *   ρ𞁑 → density
         */
        // clang-format on
        float const h𞁑 = h𞁑ₘₑₘ;
        float const ih𞁑 = 1.0f / h𞁑;
        SimdFloat const m𞁓_over_hᴰ𞁑(data.gasMass * sph_geometry::inverseVolume(ih𞁑));
        SimdFloat const m𞁓ih_D_plus_1𞁑(data.gasMass * sph_geometry::inverseGradientScale(ih𞁑));
        // Reciprocal and the mass/support-volume and mass/gradient-scale products.
        flops += 3 + sph_flops::kInverseVolume + sph_flops::kInverseGradientScale;
        SimdFloat ρ𞁑Pack(0.0f);
        SimdFloat ρ𞁑_div_v𞁑Pack(0.0f);
        Vector3<SimdFloat> ρ𞁑_curl_v𞁑Pack(SimdFloat(0.0f));    // padded: see the curl above
        // clang-format off
        /*
         * Variational operator: the unweighted sums ΣW = Σ𞁓 w(q𞁑𞁓) and Σ𞁓 q𞁑𞁓 w'(q𞁑𞁓)
         * of cosmictiger/sph/volume.hpp give Ω𞁑 and ζ𞁑 at this smoothing length.
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
            SimdFloat const q𞁑𞁓 = abs_r𞁑𞁓 * SimdFloat(ih𞁑);
            SimdFloat const w_q𞁑𞁓 = kernelW(q𞁑𞁓);
            ρ𞁑Pack += simdSelect(
                detail::sphCpuValidLanes(neighbors, first), m𞁓_over_hᴰ𞁑 * w_q𞁑𞁓, SimdFloat(0.0f));
            // Fixed scales, norm, q𞁑𞁓, density product and sum.
            flops += (sph_flops::kVectorOperation + sph_flops::kNorm + 3) * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓);
            auto const inside = detail::sphCpuValidLanes(neighbors, first) && q𞁑𞁓 < SimdFloat(1.0f);
            ΣWPack += simdSelect(inside, w_q𞁑𞁓, SimdFloat(0.0f));
            ΣqwʹPack += simdSelect(inside, q𞁑𞁓 * dkernelWDq(q𞁑𞁓), SimdFloat(0.0f));
            flops += 3 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
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
                m𞁓ih_D_plus_1𞁑 * dkernelWDq(q𞁑𞁓) /
                    simdSelect(contributes, abs_r𞁑𞁓, SimdFloat(1.0f)),
                SimdFloat(0.0f));
            SpatialVector<SimdFloat> const v𞁑(data.sourceVelocity(particleIndex));
            SpatialVector<SimdFloat> const v𞁓 = detail::sphCpuLoad3(data.sourceVelocityX,
                data.sourceVelocityY, data.sourceVelocityZ, neighbors, first, 0.0f);
            SpatialVector<SimdFloat> const v𞁑𞁓 =
                simdSelect(contributes, v𞁑 - v𞁓, SpatialVector<SimdFloat>(SimdFloat(0)));
            // Kernel-gradient coefficient (2) and relative velocity.
            flops += (2 + sph_flops::kVectorOperation) * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
            SpatialVector<SimdFloat> const m𞁓grad𞁑W𞁑𞁓 = m𞁓F𞁑 * r𞁑𞁓;
            flops += sph_flops::kVectorOperation * SimdFloat::size();
            ρ𞁑_div_v𞁑Pack -= v𞁑𞁓.dot(m𞁓grad𞁑W𞁑𞁓);
            flops += (sph_flops::kDotProduct + 1) * SimdFloat::size();
            ρ𞁑_curl_v𞁑Pack +=
                sph_geometry::paddedVector(v𞁑𞁓).cross(sph_geometry::paddedVector(m𞁓grad𞁑W𞁑𞁓));
            flops += 12 * SimdFloat::size();
        }
        float const ρ𞁑 = simdSum(ρ𞁑Pack);
        flops += SimdFloat::size() - 1;
        ALWAYS_ASSERT(h𞁑 > 0.0f && ρ𞁑 > 0.0f);
        data.sinkDensity[sinkIndex] = ρ𞁑;
        float const ΣW = simdSum(ΣWPack);
        float const Σqwʹ = simdSum(ΣqwʹPack);
        auto const gradH = sphGradHEvaluation(h𞁑, parameters.minimumSmoothingLength,
            parameters.maximumSmoothingLength, data.targetNeighborCount, data.gasMass, ΣW, Σqwʹ);
        /*
         * Reductions, layers and target (12), residual (2), Ω, ζ, the weighted sums (8).
         */
        flops += 2 * (SimdFloat::size() - 1) + 24;
        detail::recordSphCpuGradH(gradH, h𞁑, ρ𞁑, ΣW, Σqwʹ, result);
        data.sinkGradHCorrection[sinkIndex] = gradH.correction.weightedSupportCorrection;
        float const c𞁑 = std::sqrt(γ * γ_minus_1 * data.sourceInternalEnergy[particleIndex]);
        flops += 3;
        if (data.sinkBalsaraFactor) {
            // The density-weighted sums take ρ𞁑 c𞁑/h𞁑 with h𞁑 the smoothing length.
            Vector3<float> const ρ𞁑_curl_v𞁑(simdSum(ρ𞁑_curl_v𞁑Pack[kXi]),
                simdSum(ρ𞁑_curl_v𞁑Pack[kYi]), simdSum(ρ𞁑_curl_v𞁑Pack[kZi]));
            data.sinkBalsaraFactor[sinkIndex] =
                balsara::encodeFactor(balsara::balsaraFactor(simdSum(ρ𞁑_div_v𞁑Pack),
                    ρ𞁑_curl_v𞁑.norm(), ρ𞁑 * c𞁑 / h𞁑, parameters.balsaraSoundSpeedCoefficient));
            flops += 4 * (SimdFloat::size() - 1) + 12;
        }
        if (updateMorrisMonaghan) {
            /*
             * Morris & Monaghan (1997), equations (5), (6), (13), with the Rosswog et al. (2000)
             * limiter, in conformal time; the update is in
             * cosmictiger/sph/viscosity/morris_and_monaghan.hpp. The sum/rho is a² div_phys(v);
             * conformal-time rates divide by a. τ𞁑⁻¹ = C c𞁑/h_M𞁑, ∇·v𞁑 = ρ𞁑 ∇·v𞁑/(a ρ𞁑).
             * Symbol directory:
             *   ∇·v𞁑 → divergence; c𞁑 → sound speed; h_M𞁑 → viscosity smoothing length
             *   τ𞁑⁻¹ → inverse decay time; C → viscosity decay parameter; Δt → time step
             *   α𞁑 → stored bulk viscosity coefficient
             *   h_M/h → Monaghan to stored smoothing-length ratio (monaghan::lengthPerSupport).
             *   Code names: div_v_a = ∇·v𞁑; iτ𞁑 = τ𞁑⁻¹; h_M_over_h = h_M/h.
             */
            float const div_v_a = simdSum(ρ𞁑_div_v𞁑Pack) / (ρ𞁑 * parameters.scaleFactor);
            float const h_M𞁑 = h𞁑 * h_M_over_h;
            float const iτ𞁑 = parameters.viscosityDecay * c𞁑 / (parameters.scaleFactor * h_M𞁑);
            int const rung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
            bool const actualInterval = parameters.globalTimestep;
            float const Δt = actualInterval ? parameters.viscosityTimestep :
                                              parameters.viscosityTimestep / (1 << rung);
            auto& α𞁑 = data.sinkBulkViscosityCoefficient[sinkIndex];
            α𞁑 = morris_and_monaghan::evolvedBulkViscosityCoefficient(float(α𞁑),
                parameters.minimumBulkViscosityCoefficient,
                parameters.maximumBulkViscosityCoefficient, iτ𞁑, div_v_a, Δt);
            flops += SimdFloat::size() - 1 + 18;
            flops.doublePrecision += actualInterval ? 1 : 2;
        }
        data.sinkFlags[sinkIndex].setConverged(true);
        result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, h𞁑);
        result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, h𞁑);
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}
