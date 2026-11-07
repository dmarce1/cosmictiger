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
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>
#include <cosmictiger/sph/viscosity/hu_etal.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/viscosity/morris_and_monaghan.hpp>
#include <cosmictiger/sph/viscosity/price_etal.hpp>

// Monaghan (1992), ARA&A 30, 543; cosmictiger/sph/viscosity/monaghan.hpp gives the full
// citation and the notation. Particle a is the sink and particle b a pack of staged neighbors.

namespace detail {

// clang-format off
/*
 * Section 6: the resolution length follows the local density, smoothingLength ∝ ρ_a^(−1/ν) with
 * ν = 3 (equations 6.1-6.2). Instead of integrating equation 6.2 or 6.7, the solve holds
 *   (4π/3) ρ_a H_a³/m = N_target,   i.e.   ΣW = Σ_b w(|r_ab|/H_a) = 3 N_target/(4π)
 * for the selected kernel W(r, H) = w(r/H)/H³, by a bounded Newton iteration on the
 * support radius H_a with dΣW/dH_a = −Σ_b q_ab w'(q_ab)/H_a, q_ab = |r_ab|/H_a. The bounds
 * limit the support radius without modifying the density sum.
 *
 * Symbol directory (in the code below):
 *   q_ab → normalized radius
 *   |r_ab| → separation magnitude packs
 *   H_a → sink kernel smoothing length
 *   ρ_a → density
 *   ΣW → kernel sum
 *   Code names: abs_r_ab = |r_ab|.
 */
// clang-format on
bool solveSphCpuSmoothingLength(SphRunParameters const& parameters, SphRunData const& data,
    SphTreeNode const& targetNode, std::vector<int> const& neighbors, int particleIndex,
    float& smoothingLength, sph_flops::Counter& flops) {
    float const minimumSmoothingLength = parameters.minimumSmoothingLength;
    float const maximumSmoothingLength = parameters.maximumSmoothingLength;
    ALWAYS_ASSERT(minimumSmoothingLength > 0.0f &&
        maximumSmoothingLength >= minimumSmoothingLength && data.targetNeighborCount > 0.0f);
    float& H_a = smoothingLength;
    H_a = std::min(std::max(H_a, minimumSmoothingLength), maximumSmoothingLength);
    Vector3<Position> position;
    position[kXi] = data.sourcePositionX[particleIndex];
    position[kYi] = data.sourcePositionY[particleIndex];
    position[kZi] = data.sourcePositionZ[particleIndex];
    // Distances do not change during the bounded support-radius solve.
    std::vector<SimdFloat> abs_r_ab;    // |r_ab| for each pack of neighbors
    abs_r_ab.reserve((neighbors.size() + SimdFloat::size() - 1) / SimdFloat::size());
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        Vector3<SimdFloat> const separation =
            sphCpuSeparations(data, particleIndex, neighbors, first);
        abs_r_ab.push_back(simdSqrt(sph_geometry::spatialDot(separation, separation)));
        flops +=
            (2 * kSpatialDimensions + 3) * SimdFloat::size();    // XYZ scales, spatial dot, root.
    }
    float lower = minimumSmoothingLength;
    float upper = maximumSmoothingLength;
    float const targetKernelSum =
        data.targetNeighborCount * float(1.0 / sph_geometry::kUnitBallVolume);
    flops += 1;
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            flops.singlePrecision += 1;    // Single-precision fixed-point scale.
            if (targetNode.outerBox.end[dim] <
                RangeFixed(position[dim]) + RangeFixed(H_a) + RangeFixed::min()) {
                return false;
            }
            flops.singlePrecision += 1;
            if (RangeFixed(position[dim]) <
                targetNode.outerBox.begin[dim] + RangeFixed(H_a) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const inverseSmoothingLength(1.0f / H_a);
        flops += 1;
        SimdFloat ΣWPack(0.0f);
        SimdFloat kernelSumSupportDerivativePack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const q_ab = abs_r_ab[first / SimdFloat::size()] * inverseSmoothingLength;
            auto const contributes = sphCpuValidLanes(neighbors, first) && q_ab < SimdFloat(1.0f);
            ΣWPack += simdSelect(contributes, kernelW(q_ab), SimdFloat(0.0f));
            kernelSumSupportDerivativePack -= simdSelect(
                contributes, q_ab * dkernelWDq(q_ab) * inverseSmoothingLength, SimdFloat(0.0f));
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
                // ρ_a -> sink density; m -> gas mass; H_a -> sink kernel smoothing length;
                // Σ_b w(q_ab) -> kernel sum. No target-count approximation is used.
                int const sinkIndex = targetNode.sinkParticleRange.first -
                    targetNode.particleRange.first + particleIndex;
                float const inverseSupport = 1.0f / H_a;
                data.sinkDensity[sinkIndex] =
                    data.gasMass * sph_geometry::inverseVolume(inverseSupport) * ΣW;
                flops += 5;
            }
        };
        float const residual = ΣW - targetKernelSum;
        flops += SimdFloat::size() - 1 +
            2;    // Reduction, residual and tolerance; abs is a sign operation.
        if (std::abs(residual) <= kSphSmoothingLengthTolerance * targetKernelSum ||
            (H_a <= minimumSmoothingLength && residual >= 0.0f) ||
            (H_a >= maximumSmoothingLength && residual <= 0.0f)) {
            publishInitialDensity();
            return true;
        }
        if (residual > 0.0f) {
            upper = H_a;
        } else {
            lower = H_a;
        }
        float const kernelSumSupportDerivative = simdSum(kernelSumSupportDerivativePack);
        flops += SimdFloat::size() - 1 + 4;    // Reduction, change, two bounds, next H.
        float change =
            kernelSumSupportDerivative > 0.0f ? -residual / kernelSumSupportDerivative : 0.2f * H_a;
        change = std::min(std::max(change, -0.2f * H_a), 0.2f * H_a);
        float next =
            std::min(std::max(H_a + change, minimumSmoothingLength), maximumSmoothingLength);
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
                     H_a, minimumSmoothingLength, maximumSmoothingLength, data.targetNeighborCount)
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
 */
// clang-format on
void recordSphCpuGradH(SphGradHEvaluation const& gradH, float smoothingLength, float density,
    float numberSum, float numberMoment, SphRunReturn& result) {
    if (gradH.constraint != sph_volume::SupportConstraint::Free) {
        result.heldSupports++;
        return;
    }
    if (!sph_volume::isUsable(gradH.correction)) {
        std::cout << std::format(
                         "SPH grad-H correction is unusable: smoothingLength={:e} density={:e} "
                         "\u03a3W={:e} \u03a3q dw/dq={:e} \u03a9={:e} \u03b6={:e} residual={:e}\n",
                         smoothingLength, density, numberSum, numberMoment,
                         gradH.correction.supportJacobian,
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

bool sphCpuHasActiveNeighbor(SphRunParameters const& parameters, SphRunData const& data,
    std::vector<int> const& neighbors, int particleIndex, float smoothingLength,
    sph_flops::Counter& flops) {
    SimdFloat const squaredSmoothingLength(smoothingLength * smoothingLength);
    flops += 1;
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const separation = sphCpuSeparations(data, particleIndex, neighbors, first);
        auto const squaredSeparation = sph_geometry::spatialDot(separation, separation);
        auto const neighborSupport = sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 0.0f);
        SimdFloat const neighborRung([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
        });
        auto const active = (neighborRung >= SimdFloat(float(parameters.minimumRung))) &&
            (neighborRung <= SimdFloat(float(parameters.maximumRung)));
        auto const intersects =
            squaredSeparation < simdMax(squaredSmoothingLength, sqr(neighborSupport));
        flops += 9 * SimdFloat::size();    // Three fixed scales, five dot ops, support square.
        if (simdAnyOf(sphCpuValidLanes(neighbors, first) && active && intersects)) {
            return true;
        }
    }
    return false;
}

}    // namespace detail

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
            float& smoothingLength = data.sinkSmoothingLength[sinkIndex];
            bool const converged = detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, smoothingLength, flops);
            data.sinkFlags[sinkIndex].setConverged(converged);
            if (converged) {
                result.minimumSmoothingLength =
                    std::min(result.minimumSmoothingLength, smoothingLength);
                result.maximumSmoothingLength =
                    std::max(result.maximumSmoothingLength, smoothingLength);
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
 * Density and gradient sums: Monaghan (1992), ARA&A 30, 543, equation (3.8),
 * doi:10.1146/annurev.aa.30.090192.002551; Cullen & Dehnen (2010), equations
 * (15), (17), (B2), (B4), (B7), https://arxiv.org/abs/1006.1524.
 *   ρ_a = Σ_b m_b W_ab(H_a), W_ab = H_a⁻³ w(q_ab), q_ab = |r_ab|/H_a,
 *   ρ_a ∇·v_a = −Σ_b m_b v_ab · ∇_a W_ab,
 *   ρ_a ∇×v_a = Σ_b m_b v_ab × ∇_a W_ab,
 *   R_a = ρ_a⁻¹ Σ_b m_b W_ab sign(∇·v_b),
 *   T = Σ_b r_ab ⊗ r_ab w̄_ab, D = Σ_b v_ab ⊗ r_ab w̄_ab, V = D T⁻¹.
 * Direct acceleration reconstruction (Cullen & Dehnen, equation B12):
 *   C = Σ_b F_ab ⊗ r_ab w̄_ab, ∇_x F = C T⁻¹, F_ab = F_a − F_b.
 * Symbol directory (Pack denotes SIMD partial sums):
 *   H_a → sink kernel smoothing length; q_ab → normalized radius; w(q_ab) → kernel weight
 *   ρ_a → density; ρ_a ∇·v_a → density-weighted divergence
 *   ρ_a ∇×v_a → density-weighted curl; ρ_a R_a → density-weighted sign average
 *   r_ab, v_ab → separation, velocity difference (peculiar stored velocity)
 *   m_b ∇_a W_ab → mass-weighted kernel gradient
 *   T, D, C → moment, velocity, acceleration gradient sums
 *   F_ab → acceleration difference; F → source acceleration (full canonical force)
 *   Σ_b w → kernel sum; Σ_b q_ab w′ → kernel derivative moment
 *   v_sig,a → signal speed
 *   w̄_ab = −m_b F_a → gradient weight inside the corrected-gradient branch.
 *   Code names: w_q_ab = w(q_ab); ρ_a_div_v_a = ρ_a ∇·v_a; ρ_a_curl_v_a = ρ_a ∇×v_a;
 *     ρ_a_R_a = ρ_a R_a; m_b_grad_a_W_ab = m_b ∇_a W_ab; ΣW = Σ_b w; Σqwʹ = Σ_b q_ab w′;
 *     v_sig_a = v_sig,a.
 * Peculiar velocities enter all switch gradients and densityDerivativeSum.
 * Dividing densityDerivativeSum by a ρ_a Ω_a gives d ln ρ_a/dτ
 * (Price 2012, equation (27), https://arxiv.org/abs/1012.1885).
 * Physical pair velocities include Hubble flow in the signal-speed estimate.
 */
// clang-format on
namespace detail {

template <bool retainHydroDiagnostics>
SphRunReturn sphCpuDensityAndGradientsImpl(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    bool const computeGradientDiagnostics =
        parameters.gradientSwitch() && data.sinkViscosityDiagnostics != nullptr;
    bool const computeAccelerationGradient =
        computeGradientDiagnostics && parameters.accelerationGradientViscosity;
    ALWAYS_ASSERT(!computeAccelerationGradient ||
        (data.sourceAccelerationX && data.sourceAccelerationY && data.sourceAccelerationZ));
    ALWAYS_ASSERT(!computeGradientDiagnostics || !parameters.cullenDehnen ||
        computeAccelerationGradient || data.sourceVelocityDivergence);
    bool const computeSphBalsara = parameters.balsaraSwitch && !parameters.gradientSwitch();
    bool const evolveMorrisMonaghan = parameters.variableViscosity &&
        !parameters.gradientSwitch() && parameters.viscosityTimestep > 0.0;
    sph_flops::Counter flops = 1;    // γ − 1.
    float const gasAdiabaticIndex = data.adiabaticIndex;
    float const pressureEnergyRatio = gasAdiabaticIndex - 1.0f;    // P/(ρu) for every gas particle
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            parameters.isActiveRung(data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
        float& smoothingLength = data.sinkSmoothingLength[sinkIndex];
        bool const semiactive = active ||
            detail::sphCpuHasActiveNeighbor(
                parameters, data, neighbors, particleIndex, smoothingLength, flops);
        data.sinkFlags[sinkIndex].setSemiactive(semiactive);
        if (!semiactive) {
            continue;
        }
        bool const updateMorrisMonaghan =
            evolveMorrisMonaghan && active && !data.sinkFlags[sinkIndex].converged();
        bool const computeSphDivergence = computeSphBalsara || updateMorrisMonaghan;
        if (!active && !data.sinkFlags[sinkIndex].converged() &&
            !detail::solveSphCpuSmoothingLength(
                parameters, data, targetNode, neighbors, particleIndex, smoothingLength, flops)) {
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
        float const H_a = smoothingLength;
        float const inverseSmoothingLength = 1.0f / H_a;
        SimdFloat const massOverSupportVolume(
            data.gasMass * sph_geometry::inverseVolume(inverseSmoothingLength));
        flops += kSpatialDimensions + 1;    // Reciprocal and mass/support products.
        SimdFloat ρ_aPack(0.0f);
        SimdFloat densityDerivativeSumPack(0.0f);
        SimdFloat ρ_a_div_v_aPack(0.0f);
        Vector3<SimdFloat> ρ_a_curl_v_aPack(SimdFloat(0.0f));
        cullen_and_dehnen::GradientSums<SimdFloat> gradientSums{};
        SimdFloat ρ_a_R_aPack(0.0f);
        SimdFloat v_sig_aPack(0.0f);
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
        float soundSpeedForSwitch =
            0.0f;    // Sound speed for the CD signal speed or the Price target.
        if (computeGradientDiagnostics) {
            soundSpeedForSwitch = std::sqrt(
                gasAdiabaticIndex * pressureEnergyRatio * data.sourceInternalEnergy[particleIndex]);
        }
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            Vector3<SimdFloat> const r_ab =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const radius = simdSqrt(sph_geometry::spatialDot(r_ab, r_ab));
            SimdFloat const q_ab = radius * SimdFloat(inverseSmoothingLength);
            SimdFloat const w_q_ab = kernelW(q_ab);
            ρ_aPack += simdSelect(detail::sphCpuValidLanes(neighbors, first),
                massOverSupportVolume * w_q_ab, SimdFloat(0.0f));
            flops += 12 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab);
            auto const inside =
                detail::sphCpuValidLanes(neighbors, first) && q_ab < SimdFloat(1.0f);
            ΣWPack += simdSelect(inside, w_q_ab, SimdFloat(0.0f));
            ΣqwʹPack += simdSelect(inside, q_ab * dkernelWDq(q_ab), SimdFloat(0.0f));
            flops += 3 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
            if (computeSphDivergence || computeGradientDiagnostics ||
                (data.sinkLogDensityRate || retainHydroDiagnostics)) {
                // clang-format off
                /*
                 * For constant viscosity and Morris & Monaghan, GADGET-4 equation
                 * (72) uses −Σ_b m_b v_ab·gradW_ab and Σ_b m_b v_ab×gradW_ab.
                 * Gradient switches instead reuse V = D T⁻¹ below. The independent
                 * density derivative keeps its variational SPH sum.
                 *
                 * Symbol directory (in the code below):
                 *   v_ab → velocity difference
                 *   m_b → gas mass
                 *   ρ_a → density
                 */
                // clang-format on
                auto const contributes = detail::sphCpuValidLanes(neighbors, first) &&
                    radius > SimdFloat(0.0f) && q_ab < SimdFloat(1.0f);
                SimdFloat const massWeightedGradientFactor = simdSelect(contributes,
                    massOverSupportVolume * SimdFloat(inverseSmoothingLength) * dkernelWDq(q_ab) /
                        simdSelect(contributes, radius, SimdFloat(1.0f)),
                    SimdFloat(0.0f));
                Vector3<SimdFloat> const v_ab = simdSelect(contributes,
                    Vector3<SimdFloat>(SimdFloat(data.sourceVelocityX[particleIndex]),
                        SimdFloat(data.sourceVelocityY[particleIndex]),
                        SimdFloat(data.sourceVelocityZ[particleIndex])) -
                        detail::sphCpuLoad3(data.sourceVelocityX, data.sourceVelocityY,
                            data.sourceVelocityZ, neighbors, first, 0.0f),
                    Vector3<SimdFloat>(SimdFloat(0)));
                // Kernel-gradient coefficient (3) and relative velocity (3).
                flops += 6 * SimdFloat::size() +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
                if (computeSphDivergence || (data.sinkLogDensityRate || retainHydroDiagnostics)) {
                    Vector3<SimdFloat> const m_b_grad_a_W_ab = massWeightedGradientFactor * r_ab;
                    flops += 3 * SimdFloat::size();
                    if (data.sinkLogDensityRate || retainHydroDiagnostics) {
                        // Grad-H and conformal-time factors are applied below.
                        densityDerivativeSumPack += sph_geometry::spatialDot(v_ab, m_b_grad_a_W_ab);
                        flops += 6 * SimdFloat::size();
                    }
                    if (computeSphDivergence) {
                        ρ_a_div_v_aPack -= sph_geometry::spatialDot(v_ab, m_b_grad_a_W_ab);
                        flops += 6 * SimdFloat::size();
                    }
                    if (computeSphBalsara) {
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            int const next = (dim + 1) % kDimensions;
                            int const last = (dim + 2) % kDimensions;
                            ρ_a_curl_v_aPack[dim] += v_ab[next] * m_b_grad_a_W_ab[last] -
                                v_ab[last] * m_b_grad_a_W_ab[next];
                        }
                        flops += 12 * SimdFloat::size();
                    }
                }
                if (computeGradientDiagnostics) {
                    /*
                     * Cullen & Dehnen (2010), equations (B2), (B4), (B7), and
                     * Price et al. (2018), equations (51)-(52): V = D T⁻¹.
                     * Equation (B7) allows the common mass/kernel-gradient
                     * weights, independent of densities changing on other ranks.
                     */
                    SimdFloat const w̄_ab = -massWeightedGradientFactor;
                    for (int i = 0; i < kSpatialDimensions; ++i) {
                        for (int j = 0; j < kSpatialDimensions; ++j) {
                            gradientSums.moment[i][j] += w̄_ab * r_ab[i] * r_ab[j];
                            gradientSums.velocity[i][j] += w̄_ab * v_ab[i] * r_ab[j];
                        }
                    }
                    flops += 6 * kSpatialDimensions * kSpatialDimensions * SimdFloat::size();
                    if (computeAccelerationGradient) {
                        Vector3<SimdFloat> const F_ab = simdSelect(contributes,
                            Vector3<SimdFloat>(SimdFloat(data.sourceAccelerationX[particleIndex]),
                                SimdFloat(data.sourceAccelerationY[particleIndex]),
                                SimdFloat(data.sourceAccelerationZ[particleIndex])) -
                                detail::sphCpuLoad3(data.sourceAccelerationX,
                                    data.sourceAccelerationY, data.sourceAccelerationZ, neighbors,
                                    first, 0.0f),
                            Vector3<SimdFloat>(SimdFloat(0)));
                        for (int row = 0; row < kSpatialDimensions; ++row) {
                            for (int column = 0; column < kSpatialDimensions; ++column) {
                                gradientSums.acceleration[row][column] +=
                                    w̄_ab * F_ab[row] * r_ab[column];
                            }
                        }
                        flops +=
                            (3 + 3 * kSpatialDimensions * kSpatialDimensions) * SimdFloat::size();
                    }
                    if (parameters.cullenDehnen && data.sourceVelocityDivergence) {
                        // Equations (15), (17): prior neighbor signs, including self.
                        SimdFloat const neighborVelocityDivergence = detail::sphCpuLoad(
                            data.sourceVelocityDivergence, neighbors, first, 0.0f);
                        SimdFloat const neighborDivergenceSign =
                            simdSelect(neighborVelocityDivergence > SimdFloat(0), SimdFloat(1),
                                simdSelect(neighborVelocityDivergence < SimdFloat(0), SimdFloat(-1),
                                    SimdFloat(0)));
                        ρ_a_R_aPack += simdSelect(detail::sphCpuValidLanes(neighbors, first),
                            massOverSupportVolume * w_q_ab * neighborDivergenceSign, SimdFloat(0));
                        flops += 3 * SimdFloat::size();
                    }
                    if (parameters.cullenDehnen || parameters.huEtAl) {
                        // Hubble flow enters the approaching-pair signal speed only.
                        Vector3<SimdFloat> const physicalVelocityDifference =
                            v_ab + SimdFloat(parameters.expansionRate) * r_ab;
                        SimdFloat const neighborSoundSpeed =
                            simdSqrt(SimdFloat(gasAdiabaticIndex) * SimdFloat(pressureEnergyRatio) *
                                simdSelect(contributes,
                                    detail::sphCpuLoad(
                                        data.sourceInternalEnergy, neighbors, first, 1.0f),
                                    SimdFloat(1)));
                        SimdFloat const radialVelocity =
                            sph_geometry::spatialDot(physicalVelocityDifference, r_ab) /
                            simdSelect(contributes, radius, SimdFloat(1));
                        v_sig_aPack = simdMax(v_sig_aPack,
                            simdSelect(contributes,
                                SimdFloat(0.5f) *
                                        (SimdFloat(soundSpeedForSwitch) + neighborSoundSpeed) -
                                    simdMin(radialVelocity, SimdFloat(0)),
                                SimdFloat(0)));
                        flops += 18 * SimdFloat::size();
                    }
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
        if (data.sinkLogDensityRate || retainHydroDiagnostics) {
            // clang-format off
            /*
             * Price (2012), equation (27), in the stored comoving variables:
             *   dρ_c/dτ = Σ_b m_b (w_a−w_b)·∇_aW_ab / (a Ω_a).
             * Held supports use Ω_a=1. This is peculiar compression, excluding
             * homogeneous Hubble dilution; w=a v_pec and dt=a dτ.
             *
             * Symbol directory:
             *   ρ_c → density; dρ_c/dτ → density rate;
             *   w_a−w_b → velocity difference; a → scale factor;
             *   Ω_a → support Jacobian;
             *   Σ_b m_b (w_a−w_b)·∇_aW_ab → density derivative sum.
             *   Code names: dρ_c_dτ = dρ_c/dτ.
             */
            // clang-format on
            if constexpr (retainHydroDiagnostics) {
                float const dρ_c_dτ = simdSum(densityDerivativeSumPack) /
                    (parameters.scaleFactor * gradH.correction.supportJacobian);
                data.sinkHydroDiagnostics[sinkIndex].densityRate = dρ_c_dτ;
                if (data.sinkLogDensityRate) {
                    data.sinkLogDensityRate[sinkIndex] = dρ_c_dτ / ρ_a;
                    flops += 1;
                }
                flops += SimdFloat::size() - 1 + 2;
            } else {
                // Preserve the original single-division predictor path when diagnostics are off.
                data.sinkLogDensityRate[sinkIndex] = simdSum(densityDerivativeSumPack) /
                    (parameters.scaleFactor * ρ_a * gradH.correction.supportJacobian);
                flops += SimdFloat::size() - 1 + 3;
            }
        }
        if (computeSphBalsara && data.sinkBalsaraFactor) {
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
             * Morris & Monaghan (1997), equations (5), (6), (13).
             * The sum/rho is a² div_phys(v); conformal-time rates divide by a.
             */
            float const velocityDivergence =
                simdSum(ρ_a_div_v_aPack) / (ρ_a * parameters.scaleFactor);
            float const soundSpeed = std::sqrt(
                gasAdiabaticIndex * pressureEnergyRatio * data.sourceInternalEnergy[particleIndex]);
            float const smoothingLength = H_a /
                selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                float(monaghan::kViscosityWidthScale);
            float const inverseDecayTime =
                parameters.viscosityDecay * soundSpeed / (parameters.scaleFactor * smoothingLength);
            int const rung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
            bool const actualInterval = parameters.globalTimestep;
            float const timeStep = actualInterval ? parameters.viscosityTimestep :
                                                    parameters.viscosityTimestep / (1 << rung);
            auto& bulkViscosityCoefficient = data.sinkBulkViscosityCoefficient[sinkIndex];
            bulkViscosityCoefficient = morris_and_monaghan::evolvedBulkViscosityCoefficient(
                float(bulkViscosityCoefficient), parameters.minimumBulkViscosityCoefficient,
                inverseDecayTime, velocityDivergence, timeStep);
            flops += SimdFloat::size() - 1 + 19;
            flops.doublePrecision += actualInterval ? 1 : 2;
        }
        if (computeGradientDiagnostics) {
            cullen_and_dehnen::GradientSums<float> sums{};
            for (int i = 0; i < kSpatialDimensions; ++i) {
                for (int j = 0; j < kSpatialDimensions; ++j) {
                    sums.moment[i][j] = simdSum(gradientSums.moment[i][j]);
                    sums.velocity[i][j] = simdSum(gradientSums.velocity[i][j]);
                    if (computeAccelerationGradient) {
                        sums.acceleration[i][j] = simdSum(gradientSums.acceleration[i][j]);
                    }
                }
            }
            // Both sensor routes leave endpoint history and alpha unchanged.
            float velocityGradient[kDimensions][kDimensions];
            float divergenceDerivative = 0.0f;
            int rank;
            if (computeAccelerationGradient) {
                float accelerationGradient[kDimensions][kDimensions];
                rank = cullen_and_dehnen::reconstructGradients(
                    sums, velocityGradient, accelerationGradient, flops);
                if (rank != 0) {
                    divergenceDerivative = cullen_and_dehnen::peculiarDivergenceDerivative(
                        velocityGradient, accelerationGradient, parameters.scaleFactor, flops);
                }
                flops += 9 * (SimdFloat::size() - 1);
            } else {
                rank =
                    cullen_and_dehnen::reconstructVelocityGradient(sums, velocityGradient, flops);
            }
            float const canonicalDivergence =
                velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2];
            flops += 2;
            if (!computeAccelerationGradient) {
                ALWAYS_ASSERT(data.sinkScaledDivergence);
                auto const& state = data.sinkScaledDivergence[sinkIndex];
                int const currentRung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
                double const sampleInterval =
                    sphViscositySampleInterval(currentRung, parameters.viscositySampleInterval,
                        parameters.globalTimestep, parameters.viscosityInitializeHistory);
                double const previousSampleTimestep =
                    sphViscosityStoredSampleTimestep(currentRung, parameters.viscosityBaseTimestep,
                        parameters.viscosityPreviousTimestep, parameters.globalTimestep);
                if (!parameters.globalTimestep && sampleInterval > 0.0) {
                    flops.doublePrecision += 2;
                }
                if (rank != 0) {
                    divergenceDerivative = cullen_and_dehnen::divergenceTimeDerivative(
                        canonicalDivergence, parameters.scaleFactor, state, sampleInterval,
                        previousSampleTimestep, flops);
                }
            }
            if (parameters.balsaraSwitch && data.sinkBalsaraFactor) {
                data.sinkBalsaraFactor[sinkIndex] = balsara::encodeFactor(
                    balsara::factorFromVelocityGradient(velocityGradient, flops));
            }
            cullen_and_dehnen::Result<float> diagnostics;
            if (parameters.cullenDehnen) {
                diagnostics = cullen_and_dehnen::diagnosticsFromGradient(velocityGradient,
                    parameters.scaleFactor, divergenceDerivative, simdSum(ρ_a_R_aPack) / ρ_a, H_a,
                    std::max(soundSpeedForSwitch, simdMaximum(v_sig_aPack)),
                    parameters.maximumBulkViscosityCoefficient, parameters.viscosityDecayCD, flops);
                flops += SimdFloat::size() - 1;
            } else if (parameters.huEtAl) {
                diagnostics = hu_etal::diagnosticsFromGradient(velocityGradient,
                    parameters.scaleFactor, divergenceDerivative, H_a, soundSpeedForSwitch,
                    std::max(soundSpeedForSwitch, simdMaximum(v_sig_aPack)),
                    parameters.maximumBulkViscosityCoefficientHu, parameters.viscosityDecayHu,
                    flops);
            } else {
                /*
                 * Price et al. (2018), equations (46)-(48): the paper's smoothing length is
                 * the same calibrated viscosity length as Morris & Monaghan's h.
                 */
                float const smoothingLength = H_a /
                    selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                    float(monaghan::kViscosityWidthScale);
                diagnostics = price_etal::diagnosticsFromGradient(velocityGradient,
                    parameters.scaleFactor, divergenceDerivative, smoothingLength,
                    soundSpeedForSwitch, parameters.maximumBulkViscosityCoefficientPrice,
                    parameters.viscosityDecayPrice, flops);
                flops += 2;
            }
            diagnostics.divergence = canonicalDivergence;
            if (data.sinkViscosityDiagnostics != nullptr) {
                auto& pending = data.sinkViscosityDiagnostics[sinkIndex];
                pending.pendingDivergence = diagnostics.divergence;
                pending.targetBulkViscosityCoefficient = diagnostics.targetBulkViscosityCoefficient;
                pending.inverseDecayTime = diagnostics.inverseDecayTime;
                pending.limiter = diagnostics.limiter;
            }
            flops += 18 * (SimdFloat::size() - 1);
        }
        data.sinkFlags[sinkIndex].setConverged(true);
        result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, H_a);
        result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, H_a);
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}

}    // namespace detail

SphRunReturn sphCpuDensityAndGradients(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    // One dispatch per target node; the disabled specialization contains no diagnostic work.
    return data.sinkHydroDiagnostics ?
        detail::sphCpuDensityAndGradientsImpl<true>(parameters, data, targetNodeIndex) :
        detail::sphCpuDensityAndGradientsImpl<false>(parameters, data, targetNodeIndex);
}
