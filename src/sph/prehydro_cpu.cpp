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
 *   (4π/3) ρ_a H_a³/m = N_target,   i.e.   ΣW = Σ_b w(r_ab/H_a) = 3 N_target/(4π)
 * for the selected kernel W(r, H) = w(r/H)/H³, by a bounded Newton iteration on the
 * support radius H_a with dΣW/dH_a = −Σ_b q_ab w'(q_ab)/H_a, q_ab = r_ab/H_a. The bounds
 * limit the support radius without modifying the density sum.
 *
 * Symbol directory (in the code below):
 *   q_ab → normalizedRadius
 *   |r_ab| → separationMagnitudePacks
 *   H_a → sinkKernelSupportRadius
 *   ρ_a → density
 *   ΣW → kernelSum
 */
// clang-format on
bool solveSphCpuSupportRadius(SphRunParameters const& parameters, SphRunData const& data,
    SphTreeNode const& targetNode, std::vector<int> const& neighbors, int particleIndex,
    float& supportRadius, sph_flops::Counter& flops) {
    float const minimumSupportRadius = parameters.minimumSupportRadius;
    float const maximumSupportRadius = parameters.maximumSupportRadius;
    ALWAYS_ASSERT(minimumSupportRadius > 0.0f && maximumSupportRadius >= minimumSupportRadius &&
        data.targetNeighborCount > 0.0f);
    float& sinkKernelSupportRadius = supportRadius;
    sinkKernelSupportRadius =
        std::min(std::max(sinkKernelSupportRadius, minimumSupportRadius), maximumSupportRadius);
    Vector3<Position> position;
    position[kXDimension] = data.sourcePositionX[particleIndex];
    position[kYDimension] = data.sourcePositionY[particleIndex];
    position[kZDimension] = data.sourcePositionZ[particleIndex];
    // Distances do not change during the bounded support-radius solve.
    std::vector<SimdFloat> separationMagnitudePacks;    // |r_ab| for each pack of neighbors
    separationMagnitudePacks.reserve(
        (neighbors.size() + SimdFloat::size() - 1) / SimdFloat::size());
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        Vector3<SimdFloat> const separation =
            sphCpuSeparations(data, particleIndex, neighbors, first);
        separationMagnitudePacks.push_back(simdSqrt(separation.dot(separation)));
        flops += 9 * SimdFloat::size();    // Three fixed scales, dot product, square root.
    }
    float lower = minimumSupportRadius;
    float upper = maximumSupportRadius;
    float const targetKernelSum =
        data.targetNeighborCount * (3.0f / (4.0f * std::numbers::pi_v<float>) );
    flops += 1;
    for (int iteration = 0; iteration < 128; ++iteration) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            flops.singlePrecision += 1;    // Single-precision fixed-point scale.
            if (targetNode.outerBox.end[dim] < RangeFixed(position[dim]) +
                    RangeFixed(sinkKernelSupportRadius) + RangeFixed::min()) {
                return false;
            }
            flops.singlePrecision += 1;
            if (RangeFixed(position[dim]) < targetNode.outerBox.begin[dim] +
                    RangeFixed(sinkKernelSupportRadius) + RangeFixed::min()) {
                return false;
            }
        }
        SimdFloat const inverseSupportRadius(1.0f / sinkKernelSupportRadius);
        flops += 1;
        SimdFloat kernelSumPack(0.0f);
        SimdFloat kernelSumSupportDerivativePack(0.0f);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            auto const normalizedRadius =
                separationMagnitudePacks[first / SimdFloat::size()] * inverseSupportRadius;
            auto const contributes =
                sphCpuValidLanes(neighbors, first) && normalizedRadius < SimdFloat(1.0f);
            kernelSumPack += simdSelect(contributes, kernelW(normalizedRadius), SimdFloat(0.0f));
            kernelSumSupportDerivativePack -= simdSelect(contributes,
                normalizedRadius * dkernelWDq(normalizedRadius) * inverseSupportRadius,
                SimdFloat(0.0f));
            // q, two accumulations and the two derivative products, in every lane.
            flops += 5 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), normalizedRadius) +
                sph_flops::densityKernelOperations(selectedHydroKernel(), normalizedRadius, true);
        }
        float const residual = simdSum(kernelSumPack) - targetKernelSum;
        flops += SimdFloat::size() - 1 +
            2;    // Reduction, residual and tolerance; abs is a sign operation.
        if (std::abs(residual) <= kSphSupportRadiusTolerance * targetKernelSum ||
            (sinkKernelSupportRadius <= minimumSupportRadius && residual >= 0.0f) ||
            (sinkKernelSupportRadius >= maximumSupportRadius && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = sinkKernelSupportRadius;
        } else {
            lower = sinkKernelSupportRadius;
        }
        float const kernelSumSupportDerivative = simdSum(kernelSumSupportDerivativePack);
        flops += SimdFloat::size() - 1 + 4;    // Reduction, change, two bounds, next H.
        float change = kernelSumSupportDerivative > 0.0f ? -residual / kernelSumSupportDerivative :
                                                           0.2f * sinkKernelSupportRadius;
        change = std::min(
            std::max(change, -0.2f * sinkKernelSupportRadius), 0.2f * sinkKernelSupportRadius);
        float next = std::min(
            std::max(sinkKernelSupportRadius + change, minimumSupportRadius), maximumSupportRadius);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
            flops += 2;
        }
        if (next == sinkKernelSupportRadius) {
            return true;
        }
        sinkKernelSupportRadius = next;
    }
    std::cout << std::format("SPH smoothing-length solve did not converge: supportRadius={:e} "
                             "bounds=[{:e},{:e}] target={:e}\n",
                     sinkKernelSupportRadius, minimumSupportRadius, maximumSupportRadius,
                     data.targetNeighborCount)
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
 *   ζ_a → gradH.correction.weightedSupportCorrection
 */
// clang-format on
void recordSphCpuGradH(SphGradHEvaluation const& gradH, float supportRadius, float density,
    float numberSum, float numberMoment, SphRunReturn& result) {
    if (gradH.constraint != sph_volume::SupportConstraint::Free) {
        result.heldSupports++;
        return;
    }
    if (!sph_volume::isUsable(gradH.correction)) {
        std::cout << std::format(
                         "SPH grad-H correction is unusable: supportRadius={:e} density={:e} "
                         "\u03a3W={:e} \u03a3q dw/dq={:e} \u03a9={:e} \u03b6={:e} residual={:e}\n",
                         supportRadius, density, numberSum, numberMoment,
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
    std::vector<int> const& neighbors, int particleIndex, float supportRadius,
    sph_flops::Counter& flops) {
    SimdFloat const squaredSupportRadius(supportRadius * supportRadius);
    flops += 1;
    for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
        auto const separation = sphCpuSeparations(data, particleIndex, neighbors, first);
        auto const squaredSeparation = separation.dot(separation);
        auto const neighborSupport = sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 0.0f);
        SimdFloat const neighborRung([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) : 0.0f;
        });
        auto const active = (neighborRung >= SimdFloat(float(parameters.minimumRung))) &&
            (neighborRung <= SimdFloat(float(parameters.maximumRung)));
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
            float& supportRadius = data.sinkSmoothingLength[sinkIndex];
            bool const converged = detail::solveSphCpuSupportRadius(
                parameters, data, targetNode, neighbors, particleIndex, supportRadius, flops);
            data.sinkFlags[sinkIndex].setConverged(converged);
            if (converged) {
                result.minimumSupportRadius = std::min(result.minimumSupportRadius, supportRadius);
                result.maximumSupportRadius = std::max(result.maximumSupportRadius, supportRadius);
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
 * Symbol directory (Pack denotes SIMD partial sums):
 *   H_a → sinkKernelSupportRadius; q_ab → normalizedRadius; w(q_ab) → kernelWeight
 *   ρ_a → density; ρ_a ∇·v_a → densityWeightedDivergence
 *   ρ_a ∇×v_a → densityWeightedCurl; ρ_a R_a → densityWeightedSignAverage
 *   r_ab, v_ab → separation, velocityDifference (peculiar stored velocity)
 *   m_b ∇_a W_ab → massWeightedKernelGradient
 *   T, D → gradientSums.moment, gradientSums.velocity
 *   Σ_b w → kernelSum; Σ_b q_ab w′ → kernelDerivativeMoment
 *   v_sig,a → signalSpeed (CUDA), signalSpeedPack (CPU)
 *   w̄_ab = −m_b F_a → gradientWeight inside the corrected-gradient branch.
 * Peculiar velocities enter all switch gradients and densityDerivativeSum.
 * Dividing densityDerivativeSum by a ρ_a Ω_a gives d ln ρ_a/dτ
 * (Price 2012, equation (27), https://arxiv.org/abs/1012.1885).
 * Physical pair velocities include Hubble flow in the signal-speed estimate.
 */
// clang-format on
SphRunReturn sphCpuPrehydro2(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    bool const computeGradientHistory =
        parameters.gradientSwitch() && data.sinkViscosityDiagnostics != nullptr;
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
        float& supportRadius = data.sinkSmoothingLength[sinkIndex];
        bool const semiactive = active ||
            detail::sphCpuHasActiveNeighbor(
                parameters, data, neighbors, particleIndex, supportRadius, flops);
        data.sinkFlags[sinkIndex].setSemiactive(semiactive);
        if (!semiactive) {
            continue;
        }
        bool const updateMorrisMonaghan =
            evolveMorrisMonaghan && active && !data.sinkFlags[sinkIndex].converged();
        bool const computeSphDivergence = computeSphBalsara || updateMorrisMonaghan;
        if (!active && !data.sinkFlags[sinkIndex].converged() &&
            !detail::solveSphCpuSupportRadius(
                parameters, data, targetNode, neighbors, particleIndex, supportRadius, flops)) {
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
         *   H_a → sinkKernelSupportRadius
         *   m_b → data.gasMass
         *   ρ_a → density
         */
        // clang-format on
        float const sinkKernelSupportRadius = supportRadius;
        float const inverseSupportRadius = 1.0f / sinkKernelSupportRadius;
        SimdFloat const massOverSupportVolume(
            data.gasMass * inverseSupportRadius * inverseSupportRadius * inverseSupportRadius);
        flops += 4;    // Reciprocal and three mass/support products.
        SimdFloat densityPack(0.0f);
        SimdFloat densityDerivativeSumPack(0.0f);
        SimdFloat densityWeightedDivergencePack(0.0f);
        Vector3<SimdFloat> densityWeightedCurlPack(SimdFloat(0.0f));
        cullen_and_dehnen::VelocityGradientSums<SimdFloat> gradientSums{};
        SimdFloat densityWeightedSignAveragePack(0.0f);
        SimdFloat signalSpeedPack(0.0f);
        // clang-format off
        /*
         * Variational operator: the unweighted sums ΣW = Σ_b w(q_ab) and Σ_b q_ab w'(q_ab)
         * of cosmictiger/sph/volume.hpp give Ω_a and ζ_a at this support radius.
         *
         * Symbol directory (in the code below):
         *   q_ab → normalizedRadius
         *   ζ_a → gradH.correction.weightedSupportCorrection
         *   ΣW → kernelSum
         */
        // clang-format on
        SimdFloat kernelSumPack(0.0f);
        SimdFloat kernelDerivativeMomentPack(0.0f);
        float soundSpeedForSwitch =
            0.0f;    // Sound speed for the CD signal speed or the Price target.
        if (computeGradientHistory) {
            soundSpeedForSwitch = std::sqrt(
                gasAdiabaticIndex * pressureEnergyRatio * data.sourceInternalEnergy[particleIndex]);
        }
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            Vector3<SimdFloat> const separation =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const radius = simdSqrt(separation.dot(separation));
            SimdFloat const normalizedRadius = radius * SimdFloat(inverseSupportRadius);
            SimdFloat const kernelWeight = kernelW(normalizedRadius);
            densityPack += simdSelect(detail::sphCpuValidLanes(neighbors, first),
                massOverSupportVolume * kernelWeight, SimdFloat(0.0f));
            flops += 12 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), normalizedRadius);
            auto const inside =
                detail::sphCpuValidLanes(neighbors, first) && normalizedRadius < SimdFloat(1.0f);
            kernelSumPack += simdSelect(inside, kernelWeight, SimdFloat(0.0f));
            kernelDerivativeMomentPack += simdSelect(
                inside, normalizedRadius * dkernelWDq(normalizedRadius), SimdFloat(0.0f));
            flops += 3 * SimdFloat::size() +
                sph_flops::densityKernelOperations(selectedHydroKernel(), normalizedRadius, true);
            if (computeSphDivergence || computeGradientHistory || data.sinkLogDensityRate) {
                // clang-format off
                /*
                 * For constant viscosity and Morris & Monaghan, GADGET-4 equation
                 * (72) uses −Σ_b m_b v_ab·gradW_ab and Σ_b m_b v_ab×gradW_ab.
                 * Gradient switches instead reuse V = D T⁻¹ below. The independent
                 * density derivative keeps its variational SPH sum.
                 *
                 * Symbol directory (in the code below):
                 *   v_ab → velocityDifference
                 *   m_b → data.gasMass
                 *   ρ_a → density
                 */
                // clang-format on
                auto const contributes = detail::sphCpuValidLanes(neighbors, first) &&
                    radius > SimdFloat(0.0f) && normalizedRadius < SimdFloat(1.0f);
                SimdFloat const massWeightedGradientFactor = simdSelect(contributes,
                    massOverSupportVolume * SimdFloat(inverseSupportRadius) *
                        dkernelWDq(normalizedRadius) /
                        simdSelect(contributes, radius, SimdFloat(1.0f)),
                    SimdFloat(0.0f));
                Vector3<SimdFloat> const velocityDifference = simdSelect(contributes,
                    Vector3<SimdFloat>(SimdFloat(data.sourceVelocityX[particleIndex]),
                        SimdFloat(data.sourceVelocityY[particleIndex]),
                        SimdFloat(data.sourceVelocityZ[particleIndex])) -
                        detail::sphCpuLoad3(data.sourceVelocityX, data.sourceVelocityY,
                            data.sourceVelocityZ, neighbors, first, 0.0f),
                    Vector3<SimdFloat>(SimdFloat(0)));
                // Kernel-gradient coefficient (3) and relative velocity (3).
                flops += 6 * SimdFloat::size() +
                    sph_flops::densityKernelOperations(
                        selectedHydroKernel(), normalizedRadius, true);
                if (computeSphDivergence || data.sinkLogDensityRate) {
                    Vector3<SimdFloat> const massWeightedKernelGradient =
                        massWeightedGradientFactor * separation;
                    flops += 3 * SimdFloat::size();
                    if (data.sinkLogDensityRate) {
                        // Grad-H and conformal-time factors are applied below.
                        densityDerivativeSumPack +=
                            velocityDifference.dot(massWeightedKernelGradient);
                        flops += 6 * SimdFloat::size();
                    }
                    if (computeSphDivergence) {
                        densityWeightedDivergencePack -=
                            velocityDifference.dot(massWeightedKernelGradient);
                        flops += 6 * SimdFloat::size();
                    }
                    if (computeSphBalsara) {
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            int const next = (dim + 1) % kDimensions;
                            int const last = (dim + 2) % kDimensions;
                            densityWeightedCurlPack[dim] +=
                                velocityDifference[next] * massWeightedKernelGradient[last] -
                                velocityDifference[last] * massWeightedKernelGradient[next];
                        }
                        flops += 12 * SimdFloat::size();
                    }
                }
                if (computeGradientHistory) {
                    /*
                     * Cullen & Dehnen (2010), equations (B2), (B4), (B7), and
                     * Price et al. (2018), equations (51)-(52): V = D T⁻¹.
                     * Equation (B7) allows the common mass/kernel-gradient
                     * weights, independent of densities changing on other ranks.
                     */
                    SimdFloat const gradientWeight = -massWeightedGradientFactor;
                    for (int i = 0; i < kDimensions; ++i) {
                        for (int j = 0; j < kDimensions; ++j) {
                            gradientSums.moment[i][j] +=
                                gradientWeight * separation[i] * separation[j];
                            gradientSums.velocity[i][j] +=
                                gradientWeight * velocityDifference[i] * separation[j];
                        }
                    }
                    flops += 54 * SimdFloat::size();
                    if (parameters.cullenDehnen) {
                        // Equations (15), (17): prior neighbor signs, including self.
                        SimdFloat const neighborVelocityDivergence = detail::sphCpuLoad(
                            data.sourceVelocityDivergence, neighbors, first, 0.0f);
                        SimdFloat const neighborDivergenceSign =
                            simdSelect(neighborVelocityDivergence > SimdFloat(0), SimdFloat(1),
                                simdSelect(neighborVelocityDivergence < SimdFloat(0), SimdFloat(-1),
                                    SimdFloat(0)));
                        densityWeightedSignAveragePack +=
                            simdSelect(detail::sphCpuValidLanes(neighbors, first),
                                massOverSupportVolume * kernelWeight * neighborDivergenceSign,
                                SimdFloat(0));
                        flops += 3 * SimdFloat::size();
                    }
                    if (parameters.cullenDehnen || parameters.huEtAl) {
                        // Hubble flow enters the approaching-pair signal speed only.
                        Vector3<SimdFloat> const physicalVelocityDifference =
                            velocityDifference + SimdFloat(parameters.expansionRate) * separation;
                        SimdFloat const neighborSoundSpeed =
                            simdSqrt(SimdFloat(gasAdiabaticIndex) * SimdFloat(pressureEnergyRatio) *
                                simdSelect(contributes,
                                    detail::sphCpuLoad(
                                        data.sourceInternalEnergy, neighbors, first, 1.0f),
                                    SimdFloat(1)));
                        SimdFloat const radialVelocity =
                            physicalVelocityDifference.dot(separation) /
                            simdSelect(contributes, radius, SimdFloat(1));
                        signalSpeedPack = simdMax(signalSpeedPack,
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
        float const density = simdSum(densityPack);
        flops += SimdFloat::size() - 1;
        ALWAYS_ASSERT(sinkKernelSupportRadius > 0.0f && density > 0.0f);
        data.sinkDensity[sinkIndex] = density;
        float const kernelSum = simdSum(kernelSumPack);
        float const kernelDerivativeMoment = simdSum(kernelDerivativeMomentPack);
        auto const gradH = sphGradHEvaluation(sinkKernelSupportRadius,
            parameters.minimumSupportRadius, parameters.maximumSupportRadius,
            data.targetNeighborCount, data.gasMass, kernelSum, kernelDerivativeMoment);
        /*
         * Reductions, residual (2), and Ω, ζ, the weighted sums (6).
         */
        flops += 2 * (SimdFloat::size() - 1) + 8;
        detail::recordSphCpuGradH(
            gradH, sinkKernelSupportRadius, density, kernelSum, kernelDerivativeMoment, result);
        data.sinkGradHCorrection[sinkIndex] = gradH.correction.weightedSupportCorrection;
        if (data.sinkLogDensityRate) {
            // clang-format off
            /*
             * Price (2012), equation (27): d rho/d tau = sum m v_ab.grad W /
             * (a Omega). Held supports use Omega=1; no Hubble velocity belongs
             * in the derivative of the stored comoving density.
             *
             * Symbol directory (in the code below):
             *   v_ab → velocityDifference
             */
            // clang-format on
            data.sinkLogDensityRate[sinkIndex] = simdSum(densityDerivativeSumPack) /
                (parameters.scaleFactor * density * gradH.correction.supportJacobian);
            flops += SimdFloat::size() - 1 + 3;
        }
        if (computeSphBalsara && data.sinkBalsaraFactor) {
            Vector3<float> const densityWeightedCurl(simdSum(densityWeightedCurlPack[kXDimension]),
                simdSum(densityWeightedCurlPack[kYDimension]),
                simdSum(densityWeightedCurlPack[kZDimension]));
            data.sinkBalsaraFactor[sinkIndex] = balsara::encodeFactor(balsara::balsaraFactor(
                simdSum(densityWeightedDivergencePack), densityWeightedCurl.norm()));
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
                simdSum(densityWeightedDivergencePack) / (density * parameters.scaleFactor);
            float const soundSpeed = std::sqrt(
                gasAdiabaticIndex * pressureEnergyRatio * data.sourceInternalEnergy[particleIndex]);
            float const smoothingLength = sinkKernelSupportRadius /
                float(selectedHydroKernel().supportRadiusPerEffectiveRadius) *
                float(monaghan::kViscosityWidthScale);
            flops.doublePrecision += 1;    // Runtime kernel-table coefficient narrowing.
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
        if (computeGradientHistory) {
            cullen_and_dehnen::VelocityGradientSums<float> sums{};
            for (int i = 0; i < kDimensions; ++i) {
                for (int j = 0; j < kDimensions; ++j) {
                    sums.moment[i][j] = simdSum(gradientSums.moment[i][j]);
                    sums.velocity[i][j] = simdSum(gradientSums.velocity[i][j]);
                }
            }
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
            // Reconstruct V once for the viscosity model and optional Balsara factor.
            float velocityGradient[kDimensions][kDimensions];
            int const rank =
                cullen_and_dehnen::reconstructVelocityGradient(sums, velocityGradient, flops);
            float const canonicalDivergence =
                velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2];
            flops += 2;
            float const divergenceDerivative = rank == 0 ?
                0.0f :
                cullen_and_dehnen::divergenceTimeDerivative(canonicalDivergence,
                    parameters.scaleFactor, state, sampleInterval, previousSampleTimestep, flops);
            if (parameters.balsaraSwitch && data.sinkBalsaraFactor) {
                data.sinkBalsaraFactor[sinkIndex] = balsara::encodeFactor(
                    balsara::factorFromVelocityGradient(velocityGradient, flops));
            }
            cullen_and_dehnen::Result<float> diagnostics;
            if (parameters.cullenDehnen) {
                diagnostics = cullen_and_dehnen::diagnosticsFromGradient(velocityGradient,
                    parameters.scaleFactor, divergenceDerivative,
                    simdSum(densityWeightedSignAveragePack) / density, sinkKernelSupportRadius,
                    std::max(soundSpeedForSwitch, simdMaximum(signalSpeedPack)),
                    parameters.maximumBulkViscosityCoefficient, parameters.viscosityDecayCD, flops);
                flops += SimdFloat::size() - 1;
            } else if (parameters.huEtAl) {
                diagnostics =
                    hu_etal::diagnosticsFromGradient(velocityGradient, parameters.scaleFactor,
                        divergenceDerivative, sinkKernelSupportRadius, soundSpeedForSwitch,
                        std::max(soundSpeedForSwitch, simdMaximum(signalSpeedPack)),
                        parameters.maximumBulkViscosityCoefficientHu, parameters.viscosityDecayHu,
                        flops);
            } else {
                /*
                 * Price et al. (2018), equations (46)-(48): the paper's smoothing length is
                 * the same calibrated viscosity length as Morris & Monaghan's h.
                 */
                float const smoothingLength = sinkKernelSupportRadius /
                    float(selectedHydroKernel().supportRadiusPerEffectiveRadius) *
                    float(monaghan::kViscosityWidthScale);
                flops.doublePrecision += 1;
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
        result.minimumSupportRadius =
            std::min(result.minimumSupportRadius, sinkKernelSupportRadius);
        result.maximumSupportRadius =
            std::max(result.maximumSupportRadius, sinkKernelSupportRadius);
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}
