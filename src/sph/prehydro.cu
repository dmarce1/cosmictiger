/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/cuda.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/viscosity/morris_and_monaghan.hpp>

using sph_geometry::SpatialVector;

namespace detail {

struct SmoothingWorkspace {
    SharedNeighborBuffer<SphNeighborPosition, kSphNeighborSharedBytes / sizeof(SphNeighborPosition)>
        positions;
};

struct NeighborActivity {
    float smoothingLength;
    char rung;
    SpatialVector<float> velocity;
};

struct DensityWorkspace {
    // The RTX 3090 budget sweep favored 12 KiB over 8 KiB for density.
    static constexpr int kSharedBytes = 12 * 1024;
    static constexpr int kSharedCapacity =
        kSharedBytes / (sizeof(SphNeighborPosition) + sizeof(NeighborActivity));
    SharedNeighborBuffer<SphNeighborPosition, kSharedCapacity> positions;
    SharedNeighborBuffer<NeighborActivity, kSharedCapacity> particles;
};

}    // namespace detail

// h𞁑 → sink kernel smoothing length, solved in place.
__global__ void sphCudaSmoothingLengths(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    sph_flops::Counter flops = 0;
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::SmoothingWorkspace workspace;
    new (&workspace) detail::SmoothingWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    while (targetNodeIndex < data.numberOfTargetNodes) {
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        workspace.positions.resize(0);
        for (int neighborNodeIndex = targetNode.neighborRange.first;
            neighborNodeIndex < targetNode.neighborRange.second; neighborNodeIndex++) {
            auto const& neighborNode = data.treeNodes[data.neighborNodeIndices[neighborNodeIndex]];
            int const end = neighborNode.particleRange.first +
                roundUp(neighborNode.particleRange.second - neighborNode.particleRange.first,
                    ParticleInteger(blockSize));
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    contains = targetNode.outerBox.contains(
                        data.sourcePaddedPosition(sourceParticleIndex), &flops.doublePrecision);
                }
                int slot = contains;
                int total;
                computeIndices<kSmoothingLengthsBlockSize>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {data.sourcePosition(sourceParticleIndex)};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            bool const active = parameters.isActiveRung(
                data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
            if (active && !data.sinkFlags[sinkIndex].converged()) {
                float& h𞁑 = data.sinkSmoothingLength[sinkIndex];
                bool const converged = solveSphSmoothingLength<kSmoothingLengthsBlockSize>(h𞁑,
                    parameters.minimumSmoothingLength, parameters.maximumSmoothingLength,
                    data.targetNeighborCount, workspace.positions,
                    data.sourcePosition(particleIndex), data.sourcePaddedPosition(particleIndex),
                    targetNode.outerBox, flops);
                if (threadIndex == 0) {
                    data.sinkFlags[sinkIndex].setConverged(converged);
                    if (converged) {
                        atomicMin(&reduction->minimumSmoothingLength, h𞁑);
                        atomicMax(&reduction->maximumSmoothingLength, h𞁑);
                    } else {
                        atomicAdd(&reduction->pendingChanges, 1);
                    }
                }
            }
        }
        __syncthreads();
        if (threadIndex == 0) {
            targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
        }
        __syncthreads();
    }
    detail::recordSphGpuFlops(flops, reduction);
    (&workspace)->~SmoothingWorkspace();
}

// clang-format off
/*
 * Density and velocity-derivative sums: Monaghan (1992), ARA&A 30, 543, equation (3.8),
 * doi:10.1146/annurev.aa.30.090192.002551, with the divergence and curl of GADGET-4,
 * Springel et al. (2021), equation (72), cited in cosmictiger/sph/viscosity/balsara.hpp.
 *   ρ𞁑 = Σ𞁓 m𞁓 W𞁑𞁓(h𞁑), W𞁑𞁓 = h𞁑⁻³ w(q𞁑𞁓), q𞁑𞁓 = |r𞁑𞁓|/h𞁑,
 *   ρ𞁑 ∇·v𞁑 = −Σ𞁓 m𞁓 v𞁑𞁓 · ∇𞁑 W𞁑𞁓,
 *   ρ𞁑 ∇×v𞁑 = Σ𞁓 m𞁓 v𞁑𞁓 × ∇𞁑 W𞁑𞁓.
 * Symbol directory (Pack denotes SIMD partial sums):
 *   h𞁑, h𞁑⁻¹ → sink kernel smoothing length, its inverse; q𞁑𞁓 → normalized radius
 *   w(q𞁑𞁓) → kernel weight; m𞁓/h𞁑ᴰ → mass over support volume
 *   ρ𞁑 → density; ρ𞁑 ∇·v𞁑 → density-weighted divergence
 *   ρ𞁑 ∇×v𞁑 → density-weighted curl
 *   γ, γ − 1 → gas adiabatic index, P/(ρu) of all gas
 *   x𞁑 → sink position; paddedX_b → neighbor position as the record of the box tests
 *   r𞁑𞁓, |r𞁑𞁓|, r𞁑𞁓² → separation, distance, squared separation
 *   h𞁓 → neighbor kernel smoothing length
 *   v𞁑𞁓 → velocity difference (peculiar stored velocity)
 *   m𞁓 h𞁑⁻⁽ᴰ⁺¹⁾ → mass times the sink inverse gradient scale, m𞁓 F𞁑 = m𞁓 h𞁑⁻⁽ᴰ⁺¹⁾ w′(q𞁑𞁓)/|r𞁑𞁓|
 *   m𞁓 F𞁑 → mass-weighted kernel gradient factor; m𞁓 ∇𞁑 W𞁑𞁓 → mass-weighted kernel gradient
 *   h_M/h → Monaghan smoothing length per support radius (monaghan::lengthPerSupport)
 *   Σ𞁓 w → kernel sum; Σ𞁓 q𞁑𞁓 w′ → kernel derivative moment
 *   h𞁑,stored → stored smoothing length, updated by the solver
 *   Code names: ih𞁑 = h𞁑⁻¹; h𞁑ₘₑₘ = h𞁑,stored; m𞁓_over_hᴰ𞁑 = m𞁓/h𞁑ᴰ
 *     γ_minus_1 = γ − 1; abs_r𞁑𞁓 = |r𞁑𞁓|; r2𞁑𞁓 = r𞁑𞁓²; w_q𞁑𞁓 = w(q𞁑𞁓); ρ𞁑_div_v𞁑 = ρ𞁑 ∇·v𞁑; ρ𞁑_curl_v𞁑 = ρ𞁑 ∇×v𞁑
 *     m𞁓F𞁑 = m𞁓 F𞁑; m𞁓grad𞁑W𞁑𞁓 = m𞁓 ∇𞁑 W𞁑𞁓; ΣW = Σ𞁓 w; Σqwʹ = Σ𞁓 q𞁑𞁓 w′
 *     m𞁓ih_D_plus_1𞁑 = m𞁓 h𞁑⁻⁽ᴰ⁺¹⁾; h_M_over_h = h_M/h.
 * Peculiar velocities enter the divergence and the curl.
 * The curl stays a three-component record: a two-dimensional curl points along the inactive axis.
 */
// clang-format on
__global__ void sphCudaDensityAndGradients(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    bool const evolveMorrisMonaghan =
        parameters.variableViscosity && parameters.viscosityTimestep > 0.0;
    bool const loadVelocity = parameters.balsaraSwitch || evolveMorrisMonaghan;
    sph_flops::Counter flops = 2;    // γ − 1 and h_M/h.
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every particle.
    float const h_M_over_h =
        monaghan::lengthPerSupport(selectedHydroKernel().smoothingLengthPerEffectiveRadius);
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::DensityWorkspace workspace;
    new (&workspace) detail::DensityWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    while (targetNodeIndex < data.numberOfTargetNodes) {
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        workspace.positions.resize(0);
        workspace.particles.resize(0);
        for (int neighborNodeIndex = targetNode.neighborRange.first;
            neighborNodeIndex < targetNode.neighborRange.second; neighborNodeIndex++) {
            auto const& neighborNode = data.treeNodes[data.neighborNodeIndices[neighborNodeIndex]];
            int const end = neighborNode.particleRange.first +
                roundUp(neighborNode.particleRange.second - neighborNode.particleRange.first,
                    ParticleInteger(blockSize));
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    Vector3<Position> const paddedX_b =
                        data.sourcePaddedPosition(sourceParticleIndex);
                    contains = targetNode.outerBox.contains(paddedX_b, &flops.doublePrecision);
                    if (!contains) {
                        contains = sphSupportIntersects(targetNode.innerBox, paddedX_b,
                            data.sourceSmoothingLength[sourceParticleIndex], &flops);
                    }
                }
                int slot = contains;
                int total;
                computeIndices<kDensityAndGradientsBlockSize>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                workspace.particles.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {data.sourcePosition(sourceParticleIndex)};
                    auto& particle = workspace.particles[offset + slot];
                    particle.smoothingLength = data.sourceSmoothingLength[sourceParticleIndex];
                    particle.rung = data.sourceRungs[sourceParticleIndex];
                    particle.velocity = loadVelocity ? data.sourceVelocity(sourceParticleIndex) :
                                                       SpatialVector<float>(0.0f);
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            bool const active = parameters.isActiveRung(
                data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
            SpatialVector<Position> const x𞁑 = data.sourcePosition(particleIndex);
            float& h𞁑ₘₑₘ = data.sinkSmoothingLength[sinkIndex];
            int semiactive = active;
            if (!active) {
                for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                    neighborIndex += blockSize) {
                    auto const& neighbor = workspace.positions[neighborIndex];
                    auto const& particle = workspace.particles[neighborIndex];
                    if (parameters.isActiveRung(particle.rung)) {
                        auto const r𞁑𞁓 = distance(x𞁑, neighbor.position);
                        float const r2𞁑𞁓 = r𞁑𞁓.dot(r𞁑𞁓);
                        float const h𞁓 = particle.smoothingLength;
                        // Fixed distances, squared norm, two supports.
                        flops += sph_flops::kVectorOperation + sph_flops::kDotProduct + 2;
                        if (r2𞁑𞁓 < fmaxf(sqr(h𞁑ₘₑₘ), sqr(h𞁓))) {
                            semiactive++;
                        }
                    }
                }
                sharedReduceAdd<int, kDensityAndGradientsBlockSize>(semiactive);
            }
            if (threadIndex == 0) {
                data.sinkFlags[sinkIndex].setSemiactive(semiactive != 0);
            }
            // Semiactive and convergence share a byte; finish the masked write
            // before any thread reads convergence from that byte.
            __syncthreads();
            if (!semiactive) {
                continue;
            }
            bool const updateMorrisMonaghan =
                evolveMorrisMonaghan && active && !data.sinkFlags[sinkIndex].converged();
            bool const computeSphDivergence = parameters.balsaraSwitch || updateMorrisMonaghan;
            bool converged = true;
            if (!active && !data.sinkFlags[sinkIndex].converged()) {
                converged = solveSphSmoothingLength<kDensityAndGradientsBlockSize>(h𞁑ₘₑₘ,
                    parameters.minimumSmoothingLength, parameters.maximumSmoothingLength,
                    data.targetNeighborCount, workspace.positions, x𞁑,
                    data.sourcePaddedPosition(particleIndex), targetNode.outerBox, flops);
            }
            if (!converged) {
                if (threadIndex == 0) {
                    data.sinkFlags[sinkIndex].setConverged(false);
                    atomicAdd(&reduction->pendingChanges, 1);
                }
                continue;
            }
            // clang-format off
            /*
             * Equation 3.8 of Monaghan (1992) (the interpolant 2.7 at r𞁑): ρ𞁑 = Σ𞁓 m𞁓 W𞁑𞁓,
             * with the sink's own support radius in W(r, h𞁑) = w(r/h𞁑)/h𞁑³, a gather form of
             * equation 6.3 (which uses h_M𞁓). Every gas particle has mass m𞁓 = gasMass, and the
             * sum includes b = a. Each thread accumulates part of the sum.
             *
             * Symbol directory (in the code below):
             *   h𞁑 → sink kernel smoothing length
             *   m𞁓 → gas mass
             *   ρ𞁑 → density
             */
            // clang-format on

            float const h𞁑 = h𞁑ₘₑₘ;
            float const ih𞁑 = 1.0f / h𞁑;
            float const m𞁓_over_hᴰ𞁑 = data.gasMass * sph_geometry::inverseVolume(ih𞁑);
            float const m𞁓ih_D_plus_1𞁑 = data.gasMass * sph_geometry::inverseGradientScale(ih𞁑);
            // Reciprocal and the mass/support-volume and mass/gradient-scale products.
            flops += 3 + sph_flops::kInverseVolume + sph_flops::kInverseGradientScale;
            float ρ𞁑 = 0.0f;
            float ρ𞁑_div_v𞁑 = 0.0f;
            Vector3<float> ρ𞁑_curl_v𞁑(0.0f);    // padded: see the curl above
            // clang-format off
            /*
             * Variational operator: ΣW = Σ𞁓 w(q𞁑𞁓) and Σ𞁓 q𞁑𞁓 w'(q𞁑𞁓) give Ω𞁑 and ζ𞁑.
             *
             * Symbol directory (in the code below):
             *   q𞁑𞁓 → normalized radius
             *   ζ𞁑 → weighted support correction
             *   ΣW → kernel sum
             */
            // clang-format on
            float ΣW = 0.0f;
            float Σqwʹ = 0.0f;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.positions[neighborIndex];
                auto const r𞁑𞁓 = distance(x𞁑, neighbor.position);
                float const abs_r𞁑𞁓 = r𞁑𞁓.norm();
                float const q𞁑𞁓 = abs_r𞁑𞁓 * ih𞁑;    // |r𞁑𞁓|/h𞁑
                float const w_q𞁑𞁓 = kernelW(q𞁑𞁓);
                ρ𞁑 += m𞁓_over_hᴰ𞁑 * w_q𞁑𞁓;
                // Fixed distances, norm, q𞁑𞁓, density product and sum.
                flops += sph_flops::kVectorOperation + sph_flops::kNorm + 3 +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓);
                if (q𞁑𞁓 < 1.0f) {
                    ΣW += w_q𞁑𞁓;
                    Σqwʹ += q𞁑𞁓 * dkernelWDq(q𞁑𞁓);
                    flops +=
                        3 + sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
                }
                if (computeSphDivergence && abs_r𞁑𞁓 > 0.0f && q𞁑𞁓 < 1.0f) {
                    // clang-format off
                    /*
                     * GADGET-4, Springel et al. (2021), equation (72) uses −Σ𞁓 m𞁓 v𞁑𞁓·gradW_ab
                     * and Σ𞁓 m𞁓 v𞁑𞁓×gradW_ab.
                     *
                     * Symbol directory (in the code below):
                     *   v𞁑, v𞁓 → sink, neighbor stored velocity; v𞁑𞁓 → velocity difference;
                     *   m𞁓 → gas mass; ρ𞁑 → density.
                     */
                    // clang-format on
                    float const m𞁓F𞁑 = m𞁓ih_D_plus_1𞁑 * dkernelWDq(q𞁑𞁓) / abs_r𞁑𞁓;
                    auto const& particle = workspace.particles[neighborIndex];
                    SpatialVector<float> const v𞁑 = data.sourceVelocity(particleIndex);
                    SpatialVector<float> const& v𞁓 = particle.velocity;
                    SpatialVector<float> const v𞁑𞁓 = v𞁑 - v𞁓;
                    // Kernel-gradient coefficient (2) and relative velocity.
                    flops += 2 + sph_flops::kVectorOperation +
                        sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑𞁓, true);
                    SpatialVector<float> const m𞁓grad𞁑W𞁑𞁓 = m𞁓F𞁑 * r𞁑𞁓;
                    flops += sph_flops::kVectorOperation;
                    ρ𞁑_div_v𞁑 -= v𞁑𞁓.dot(m𞁓grad𞁑W𞁑𞁓);
                    flops += sph_flops::kDotProduct + 1;
                    if (parameters.balsaraSwitch) {
                        ρ𞁑_curl_v𞁑 += sph_geometry::paddedVector(v𞁑𞁓).cross(
                            sph_geometry::paddedVector(m𞁓grad𞁑W𞁑𞁓));
                        flops += 12;
                    }
                }
            }
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ρ𞁑);
            flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ΣW);
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(Σqwʹ);
            flops += 2 * detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            if (computeSphDivergence) {
                sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ρ𞁑_div_v𞁑);
                flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }
            if (parameters.balsaraSwitch) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ρ𞁑_curl_v𞁑[dim]);
                }
                flops += 3 * detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }

            if (threadIndex == 0) {
                ALWAYS_ASSERT(h𞁑 > 0.0f && ρ𞁑 > 0.0f);
                data.sinkDensity[sinkIndex] = ρ𞁑;

                auto const gradH = sphGradHEvaluation(h𞁑, parameters.minimumSmoothingLength,
                    parameters.maximumSmoothingLength, data.targetNeighborCount, data.gasMass, ΣW,
                    Σqwʹ);
                flops += 8;    // Residual (2), and Ω, ζ, the weighted sums (6).
                if (gradH.constraint != sph_volume::SupportConstraint::Free) {
                    atomicAdd(&reduction->heldSupports, 1);
                } else {
                    // clang-format off
                    /*
                     * A held support uses ζ𞁑 = 0. An unusable free correction stops the
                     * run; a small positive Ω𞁑 is counted.
                     *
                     * Symbol directory (in the code below):
                     *   ζ𞁑 → weighted support correction
                     */
                    // clang-format on
                    if (!sph_volume::isUsable(gradH.correction)) {
                        printf("SPH grad-H correction is unusable: smoothingLength=%e "
                               "density=%e ΣW=%e Σq dw/dq=%e Ω=%e ζ=%e residual=%e\n",
                            h𞁑, ρ𞁑, ΣW, Σqwʹ, gradH.correction.supportJacobian,
                            gradH.correction.weightedSupportCorrection, gradH.relativeResidual);
                        __trap();
                    }
                    atomicMin(&reduction->minimumGradHOmega, gradH.correction.supportJacobian);
                    atomicMax(&reduction->maximumSupportResidual, fabsf(gradH.relativeResidual));
                    if (!sph_volume::isWellConditioned(gradH.correction)) {
                        atomicAdd(&reduction->illConditionedSupports, 1);
                    }
                }
                data.sinkGradHCorrection[sinkIndex] = gradH.correction.weightedSupportCorrection;

                if (parameters.balsaraSwitch && data.sinkBalsaraFactor) {
                    data.sinkBalsaraFactor[sinkIndex] =
                        balsara::encodeFactor(balsara::balsaraFactor(ρ𞁑_div_v𞁑, ρ𞁑_curl_v𞁑.norm()));
                    flops += 8;
                } else if (!parameters.balsaraSwitch && data.sinkBalsaraFactor) {
                    data.sinkBalsaraFactor[sinkIndex] = 255;
                }
                if (updateMorrisMonaghan) {
                    /*
                     * Morris & Monaghan (1997), equations (5), (6), (13), in conformal time; the
                     * update is in cosmictiger/sph/viscosity/morris_and_monaghan.hpp. The sum/rho
                     * is a² div_phys(v); conformal-time rates divide by a. τ𞁑⁻¹ = C₁ c𞁑/h_M𞁑,
                     * ∇·v𞁑 = ρ𞁑 ∇·v𞁑/(a ρ𞁑). Symbol directory: ∇·v𞁑 → divergence; c𞁑 → sound
                     * speed; h_M𞁑 → viscosity smoothing length; τ𞁑⁻¹ → inverse decay time; C₁ →
                     * viscosity decay parameter; Δt → time step; α𞁑 → stored bulk viscosity
                     * coefficient; h_M/h → Monaghan smoothing length per support radius.
                     * Code names: div_v_a = ∇·v𞁑; iτ𞁑 = τ𞁑⁻¹; h_M_over_h = h_M/h.
                     */
                    float const div_v_a = ρ𞁑_div_v𞁑 / (ρ𞁑 * parameters.scaleFactor);
                    float const c𞁑 =
                        sqrtf(γ * γ_minus_1 * data.sourceInternalEnergy[particleIndex]);
                    float const h_M𞁑 = h𞁑 * h_M_over_h;
                    float const iτ𞁑 =
                        parameters.viscosityDecay * c𞁑 / (parameters.scaleFactor * h_M𞁑);
                    int const rung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
                    bool const actualInterval = parameters.globalTimestep;
                    float const Δt = actualInterval ? parameters.viscosityTimestep :
                                                      parameters.viscosityTimestep / (1 << rung);
                    auto& α𞁑 = data.sinkBulkViscosityCoefficient[sinkIndex];
                    α𞁑 = morris_and_monaghan::evolvedBulkViscosityCoefficient(
                        float(α𞁑), parameters.minimumBulkViscosityCoefficient, iτ𞁑, div_v_a, Δt);

                    flops += 18;
                    flops.doublePrecision += actualInterval ? 1 : 2;
                }
                data.sinkFlags[sinkIndex].setConverged(true);
                atomicMin(&reduction->minimumSmoothingLength, h𞁑ₘₑₘ);
                atomicMax(&reduction->maximumSmoothingLength, h𞁑ₘₑₘ);
            }
        }
        __syncthreads();
        if (threadIndex == 0) {
            targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
        }
        __syncthreads();
    }
    detail::recordSphGpuFlops(flops, reduction);
    (&workspace)->~DensityWorkspace();
}
