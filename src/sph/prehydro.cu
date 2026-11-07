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
    SpatialVector<float> v;
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

// H_a → sink kernel smoothing length, solved in place.
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
                float& H_a = data.sinkSmoothingLength[sinkIndex];
                bool const converged = solveSphSmoothingLength<kSmoothingLengthsBlockSize>(H_a,
                    parameters.minimumSmoothingLength, parameters.maximumSmoothingLength,
                    data.targetNeighborCount, workspace.positions,
                    data.sourcePosition(particleIndex), data.sourcePaddedPosition(particleIndex),
                    targetNode.outerBox, flops);
                if (threadIndex == 0) {
                    data.sinkFlags[sinkIndex].setConverged(converged);
                    if (converged) {
                        atomicMin(&reduction->minimumSmoothingLength, H_a);
                        atomicMax(&reduction->maximumSmoothingLength, H_a);
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
 * doi:10.1146/annurev.aa.30.090192.002551, with the divergence and curl of GADGET-4
 * equation (72), cited in cosmictiger/sph/viscosity/balsara.hpp.
 *   ρ_a = Σ_b m_b W_ab(H_a), W_ab = H_a⁻³ w(q_ab), q_ab = |r_ab|/H_a,
 *   ρ_a ∇·v_a = −Σ_b m_b v_ab · ∇_a W_ab,
 *   ρ_a ∇×v_a = Σ_b m_b v_ab × ∇_a W_ab.
 * Symbol directory (Pack denotes SIMD partial sums):
 *   H_a, H_a⁻¹ → sink kernel smoothing length, its inverse; q_ab → normalized radius
 *   w(q_ab) → kernel weight; m_b/H_aᴰ → mass over support volume
 *   ρ_a → density; ρ_a ∇·v_a → density-weighted divergence
 *   ρ_a ∇×v_a → density-weighted curl
 *   γ, γ − 1 → gas adiabatic index, P/(ρu) of all gas
 *   x_a → sink position; paddedX_b → neighbor position as the record of the box tests
 *   r_ab, |r_ab|, r_ab² → separation, distance, squared separation
 *   H_b → neighbor kernel smoothing length
 *   v_ab → velocity difference (peculiar stored velocity)
 *   m_b F_a → mass-weighted kernel gradient factor; m_b ∇_a W_ab → mass-weighted kernel gradient
 *   Σ_b w → kernel sum; Σ_b q_ab w′ → kernel derivative moment
 *   H_a,stored → stored smoothing length, updated by the solver
 *   Code names: H_inv_a = H_a⁻¹; H_a_stored = H_a,stored; m_b_over_Hᴰ_a = m_b/H_aᴰ;
 *     γ_minus_1 = γ − 1; abs_r_ab = |r_ab|; r2_ab = r_ab²; w_q_ab = w(q_ab); ρ_a_div_v_a = ρ_a ∇·v_a; ρ_a_curl_v_a = ρ_a ∇×v_a;
 *     m_b_F_a = m_b F_a; m_b_grad_a_W_ab = m_b ∇_a W_ab; ΣW = Σ_b w; Σqwʹ = Σ_b q_ab w′.
 * Peculiar velocities enter the divergence and the curl.
 * The curl stays a three-component record: a two-dimensional curl points along the inactive axis.
 */
// clang-format on
__global__ void sphCudaDensityAndGradients(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    bool const evolveMorrisMonaghan =
        parameters.variableViscosity && parameters.viscosityTimestep > 0.0;
    bool const loadVelocity = parameters.balsaraSwitch || evolveMorrisMonaghan;
    sph_flops::Counter flops = 1;    // γ − 1.
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every particle.
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
                    workspace.particles[offset + slot] = {
                        data.sourceSmoothingLength[sourceParticleIndex],
                        data.sourceRungs[sourceParticleIndex],
                        loadVelocity ? data.sourceVelocity(sourceParticleIndex) :
                                       SpatialVector<float>(0.0f)};
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
            SpatialVector<Position> const x_a = data.sourcePosition(particleIndex);
            float& H_a_stored = data.sinkSmoothingLength[sinkIndex];
            int semiactive = active;
            if (!active) {
                for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                    neighborIndex += blockSize) {
                    auto const& neighbor = workspace.positions[neighborIndex];
                    auto const& particle = workspace.particles[neighborIndex];
                    if (parameters.isActiveRung(particle.rung)) {
                        auto const r_ab = distance(x_a, neighbor.x);
                        float const r2_ab = r_ab.dot(r_ab);
                        float const H_b = particle.smoothingLength;
                        // Fixed distances, squared norm, two supports.
                        flops += sph_flops::kVectorOperation + sph_flops::kDotProduct + 2;
                        if (r2_ab < fmaxf(sqr(H_a_stored), sqr(H_b))) {
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
                converged = solveSphSmoothingLength<kDensityAndGradientsBlockSize>(H_a_stored,
                    parameters.minimumSmoothingLength, parameters.maximumSmoothingLength,
                    data.targetNeighborCount, workspace.positions, x_a,
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
             * Equation 3.8 of Monaghan (1992) (the interpolant 2.7 at r_a): ρ_a = Σ_b m_b W_ab,
             * with the sink's own support radius in W(r, H_a) = w(r/H_a)/H_a³, a gather form of
             * equation 6.3 (which uses h_b). Every gas particle has mass m_b = gasMass, and the
             * sum includes b = a. Each thread accumulates part of the sum.
             *
             * Symbol directory (in the code below):
             *   H_a → sink kernel smoothing length
             *   m_b → gas mass
             *   ρ_a → density
             */
            // clang-format on

            float const H_a = H_a_stored;
            float const H_inv_a = 1.0f / H_a;
            float const m_b_over_Hᴰ_a = data.gasMass * sph_geometry::inverseVolume(H_inv_a);
            flops += 2 + sph_flops::kInverseVolume;    // Reciprocal and mass/support products.
            float ρ_a = 0.0f;
            float ρ_a_div_v_a = 0.0f;
            Vector3<float> ρ_a_curl_v_a(0.0f);    // padded: see the curl above
            // clang-format off
            /*
             * Variational operator: ΣW = Σ_b w(q_ab) and Σ_b q_ab w'(q_ab) give Ω_a and ζ_a.
             *
             * Symbol directory (in the code below):
             *   q_ab → normalized radius
             *   ζ_a → weighted support correction
             *   ΣW → kernel sum
             */
            // clang-format on
            float ΣW = 0.0f;
            float Σqwʹ = 0.0f;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.positions.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.positions[neighborIndex];
                auto const r_ab = distance(x_a, neighbor.x);
                float const abs_r_ab = r_ab.norm();
                float const q_ab = abs_r_ab * H_inv_a;    // |r_ab|/H_a
                float const w_q_ab = kernelW(q_ab);
                ρ_a += m_b_over_Hᴰ_a * w_q_ab;
                // Fixed distances, norm, q_ab, density product and sum.
                flops += sph_flops::kVectorOperation + sph_flops::kNorm + 3 +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab);
                if (q_ab < 1.0f) {
                    ΣW += w_q_ab;
                    Σqwʹ += q_ab * dkernelWDq(q_ab);
                    flops +=
                        3 + sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
                }
                if (computeSphDivergence && abs_r_ab > 0.0f && q_ab < 1.0f) {
                    // clang-format off
                    /*
                     * GADGET-4 equation (72) uses −Σ_b m_b v_ab·gradW_ab and Σ_b m_b v_ab×gradW_ab.
                     *
                     * Symbol directory (in the code below):
                     *   v_a, v_b → sink, neighbor stored velocity; v_ab → velocity difference;
                     *   m_b → gas mass; ρ_a → density.
                     */
                    // clang-format on
                    float const m_b_F_a = m_b_over_Hᴰ_a * H_inv_a * dkernelWDq(q_ab) / abs_r_ab;
                    auto const& particle = workspace.particles[neighborIndex];
                    SpatialVector<float> const v_a = data.sourceVelocity(particleIndex);
                    SpatialVector<float> const& v_b = particle.v;
                    SpatialVector<float> const v_ab = v_a - v_b;
                    // Kernel-gradient coefficient (3) and relative velocity.
                    flops += 3 + sph_flops::kVectorOperation +
                        sph_flops::densityKernelOperations(selectedHydroKernel(), q_ab, true);
                    SpatialVector<float> const m_b_grad_a_W_ab = m_b_F_a * r_ab;
                    flops += sph_flops::kVectorOperation;
                    ρ_a_div_v_a -= v_ab.dot(m_b_grad_a_W_ab);
                    flops += sph_flops::kDotProduct + 1;
                    if (parameters.balsaraSwitch) {
                        ρ_a_curl_v_a += sph_geometry::paddedVector(v_ab).cross(
                            sph_geometry::paddedVector(m_b_grad_a_W_ab));
                        flops += 12;
                    }
                }
            }
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ρ_a);
            flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ΣW);
            sharedReduceAdd<float, kDensityAndGradientsBlockSize>(Σqwʹ);
            flops += 2 * detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            if (computeSphDivergence) {
                sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ρ_a_div_v_a);
                flops += detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }
            if (parameters.balsaraSwitch) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    sharedReduceAdd<float, kDensityAndGradientsBlockSize>(ρ_a_curl_v_a[dim]);
                }
                flops += 3 * detail::sphReductionFlops<kDensityAndGradientsBlockSize>();
            }

            if (threadIndex == 0) {
                ALWAYS_ASSERT(H_a > 0.0f && ρ_a > 0.0f);
                data.sinkDensity[sinkIndex] = ρ_a;

                auto const gradH = sphGradHEvaluation(H_a, parameters.minimumSmoothingLength,
                    parameters.maximumSmoothingLength, data.targetNeighborCount, data.gasMass, ΣW,
                    Σqwʹ);
                flops += 8;    // Residual (2), and Ω, ζ, the weighted sums (6).
                if (gradH.constraint != sph_volume::SupportConstraint::Free) {
                    atomicAdd(&reduction->heldSupports, 1);
                } else {
                    // clang-format off
                    /*
                     * A held support uses ζ_a = 0. An unusable free correction stops the
                     * run; a small positive Ω_a is counted.
                     *
                     * Symbol directory (in the code below):
                     *   ζ_a → weighted support correction
                     */
                    // clang-format on
                    if (!sph_volume::isUsable(gradH.correction)) {
                        printf("SPH grad-H correction is unusable: smoothingLength=%e "
                               "density=%e ΣW=%e Σq dw/dq=%e Ω=%e ζ=%e residual=%e\n",
                            H_a, ρ_a, ΣW, Σqwʹ, gradH.correction.supportJacobian,
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
                    data.sinkBalsaraFactor[sinkIndex] = balsara::encodeFactor(
                        balsara::balsaraFactor(ρ_a_div_v_a, ρ_a_curl_v_a.norm()));
                    flops += 8;
                } else if (!parameters.balsaraSwitch && data.sinkBalsaraFactor) {
                    data.sinkBalsaraFactor[sinkIndex] = 255;
                }
                if (updateMorrisMonaghan) {
                    /*
                     * Morris & Monaghan (1997), equations (5), (6), (13), in conformal time; the
                     * update is in cosmictiger/sph/viscosity/morris_and_monaghan.hpp. The sum/rho
                     * is a² div_phys(v); conformal-time rates divide by a. τ_a⁻¹ = C₁ c_a/h_a,
                     * ∇·v_a = ρ_a ∇·v_a/(a ρ_a). Symbol directory: ∇·v_a → divergence; c_a → sound
                     * speed; h_a → viscosity smoothing length τ_a⁻¹ → inverse decay time; C₁ →
                     * viscosity decay parameter; Δt → time step α_a → stored bulk viscosity
                     * coefficient. Code names: div_v_a = ∇·v_a; τ_inv_a = τ_a⁻¹.
                     */
                    float const div_v_a = ρ_a_div_v_a / (ρ_a * parameters.scaleFactor);
                    float const c_a =
                        sqrtf(γ * γ_minus_1 * data.sourceInternalEnergy[particleIndex]);
                    float const h_a = H_a /
                        selectedHydroKernel().smoothingLengthPerEffectiveRadius *
                        float(monaghan::kViscosityWidthScale);
                    float const τ_inv_a =
                        parameters.viscosityDecay * c_a / (parameters.scaleFactor * h_a);
                    int const rung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
                    bool const actualInterval = parameters.globalTimestep;
                    float const Δt = actualInterval ? parameters.viscosityTimestep :
                                                      parameters.viscosityTimestep / (1 << rung);
                    auto& α_a = data.sinkBulkViscosityCoefficient[sinkIndex];
                    α_a = morris_and_monaghan::evolvedBulkViscosityCoefficient(float(α_a),
                        parameters.minimumBulkViscosityCoefficient, τ_inv_a, div_v_a, Δt);

                    flops += 19;
                    flops.doublePrecision += actualInterval ? 1 : 2;
                }
                data.sinkFlags[sinkIndex].setConverged(true);
                atomicMin(&reduction->minimumSmoothingLength, H_a_stored);
                atomicMax(&reduction->maximumSmoothingLength, H_a_stored);
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
