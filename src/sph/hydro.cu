/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/conduction.hpp>
#include <cosmictiger/sph/cuda.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/variable_smoothing_lengths.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/volume.hpp>

using sph_geometry::SpatialVector;

namespace detail {

struct HydroNeighbor {
    SpatialVector<Position> x;
    float smoothingLength;
    SpatialVector<float> v;
    float internalEnergy;
    float density;
    float balsaraFactor;
    float bulkViscosityCoefficient;
    float gradHCorrection;    // ζ_b from the current density pass.
    int rung;
    int sourceIndex;
};

struct HydroWorkspace {
    SharedNeighborBuffer<HydroNeighbor, kSphNeighborSharedBytes / sizeof(HydroNeighbor)> neighbors;
};

}    // namespace detail

/*
 * Variational pressure force and compatible work use the equations documented in
 * cosmictiger/sph/variable_smoothing_lengths.hpp. Artificial viscosity uses Monaghan (1992),
 * equations 4.1-4.2 and the symmetric kernel of p. 554; its heating follows equation
 * 6.5 and its time step section 10.3, p. 563. The viscosity header gives the full
 * citation. Particle a is the sink and particle b a staged neighbor; the computation
 * uses the papers' symbols, described below. Positions and density are comoving,
 * and stored velocities and internal energies carry the factors a and a² of
 * docs/variational_sph.md.
 */

// clang-format off
/*
 * Production pair equations: Hopkins (2013), MNRAS 428, 2840, equations (12)–(13)
 * (https://arxiv.org/abs/1206.5006), and Price (2012), JCP 231, 759, (30), (35)
 * (https://arxiv.org/abs/1012.1885). Equal masses make the volume weight m_a and the
 * weighted density ρ_a:
 *   Q_ab/m_a = P_a(m_b − ζ_a)/ρ_a²,
 *   dv_a/dτ = −a⁻¹ Σ_b [(Q_ab/m_a)F_a + (Q_ba/m_a)F_b] r_ab,
 *   du_a/dτ = a⁻¹ Σ_b (Q_ab/m_a)F_a (v_ab · r_ab),
 *   ∇_a W_ab(H_a) = F_a r_ab, F_a = w′(q_a)/(H_a⁴ |r_ab|), q_a = |r_ab|/H_a.
 * These are derived comoving forms; the pair viscosity below adds to the same pair force.
 * Symbol directory:
 *   a, a⁻¹ → scale factor, its inverse; a ȧ → expansion rate, a da/dt = da/dτ
 *   m_b, m_b/a → gas mass, scaled gas mass; γ, γ − 1 → gas adiabatic index, P/(ρu) of all gas
 *   x_a → sink position; paddedX_b → neighbor position as the record of the box tests
 *   v_a, v_b → sink stored velocity, neighbor stored velocity
 *   ρ_a, P_a, u_a, c_a → density, pressure, internal energy, sound speed
 *   ρ_b, P_b, u_b, c_b → neighbor density, pressure, internal energy, sound speed
 *   H_a, H_b → sink kernel smoothing length, neighbor smoothing length
 *   H_a⁻¹, H_b⁻¹ → inverse smoothing lengths; H_a⁻⁽ᴰ⁺¹⁾ → sink inverse gradient scale
 *   r_ab, |r_ab|, r_ab² → separation, pair distance, squared separation
 *   v_ab → velocity difference; v_ab · r_ab → velocity dot separation
 *   (a v_phys,ab) · r_ab → velocity dot separation including the Hubble flow a ȧ r_ab
 *   q_a, q_b → radius normalized by the sink support, by the neighbor support
 *   ζ_a, ζ_b → sink grad-H correction, neighbor grad-H correction
 *   Q_ab/m_a, Q_ba/m_a → sink pressure coefficient, neighbor pressure coefficient
 *   F_a, F_b, F_ab → sink gradient factor, neighbor gradient factor, symmetric gradient factor
 *   dv_a/dτ, du_a/dτ → acceleration, reversible internal energy rate
 *   α_a, α_b, α_ab → bulk viscosity coefficient, neighbor bulk viscosity coefficient, pair bulk viscosity coefficient
 *   β_ab → pair von Neumann–Richtmyer coefficient; Π_ab → pair artificial viscosity
 *   μ_ab → viscosity velocity; ε → η²/h² of p. 551
 *   fᴬⱽ_ab → pair shear factor, the mean of the two Balsara factors
 *   f_P,ab, f_Π,ab → pressure force factor, viscous force factor of one pair
 *   f_ab → pair force, (f_P,ab + f_Π,ab) r_ab; a_ab → pair acceleration of a, −f_ab
 *   (du/dτ)_Π,ab → viscous heating of each endpoint by one pair
 *   C_ab/a → pair conduction exchange rate; (du_a/dτ)_cond,ab → conductive heating of a by one pair
 *   Σ_b C_ab/a → summed exchange rate, which sets the conduction timestep bound
 *   (du_a/dτ)_diss,ab, (du_b/dτ)_diss,ab → sink, neighbor dissipative heating by one pair
 *   (du_a/dτ)_diss → total dissipative heating, which becomes the entropy rate
 *   Ȧ_ab, Ȧ_ba → sink, neighbor entropy rate of one pair; Δτ → pair kick interval
 *   (dv_a/dτ)_grav, (dv_a/dτ)_ext → gravity acceleration, external acceleration
 *   5 − 3γ → homogeneous expansion coefficient of the stored energy
 *   Code names: a_inv = a⁻¹; aȧ = a ȧ; m_b_over_a = m_b/a; γ_minus_1 = γ − 1;
 *     H_inv_a = H_a⁻¹; H_inv_b = H_b⁻¹; H_inv_D_plus_1_a = H_a⁻⁽ᴰ⁺¹⁾; abs_r_ab = |r_ab|;
 *     r2_ab = r_ab²; v_ab_dot_r_ab = v_ab · r_ab; a_v_phys_ab_dot_r_ab = (a v_phys,ab) · r_ab;
 *     Q_ab_over_m_a = Q_ab/m_a; Q_ba_over_m_a = Q_ba/m_a; dv_a_dτ = dv_a/dτ; du_a_dτ = du_a/dτ;
 *     f_P_ab = f_P,ab; f_Π_ab = f_Π,ab; du_dτ_Π_ab = (du/dτ)_Π,ab; C_ab_over_a = C_ab/a;
 *     du_a_dτ_cond_ab = (du_a/dτ)_cond,ab; ΣC_ab_over_a = Σ_b C_ab/a;
 *     du_a_dτ_diss_ab = (du_a/dτ)_diss,ab; du_b_dτ_diss_ab = (du_b/dτ)_diss,ab;
 *     du_a_dτ_diss = (du_a/dτ)_diss; dv_a_dτ_grav = (dv_a/dτ)_grav;
 *     dv_a_dτ_ext = (dv_a/dτ)_ext; five_minus_3γ = 5 − 3γ.
 *   SIMD values append Pack; CUDA partial sums are reduced in place.
 */
// clang-format on
__global__ void sphCudaHydro(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    sph_flops::Counter flops = 3;    // Inverse scale factor, gas mass scaling and γ − 1.
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::HydroWorkspace workspace;
    new (&workspace) detail::HydroWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    float const a_inv = 1.0f / parameters.scaleFactor;
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every particle.
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4
    // defaults after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    float const smoothingLengthPerEffectiveRadius =
        selectedHydroKernel().smoothingLengthPerEffectiveRadius;
    float const m_b = data.gasMass;    // every gas particle has the same mass
    float const m_b_over_a = m_b * a_inv;
    float const ε = parameters.viscosityEpsilon;    // η²/h² of p. 551
    float const aȧ = parameters.expansionRate;      // a·da/dt, the Hubble flow of a pair
    bool const applyPairKicks = data.sourcePairUpdates && parameters.hydroKickTimestep > 0.0;
    float const Δτ = applyPairKicks ? float(parameters.hydroKickTimestep) : 0.0f;
    flops.doublePrecision += int(applyPairKicks);
    ALWAYS_ASSERT(!applyPairKicks || (parameters.midpointHydro && data.sourceIds));
    int maximumHydroRung = 0;
    int maximumGravityRung = 0;
    int maximumRung = 0;
    float maximumSignalSpeed = 0.0f;
    while (targetNodeIndex < data.numberOfTargetNodes) {
        __syncthreads();
        SphTreeNode const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
        workspace.neighbors.resize(0);
        // The union of both supports is required by the symmetric gradient.
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
                computeIndices<kHydroBlockSize>(slot, total);
                int const offset = workspace.neighbors.size();
                __syncthreads();
                workspace.neighbors.resize(offset + total);
                if (contains) {
                    workspace.neighbors[offset + slot] = {data.sourcePosition(sourceParticleIndex),
                        data.sourceSmoothingLength[sourceParticleIndex],
                        data.sourceVelocity(sourceParticleIndex),
                        data.sourceInternalEnergy[sourceParticleIndex],
                        data.sourceDensity[sourceParticleIndex],
                        parameters.balsaraSwitch ? data.sourceBalsaraFactor[sourceParticleIndex] :
                                                   1.0f,
                        parameters.variableViscosity ?
                            data.sourceBulkViscosityCoefficient[sourceParticleIndex] :
                            parameters.bulkViscosityCoefficient,
                        data.sourceGradHCorrection[sourceParticleIndex],
                        parameters.midpointHydro ? int(data.sourceRungs[sourceParticleIndex]) : 0,
                        sourceParticleIndex};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            ParticleInteger const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            int const previousRung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
            if (!parameters.isActiveRung(previousRung)) {
                continue;
            }
            // clang-format off
            /*
             * Particle a. Section 3.5 leaves the equation of state free; the ideal gas gives
             * P_a = (γ − 1) ρ_a u_a and c_a² = γ P_a/ρ_a with the single configured γ.
             *
             * Symbol directory (in the code below):
             *   H_a → support radius, 2h_a for the cubic spline
             *   P_a → pressure
             *   u_a → internal energy
             *   ρ_a → density
             *   c_a → sound speed
             *   γ → gas adiabatic index
             */
            // clang-format on
            float const H_a = data.sourceSmoothingLength[particleIndex];
            float const ρ_a = data.sourceDensity[particleIndex];
            float const u_a = data.sourceInternalEnergy[particleIndex];
            ALWAYS_ASSERT(H_a > 0.0f && ρ_a > 0.0f && u_a > 0.0f);
            float const P_a = γ_minus_1 * ρ_a * u_a;
            float const c_a = sqrtf(γ * γ_minus_1 * u_a);
            SpatialVector<Position> const x_a = data.sourcePosition(particleIndex);
            SpatialVector<float> const v_a = data.sourceVelocity(particleIndex);
            float const H_inv_a = 1.0f / H_a;
            float const H_inv_D_plus_1_a =
                sph_geometry::inverseGradientScale(H_inv_a);    // H_a^-(D+1)
            flops += 6 + sph_flops::kInverseGradientScale;
            // clang-format off
            /*
             * Variational operator (cosmictiger/sph/variable_smoothing_lengths.hpp): every
             * gas particle has the mass m, which is also its volume weight, so the weighted
             * density is ρ and Q_ab/m_a = (P_a/ρ_a²)(m − ζ_a). Comoving Q and kernel gradients
             * give the rates dv_a/dτ = −(1/a) Σ_b (Q_ab g_a + Q_ba g_b)/m_a and
             * du_a/dτ = (1/a) Σ_b Q_ab v_ab·g_a/m_a.
             *
             * Symbol directory (in the code below):
             *   v_ab → velocity difference
             *   P_a → pressure
             *   g_a → sink kernel gradient (F_a r_ab)
             *   g_b → neighbor kernel gradient (F_b r_ab)
             *   ζ_a → grad-H correction
             */
            // clang-format on
            float const ζ_a = data.sourceGradHCorrection[particleIndex];
            float const Q_ab_over_m_a = variational::directedCoefficient(
                variational::pressureVolumeFactor(P_a, 1.0f, ρ_a), m_b, ζ_a);
            flops += 5;
            SpatialVector<float> dv_a_dτ(0.0f);    // this thread's acceleration sum
            float du_a_dτ = 0.0f;
            float du_a_dτ_diss = 0.0f;    // this thread's viscous/conductive sum
            float ΣC_ab_over_a = 0.0f;
            float const α_a = parameters.variableViscosity ?
                data.sourceBulkViscosityCoefficient[particleIndex] :
                parameters.bulkViscosityCoefficient;
            float α = α_a;
            float max_b_μ_ab = 0.0f;    // max_b(−μ_ab) for δt_cv
            int maximumNeighborRung = 0;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.neighbors.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.neighbors[neighborIndex];
                // Particle b. Pairs outside both supports or at zero separation are skipped.
                auto const r_ab = distance(x_a, neighbor.x);
                float const r2_ab = r_ab.dot(r_ab);    // |r_ab|²
                float const H_b = neighbor.smoothingLength;
                // Distances, dot, support squares.
                flops +=
                    sph_flops::kVectorOperation + sph_flops::kDotProduct + (r2_ab == 0.0f ? 0 : 2);
                bool const supportsIntersect = r2_ab < fmaxf(H_a * H_a, H_b * H_b);
                if (parameters.midpointHydro && parameters.phase == 1 && supportsIntersect) {
                    maximumNeighborRung = max(maximumNeighborRung, neighbor.rung);
                }
                if (r2_ab == 0.0f || !supportsIntersect) {
                    continue;
                }
                float const ρ_b = neighbor.density;
                float const u_b = neighbor.internalEnergy;
                float const P_b = γ_minus_1 * ρ_b * u_b;
                float const c_b = sqrtf(γ * γ_minus_1 * u_b);
                float const abs_r_ab = sqrtf(r2_ab);    // |r_ab|
                // clang-format off
                /*
                 * Symmetric kernel of p. 554: ∇_a W̃_ab = ½[∇_a W(r_ab, H_a) + ∇_a
                 * W(r_ab, H_b)]. The selected kernel is W(r, H) = w(r/H)/H³, so
                 * each gradient equals w'(r/H) r_ab/(H⁴ r) and ∇_a W̃_ab = F_ab r_ab with the scalar
                 * F_ab.
                 *
                 * Symbol directory (in the code below):
                 *   F_ab → symmetric gradient factor
                 *   r_ab → separation
                 */
                // clang-format on
                float const H_inv_b = 1.0f / H_b;
                float const q_a = abs_r_ab * H_inv_a;
                float const q_b = abs_r_ab * H_inv_b;
                float const F_a = dkernelWDq(q_a) * H_inv_D_plus_1_a / abs_r_ab;
                float const F_b =
                    dkernelWDq(q_b) * sph_geometry::inverseGradientScale(H_inv_b) / abs_r_ab;
                float const F_ab = 0.5f * (F_a + F_b);
                // v_ab = v_a − v_b from the stored peculiar velocities. The physical relative
                // velocity adds the Hubble flow (a da/dt) r_ab and enters only the viscosity.
                SpatialVector<float> const& v_b = neighbor.v;
                SpatialVector<float> const v_ab = v_a - v_b;
                float const v_ab_dot_r_ab = v_ab.dot(r_ab);
                float const a_v_phys_ab_dot_r_ab = v_ab_dot_r_ab + aȧ * r2_ab;
                float const μ_ab = monaghan::viscosityMu(r2_ab, a_v_phys_ab_dot_r_ab, H_a, H_b, ε,
                    smoothingLengthPerEffectiveRadius, float(monaghan::kViscosityWidthScale));
                float const α_b =
                    parameters.variableViscosity ? neighbor.bulkViscosityCoefficient : α_a;
                float const α_ab = parameters.variableViscosity ? 0.5f * α_a + 0.5f * α_b : α_a;
                float const β_ab = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                    parameters.vonNeumannRichtmyerCoefficient * α_ab :
                    parameters.vonNeumannRichtmyerCoefficient;
                α = fmaxf(α, α_ab);
                flops += 3 * int(parameters.variableViscosity) +
                    int(parameters.vonNeumannRichtmyerCoefficientFollowsBulk());
                float fᴬⱽ_ab = 1.0f;
                if (parameters.balsaraSwitch) {
                    fᴬⱽ_ab =
                        0.5f * (data.sourceBalsaraFactor[particleIndex] + neighbor.balsaraFactor);
                    flops += 2;
                }
                // clang-format off
                /*
                 * Hopkins (2013) equation 12 with Price (2012) equation 35: separate
                 * gradients g_a = F_a r_ab and g_b = F_b r_ab, the directed Q_ab and Q_ba, and
                 * the same Q_ab in the reversible work, with the peculiar velocity; the
                 * analytic expansion term supplies the Hubble-flow work exactly. The
                 * viscous force and heating below share the same pair coefficients.
                 *
                 * Symbol directory (in the code below):
                 *   r_ab → separation
                 *   F_a → sink gradient factor
                 *   F_b → neighbor gradient factor
                 *   g_a → sink kernel gradient (F_a r_ab)
                 *   g_b → neighbor kernel gradient (F_b r_ab)
                 */
                // clang-format on
                float const ζ_b = neighbor.gradHCorrection;
                float const Q_ba_over_m_a = variational::directedCoefficient(
                    variational::pressureVolumeFactor(P_b, 1.0f, ρ_b), m_b, ζ_b);
                float const f_P_ab =
                    a_inv * variational::pairForceFactor(Q_ab_over_m_a, Q_ba_over_m_a, F_a, F_b);
                float Π_ab = monaghan::artificialViscosity(ρ_a, ρ_b, c_a, c_b, μ_ab, α_ab, β_ab);
                if (parameters.balsaraSwitch) {
                    Π_ab *= fᴬⱽ_ab;
                    flops += 1;
                }
                float const f_Π_ab = m_b_over_a * Π_ab * F_ab;
                float const du_dτ_Π_ab = m_b_over_a * 0.5f * Π_ab * F_ab * a_v_phys_ab_dot_r_ab;
                flops += sph_flops::kViscosityMu + sph_flops::kArtificialViscosity + 6;
                float du_a_dτ_diss_ab = du_dτ_Π_ab;
                float du_b_dτ_diss_ab = du_dτ_Π_ab;
                if (parameters.conductionCoefficient > 0.0f) {
                    float const C_ab_over_a = sph_conduction::exchangeRate(ρ_a, ρ_b, c_a, c_b, H_a,
                        H_b, r2_ab, a_v_phys_ab_dot_r_ab, F_ab, parameters.conductionCoefficient, ε,
                        smoothingLengthPerEffectiveRadius, m_b_over_a);
                    float const du_a_dτ_cond_ab = C_ab_over_a * (u_b - u_a);
                    du_a_dτ_diss_ab += du_a_dτ_cond_ab;
                    du_b_dτ_diss_ab -= du_a_dτ_cond_ab;
                    ΣC_ab_over_a += C_ab_over_a;
                    flops += 27;    // 22 exchange + 5 heating/row-sum operations.
                }
                SpatialVector<float> const f_ab = (f_P_ab + f_Π_ab) * r_ab;
                dv_a_dτ -= f_ab;
                if (applyPairKicks &&
                    parameters.isActiveRung(max(previousRung, int(neighbor.rung))) &&
                    sphOwnsPairKick(data.sourceIds[particleIndex],
                        data.sourceIds[neighbor.sourceIndex], previousRung, neighbor.rung)) {
                    SpatialVector<float> const a_ab = -f_ab;
                    float const Ȧ_ab = sph_entropy::fromStoredEnergy(
                        du_a_dτ_diss_ab, ρ_a, γ, parameters.scaleFactor);
                    float const Ȧ_ba = sph_entropy::fromStoredEnergy(
                        du_b_dτ_diss_ab, ρ_b, γ, parameters.scaleFactor);
                    flops += sphAccumulatePairKick(data.sourcePairUpdates[particleIndex],
                        data.sourcePairUpdates[neighbor.sourceIndex], a_ab, Ȧ_ab, Ȧ_ba, Δτ);
                    flops += 2 * sph_flops::entropyConversionFlops(γ);
                }
                du_a_dτ += a_inv * variational::pairWork(Q_ab_over_m_a, F_a, v_ab_dot_r_ab);
                du_a_dτ_diss += du_a_dτ_diss_ab;
                // Common geometry, pressure, and accumulation; the vector terms are v_ab,
                // v_ab · r_ab, the pair force and its sum.
                flops += 38 + sph_flops::kInverseGradientScale + 3 * sph_flops::kVectorOperation +
                    sph_flops::kDotProduct;
                max_b_μ_ab = fmaxf(max_b_μ_ab, -μ_ab);
                flops += sph_flops::densityKernelOperations(selectedHydroKernel(), q_a, true) +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q_b, true);
            }
            for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                sharedReduceAdd<float, kHydroBlockSize>(dv_a_dτ[dim]);
            }
            sharedReduceAdd<float, kHydroBlockSize>(du_a_dτ);
            sharedReduceAdd<float, kHydroBlockSize>(du_a_dτ_diss);
            if (parameters.conductionCoefficient > 0.0f) {
                sharedReduceAdd<float, kHydroBlockSize>(ΣC_ab_over_a);
                flops += detail::sphReductionFlops<kHydroBlockSize>();
            }
            sharedReduceMax<kHydroBlockSize>(max_b_μ_ab);
            sharedReduceMax<kHydroBlockSize>(α);
            if (parameters.midpointHydro && parameters.phase == 1) {
                sharedReduceMax<kHydroBlockSize>(maximumNeighborRung);
            }
            flops +=
                (sph_flops::kVectorOperation + 2) * detail::sphReductionFlops<kHydroBlockSize>();
            if (threadIndex == 0) {
                // Corrections are derived, never checkpointed or migrated; slots that the
                // density pass has not rebuilt hold NaN, so a stale source cannot pass silently.
                if (!(sph_volume::isFiniteValue(dv_a_dτ.dot(dv_a_dτ)) &&
                        sph_volume::isFiniteValue(du_a_dτ) &&
                        sph_volume::isFiniteValue(du_a_dτ_diss) &&
                        sph_volume::isFiniteValue(ΣC_ab_over_a))) {
                    printf("SPH variational rates are not finite: smoothingLength=%e density=%e "
                           "ζ=%e\n",
                        H_a, ρ_a, ζ_a);
                    __trap();
                }
                SpatialVector<float> dv_a_dτ_grav(0.0f);
#if COSMICTIGER_WITH_GRAVITY
                if (!parameters.midpointHydro && !parameters.deferAccelerationSources &&
                    data.selfGravityEnabled) {
                    dv_a_dτ_grav = data.sinkGravityAcceleration(sinkIndex);
                }
#endif
                if (!parameters.deferAccelerationSources) {
                    dv_a_dτ = sphCompleteAcceleration(parameters,
                        data.sourcePaddedPosition(particleIndex), v_a, dv_a_dτ, dv_a_dτ_grav,
                        data.centralAcceleration, data.centralSmoothingLength, flops);
                } else {
                    ALWAYS_ASSERT(parameters.phase == 0);
                }
                if (applyPairKicks) {
                    SpatialVector<float> const dv_a_dτ_ext = sphCompleteAcceleration(parameters,
                        data.sourcePaddedPosition(particleIndex), v_a, SpatialVector<float>(0.0f),
                        SpatialVector<float>(0.0f), data.centralAcceleration,
                        data.centralSmoothingLength, flops);
                    flops += sphAccumulateExternalKick(
                        data.sourcePairUpdates[particleIndex], dv_a_dτ_ext, Δτ);
                }
                if (data.sinkAccelerations) {
                    for (int dim = 0; dim < kSpatialDimensions; dim++) {
                        data.sinkAccelerations[sinkIndex][dim] = dv_a_dτ[dim];
                    }
                }
                // E=a^2 u and homogeneous work du/dt=-3 H P/rho give
                // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E = (5-3 gamma) (da/dtau)/a E.
                float const five_minus_3γ = 5.0f - 3.0f * γ;
                du_a_dτ += five_minus_3γ * aȧ * a_inv * u_a;
                flops += 6;    // Homogeneous thermal expansion; force sources count separately.
                if (data.sinkInternalEnergyRate) {
                    data.sinkInternalEnergyRate[sinkIndex] = du_a_dτ + du_a_dτ_diss;
                    flops += 1;
                }
                if (data.sinkEntropyRate) {
                    // Reversible PdV and homogeneous expansion leave A unchanged.
                    data.sinkEntropyRate[sinkIndex] =
                        sph_entropy::fromStoredEnergy(du_a_dτ_diss, ρ_a, γ, parameters.scaleFactor);
                    flops += sph_flops::entropyConversionFlops(γ);
                }
                if (parameters.phase == 1) {
                    // clang-format off
                    /*
                     * Section 10.3, p. 563:
                     *   δt_f = min_a (h_a/|f_a|),
                     *   δt_cv = min_a h_a/(c_a + 0.6 (α c_a + β max_b μ_ab)),
                     *   δt = 0.25 min(δt_f, δt_cv).
                     * h_a is the common effective radius 2σ (docs/monaghan_sph.md). The
                     * printed δt_f lacks a square root and is used as sqrt(h_a/|f_a|);
                     * max_b μ_ab is read as max_b(−μ_ab); the Courant factor and the calibrated
                     * acceleration coefficient replace the fixed 0.25. The reciprocals are kept
                     * apart for the diagnostics.
                     *
                     * Symbol directory (in the code below):
                     *   μ_ab → viscosity velocity; max_b μ_ab → maximum compression speed
                     *   |f_a| → acceleration magnitude
                     *   h_a → effective radius; c_a → sound speed
                     *   δt → time step; δt_g → gravity time step
                     *   δt_f⁻¹, δt_cv⁻¹ → inverse acceleration time step, inverse Courant time step
                     *   δt_c⁻¹, δt_v⁻¹ → sound and viscosity terms of δt_cv⁻¹
                     *   α → maximum bulk viscosity coefficient
                     *   β → maximum von Neumann–Richtmyer coefficient
                     *   Code names: abs_f_a = |f_a|; max_b_μ_ab = max_b μ_ab; δt_inv_f = δt_f⁻¹;
                     *     δt_inv_cv = δt_cv⁻¹; δt_inv_c = δt_c⁻¹; δt_inv_v = δt_v⁻¹.
                     */
                    // clang-format on
                    float const h_a = H_a / smoothingLengthPerEffectiveRadius;
                    float const δt_inv_c = c_a / h_a;
                    float const β = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                        parameters.vonNeumannRichtmyerCoefficient * α :
                        parameters.vonNeumannRichtmyerCoefficient;
                    float const δt_inv_v = 0.6f * (α * c_a + β * max_b_μ_ab) / h_a;
                    float const δt_inv_cv = δt_inv_c + δt_inv_v;
                    float const abs_f_a = dv_a_dτ.norm();
                    float const δt_inv_f = parameters.courantFactor *
                        sqrtf(parameters.scaleFactor * abs_f_a / h_a) /
                        (data.accelerationTimestepFactor * kHydroAccelerationScale);
                    float δt = fminf(parameters.maximumTimestep,
                        parameters.courantFactor * parameters.scaleFactor /
                            fmaxf(fmaxf(δt_inv_cv, δt_inv_f), 1.0e-30f));
                    // Bound reversible work; the approach-speed CFL term controls viscosity.
                    if (du_a_dτ != 0.0f) {
                        flops += 3;
                        δt = fminf(δt,
                            parameters.courantFactor / float(monaghan::kViscosityWidthScale) * u_a /
                                fabsf(du_a_dτ));
                    }
                    // Derived positivity bound, matching the CPU sum over all neighbors.
                    if (ΣC_ab_over_a > 0.0f) {
                        δt = fminf(δt, 0.5f / ΣC_ab_over_a);
                        flops += 1;
                    }
                    ALWAYS_ASSERT(isfinite(δt) && δt > 0.0f);
                    atomicMin(&reduction->minimumTimestep, δt);
                    int const hydroRung = parameters.globalTimestep ?
                        0 :
                        max(parameters.midpointHydro ? 0 : 1,
                            int(ceilf(log2f(parameters.baseTimestep / δt))));
                    int const minimumProposedRung = parameters.midpointHydro ?
                        parameters.minimumProposedRung :
                        parameters.minimumRung;
                    int nextRung = max(max(minimumProposedRung, previousRung - 1), hydroRung);
                    if (parameters.midpointHydro) {
                        // Commit the touching-neighbor bound at the next gravity boundary.
                        nextRung = max(nextRung, maximumNeighborRung - kMaxNeighborRungDifference);
                    }
#if COSMICTIGER_WITH_GRAVITY
                    float δt_g = parameters.maximumTimestep;
                    int gravityRung = 0;
                    if (!parameters.midpointHydro) {
                        δt_g = fminf(parameters.maximumTimestep,
                            data.accelerationTimestepFactor *
                                sqrtf(parameters.scaleFactor * data.gravityEffectiveRadius /
                                    fmaxf(dv_a_dτ_grav.norm(), 1.0e-30f)));
                        if (data.selfGravityEnabled) {
                            atomicMin(&reduction->minimumTimestep, δt_g);
                        }
                        gravityRung = parameters.globalTimestep ?
                            0 :
                            max(1, int(ceilf(log2f(parameters.baseTimestep / δt_g))));
                        nextRung = max(nextRung, data.selfGravityEnabled ? gravityRung : 1);
                        maximumGravityRung = max(maximumGravityRung, gravityRung);
                        // Exclude the bound computed only for disabled-gravity diagnostics.
                        flops +=
                            data.selfGravityEnabled ? (parameters.globalTimestep ? 10 : 13) : 0;
                    }
#endif
                    if (parameters.globalTimestep) {
                        nextRung = 0;
                    }
                    if (nextRung >= kMaxRung &&
                        reduction->timestepError.record(
                            {data.sinkParticleIndices[sinkIndex], nextRung})) {
                        printf(
                            "SPH timestep exceeds rung range: particle=%lli rung=%i hydroRung=%i "
                            "previousRung=%i minimumRung=%i baseDt=%e selectedDt=%e capDt=%e "
                            "courantDt=%e accelerationDt=%e thermalDt=%e H=%e rho=%e u=%e "
                            "duDt=%e acceleration=%e soundSpeed=%e compression=%e alpha=%e "
                            "omegaFromZeta=%e scaleFactor=%e\n",
                            static_cast<long long>(data.sinkParticleIndices[sinkIndex]), nextRung,
                            hydroRung, previousRung, parameters.minimumRung,
                            parameters.baseTimestep, δt, parameters.maximumTimestep,
                            parameters.courantFactor * parameters.scaleFactor /
                                fmaxf(δt_inv_cv, 1.0e-30f),
                            parameters.courantFactor * parameters.scaleFactor /
                                fmaxf(δt_inv_f, 1.0e-30f),
                            du_a_dτ != 0.0f ? parameters.courantFactor /
                                    float(monaghan::kViscosityWidthScale) * u_a / fabsf(du_a_dτ) :
                                              parameters.maximumTimestep,
                            H_a, ρ_a, u_a, du_a_dτ, abs_f_a, c_a, max_b_μ_ab, α, m_b / (m_b - ζ_a),
                            parameters.scaleFactor);
#if COSMICTIGER_WITH_GRAVITY
                        printf("SPH gravity timestep: enabled=%i gravityRung=%i gravityDt=%e\n",
                            int(data.selfGravityEnabled), gravityRung, δt_g);
#endif
                    }
                    // Publish only representable rungs. Every thread still reaches
                    // the block barriers; the host throws after stream completion.
                    if (nextRung < kMaxRung) {
                        if (data.proposeRungs) {
                            if (parameters.retainProposedRungs) {
                                nextRung = max(nextRung, data.sinkFlags[sinkIndex].proposedRung());
                            }
                            data.sinkFlags[sinkIndex].setProposedRung(nextRung);
                        } else {
                            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
                        }
                    }
                    maximumHydroRung = max(maximumHydroRung, hydroRung);
                    maximumRung = max(maximumRung, nextRung);
                    maximumSignalSpeed = fmaxf(maximumSignalSpeed, δt_inv_cv * h_a);
                    // Exclude the signal-speed report.
                    flops += sph_flops::kNorm + (parameters.globalTimestep ? 16 : 19);
                    atomicMax(&reduction->inverseSoundTimestep, δt_inv_c);
                    atomicMax(&reduction->inverseViscosityTimestep, δt_inv_v);
                    atomicMax(&reduction->inverseAccelerationTimestep, δt_inv_f);
                }
            }
        }
        __syncthreads();
        if (threadIndex == 0) {
            // Stop assigning nodes after a failed timestep. This decision is
            // block-uniform, so shared workspace cleanup and barriers stay valid.
            targetNodeIndex = reduction->timestepError.pending() ?
                data.numberOfTargetNodes :
                atomicAdd(&reduction->nextNodeIndex, 1);
        }
        __syncthreads();
    }
    if (threadIndex == 0) {
        atomicMax(&reduction->maximumSignalSpeed, maximumSignalSpeed);
        atomicMax(&reduction->maximumRung, maximumRung);
        atomicMax(&reduction->maximumHydroRung, maximumHydroRung);
        atomicMax(&reduction->maximumGravityRung, maximumGravityRung);
    }
    detail::recordSphGpuFlops(flops, reduction);
    (&workspace)->~HydroWorkspace();
}
