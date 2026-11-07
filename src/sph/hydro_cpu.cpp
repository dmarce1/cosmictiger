/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <vector>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/sph/conduction.hpp>
#include <cosmictiger/sph/cpu.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/variable_smoothing_lengths.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/volume.hpp>

using sph_geometry::SpatialVector;

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
 *   v_a, v_b → sink stored velocity, neighbor stored velocity
 *   ρ_a, P_a, u_a, c_a → density, pressure, internal energy, sound speed
 *   ρ_b, P_b, u_b, c_b → neighbor density, pressure, internal energy, sound speed
 *   H_a, H_b → sink kernel smoothing length, neighbor smoothing length
 *   H_a⁻¹, H_b⁻¹ → inverse smoothing lengths; H_a⁻⁽ᴰ⁺¹⁾ → sink inverse gradient scale
 *   r_ab, |r_ab|, r_ab² → separation, pair distance, squared separation
 *   v_ab → velocity difference; v_ab · r_ab → velocity dot separation
 *   (a v_phys,ab) · r_ab → velocity dot separation including the Hubble flow a ȧ r_ab
 *   r_ab² staged, H_b staged → r_ab², H_b before excluded lanes are replaced
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
 *     dv_a_dτ_ext = (dv_a/dτ)_ext; five_minus_3γ = 5 − 3γ; r2_ab_staged = r_ab² staged;
 *     H_b_staged = H_b staged.
 *   SIMD values append Pack; CUDA partial sums are reduced in place.
 */
// clang-format on
SphRunReturn sphCpuHydro(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    sph_flops::Counter flops = 3;    // Scale-factor reciprocal, gas mass scaling and γ − 1.
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    float const a_inv = 1.0f / parameters.scaleFactor;
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every gas particle
    SimdFloat const γPack(γ);
    SimdFloat const γ_minus_1Pack(γ_minus_1);
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4 defaults
    // after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    float const smoothingLengthPerEffectiveRadius =
        selectedHydroKernel().smoothingLengthPerEffectiveRadius;
    SimdFloat const zero(0.0f);
    SimdFloat const one(1.0f);
    SimdFloat const half(0.5f);
    float const m_b = data.gasMass;    // every gas particle has the same mass
    SimdFloat const m_b_over_a(m_b * a_inv);
    float const ε = parameters.viscosityEpsilon;    // η²/h² of p. 551
    float const aȧ = parameters.expansionRate;      // a·da/dt, the Hubble flow of a pair
    bool const applyPairKicks = data.sourcePairUpdates && parameters.hydroKickTimestep > 0.0;
    float const Δτ = applyPairKicks ? float(parameters.hydroKickTimestep) : 0.0f;
    flops.doublePrecision += int(applyPairKicks);
    ALWAYS_ASSERT(!applyPairKicks || (parameters.midpointHydro && data.sourceIds));
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; particleIndex++) {
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
        float const c_a = std::sqrt(γ * γ_minus_1 * u_a);
        SpatialVector<float> const v_a = data.sourceVelocity(particleIndex);
        SpatialVector<SimdFloat> const v_aPack(v_a);
        float const α_a = parameters.variableViscosity ?
            data.sourceBulkViscosityCoefficient[particleIndex] :
            parameters.bulkViscosityCoefficient;
        SimdFloat αPack(α_a);
        float const H_inv_a = 1.0f / H_a;
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
        float const H_inv_D_plus_1_a = sph_geometry::inverseGradientScale(H_inv_a);    // H_a^-(D+1)
        SpatialVector<SimdFloat> dv_a_dτPack(zero);    // lane partial sums of acceleration
        SimdFloat du_a_dτ_dissPack(zero);
        SimdFloat ΣC_ab_over_aPack(zero);
        SimdFloat du_a_dτPack(zero);       // lane partial sums of thermal work
        SimdFloat max_b_μ_abPack(zero);    // lane maxima of −μ_ab for δt_cv
        SimdFloat maximumNeighborRungPack(zero);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            // Particle b: one SIMD pack of staged neighbors. Lanes beyond the list, outside
            // both supports, or at zero separation are excluded. Their inputs are replaced
            // before any division or square root; masking only the result can still trap.
            SpatialVector<SimdFloat> const r_ab =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const r2_ab_staged = r_ab.dot(r_ab);
            SimdFloat const H_b_staged =
                detail::sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 1.0f);
            auto const supportsIntersect = detail::sphCpuValidLanes(neighbors, first) &
                (r2_ab_staged < simdMax(SimdFloat(H_a * H_a), sqr(H_b_staged)));
            if (parameters.midpointHydro && parameters.phase == 1) {
                SimdFloat const neighborRungs([&](auto lane) {
                    size_t const index = first + size_t(lane);
                    return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) :
                                                      0.0f;
                });
                maximumNeighborRungPack = simdMax(
                    maximumNeighborRungPack, simdSelect(supportsIntersect, neighborRungs, zero));
            }
            auto const interacts = supportsIntersect & (r2_ab_staged > zero);
            // Fixed scales, dot operations and H_b² per lane; H_a² is scalar.
            flops +=
                (sph_flops::kVectorOperation + sph_flops::kDotProduct + 1) * SimdFloat::size() + 1;
            if (!simdAnyOf(interacts)) {
                continue;
            }
            SimdFloat const H_b = simdSelect(interacts, H_b_staged, one);
            SimdFloat const ρ_b = simdSelect(
                interacts, detail::sphCpuLoad(data.sourceDensity, neighbors, first, 1.0f), one);
            SimdFloat const u_b = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceInternalEnergy, neighbors, first, 1.0f), one);
            SimdFloat const P_b = γ_minus_1Pack * ρ_b * u_b;
            SimdFloat const c_b = simdSqrt(γPack * γ_minus_1Pack * u_b);
            SimdFloat const r2_ab = simdSelect(interacts, r2_ab_staged, one);    // |r_ab|²
            SimdFloat const abs_r_ab = simdSqrt(r2_ab);                          // |r_ab|
            // clang-format off
            /*
             * Symmetric kernel of p. 554: ∇_a W̃_ab = ½[∇_a W(r_ab, H_a) + ∇_a W(r_ab,
             * H_b)]. The selected kernel is W(r, H) = w(r/H)/H³, so each
             * gradient equals w'(r/H) r_ab/(H⁴ r) and ∇_a W̃_ab = F_ab r_ab with the scalar F_ab.
             *
             * Symbol directory (in the code below):
             *   F_ab → symmetric gradient factor
             *   r_ab → separation
             */
            // clang-format on
            SimdFloat const H_inv_b = one / H_b;
            SimdFloat const q_a = abs_r_ab * SimdFloat(H_inv_a);
            SimdFloat const q_b = abs_r_ab * H_inv_b;
            SimdFloat const F_a = dkernelWDq(q_a) * SimdFloat(H_inv_D_plus_1_a) / abs_r_ab;
            SimdFloat const F_b =
                dkernelWDq(q_b) * sph_geometry::inverseGradientScale(H_inv_b) / abs_r_ab;
            flops += sph_flops::densityKernelOperations(selectedHydroKernel(), q_a, true) +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q_b, true);
            SimdFloat const F_ab = simdSelect(interacts, half * (F_a + F_b), zero);
            // v_ab = v_a − v_b from the stored peculiar velocities. The physical relative
            // velocity adds the Hubble flow (a da/dt) r_ab and enters only the viscosity.
            SpatialVector<SimdFloat> const v_b = detail::sphCpuLoad3(data.sourceVelocityX,
                data.sourceVelocityY, data.sourceVelocityZ, neighbors, first, 0.0f);
            SpatialVector<SimdFloat> const v_ab =
                simdSelect(interacts, v_aPack - v_b, SpatialVector<SimdFloat>(zero));
            SimdFloat const v_ab_dot_r_ab = v_ab.dot(r_ab);
            SimdFloat const a_v_phys_ab_dot_r_ab =
                v_ab_dot_r_ab + SimdFloat(aȧ) * simdSelect(interacts, r2_ab, zero);
            SimdFloat const μ_ab = monaghan::viscosityMu(r2_ab, a_v_phys_ab_dot_r_ab,
                SimdFloat(H_a), H_b, SimdFloat(ε), SimdFloat(smoothingLengthPerEffectiveRadius),
                SimdFloat(float(monaghan::kViscosityWidthScale)));
            SimdFloat const α_b = parameters.variableViscosity ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourceBulkViscosityCoefficient, neighbors, first, α_a),
                    SimdFloat(α_a)) :
                SimdFloat(α_a);
            SimdFloat const α_ab =
                parameters.variableViscosity ? half * SimdFloat(α_a) + half * α_b : SimdFloat(α_a);
            SimdFloat const β_ab = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                SimdFloat(parameters.vonNeumannRichtmyerCoefficient) * α_ab :
                SimdFloat(parameters.vonNeumannRichtmyerCoefficient);
            αPack = simdMax(αPack, simdSelect(interacts, α_ab, SimdFloat(α_a)));
            flops += (3 * int(parameters.variableViscosity) +
                         int(parameters.vonNeumannRichtmyerCoefficientFollowsBulk())) *
                SimdFloat::size();
            SimdFloat fᴬⱽ_ab(one);
            if (parameters.balsaraSwitch) {
                // Preserve the optional symmetric Balsara multiplier.
                fᴬⱽ_ab = half *
                    (SimdFloat(data.sourceBalsaraFactor[particleIndex]) +
                        detail::sphCpuLoad(data.sourceBalsaraFactor, neighbors, first, 0.0f));
                flops += 2 * SimdFloat::size();
            }
            // clang-format off
            /*
             * Hopkins (2013) equation 12 with Price (2012) equation 35: separate gradients
             * g_a = F_a r_ab and g_b = F_b r_ab, the directed Q_ab and Q_ba, and the same
             * Q_ab in the reversible work. The pressure work takes the peculiar velocity;
             * the analytic expansion term below supplies the Hubble-flow part exactly.
             * Viscous force and shock heating below use the same pair coefficients.
             *
             * Symbol directory (in the code below):
             *   r_ab → separation
             *   F_a → sink gradient factor
             *   F_b → neighbor gradient factor
             *   g_a → sink kernel gradient (F_a r_ab)
             *   g_b → neighbor kernel gradient (F_b r_ab)
             *   maskedF_a, maskedF_b → F_a, F_b with excluded SIMD lanes zeroed
             */
            // clang-format on
            SimdFloat const ζ_b = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceGradHCorrection, neighbors, first, 0.0f), zero);
            SimdFloat const Q_ba_over_m_a = variational::directedCoefficient(
                variational::pressureVolumeFactor(P_b, one, ρ_b), SimdFloat(m_b), ζ_b);
            SimdFloat const maskedF_a = simdSelect(interacts, F_a, zero);
            SimdFloat const maskedF_b = simdSelect(interacts, F_b, zero);
            SimdFloat const f_P_ab = SimdFloat(a_inv) *
                variational::pairForceFactor(
                    SimdFloat(Q_ab_over_m_a), Q_ba_over_m_a, maskedF_a, maskedF_b);
            SimdFloat Π_ab = monaghan::artificialViscosity(
                SimdFloat(ρ_a), ρ_b, SimdFloat(c_a), c_b, μ_ab, α_ab, β_ab);
            if (parameters.balsaraSwitch) {
                Π_ab *= fᴬⱽ_ab;
                flops += SimdFloat::size();
            }
            SimdFloat const f_Π_ab = m_b_over_a * Π_ab * F_ab;
            SimdFloat const du_dτ_Π_ab = m_b_over_a * half * Π_ab * F_ab * a_v_phys_ab_dot_r_ab;
            flops +=
                (sph_flops::kViscosityMu + sph_flops::kArtificialViscosity + 6) * SimdFloat::size();
            SimdFloat du_a_dτ_diss_ab = du_dτ_Π_ab;
            SimdFloat du_b_dτ_diss_ab = du_dτ_Π_ab;
            if (parameters.conductionCoefficient > 0.0f) {
                SimdFloat const C_ab_over_a = sph_conduction::exchangeRate(SimdFloat(ρ_a), ρ_b,
                    SimdFloat(c_a), c_b, SimdFloat(H_a), H_b, r2_ab, a_v_phys_ab_dot_r_ab, F_ab,
                    SimdFloat(parameters.conductionCoefficient), SimdFloat(ε),
                    SimdFloat(smoothingLengthPerEffectiveRadius), m_b_over_a);
                SimdFloat const du_a_dτ_cond_ab = C_ab_over_a * (u_b - SimdFloat(u_a));
                du_a_dτ_diss_ab += du_a_dτ_cond_ab;
                du_b_dτ_diss_ab -= du_a_dτ_cond_ab;
                ΣC_ab_over_aPack += C_ab_over_a;
                flops += 27 * SimdFloat::size();    // 22 exchange + 5 heating/row-sum operations.
            }
            SpatialVector<SimdFloat> const f_abPack = (f_P_ab + f_Π_ab) * r_ab;
            dv_a_dτPack -= f_abPack;
            if (applyPairKicks) {
                for (size_t lane = 0; lane < SimdFloat::size(); ++lane) {
                    if (!interacts[lane]) {
                        continue;
                    }
                    int const neighborIndex = neighbors[first + lane];
                    if (!parameters.isActiveRung(
                            std::max(previousRung, int(data.sourceRungs[neighborIndex]))) ||
                        !sphOwnsPairKick(data.sourceIds[particleIndex],
                            data.sourceIds[neighborIndex], previousRung,
                            data.sourceRungs[neighborIndex])) {
                        continue;
                    }
                    SpatialVector<float> const a_ab = -detail::sphCpuLane(f_abPack, lane);
                    float const Ȧ_ab = sph_entropy::fromStoredEnergy(
                        float(du_a_dτ_diss_ab[lane]), ρ_a, γ, parameters.scaleFactor);
                    float const Ȧ_ba = sph_entropy::fromStoredEnergy(
                        float(du_b_dτ_diss_ab[lane]), float(ρ_b[lane]), γ, parameters.scaleFactor);
                    flops += sphAccumulatePairKick(data.sourcePairUpdates[particleIndex],
                        data.sourcePairUpdates[neighborIndex], a_ab, Ȧ_ab, Ȧ_ba, Δτ);
                    flops += 2 * sph_flops::entropyConversionFlops(γ);
                }
            }
            du_a_dτPack += SimdFloat(a_inv) *
                variational::pairWork(SimdFloat(Q_ab_over_m_a), maskedF_a, v_ab_dot_r_ab);
            du_a_dτ_dissPack += du_a_dτ_diss_ab;
            // Common geometry, pressure, and accumulation; the vector terms are v_ab, v_ab · r_ab,
            // the pair force and its sum.
            flops += (38 + sph_flops::kInverseGradientScale + 3 * sph_flops::kVectorOperation +
                         sph_flops::kDotProduct) *
                SimdFloat::size();
            max_b_μ_abPack = simdMax(max_b_μ_abPack, -μ_ab);
        }
        SpatialVector<float> dv_a_dτ = detail::sphCpuSum(dv_a_dτPack);
        float du_a_dτ = simdSum(du_a_dτPack);
        float const ΣC_ab_over_a =
            parameters.conductionCoefficient > 0.0f ? simdSum(ΣC_ab_over_aPack) : 0.0f;
        if (parameters.conductionCoefficient > 0.0f) {
            flops += SimdFloat::size() - 1;
        }
        // The acceleration components and the work; maxima are comparisons.
        flops += (sph_flops::kVectorOperation + 1) * (SimdFloat::size() - 1);
        // Corrections are derived, never checkpointed or migrated; slots that the density
        // pass has not rebuilt hold NaN, so a stale source cannot pass silently.
        if (!(sph_volume::isFiniteValue(dv_a_dτ.dot(dv_a_dτ)) &&
                sph_volume::isFiniteValue(du_a_dτ) &&
                sph_volume::isFiniteValue(simdSum(du_a_dτ_dissPack)) &&
                sph_volume::isFiniteValue(ΣC_ab_over_a))) {
            std::cout << std::format("SPH variational rates are not finite: smoothingLength={:e} "
                                     "density={:e} \u03b6={:e}\n",
                             H_a, ρ_a, ζ_a)
                      << std::flush;
            ALWAYS_ASSERT(false);
        }
        float const α = simdMaximum(αPack);
        float const β = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
            parameters.vonNeumannRichtmyerCoefficient * α :
            parameters.vonNeumannRichtmyerCoefficient;
        float const max_b_μ_ab =
            simdMaximum(max_b_μ_abPack);    // max_b(−μ_ab) over approaching pairs
        // The pair calculation can overlap gravity. Only source assembly and
        // the following corrector require the fresh endpoint acceleration.
        SpatialVector<float> dv_a_dτ_grav(0.0f);
#if COSMICTIGER_WITH_GRAVITY
        if (!parameters.midpointHydro && !parameters.deferAccelerationSources &&
            data.selfGravityEnabled) {
            dv_a_dτ_grav = data.sinkGravityAcceleration(sinkIndex);
        }
#endif
        if (!parameters.deferAccelerationSources) {
            dv_a_dτ = sphCompleteAcceleration(parameters, data.sourcePaddedPosition(particleIndex),
                v_a, dv_a_dτ, dv_a_dτ_grav, data.centralAcceleration, data.centralSmoothingLength,
                flops);
        } else {
            ALWAYS_ASSERT(parameters.phase == 0);
        }
        if (applyPairKicks) {
            // External forces have no reciprocal gas endpoint. Integrate them
            // once on the target's own rung, separately from pair reductions.
            SpatialVector<float> const dv_a_dτ_ext =
                sphCompleteAcceleration(parameters, data.sourcePaddedPosition(particleIndex), v_a,
                    SpatialVector<float>(0.0f), SpatialVector<float>(0.0f),
                    data.centralAcceleration, data.centralSmoothingLength, flops);
            flops +=
                sphAccumulateExternalKick(data.sourcePairUpdates[particleIndex], dv_a_dτ_ext, Δτ);
        }
        if (data.sinkAccelerations) {
            for (int dim = 0; dim < kSpatialDimensions; dim++) {
                data.sinkAccelerations[sinkIndex][dim] = dv_a_dτ[dim];
            }
        }
        // E=a^2 u and homogeneous adiabatic work du/dt=-3 H P/rho give
        // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E = (5-3 gamma) (da/dtau)/a E.
        // The variational pair work also takes the peculiar velocity: with V_i = a^3 V^c_i
        // and V^c_i a function of comoving positions and supports, this term supplies the
        // Hubble-flow work exactly for free supports and for supports held at comoving
        // bounds (docs/variational_sph.md). Free supports also satisfy
        // sum_j (x_j - zeta_i) r_ij . g_i = -3 y_i, so physical pair velocities would count
        // that work twice; held supports have no such identity.
        float const five_minus_3γ = 5.0f - 3.0f * γ;
        du_a_dτ += five_minus_3γ * aȧ * a_inv * u_a;
        flops += 6;    // Homogeneous thermal expansion; force sources count separately.
        if (data.sinkInternalEnergyRate) {
            data.sinkInternalEnergyRate[sinkIndex] = du_a_dτ + simdSum(du_a_dτ_dissPack);
            flops += SimdFloat::size();
        }
        if (data.sinkEntropyRate) {
            // Reversible PdV and homogeneous expansion leave A unchanged.
            data.sinkEntropyRate[sinkIndex] = sph_entropy::fromStoredEnergy(
                simdSum(du_a_dτ_dissPack), ρ_a, γ, parameters.scaleFactor);
            flops += sph_flops::entropyConversionFlops(γ);
            flops += SimdFloat::size() - 1;
        }
        if (parameters.phase == 1) {
            // clang-format off
            /*
             * Section 10.3, p. 563:
             *   δt_f = min_a (h_a/|f_a|),   δt_cv = min_a h_a/(c_a +
             *   0.6 (α c_a + β max_b μ_ab)), δt = 0.25 min(δt_f, δt_cv).
             * h_a is the common effective radius 2σ (docs/monaghan_sph.md). The printed
             * δt_f lacks a square root and is used as sqrt(h_a/|f_a|); max_b μ_ab is
             * read as the compression magnitude max_b(−μ_ab); the Courant factor and the
             * calibrated acceleration coefficient replace the fixed 0.25. The reciprocals are
             * kept apart for the diagnostics in SphRunReturn.
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
            float const δt_inv_v = 0.6f * (α * c_a + β * max_b_μ_ab) / h_a;
            float const δt_inv_cv = δt_inv_c + δt_inv_v;
            float const abs_f_a = dv_a_dτ.norm();
            float const δt_inv_f = parameters.courantFactor *
                std::sqrt(parameters.scaleFactor * abs_f_a / h_a) /
                (data.accelerationTimestepFactor * kHydroAccelerationScale);
            float δt = std::fmin(parameters.maximumTimestep,
                parameters.courantFactor * parameters.scaleFactor /
                    std::fmax(std::fmax(δt_inv_cv, δt_inv_f), 1.0e-30f));
            // Bound reversible work; the approach-speed CFL term controls viscous heating.
            if (du_a_dτ != 0.0f) {
                δt = std::fmin(δt,
                    parameters.courantFactor / float(monaghan::kViscosityWidthScale) * u_a /
                        std::fabs(du_a_dτ));
                flops += 3;    // Two divisions and a product; abs is a sign operation.
            }
            // Derived forward-Euler diffusion bound: δt Σ_b C_ab/a ≤ 1/2.
            // It uses the sum of positive exchange rates, even when net heating cancels.
            if (ΣC_ab_over_a > 0.0f) {
                δt = std::fmin(δt, 0.5f / ΣC_ab_over_a);
                flops += 1;
            }
            ALWAYS_ASSERT(std::isfinite(δt) && δt > 0.0f);
            result.minimumTimestep = std::min(result.minimumTimestep, double(δt));
            flops.doublePrecision += 1;
            int const hydroRung = parameters.globalTimestep ?
                0 :
                std::max(parameters.midpointHydro ? 0 : 1,
                    int(std::ceil(std::log2(parameters.baseTimestep / δt))));
            int const minimumProposedRung =
                parameters.midpointHydro ? parameters.minimumProposedRung : parameters.minimumRung;
            int nextRung = std::max(std::max(minimumProposedRung, previousRung - 1), hydroRung);
            if (parameters.midpointHydro) {
                // Keep the existing touching-neighbor timestep limit in the proposal.
                // The gravity boundary, not this midpoint, commits the new live rung.
                nextRung = std::max(nextRung,
                    int(simdMaximum(maximumNeighborRungPack)) - kMaxNeighborRungDifference);
            }
#if COSMICTIGER_WITH_GRAVITY
            float δt_g = parameters.maximumTimestep;
            int gravityRung = 0;
            if (!parameters.midpointHydro) {
                δt_g = std::fmin(parameters.maximumTimestep,
                    data.accelerationTimestepFactor *
                        std::sqrt(parameters.scaleFactor * data.gravityEffectiveRadius /
                            std::fmax(dv_a_dτ_grav.norm(), 1.0e-30f)));
                if (data.selfGravityEnabled) {
                    result.minimumTimestep = std::min(result.minimumTimestep, double(δt_g));
                    flops.doublePrecision += 1;
                }
                gravityRung = parameters.globalTimestep ?
                    0 :
                    std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / δt_g))));
                nextRung = std::max(nextRung, data.selfGravityEnabled ? gravityRung : 1);
                result.maximumGravityRung = std::max(result.maximumGravityRung, gravityRung);
                // Exclude the bound computed only for disabled-gravity diagnostics.
                flops += data.selfGravityEnabled ? (parameters.globalTimestep ? 10 : 13) : 0;
            }
#endif
            if (parameters.globalTimestep) {
                nextRung = 0;
            }
            if (nextRung >= kMaxRung) {
                std::cout << std::format(
                                 "SPH timestep exceeds rung range: particle={:d} rung={:d} "
                                 "hydroRung={:d} previousRung={:d} minimumRung={:d} baseDt={:e} "
                                 "selectedDt={:e} capDt={:e} courantDt={:e} accelerationDt={:e} "
                                 "thermalDt={:e} H={:e} rho={:e} u={:e} duDt={:e} "
                                 "acceleration={:e} soundSpeed={:e} compression={:e} alpha={:e} "
                                 "omegaFromZeta={:e} scaleFactor={:e}\n",
                                 static_cast<long long>(data.sinkParticleIndices[sinkIndex]),
                                 nextRung, hydroRung, previousRung, parameters.minimumRung,
                                 parameters.baseTimestep, δt, parameters.maximumTimestep,
                                 parameters.courantFactor * parameters.scaleFactor /
                                     std::fmax(δt_inv_cv, 1.0e-30f),
                                 parameters.courantFactor * parameters.scaleFactor /
                                     std::fmax(δt_inv_f, 1.0e-30f),
                                 du_a_dτ != 0.0f ? parameters.courantFactor /
                                         float(monaghan::kViscosityWidthScale) * u_a /
                                         std::fabs(du_a_dτ) :
                                                   parameters.maximumTimestep,
                                 H_a, ρ_a, u_a, du_a_dτ, abs_f_a, c_a, max_b_μ_ab, α,
                                 m_b / (m_b - ζ_a), parameters.scaleFactor)
                          << std::flush;
#if COSMICTIGER_WITH_GRAVITY
                std::cout
                    << std::format(
                           "SPH gravity timestep: enabled={:d} gravityRung={:d} gravityDt={:e}\n",
                           int(data.selfGravityEnabled), gravityRung, δt_g)
                    << std::flush;
#endif
            }
            ALWAYS_ASSERT(nextRung < kMaxRung);
            if (data.proposeRungs) {
                if (parameters.retainProposedRungs) {
                    nextRung = std::max(nextRung, data.sinkFlags[sinkIndex].proposedRung());
                }
                data.sinkFlags[sinkIndex].setProposedRung(nextRung);
            } else {
                data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
            }
            result.maximumHydroRung = std::max(result.maximumHydroRung, hydroRung);
            result.maximumRung = std::max(result.maximumRung, nextRung);
            result.maximumSignalSpeed = std::fmax(result.maximumSignalSpeed, δt_inv_cv * h_a);
            result.inverseSoundTimestep = std::fmax(result.inverseSoundTimestep, δt_inv_c);
            result.inverseViscosityTimestep = std::fmax(result.inverseViscosityTimestep, δt_inv_v);
            result.inverseAccelerationTimestep =
                std::fmax(result.inverseAccelerationTimestep, δt_inv_f);
            // h/c/visc (8), acceleration norm and limit (6), timestep (2), hydro rung (3).
            // Exclude the signal-speed diagnostic.
            flops += sph_flops::kNorm + (parameters.globalTimestep ? 16 : 19);
        }
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}
