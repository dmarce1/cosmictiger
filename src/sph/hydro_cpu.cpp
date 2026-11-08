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
 * uses the symbols of Monaghan (1992), Hopkins (2013) and Price (2012), described
 * below. Positions and density are comoving, and stored velocities and internal
 * energies carry the factors a and a² of docs/variational_sph.md.
 */

// clang-format off
/*
 * Production pair equations: Hopkins (2013), MNRAS 428, 2840, equations (12)–(13)
 * (https://arxiv.org/abs/1206.5006), and Price (2012), JCP 231, 759, (30), (35)
 * (https://arxiv.org/abs/1012.1885). Equal masses make the volume weight m𞁑 and the
 * weighted density ρ𞁑:
 *   Q𞁑𞁓/m𞁑 = P𞁑(m𞁓 − ζ𞁑)/ρ𞁑²,
 *   dv𞁑/dτ = −a⁻¹ Σ𞁓 [(Q𞁑𞁓/m𞁑)F𞁑 + (Q𞁓𞁑/m𞁑)F𞁓] r𞁑𞁓,
 *   du𞁑/dτ = a⁻¹ Σ𞁓 (Q𞁑𞁓/m𞁑)F𞁑 (v𞁑𞁓 · r𞁑𞁓),
 *   ∇𞁑 W𞁑𞁓(h𞁑) = F𞁑 r𞁑𞁓, F𞁑 = w′(q𞁑)/(h𞁑⁴ |r𞁑𞁓|), q𞁑 = |r𞁑𞁓|/h𞁑.
 * These are derived comoving forms; the pair viscosity below adds to the same pair force.
 * Symbol directory:
 *   a, a⁻¹ → scale factor, its inverse; a ȧ → expansion rate, a da/dt = da/dτ
 *   m𞁓, m𞁓/a → gas mass, scaled gas mass; γ, γ − 1 → gas adiabatic index, P/(ρu) of all gas
 *   v𞁑, v𞁓 → sink stored velocity, neighbor stored velocity
 *   ρ𞁑, P𞁑, u𞁑, c𞁑 → density, pressure, internal energy, sound speed
 *   ρ𞁓, P𞁓, u𞁓, c𞁓 → neighbor density, pressure, internal energy, sound speed
 *   h𞁑, h𞁓 → sink kernel smoothing length, neighbor smoothing length
 *   h𞁑⁻¹, h𞁓⁻¹ → inverse smoothing lengths; h𞁑⁻⁽ᴰ⁺¹⁾ → sink inverse gradient scale
 *   r𞁑𞁓, |r𞁑𞁓|, r𞁑𞁓² → separation, pair distance, squared separation
 *   v𞁑𞁓 → velocity difference; v𞁑𞁓 · r𞁑𞁓 → velocity dot separation
 *   (a vₚₕ𞁟ₛ𞁑𞁓) · r𞁑𞁓 → velocity dot separation including the Hubble flow a ȧ r𞁑𞁓
 *   r𞁑𞁓² staged, h𞁓 staged → r𞁑𞁓², h𞁓 before excluded lanes are replaced
 *   q𞁑, q𞁓 → radius normalized by the sink support, by the neighbor support
 *   ζ𞁑, ζ𞁓 → sink grad-H correction, neighbor grad-H correction
 *   Q𞁑𞁓/m𞁑, Q𞁓𞁑/m𞁑 → sink pressure coefficient, neighbor pressure coefficient
 *   F𞁑, F𞁓, F𞁑𞁓 → sink gradient factor, neighbor gradient factor, symmetric gradient factor
 *   dv𞁑/dτ, du𞁑/dτ → acceleration, reversible internal energy rate
 *   α𞁑, α𞁓, α𞁑𞁓 → bulk viscosity coefficient, neighbor bulk viscosity coefficient, pair bulk viscosity coefficient
 *   β𞁑𞁓 → pair von Neumann–Richtmyer coefficient; Π𞁑𞁓 → pair artificial viscosity
 *   μ𞁑𞁓 → viscosity velocity; ε → η²/h² of Monaghan (1992), p. 551
 *   h_M/h → Monaghan smoothing length per support radius (monaghan::lengthPerSupport)
 *   h̄_M𞁑𞁓 → pair Monaghan smoothing length ½(h𞁑 + h𞁓) h_M/h, shared by viscosity and conduction
 *   fᴬⱽ𞁑𞁓 → pair shear factor, the mean of the two Balsara factors
 *   f_P𞁑𞁓, f_Π𞁑𞁓 → pressure force factor, viscous force factor of one pair
 *   f𞁑𞁓 → pair force, (f_P𞁑𞁓 + f_Π𞁑𞁓) r𞁑𞁓; a𞁑𞁓 → pair acceleration of a, −f𞁑𞁓
 *   (du/dτ)_Π𞁑𞁓 → viscous heating of each endpoint by one pair
 *   C𞁑𞁓/a → pair conduction exchange rate; (du𞁑/dτ)𞁞ₒₙ𞁑𞁓 → conductive heating of a by one pair
 *   Σ𞁓 C𞁑𞁓/a → summed exchange rate, which sets the conduction timestep bound
 *   (du𞁑/dτ)ₕₑₐₜ𞁑𞁓, (du𞁓/dτ)ₕₑₐₜ𞁑𞁓 → sink, neighbor dissipative heating by one pair
 *   (du𞁑/dτ)ₕₑₐₜ → total dissipative heating, which becomes the entropy rate
 *   Ȧ𞁑𞁓, Ȧ𞁓𞁑 → sink, neighbor entropy rate of one pair; Δτ → pair kick interval
 *   (dv𞁑/dτ)_G, (dv𞁑/dτ)ₑₓₜ → gravity acceleration, external acceleration
 *   5 − 3γ → homogeneous expansion coefficient of the stored energy
 *   Code names: ia = a⁻¹; aȧ = a ȧ; m𞁓_over_a = m𞁓/a; γ_minus_1 = γ − 1
 *     ih𞁑 = h𞁑⁻¹; ih𞁓 = h𞁓⁻¹; ih_D_plus_1𞁑 = h𞁑⁻⁽ᴰ⁺¹⁾; abs_r𞁑𞁓 = |r𞁑𞁓|
 *     r2𞁑𞁓 = r𞁑𞁓²; v𞁑𞁓_dot_r𞁑𞁓 = v𞁑𞁓 · r𞁑𞁓; a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓 = (a vₚₕ𞁟ₛ𞁑𞁓) · r𞁑𞁓
 *     Q𞁑𞁓_over_m𞁑 = Q𞁑𞁓/m𞁑; Q𞁓𞁑_over_m𞁑 = Q𞁓𞁑/m𞁑; dv𞁑_dτ = dv𞁑/dτ; du𞁑_dτ = du𞁑/dτ
 *     du_dτ_Π𞁑𞁓 = (du/dτ)_Π𞁑𞁓; C𞁑𞁓_over_a = C𞁑𞁓/a
 *     du𞁑_dτ𞁞ₒₙ𞁑𞁓 = (du𞁑/dτ)𞁞ₒₙ𞁑𞁓; ΣC𞁑𞁓_over_a = Σ𞁓 C𞁑𞁓/a
 *     du𞁑_dτₕₑₐₜ𞁑𞁓 = (du𞁑/dτ)ₕₑₐₜ𞁑𞁓; du𞁓_dτₕₑₐₜ𞁑𞁓 = (du𞁓/dτ)ₕₑₐₜ𞁑𞁓
 *     du𞁑_dτₕₑₐₜ = (du𞁑/dτ)ₕₑₐₜ; dv𞁑_dτ_G = (dv𞁑/dτ)_G; h_M_over_h = h_M/h
 *     dv𞁑_dτₑₓₜ = (dv𞁑/dτ)ₑₓₜ; five_minus_3γ = 5 − 3γ; r2𞁑𞁓ₚᵣₑ = r𞁑𞁓² staged
 *     h𞁓ₚᵣₑ = h𞁓 staged.
 *   SIMD values append Pack; CUDA partial sums are reduced in place.
 */
// clang-format on
SphRunReturn sphCpuHydro(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    // Scale-factor reciprocal, gas mass scaling, γ − 1 and h_M/h.
    sph_flops::Counter flops = 4;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    float const ia = 1.0f / parameters.scaleFactor;
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every gas particle
    SimdFloat const γPack(γ);
    SimdFloat const γ_minus_1Pack(γ_minus_1);
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4 defaults
    // after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    // Calibration of the reversible-work bound δt ≤ C_cfl u𞁑/(k_u |du𞁑/dτ|): k_u equals the
    // cubic Monaghan viscosity width scale, which reproduces the former cubic-spline bound.
    constexpr float kThermalTimestepScale = float(kSpatialDimensions == 1 ?
            0.8660254037844386 :
            (kSpatialDimensions == 2 ? 0.8890008890013335 : 0.9128709291752769));
    static_assert(kThermalTimestepScale == float(monaghan::kViscosityWidthScale));
    float const smoothingLengthPerEffectiveRadius =
        selectedHydroKernel().smoothingLengthPerEffectiveRadius;
    SimdFloat const h_M_over_h(monaghan::lengthPerSupport(smoothingLengthPerEffectiveRadius));
    SimdFloat const zero(0.0f);
    SimdFloat const one(1.0f);
    SimdFloat const half(0.5f);
    float const m𞁓 = data.gasMass;    // every gas particle has the same mass
    SimdFloat const m𞁓_over_a(m𞁓 * ia);
    float const ε = parameters.viscosityEpsilon;    // η²/h² of Monaghan (1992), p. 551
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
         * Particle a. Section 3.5 of Monaghan (1992) leaves the equation of state free; the
         * ideal gas gives P𞁑 = (γ − 1) ρ𞁑 u𞁑 and c𞁑² = γ P𞁑/ρ𞁑 with the single configured γ.
         *
         * Symbol directory (in the code below):
         *   h𞁑 → support radius, 2h_M for the cubic spline
         *   P𞁑 → pressure
         *   u𞁑 → internal energy
         *   ρ𞁑 → density
         *   c𞁑 → sound speed
         *   γ → gas adiabatic index
         */
        // clang-format on
        float const h𞁑 = data.sourceSmoothingLength[particleIndex];
        float const ρ𞁑 = data.sourceDensity[particleIndex];
        float const u𞁑 = data.sourceInternalEnergy[particleIndex];
        ALWAYS_ASSERT(h𞁑 > 0.0f && ρ𞁑 > 0.0f && u𞁑 > 0.0f);
        float const P𞁑 = γ_minus_1 * ρ𞁑 * u𞁑;
        float const c𞁑 = std::sqrt(γ * γ_minus_1 * u𞁑);
        SpatialVector<float> const v𞁑 = data.sourceVelocity(particleIndex);
        SpatialVector<SimdFloat> const v𞁑Pack(v𞁑);
        float const α𞁑 = parameters.variableViscosity ?
            data.sourceBulkViscosityCoefficient[particleIndex] :
            parameters.bulkViscosityCoefficient;
        SimdFloat αPack(α𞁑);
        float const ih𞁑 = 1.0f / h𞁑;
        flops += 6 + sph_flops::kInverseGradientScale;
        // clang-format off
        /*
         * Variational operator (cosmictiger/sph/variable_smoothing_lengths.hpp): every
         * gas particle has the mass m, which is also its volume weight, so the weighted
         * density is ρ and Q𞁑𞁓/m𞁑 = (P𞁑/ρ𞁑²)(m − ζ𞁑). Comoving Q and kernel gradients
         * give the rates dv𞁑/dτ = −(1/a) Σ𞁓 (Q𞁑𞁓 g𞁑 + Q𞁓𞁑 g𞁓)/m𞁑 and
         * du𞁑/dτ = (1/a) Σ𞁓 Q𞁑𞁓 v𞁑𞁓·g𞁑/m𞁑.
         *
         * Symbol directory (in the code below):
         *   v𞁑𞁓 → velocity difference
         *   P𞁑 → pressure
         *   g𞁑 → sink kernel gradient (F𞁑 r𞁑𞁓)
         *   g𞁓 → neighbor kernel gradient (F𞁓 r𞁑𞁓)
         *   ζ𞁑 → grad-H correction
         */
        // clang-format on
        float const ζ𞁑 = data.sourceGradHCorrection[particleIndex];
        float const Q𞁑𞁓_over_m𞁑 = variational::directedCoefficient(
            variational::pressureVolumeFactor(P𞁑, 1.0f, ρ𞁑), m𞁓, ζ𞁑);
        flops += 5;
        float const ih_D_plus_1𞁑 = sph_geometry::inverseGradientScale(ih𞁑);    // h𞁑^-(D+1)
        SpatialVector<SimdFloat> dv𞁑_dτPack(zero);    // lane partial sums of acceleration
        SimdFloat du𞁑_dτₕₑₐₜPack(zero);
        SimdFloat ΣC𞁑𞁓_over_aPack(zero);
        SimdFloat du𞁑_dτPack(zero);     // lane partial sums of thermal work
        SimdFloat max𞁓μ𞁑𞁓Pack(zero);    // lane maxima of −μ𞁑𞁓 for δt𞁞ᵥ
        SimdFloat maximumNeighborRungPack(zero);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            // Particle b: one SIMD pack of staged neighbors. Lanes beyond the list, outside
            // both supports, or at zero separation are excluded. Their inputs are replaced
            // before any division or square root; masking only the result can still trap.
            SpatialVector<SimdFloat> const r𞁑𞁓 =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);
            SimdFloat const r2𞁑𞁓ₚᵣₑ = r𞁑𞁓.dot(r𞁑𞁓);
            SimdFloat const h𞁓ₚᵣₑ =
                detail::sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 1.0f);
            auto const supportsIntersect = detail::sphCpuValidLanes(neighbors, first) &
                (r2𞁑𞁓ₚᵣₑ < simdMax(SimdFloat(h𞁑 * h𞁑), sqr(h𞁓ₚᵣₑ)));
            if (parameters.midpointHydro && parameters.phase == 1) {
                SimdFloat const neighborRungs([&](auto lane) {
                    size_t const index = first + size_t(lane);
                    return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) :
                                                      0.0f;
                });
                maximumNeighborRungPack = simdMax(
                    maximumNeighborRungPack, simdSelect(supportsIntersect, neighborRungs, zero));
            }
            auto const interacts = supportsIntersect & (r2𞁑𞁓ₚᵣₑ > zero);
            // Fixed scales, dot operations and h𞁓² per lane; h𞁑² is scalar.
            flops +=
                (sph_flops::kVectorOperation + sph_flops::kDotProduct + 1) * SimdFloat::size() + 1;
            if (!simdAnyOf(interacts)) {
                continue;
            }
            SimdFloat const h𞁓 = simdSelect(interacts, h𞁓ₚᵣₑ, one);
            SimdFloat const ρ𞁓 = simdSelect(
                interacts, detail::sphCpuLoad(data.sourceDensity, neighbors, first, 1.0f), one);
            SimdFloat const u𞁓 = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceInternalEnergy, neighbors, first, 1.0f), one);
            SimdFloat const P𞁓 = γ_minus_1Pack * ρ𞁓 * u𞁓;
            SimdFloat const c𞁓 = simdSqrt(γPack * γ_minus_1Pack * u𞁓);
            SimdFloat const r2𞁑𞁓 = simdSelect(interacts, r2𞁑𞁓ₚᵣₑ, one);    // |r𞁑𞁓|²
            SimdFloat const abs_r𞁑𞁓 = simdSqrt(r2𞁑𞁓);                      // |r𞁑𞁓|
            // clang-format off
            /*
             * Symmetric kernel of Monaghan (1992), p. 554: ∇𞁑 W̃𞁑𞁓 = ½[∇𞁑 W(r𞁑𞁓, h𞁑) +
             * ∇𞁑 W(r𞁑𞁓, h𞁓)]. The selected kernel is W(r, h) = w(r/h)/h³, so each
             * gradient equals w'(r/h) r𞁑𞁓/(h⁴ r) and ∇𞁑 W̃𞁑𞁓 = F𞁑𞁓 r𞁑𞁓 with the scalar F𞁑𞁓.
             *
             * Symbol directory (in the code below):
             *   F𞁑𞁓 → symmetric gradient factor
             *   r𞁑𞁓 → separation
             */
            // clang-format on
            SimdFloat const ih𞁓 = one / h𞁓;
            SimdFloat const q𞁑 = abs_r𞁑𞁓 * SimdFloat(ih𞁑);
            SimdFloat const q𞁓 = abs_r𞁑𞁓 * ih𞁓;
            SimdFloat const F𞁑 = dkernelWDq(q𞁑) * SimdFloat(ih_D_plus_1𞁑) / abs_r𞁑𞁓;
            SimdFloat const F𞁓 = dkernelWDq(q𞁓) * sph_geometry::inverseGradientScale(ih𞁓) / abs_r𞁑𞁓;
            flops += sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑, true) +
                sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁓, true);
            SimdFloat const F𞁑𞁓 = simdSelect(interacts, half * (F𞁑 + F𞁓), zero);
            // v𞁑𞁓 = v𞁑 − v𞁓 from the stored peculiar velocities. The physical relative
            // velocity adds the Hubble flow (a da/dt) r𞁑𞁓 and enters only the viscosity.
            SpatialVector<SimdFloat> const v𞁓 = detail::sphCpuLoad3(data.sourceVelocityX,
                data.sourceVelocityY, data.sourceVelocityZ, neighbors, first, 0.0f);
            SpatialVector<SimdFloat> const v𞁑𞁓 =
                simdSelect(interacts, v𞁑Pack - v𞁓, SpatialVector<SimdFloat>(zero));
            SimdFloat const v𞁑𞁓_dot_r𞁑𞁓 = v𞁑𞁓.dot(r𞁑𞁓);
            SimdFloat const a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓 =
                v𞁑𞁓_dot_r𞁑𞁓 + SimdFloat(aȧ) * simdSelect(interacts, r2𞁑𞁓, zero);
            SimdFloat const h̄_M𞁑𞁓 = monaghan::pairLength(SimdFloat(h𞁑), h𞁓, h_M_over_h);
            SimdFloat const μ𞁑𞁓 =
                monaghan::viscosityMu(r2𞁑𞁓, a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓, h̄_M𞁑𞁓, SimdFloat(ε));
            SimdFloat const α𞁓 = parameters.variableViscosity ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourceBulkViscosityCoefficient, neighbors, first, α𞁑),
                    SimdFloat(α𞁑)) :
                SimdFloat(α𞁑);
            SimdFloat const α𞁑𞁓 =
                parameters.variableViscosity ? half * SimdFloat(α𞁑) + half * α𞁓 : SimdFloat(α𞁑);
            SimdFloat const β𞁑𞁓 = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                SimdFloat(parameters.vonNeumannRichtmyerCoefficient) * α𞁑𞁓 :
                SimdFloat(parameters.vonNeumannRichtmyerCoefficient);
            αPack = simdMax(αPack, simdSelect(interacts, α𞁑𞁓, SimdFloat(α𞁑)));
            flops += (3 * int(parameters.variableViscosity) +
                         int(parameters.vonNeumannRichtmyerCoefficientFollowsBulk())) *
                SimdFloat::size();
            SimdFloat fᴬⱽ𞁑𞁓(one);
            if (parameters.balsaraSwitch) {
                // Preserve the optional symmetric Balsara multiplier.
                fᴬⱽ𞁑𞁓 = half *
                    (SimdFloat(data.sourceBalsaraFactor[particleIndex]) +
                        detail::sphCpuLoad(data.sourceBalsaraFactor, neighbors, first, 0.0f));
                flops += 2 * SimdFloat::size();
            }
            // clang-format off
            /*
             * Hopkins (2013) equation 12 with Price (2012) equation 35: separate gradients
             * g𞁑 = F𞁑 r𞁑𞁓 and g𞁓 = F𞁓 r𞁑𞁓, the directed Q𞁑𞁓 and Q𞁓𞁑, and the same
             * Q𞁑𞁓 in the reversible work. The pressure work takes the peculiar velocity;
             * the analytic expansion term below supplies the Hubble-flow part exactly.
             * Viscous force and shock heating below use the same pair coefficients.
             *
             * Symbol directory (in the code below):
             *   r𞁑𞁓 → separation
             *   F𞁑 → sink gradient factor
             *   F𞁓 → neighbor gradient factor
             *   g𞁑 → sink kernel gradient (F𞁑 r𞁑𞁓)
             *   g𞁓 → neighbor kernel gradient (F𞁓 r𞁑𞁓)
             *   maskedF_a, maskedF_b → F𞁑, F𞁓 with excluded SIMD lanes zeroed
             */
            // clang-format on
            SimdFloat const ζ𞁓 = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceGradHCorrection, neighbors, first, 0.0f), zero);
            SimdFloat const Q𞁓𞁑_over_m𞁑 = variational::directedCoefficient(
                variational::pressureVolumeFactor(P𞁓, one, ρ𞁓), SimdFloat(m𞁓), ζ𞁓);
            SimdFloat const maskedF_a = simdSelect(interacts, F𞁑, zero);
            SimdFloat const maskedF_b = simdSelect(interacts, F𞁓, zero);
            SimdFloat const f_P𞁑𞁓 = SimdFloat(ia) *
                variational::pairForceFactor(
                    SimdFloat(Q𞁑𞁓_over_m𞁑), Q𞁓𞁑_over_m𞁑, maskedF_a, maskedF_b);
            SimdFloat Π𞁑𞁓 =
                monaghan::artificialViscosity(SimdFloat(ρ𞁑), ρ𞁓, SimdFloat(c𞁑), c𞁓, μ𞁑𞁓, α𞁑𞁓, β𞁑𞁓);
            if (parameters.balsaraSwitch) {
                Π𞁑𞁓 *= fᴬⱽ𞁑𞁓;
                flops += SimdFloat::size();
            }
            SimdFloat const f_Π𞁑𞁓 = m𞁓_over_a * Π𞁑𞁓 * F𞁑𞁓;
            SimdFloat const du_dτ_Π𞁑𞁓 = m𞁓_over_a * half * Π𞁑𞁓 * F𞁑𞁓 * a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓;
            // Pair length (3), μ𞁑𞁓, Π𞁑𞁓 and the force and heating factors.
            flops += (3 + sph_flops::kViscosityMu + sph_flops::kArtificialViscosity + 6) *
                SimdFloat::size();
            SimdFloat du𞁑_dτₕₑₐₜ𞁑𞁓 = du_dτ_Π𞁑𞁓;
            SimdFloat du𞁓_dτₕₑₐₜ𞁑𞁓 = du_dτ_Π𞁑𞁓;
            if (parameters.conductionCoefficient > 0.0f) {
                SimdFloat const C𞁑𞁓_over_a = sph_conduction::exchangeRate(SimdFloat(ρ𞁑), ρ𞁓,
                    SimdFloat(c𞁑), c𞁓, h̄_M𞁑𞁓, r2𞁑𞁓, a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓, F𞁑𞁓,
                    SimdFloat(parameters.conductionCoefficient), SimdFloat(ε), m𞁓_over_a);
                SimdFloat const du𞁑_dτ𞁞ₒₙ𞁑𞁓 = C𞁑𞁓_over_a * (u𞁓 - SimdFloat(u𞁑));
                du𞁑_dτₕₑₐₜ𞁑𞁓 += du𞁑_dτ𞁞ₒₙ𞁑𞁓;
                du𞁓_dτₕₑₐₜ𞁑𞁓 -= du𞁑_dτ𞁞ₒₙ𞁑𞁓;
                ΣC𞁑𞁓_over_aPack += C𞁑𞁓_over_a;
                flops += 23 * SimdFloat::size();    // 18 exchange + 5 heating/row-sum operations.
            }
            SpatialVector<SimdFloat> const f𞁑𞁓Pack = (f_P𞁑𞁓 + f_Π𞁑𞁓) * r𞁑𞁓;
            dv𞁑_dτPack -= f𞁑𞁓Pack;
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
                    SpatialVector<float> const a𞁑𞁓 = -detail::sphCpuLane(f𞁑𞁓Pack, lane);
                    float const Ȧ𞁑𞁓 = sph_entropy::fromStoredEnergy(
                        float(du𞁑_dτₕₑₐₜ𞁑𞁓[lane]), ρ𞁑, γ, parameters.scaleFactor);
                    float const Ȧ𞁓𞁑 = sph_entropy::fromStoredEnergy(
                        float(du𞁓_dτₕₑₐₜ𞁑𞁓[lane]), float(ρ𞁓[lane]), γ, parameters.scaleFactor);
                    flops += sphAccumulatePairKick(data.sourcePairUpdates[particleIndex],
                        data.sourcePairUpdates[neighborIndex], a𞁑𞁓, Ȧ𞁑𞁓, Ȧ𞁓𞁑, Δτ);
                    flops += 2 * sph_flops::entropyConversionFlops(γ);
                }
            }
            du𞁑_dτPack += SimdFloat(ia) *
                variational::pairWork(SimdFloat(Q𞁑𞁓_over_m𞁑), maskedF_a, v𞁑𞁓_dot_r𞁑𞁓);
            du𞁑_dτₕₑₐₜPack += du𞁑_dτₕₑₐₜ𞁑𞁓;
            // Common geometry, pressure, and accumulation; the vector terms are v𞁑𞁓, v𞁑𞁓 · r𞁑𞁓,
            // the pair force and its sum.
            flops += (38 + sph_flops::kInverseGradientScale + 3 * sph_flops::kVectorOperation +
                         sph_flops::kDotProduct) *
                SimdFloat::size();
            max𞁓μ𞁑𞁓Pack = simdMax(max𞁓μ𞁑𞁓Pack, -μ𞁑𞁓);
        }
        SpatialVector<float> dv𞁑_dτ = detail::sphCpuSum(dv𞁑_dτPack);
        float du𞁑_dτ = simdSum(du𞁑_dτPack);
        float const ΣC𞁑𞁓_over_a =
            parameters.conductionCoefficient > 0.0f ? simdSum(ΣC𞁑𞁓_over_aPack) : 0.0f;
        if (parameters.conductionCoefficient > 0.0f) {
            flops += SimdFloat::size() - 1;
        }
        // The acceleration components and the work; maxima are comparisons.
        flops += (sph_flops::kVectorOperation + 1) * (SimdFloat::size() - 1);
        // Corrections are derived, never checkpointed or migrated; slots that the density
        // pass has not rebuilt hold NaN, so a stale source cannot pass silently.
        if (!(sph_volume::isFiniteValue(dv𞁑_dτ.dot(dv𞁑_dτ)) && sph_volume::isFiniteValue(du𞁑_dτ) &&
                sph_volume::isFiniteValue(simdSum(du𞁑_dτₕₑₐₜPack)) &&
                sph_volume::isFiniteValue(ΣC𞁑𞁓_over_a))) {
            std::cout << std::format("SPH variational rates are not finite: smoothingLength={:e} "
                                     "density={:e} \u03b6={:e}\n",
                             h𞁑, ρ𞁑, ζ𞁑)
                      << std::flush;
            ALWAYS_ASSERT(false);
        }
        float const α = simdMaximum(αPack);
        float const β = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
            parameters.vonNeumannRichtmyerCoefficient * α :
            parameters.vonNeumannRichtmyerCoefficient;
        float const max𞁓μ𞁑𞁓 = simdMaximum(max𞁓μ𞁑𞁓Pack);    // max𞁓(−μ𞁑𞁓) over approaching pairs
        // The pair calculation can overlap gravity. Only source assembly and
        // the following corrector require the fresh endpoint acceleration.
        SpatialVector<float> dv𞁑_dτ_G(0.0f);
#if COSMICTIGER_WITH_GRAVITY
        if (!parameters.midpointHydro && !parameters.deferAccelerationSources &&
            data.selfGravityEnabled) {
            dv𞁑_dτ_G = data.sinkGravityAcceleration(sinkIndex);
        }
#endif
        if (!parameters.deferAccelerationSources) {
            dv𞁑_dτ = sphCompleteAcceleration(parameters, data.sourcePaddedPosition(particleIndex),
                v𞁑, dv𞁑_dτ, dv𞁑_dτ_G, data.centralAcceleration, data.centralSmoothingLength, flops);
        } else {
            ALWAYS_ASSERT(parameters.phase == 0);
        }
        if (applyPairKicks) {
            // External forces have no reciprocal gas endpoint. Integrate them
            // once on the target's own rung, separately from pair reductions.
            SpatialVector<float> const dv𞁑_dτₑₓₜ =
                sphCompleteAcceleration(parameters, data.sourcePaddedPosition(particleIndex), v𞁑,
                    SpatialVector<float>(0.0f), SpatialVector<float>(0.0f),
                    data.centralAcceleration, data.centralSmoothingLength, flops);
            flops +=
                sphAccumulateExternalKick(data.sourcePairUpdates[particleIndex], dv𞁑_dτₑₓₜ, Δτ);
        }
        // E=a^2 u and homogeneous adiabatic work du/dt=-3 H P/rho give
        // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E = (5-3 gamma) (da/dtau)/a E.
        // The variational pair work also takes the peculiar velocity: with Vᵢ = a^3 V^cᵢ
        // and V^cᵢ a function of comoving positions and supports, this term supplies the
        // Hubble-flow work exactly for free supports and for supports held at comoving
        // bounds (docs/variational_sph.md). Free supports also satisfy
        // sum_j (xⱼ - zeta_i) rᵢⱼ . gᵢ = -3 yᵢ, so physical pair velocities would count
        // that work twice; held supports have no such identity.
        float const five_minus_3γ = 5.0f - 3.0f * γ;
        du𞁑_dτ += five_minus_3γ * aȧ * ia * u𞁑;
        flops += 6;    // Homogeneous thermal expansion; force sources count separately.
        if (data.sinkEntropyRate) {
            // Reversible PdV and homogeneous expansion leave A unchanged.
            data.sinkEntropyRate[sinkIndex] = sph_entropy::fromStoredEnergy(
                simdSum(du𞁑_dτₕₑₐₜPack), ρ𞁑, γ, parameters.scaleFactor);
            flops += sph_flops::entropyConversionFlops(γ);
            flops += SimdFloat::size() - 1;
        }
        if (parameters.phase == 1) {
            // clang-format off
            /*
             * Monaghan (1992), section 10.3, p. 563:
             *   δt_f = min𞁑 (h̃𞁑/|f𞁑|),   δt𞁞ᵥ = min𞁑 h̃𞁑/(c𞁑 +
             *   0.6 (α c𞁑 + β max𞁓 μ𞁑𞁓)), δt = 0.25 min(δt_f, δt𞁞ᵥ).
             * h̃𞁑 is the common effective radius 2σ (docs/monaghan_sph.md), used in place of
             * h_M. The δt_f printed by Monaghan (1992) lacks a square root and is used as
             * sqrt(h̃𞁑/|f𞁑|); max𞁓 μ𞁑𞁓 is read as the compression magnitude max𞁓(−μ𞁑𞁓); the
             * Courant factor and the calibrated acceleration coefficient replace the fixed
             * 0.25. The reciprocals are kept apart for the diagnostics in SphRunReturn.
             *
             * Symbol directory (in the code below):
             *   μ𞁑𞁓 → viscosity velocity; max𞁓 μ𞁑𞁓 → maximum compression speed
             *   |f𞁑| → acceleration magnitude
             *   h̃𞁑 → effective radius; c𞁑 → sound speed
             *   δt → time step; δt_G → gravity time step
             *   δt_f⁻¹, δt𞁞ᵥ⁻¹ → inverse acceleration time step, inverse Courant time step
             *   δt𞁞⁻¹, δtᵥ⁻¹ → sound and viscosity terms of δt𞁞ᵥ⁻¹
             *   α → maximum bulk viscosity coefficient
             *   β → maximum von Neumann–Richtmyer coefficient
             *   Code names: abs_f𞁑 = |f𞁑|; iδt_f = δt_f⁻¹
             *     iδt𞁞ᵥ = δt𞁞ᵥ⁻¹; iδt𞁞 = δt𞁞⁻¹; iδtᵥ = δtᵥ⁻¹.
             */
            // clang-format on
            float const h̃𞁑 = h𞁑 / smoothingLengthPerEffectiveRadius;
            float const iδt𞁞 = c𞁑 / h̃𞁑;
            float const iδtᵥ = 0.6f * (α * c𞁑 + β * max𞁓μ𞁑𞁓) / h̃𞁑;
            float const iδt𞁞ᵥ = iδt𞁞 + iδtᵥ;
            float const abs_f𞁑 = dv𞁑_dτ.norm();
            float const iδt_f = parameters.courantFactor *
                std::sqrt(parameters.scaleFactor * abs_f𞁑 / h̃𞁑) /
                (data.accelerationTimestepFactor * kHydroAccelerationScale);
            float δt = std::fmin(parameters.maximumTimestep,
                parameters.courantFactor * parameters.scaleFactor /
                    std::fmax(std::fmax(iδt𞁞ᵥ, iδt_f), 1.0e-30f));
            // Bound reversible work; the approach-speed CFL term controls viscous heating.
            if (du𞁑_dτ != 0.0f) {
                δt = std::fmin(
                    δt, parameters.courantFactor / kThermalTimestepScale * u𞁑 / std::fabs(du𞁑_dτ));
                flops += 3;    // Two divisions and a product; abs is a sign operation.
            }
            // Derived forward-Euler diffusion bound: δt Σ𞁓 C𞁑𞁓/a ≤ 1/2.
            // It uses the sum of positive exchange rates, even when net heating cancels.
            if (ΣC𞁑𞁓_over_a > 0.0f) {
                δt = std::fmin(δt, 0.5f / ΣC𞁑𞁓_over_a);
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
            float δt_G = parameters.maximumTimestep;
            int gravityRung = 0;
            if (!parameters.midpointHydro) {
                δt_G = std::fmin(parameters.maximumTimestep,
                    data.accelerationTimestepFactor *
                        std::sqrt(parameters.scaleFactor * data.gravityEffectiveRadius /
                            std::fmax(dv𞁑_dτ_G.norm(), 1.0e-30f)));
                if (data.selfGravityEnabled) {
                    result.minimumTimestep = std::min(result.minimumTimestep, double(δt_G));
                    flops.doublePrecision += 1;
                }
                gravityRung = parameters.globalTimestep ?
                    0 :
                    std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / δt_G))));
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
                                     std::fmax(iδt𞁞ᵥ, 1.0e-30f),
                                 parameters.courantFactor * parameters.scaleFactor /
                                     std::fmax(iδt_f, 1.0e-30f),
                                 du𞁑_dτ != 0.0f ? parameters.courantFactor / kThermalTimestepScale *
                                         u𞁑 / std::fabs(du𞁑_dτ) :
                                                  parameters.maximumTimestep,
                                 h𞁑, ρ𞁑, u𞁑, du𞁑_dτ, abs_f𞁑, c𞁑, max𞁓μ𞁑𞁓, α, m𞁓 / (m𞁓 - ζ𞁑),
                                 parameters.scaleFactor)
                          << std::flush;
#if COSMICTIGER_WITH_GRAVITY
                std::cout
                    << std::format(
                           "SPH gravity timestep: enabled={:d} gravityRung={:d} gravityDt={:e}\n",
                           int(data.selfGravityEnabled), gravityRung, δt_G)
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
            result.maximumSignalSpeed = std::fmax(result.maximumSignalSpeed, iδt𞁞ᵥ * h̃𞁑);
            result.inverseSoundTimestep = std::fmax(result.inverseSoundTimestep, iδt𞁞);
            result.inverseViscosityTimestep = std::fmax(result.inverseViscosityTimestep, iδtᵥ);
            result.inverseAccelerationTimestep =
                std::fmax(result.inverseAccelerationTimestep, iδt_f);
            // h̃/c/visc (8), acceleration norm and limit (6), timestep (2), hydro rung (3).
            // Exclude the signal-speed diagnostic.
            flops += sph_flops::kNorm + (parameters.globalTimestep ? 16 : 19);
        }
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}
