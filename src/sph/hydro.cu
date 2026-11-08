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
    SpatialVector<Position> position;
    float smoothingLength;
    SpatialVector<float> velocity;
    float internalEnergy;
    float density;
    float balsaraFactor;
    float bulkViscosityCoefficient;
    float gradHCorrection;    // ζ𞁓 from the current density pass.
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
 *   x𞁑 → sink position; paddedX_b → neighbor position as the record of the box tests
 *   v𞁑, v𞁓 → sink stored velocity, neighbor stored velocity
 *   ρ𞁑, P𞁑, u𞁑, c𞁑 → density, pressure, internal energy, sound speed
 *   ρ𞁓, P𞁓, u𞁓, c𞁓 → neighbor density, pressure, internal energy, sound speed
 *   h𞁑, h𞁓 → sink kernel smoothing length, neighbor smoothing length
 *   h𞁑⁻¹, h𞁓⁻¹ → inverse smoothing lengths; h𞁑⁻⁽ᴰ⁺¹⁾ → sink inverse gradient scale
 *   r𞁑𞁓, |r𞁑𞁓|, r𞁑𞁓² → separation, pair distance, squared separation
 *   v𞁑𞁓 → velocity difference; v𞁑𞁓 · r𞁑𞁓 → velocity dot separation
 *   (a vₚₕ𞁟ₛ𞁑𞁓) · r𞁑𞁓 → velocity dot separation including the Hubble flow a ȧ r𞁑𞁓
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
 *     dv𞁑_dτₑₓₜ = (dv𞁑/dτ)ₑₓₜ; five_minus_3γ = 5 − 3γ.
 *   SIMD values append Pack; CUDA partial sums are reduced in place.
 */
// clang-format on
__global__ void sphCudaHydro(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    // Inverse scale factor, gas mass scaling, γ − 1 and h_M/h.
    sph_flops::Counter flops = 4;
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int targetNodeIndex;
    __shared__ detail::HydroWorkspace workspace;
    new (&workspace) detail::HydroWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    float const ia = 1.0f / parameters.scaleFactor;
    float const γ = data.adiabaticIndex;
    float const γ_minus_1 = γ - 1.0f;    // P/(ρu) for every particle.
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4
    // defaults after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    // Calibration of the reversible-work bound δt ≤ C_cfl u𞁑/(k_u |du𞁑/dτ|): k_u equals the
    // cubic Monaghan viscosity width scale, which reproduces the former cubic-spline bound.
    constexpr float kThermalTimestepScale = float(kSpatialDimensions == 1 ?
            0.8660254037844386 :
            (kSpatialDimensions == 2 ? 0.8890008890013335 : 0.9128709291752769));
    static_assert(kThermalTimestepScale == float(monaghan::kViscosityWidthScale));
    float const smoothingLengthPerEffectiveRadius =
        selectedHydroKernel().smoothingLengthPerEffectiveRadius;
    float const h_M_over_h = monaghan::lengthPerSupport(smoothingLengthPerEffectiveRadius);
    float const m𞁓 = data.gasMass;    // every gas particle has the same mass
    float const m𞁓_over_a = m𞁓 * ia;
    float const ε = parameters.viscosityEpsilon;    // η²/h² of Monaghan (1992), p. 551
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
                    auto& neighbor = workspace.neighbors[offset + slot];
                    neighbor.position = data.sourcePosition(sourceParticleIndex);
                    neighbor.smoothingLength = data.sourceSmoothingLength[sourceParticleIndex];
                    neighbor.velocity = data.sourceVelocity(sourceParticleIndex);
                    neighbor.internalEnergy = data.sourceInternalEnergy[sourceParticleIndex];
                    neighbor.density = data.sourceDensity[sourceParticleIndex];
                    neighbor.balsaraFactor = parameters.balsaraSwitch ?
                        data.sourceBalsaraFactor[sourceParticleIndex] :
                        1.0f;
                    neighbor.bulkViscosityCoefficient = parameters.variableViscosity ?
                        data.sourceBulkViscosityCoefficient[sourceParticleIndex] :
                        parameters.bulkViscosityCoefficient;
                    neighbor.gradHCorrection = data.sourceGradHCorrection[sourceParticleIndex];
                    neighbor.rung =
                        parameters.midpointHydro ? int(data.sourceRungs[sourceParticleIndex]) : 0;
                    neighbor.sourceIndex = sourceParticleIndex;
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
            float const c𞁑 = sqrtf(γ * γ_minus_1 * u𞁑);
            SpatialVector<Position> const x𞁑 = data.sourcePosition(particleIndex);
            SpatialVector<float> const v𞁑 = data.sourceVelocity(particleIndex);
            float const ih𞁑 = 1.0f / h𞁑;
            float const ih_D_plus_1𞁑 = sph_geometry::inverseGradientScale(ih𞁑);    // h𞁑^-(D+1)
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
            SpatialVector<float> dv𞁑_dτ(0.0f);    // this thread's acceleration sum
            float du𞁑_dτ = 0.0f;
            float du𞁑_dτₕₑₐₜ = 0.0f;    // this thread's viscous/conductive sum
            float ΣC𞁑𞁓_over_a = 0.0f;
            float const α𞁑 = parameters.variableViscosity ?
                data.sourceBulkViscosityCoefficient[particleIndex] :
                parameters.bulkViscosityCoefficient;
            float α = α𞁑;
            float max𞁓μ𞁑𞁓 = 0.0f;    // max𞁓(−μ𞁑𞁓) for δt𞁞ᵥ
            int maximumNeighborRung = 0;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.neighbors.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.neighbors[neighborIndex];
                // Particle b. Pairs outside both supports or at zero separation are skipped.
                auto const r𞁑𞁓 = distance(x𞁑, neighbor.position);
                float const r2𞁑𞁓 = r𞁑𞁓.dot(r𞁑𞁓);    // |r𞁑𞁓|²
                float const h𞁓 = neighbor.smoothingLength;
                // Distances, dot, support squares.
                flops +=
                    sph_flops::kVectorOperation + sph_flops::kDotProduct + (r2𞁑𞁓 == 0.0f ? 0 : 2);
                bool const supportsIntersect = r2𞁑𞁓 < fmaxf(h𞁑 * h𞁑, h𞁓 * h𞁓);
                if (parameters.midpointHydro && parameters.phase == 1 && supportsIntersect) {
                    maximumNeighborRung = max(maximumNeighborRung, neighbor.rung);
                }
                if (r2𞁑𞁓 == 0.0f || !supportsIntersect) {
                    continue;
                }
                float const ρ𞁓 = neighbor.density;
                float const u𞁓 = neighbor.internalEnergy;
                float const P𞁓 = γ_minus_1 * ρ𞁓 * u𞁓;
                float const c𞁓 = sqrtf(γ * γ_minus_1 * u𞁓);
                float const abs_r𞁑𞁓 = sqrtf(r2𞁑𞁓);    // |r𞁑𞁓|
                // clang-format off
                /*
                 * Symmetric kernel of Monaghan (1992), p. 554: ∇𞁑 W̃𞁑𞁓 = ½[∇𞁑 W(r𞁑𞁓, h𞁑) +
                 * ∇𞁑 W(r𞁑𞁓, h𞁓)]. The selected kernel is W(r, h) = w(r/h)/h³, so
                 * each gradient equals w'(r/h) r𞁑𞁓/(h⁴ r) and ∇𞁑 W̃𞁑𞁓 = F𞁑𞁓 r𞁑𞁓 with the scalar
                 * F𞁑𞁓.
                 *
                 * Symbol directory (in the code below):
                 *   F𞁑𞁓 → symmetric gradient factor
                 *   r𞁑𞁓 → separation
                 */
                // clang-format on
                float const ih𞁓 = 1.0f / h𞁓;
                float const q𞁑 = abs_r𞁑𞁓 * ih𞁑;
                float const q𞁓 = abs_r𞁑𞁓 * ih𞁓;
                float const F𞁑 = dkernelWDq(q𞁑) * ih_D_plus_1𞁑 / abs_r𞁑𞁓;
                float const F𞁓 = dkernelWDq(q𞁓) * sph_geometry::inverseGradientScale(ih𞁓) / abs_r𞁑𞁓;
                float const F𞁑𞁓 = 0.5f * (F𞁑 + F𞁓);
                // v𞁑𞁓 = v𞁑 − v𞁓 from the stored peculiar velocities. The physical relative
                // velocity adds the Hubble flow (a da/dt) r𞁑𞁓 and enters only the viscosity.
                SpatialVector<float> const& v𞁓 = neighbor.velocity;
                SpatialVector<float> const v𞁑𞁓 = v𞁑 - v𞁓;
                float const v𞁑𞁓_dot_r𞁑𞁓 = v𞁑𞁓.dot(r𞁑𞁓);
                float const a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓 = v𞁑𞁓_dot_r𞁑𞁓 + aȧ * r2𞁑𞁓;
                float const h̄_M𞁑𞁓 = monaghan::pairLength(h𞁑, h𞁓, h_M_over_h);
                float const μ𞁑𞁓 = monaghan::viscosityMu(r2𞁑𞁓, a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓, h̄_M𞁑𞁓, ε);
                float const α𞁓 =
                    parameters.variableViscosity ? neighbor.bulkViscosityCoefficient : α𞁑;
                float const α𞁑𞁓 = parameters.variableViscosity ? 0.5f * α𞁑 + 0.5f * α𞁓 : α𞁑;
                float const β𞁑𞁓 = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                    parameters.vonNeumannRichtmyerCoefficient * α𞁑𞁓 :
                    parameters.vonNeumannRichtmyerCoefficient;
                α = fmaxf(α, α𞁑𞁓);
                flops += 3 * int(parameters.variableViscosity) +
                    int(parameters.vonNeumannRichtmyerCoefficientFollowsBulk());
                float fᴬⱽ𞁑𞁓 = 1.0f;
                if (parameters.balsaraSwitch) {
                    fᴬⱽ𞁑𞁓 =
                        0.5f * (data.sourceBalsaraFactor[particleIndex] + neighbor.balsaraFactor);
                    flops += 2;
                }
                // clang-format off
                /*
                 * Hopkins (2013) equation 12 with Price (2012) equation 35: separate
                 * gradients g𞁑 = F𞁑 r𞁑𞁓 and g𞁓 = F𞁓 r𞁑𞁓, the directed Q𞁑𞁓 and Q𞁓𞁑, and
                 * the same Q𞁑𞁓 in the reversible work, with the peculiar velocity; the
                 * analytic expansion term supplies the Hubble-flow work exactly. The
                 * viscous force and heating below share the same pair coefficients.
                 *
                 * Symbol directory (in the code below):
                 *   r𞁑𞁓 → separation
                 *   F𞁑 → sink gradient factor
                 *   F𞁓 → neighbor gradient factor
                 *   g𞁑 → sink kernel gradient (F𞁑 r𞁑𞁓)
                 *   g𞁓 → neighbor kernel gradient (F𞁓 r𞁑𞁓)
                 */
                // clang-format on
                float const ζ𞁓 = neighbor.gradHCorrection;
                float const Q𞁓𞁑_over_m𞁑 = variational::directedCoefficient(
                    variational::pressureVolumeFactor(P𞁓, 1.0f, ρ𞁓), m𞁓, ζ𞁓);
                float const f_P𞁑𞁓 =
                    ia * variational::pairForceFactor(Q𞁑𞁓_over_m𞁑, Q𞁓𞁑_over_m𞁑, F𞁑, F𞁓);
                float Π𞁑𞁓 = monaghan::artificialViscosity(ρ𞁑, ρ𞁓, c𞁑, c𞁓, μ𞁑𞁓, α𞁑𞁓, β𞁑𞁓);
                if (parameters.balsaraSwitch) {
                    Π𞁑𞁓 *= fᴬⱽ𞁑𞁓;
                    flops += 1;
                }
                float const f_Π𞁑𞁓 = m𞁓_over_a * Π𞁑𞁓 * F𞁑𞁓;
                float const du_dτ_Π𞁑𞁓 = m𞁓_over_a * 0.5f * Π𞁑𞁓 * F𞁑𞁓 * a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓;
                // Pair length (3), μ𞁑𞁓, Π𞁑𞁓 and the force and heating factors.
                flops += 3 + sph_flops::kViscosityMu + sph_flops::kArtificialViscosity + 6;
                float du𞁑_dτₕₑₐₜ𞁑𞁓 = du_dτ_Π𞁑𞁓;
                float du𞁓_dτₕₑₐₜ𞁑𞁓 = du_dτ_Π𞁑𞁓;
                if (parameters.conductionCoefficient > 0.0f) {
                    float const C𞁑𞁓_over_a =
                        sph_conduction::exchangeRate(ρ𞁑, ρ𞁓, c𞁑, c𞁓, h̄_M𞁑𞁓, r2𞁑𞁓, a_vₚₕ𞁟ₛ𞁑𞁓_dot_r𞁑𞁓,
                            F𞁑𞁓, parameters.conductionCoefficient, ε, m𞁓_over_a);
                    float const du𞁑_dτ𞁞ₒₙ𞁑𞁓 = C𞁑𞁓_over_a * (u𞁓 - u𞁑);
                    du𞁑_dτₕₑₐₜ𞁑𞁓 += du𞁑_dτ𞁞ₒₙ𞁑𞁓;
                    du𞁓_dτₕₑₐₜ𞁑𞁓 -= du𞁑_dτ𞁞ₒₙ𞁑𞁓;
                    ΣC𞁑𞁓_over_a += C𞁑𞁓_over_a;
                    flops += 23;    // 18 exchange + 5 heating/row-sum operations.
                }
                SpatialVector<float> const f𞁑𞁓 = (f_P𞁑𞁓 + f_Π𞁑𞁓) * r𞁑𞁓;
                dv𞁑_dτ -= f𞁑𞁓;
                if (applyPairKicks &&
                    parameters.isActiveRung(max(previousRung, int(neighbor.rung))) &&
                    sphOwnsPairKick(data.sourceIds[particleIndex],
                        data.sourceIds[neighbor.sourceIndex], previousRung, neighbor.rung)) {
                    SpatialVector<float> const a𞁑𞁓 = -f𞁑𞁓;
                    float const Ȧ𞁑𞁓 =
                        sph_entropy::fromStoredEnergy(du𞁑_dτₕₑₐₜ𞁑𞁓, ρ𞁑, γ, parameters.scaleFactor);
                    float const Ȧ𞁓𞁑 =
                        sph_entropy::fromStoredEnergy(du𞁓_dτₕₑₐₜ𞁑𞁓, ρ𞁓, γ, parameters.scaleFactor);
                    flops += sphAccumulatePairKick(data.sourcePairUpdates[particleIndex],
                        data.sourcePairUpdates[neighbor.sourceIndex], a𞁑𞁓, Ȧ𞁑𞁓, Ȧ𞁓𞁑, Δτ);
                    flops += 2 * sph_flops::entropyConversionFlops(γ);
                }
                du𞁑_dτ += ia * variational::pairWork(Q𞁑𞁓_over_m𞁑, F𞁑, v𞁑𞁓_dot_r𞁑𞁓);
                du𞁑_dτₕₑₐₜ += du𞁑_dτₕₑₐₜ𞁑𞁓;
                // Common geometry, pressure, and accumulation; the vector terms are v𞁑𞁓,
                // v𞁑𞁓 · r𞁑𞁓, the pair force and its sum.
                flops += 38 + sph_flops::kInverseGradientScale + 3 * sph_flops::kVectorOperation +
                    sph_flops::kDotProduct;
                max𞁓μ𞁑𞁓 = fmaxf(max𞁓μ𞁑𞁓, -μ𞁑𞁓);
                flops += sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁑, true) +
                    sph_flops::densityKernelOperations(selectedHydroKernel(), q𞁓, true);
            }
            for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                sharedReduceAdd<float, kHydroBlockSize>(dv𞁑_dτ[dim]);
            }
            sharedReduceAdd<float, kHydroBlockSize>(du𞁑_dτ);
            sharedReduceAdd<float, kHydroBlockSize>(du𞁑_dτₕₑₐₜ);
            if (parameters.conductionCoefficient > 0.0f) {
                sharedReduceAdd<float, kHydroBlockSize>(ΣC𞁑𞁓_over_a);
                flops += detail::sphReductionFlops<kHydroBlockSize>();
            }
            sharedReduceMax<kHydroBlockSize>(max𞁓μ𞁑𞁓);
            sharedReduceMax<kHydroBlockSize>(α);
            if (parameters.midpointHydro && parameters.phase == 1) {
                sharedReduceMax<kHydroBlockSize>(maximumNeighborRung);
            }
            flops +=
                (sph_flops::kVectorOperation + 2) * detail::sphReductionFlops<kHydroBlockSize>();
            if (threadIndex == 0) {
                // Corrections are derived, never checkpointed or migrated; slots that the
                // density pass has not rebuilt hold NaN, so a stale source cannot pass silently.
                if (!(sph_volume::isFiniteValue(dv𞁑_dτ.dot(dv𞁑_dτ)) &&
                        sph_volume::isFiniteValue(du𞁑_dτ) &&
                        sph_volume::isFiniteValue(du𞁑_dτₕₑₐₜ) &&
                        sph_volume::isFiniteValue(ΣC𞁑𞁓_over_a))) {
                    printf("SPH variational rates are not finite: smoothingLength=%e density=%e "
                           "ζ=%e\n",
                        h𞁑, ρ𞁑, ζ𞁑);
                    __trap();
                }
                SpatialVector<float> dv𞁑_dτ_G(0.0f);
#if COSMICTIGER_WITH_GRAVITY
                if (!parameters.midpointHydro && !parameters.deferAccelerationSources &&
                    data.selfGravityEnabled) {
                    dv𞁑_dτ_G = data.sinkGravityAcceleration(sinkIndex);
                }
#endif
                if (!parameters.deferAccelerationSources) {
                    dv𞁑_dτ = sphCompleteAcceleration(parameters,
                        data.sourcePaddedPosition(particleIndex), v𞁑, dv𞁑_dτ, dv𞁑_dτ_G,
                        data.centralAcceleration, data.centralSmoothingLength, flops);
                } else {
                    ALWAYS_ASSERT(parameters.phase == 0);
                }
                if (applyPairKicks) {
                    SpatialVector<float> const dv𞁑_dτₑₓₜ = sphCompleteAcceleration(parameters,
                        data.sourcePaddedPosition(particleIndex), v𞁑, SpatialVector<float>(0.0f),
                        SpatialVector<float>(0.0f), data.centralAcceleration,
                        data.centralSmoothingLength, flops);
                    flops += sphAccumulateExternalKick(
                        data.sourcePairUpdates[particleIndex], dv𞁑_dτₑₓₜ, Δτ);
                }
                // E=a^2 u and homogeneous work du/dt=-3 H P/rho give
                // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E = (5-3 gamma) (da/dtau)/a E.
                float const five_minus_3γ = 5.0f - 3.0f * γ;
                du𞁑_dτ += five_minus_3γ * aȧ * ia * u𞁑;
                flops += 6;    // Homogeneous thermal expansion; force sources count separately.
                if (data.sinkEntropyRate) {
                    // Reversible PdV and homogeneous expansion leave A unchanged.
                    data.sinkEntropyRate[sinkIndex] =
                        sph_entropy::fromStoredEnergy(du𞁑_dτₕₑₐₜ, ρ𞁑, γ, parameters.scaleFactor);
                    flops += sph_flops::entropyConversionFlops(γ);
                }
                if (parameters.phase == 1) {
                    // clang-format off
                    /*
                     * Monaghan (1992), section 10.3, p. 563:
                     *   δt_f = min𞁑 (h̃𞁑/|f𞁑|),
                     *   δt𞁞ᵥ = min𞁑 h̃𞁑/(c𞁑 + 0.6 (α c𞁑 + β max𞁓 μ𞁑𞁓)),
                     *   δt = 0.25 min(δt_f, δt𞁞ᵥ).
                     * h̃𞁑 is the common effective radius 2σ (docs/monaghan_sph.md), used in
                     * place of h_M. The δt_f printed by Monaghan (1992) lacks a square
                     * root and is used as sqrt(h̃𞁑/|f𞁑|); max𞁓 μ𞁑𞁓 is read as max𞁓(−μ𞁑𞁓);
                     * the Courant factor and the calibrated acceleration coefficient replace
                     * the fixed 0.25. The reciprocals are kept apart for the diagnostics.
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
                    float const β = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                        parameters.vonNeumannRichtmyerCoefficient * α :
                        parameters.vonNeumannRichtmyerCoefficient;
                    float const iδtᵥ = 0.6f * (α * c𞁑 + β * max𞁓μ𞁑𞁓) / h̃𞁑;
                    float const iδt𞁞ᵥ = iδt𞁞 + iδtᵥ;
                    float const abs_f𞁑 = dv𞁑_dτ.norm();
                    float const iδt_f = parameters.courantFactor *
                        sqrtf(parameters.scaleFactor * abs_f𞁑 / h̃𞁑) /
                        (data.accelerationTimestepFactor * kHydroAccelerationScale);
                    float δt = fminf(parameters.maximumTimestep,
                        parameters.courantFactor * parameters.scaleFactor /
                            fmaxf(fmaxf(iδt𞁞ᵥ, iδt_f), 1.0e-30f));
                    // Bound reversible work; the approach-speed CFL term controls viscosity.
                    if (du𞁑_dτ != 0.0f) {
                        flops += 3;
                        δt = fminf(δt,
                            parameters.courantFactor / kThermalTimestepScale * u𞁑 / fabsf(du𞁑_dτ));
                    }
                    // Derived positivity bound, matching the CPU sum over all neighbors.
                    if (ΣC𞁑𞁓_over_a > 0.0f) {
                        δt = fminf(δt, 0.5f / ΣC𞁑𞁓_over_a);
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
                    float δt_G = parameters.maximumTimestep;
                    int gravityRung = 0;
                    if (!parameters.midpointHydro) {
                        δt_G = fminf(parameters.maximumTimestep,
                            data.accelerationTimestepFactor *
                                sqrtf(parameters.scaleFactor * data.gravityEffectiveRadius /
                                    fmaxf(dv𞁑_dτ_G.norm(), 1.0e-30f)));
                        if (data.selfGravityEnabled) {
                            atomicMin(&reduction->minimumTimestep, δt_G);
                        }
                        gravityRung = parameters.globalTimestep ?
                            0 :
                            max(1, int(ceilf(log2f(parameters.baseTimestep / δt_G))));
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
                                fmaxf(iδt𞁞ᵥ, 1.0e-30f),
                            parameters.courantFactor * parameters.scaleFactor /
                                fmaxf(iδt_f, 1.0e-30f),
                            du𞁑_dτ != 0.0f ? parameters.courantFactor / kThermalTimestepScale * u𞁑 /
                                    fabsf(du𞁑_dτ) :
                                             parameters.maximumTimestep,
                            h𞁑, ρ𞁑, u𞁑, du𞁑_dτ, abs_f𞁑, c𞁑, max𞁓μ𞁑𞁓, α, m𞁓 / (m𞁓 - ζ𞁑),
                            parameters.scaleFactor);
#if COSMICTIGER_WITH_GRAVITY
                        printf("SPH gravity timestep: enabled=%i gravityRung=%i gravityDt=%e\n",
                            int(data.selfGravityEnabled), gravityRung, δt_G);
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
                    maximumSignalSpeed = fmaxf(maximumSignalSpeed, iδt𞁞ᵥ * h̃𞁑);
                    // Exclude the signal-speed report.
                    flops += sph_flops::kNorm + (parameters.globalTimestep ? 16 : 19);
                    atomicMax(&reduction->inverseSoundTimestep, iδt𞁞);
                    atomicMax(&reduction->inverseViscosityTimestep, iδtᵥ);
                    atomicMax(&reduction->inverseAccelerationTimestep, iδt_f);
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
