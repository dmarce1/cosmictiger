/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/conduction.hpp>
#include <cosmictiger/sph/cuda.hpp>
#include <cosmictiger/sph/variable_smoothing_lengths.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/viscosity/price_etal.hpp>
#include <cosmictiger/sph/volume.hpp>

namespace detail {

struct HydroNeighbor {
    Position positionX;
    Position positionY;
    Position positionZ;
    float smoothingLength;
    float velocityX;
    float velocityY;
    float velocityZ;
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
 * citation. Price mode instead uses Phantom (2018), equations (39)-(42), below.
 * Particle a is the sink and particle b a staged neighbor; the computation
 * uses the papers' symbols, described below. Positions and density are comoving,
 * and stored velocities and internal energies carry the factors a and a² of
 * docs/variational_sph.md.
 */

// clang-format off
/*
 * Production pair equations: Hopkins (2013), MNRAS 428, 2840, equations (12)–(13)
 * (https://arxiv.org/abs/1206.5006), and Price (2012), JCP 231, 759, (30), (35)
 * (https://arxiv.org/abs/1012.1885). Equal masses, x_a = m_a, y_a = ρ_a:
 *   Q_ab/m_a = P_a(m_b − ζ_a)/ρ_a²,
 *   dv_a/dτ = −a⁻¹ Σ_b [(Q_ab/m_a)F_a + (Q_ba/m_a)F_b] r_ab,
 *   du_a/dτ = a⁻¹ Σ_b (Q_ab/m_a)F_a (v_ab · r_ab),
 *   ∇_a W_ab(H_a) = F_a r_ab, F_a = w′(r_ab/H_a)/(H_a⁴ |r_ab|).
 * These are derived comoving forms; viscosity is added in the model branches below.
 * Symbol directory:
 *   a → scale factor; m_b → gas mass; γ → gas adiabatic index
 *   ρ_a, P_a, u_a → density, pressure, internal energy
 *   H_a, H_b → sink kernel smoothing length, neighbor smoothing length
 *   r_ab, |r_ab| → separation, pair distance; v_ab → velocity difference
 *   ζ_a → grad-H correction; Q_ab/m_a, Q_ba/m_a → sink pressure coefficient, neighbor pressure coefficient
 *   F_a, F_b → sink gradient factor, neighbor gradient factor
 *   dv_a/dτ, du_a/dτ → acceleration, internal energy rate
 *   α_a, α_b, α_ab → bulk viscosity coefficient, neighbor bulk viscosity coefficient, pair bulk viscosity coefficient
 *   β_ab → pair von Neumann–Richtmyer coefficient; Π_ab → pair artificial viscosity
 *   μ_ab → viscosity velocity (Phantom instead stores min(v_ab · r̂_ab, 0) here)
 *   q_abᵃ, q_abᵇ → sink viscous pressure, neighbor viscous pressure
 *   C_ab, C_ba → sink viscosity coefficient, neighbor viscosity coefficient
 *   Code names: abs_r_ab = |r_ab|; Q_ab_over_m_a = Q_ab/m_a; Q_ba_over_m_a = Q_ba/m_a;
 *     dv_a_dτ = dv_a/dτ; du_a_dτ = du_a/dτ.
 *   SIMD partial sums append Pack; CUDA partial sums are reduced in place.
 */
// clang-format on
namespace detail {

template <bool retainHydroDiagnostics>
__device__ __forceinline__ void sphCudaHydroImpl(SphRunParameters parameters, SphRunCudaData data,
    SphReduction* reduction, HydroWorkspace& workspace, int& targetNodeIndex) {
    sph_flops::Counter flops = 3;    // Inverse scale factor, gas mass scaling and γ − 1.
    int const threadIndex = threadIdx.x;
    int const blockSize = blockDim.x;
    new (&workspace) detail::HydroWorkspace();
    if (threadIndex == 0) {
        targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
    }
    __syncthreads();
    float const inverseScaleFactor = 1.0f / parameters.scaleFactor;
    float const γ = data.adiabaticIndex;
    float const pressureEnergyRatio = γ - 1.0f;    // P/(ρu) for every particle.
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4
    // defaults after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    float const smoothingLengthPerEffectiveRadius =
        selectedHydroKernel().smoothingLengthPerEffectiveRadius;
    float const m_b = data.gasMass;    // every gas particle has the same mass
    float const scaledGasMass = m_b * inverseScaleFactor;
    float const viscosityRegularization = parameters.viscosityEpsilon;    // η²/h² of p. 551
    float const expansionRate = parameters.expansionRate;    // a·da/dt, the Hubble flow of a pair
    bool const applyPairKicks = data.sourcePairUpdates && parameters.hydroKickTimestep > 0.0;
    float const kickTimestep = applyPairKicks ? float(parameters.hydroKickTimestep) : 0.0f;
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
                Vector3<Position> position;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    position[kXi] = data.sourcePositionX[sourceParticleIndex];
                    position[kYi] = data.sourcePositionY[sourceParticleIndex];
                    position[kZi] = data.sourcePositionZ[sourceParticleIndex];
                    contains = targetNode.outerBox.contains(position, &flops.doublePrecision);
                    if (!contains) {
                        contains = sphSupportIntersects(targetNode.innerBox, position,
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
                    workspace.neighbors[offset + slot] = {position[kXi], position[kYi],
                        position[kZi], data.sourceSmoothingLength[sourceParticleIndex],
                        data.sourceVelocityX[sourceParticleIndex],
                        data.sourceVelocityY[sourceParticleIndex],
                        data.sourceVelocityZ[sourceParticleIndex],
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
            float const sinkSmoothingLength = data.sourceSmoothingLength[particleIndex];
            float const sinkDensity = data.sourceDensity[particleIndex];
            float const sinkInternalEnergy = data.sourceInternalEnergy[particleIndex];
            ALWAYS_ASSERT(
                sinkSmoothingLength > 0.0f && sinkDensity > 0.0f && sinkInternalEnergy > 0.0f);
            // clang-format off
            /*
             * Particle a. Section 3.5 leaves the equation of state free; the ideal gas gives
             * P_a = (γ − 1) ρ_a u_a and soundSpeed² = γ P_a/ρ_a with the single configured γ.
             *
             * Symbol directory (in the code below):
             *   P_a → pressure
             *   u_a → internal energy
             *   ρ_a → density
             *   γ → gas adiabatic index
             */
            // clang-format on
            float const H_a = sinkSmoothingLength;    // support radius, 2h_a for the cubic spline
            float const ρ_a = sinkDensity;
            float const u_a = sinkInternalEnergy;
            float const P_a = pressureEnergyRatio * ρ_a * u_a;
            float const soundSpeed = sqrtf(γ * pressureEnergyRatio * u_a);
            Position const positionX = data.sourcePositionX[particleIndex];
            Position const positionY = data.sourcePositionY[particleIndex];
            Position const positionZ = data.sourcePositionZ[particleIndex];
            Vector3<float> const velocity(data.sourceVelocityX[particleIndex],
                data.sourceVelocityY[particleIndex], data.sourceVelocityZ[particleIndex]);
            float const inverseSmoothingLength = 1.0f / H_a;
            float const inverseSmoothingLengthGradientScale =
                sph_geometry::inverseGradientScale(inverseSmoothingLength);    // H_a^-(D+1)
            flops += 8;
            // clang-format off
            /*
             * Variational operator (cosmictiger/sph/variable_smoothing_lengths.hpp): every
             * gas particle has the mass m and the volume weight x = m, so y = ρ and
             * Q_ab/m_a = (P_a/ρ_a²)(m − ζ_a). Comoving Q and kernel gradients give the rates
             * dv_a/dτ = −(1/a) Σ_b (Q_ab g_a + Q_ba g_b)/m_a and
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
            Vector3<float> dv_a_dτ(0.0f);    // this thread's acceleration sum
            Vector3<float> pressureAcceleration(0.0f);
            Vector3<float> viscosityAcceleration(0.0f);
            float du_a_dτ = 0.0f;
            float dissipativeEnergyRate = 0.0f;    // this thread's viscous/conductive sum
            float viscosityEnergyRate = 0.0f;
            float conductionEnergyRate = 0.0f;
            float conductionExchangeRate = 0.0f;
            float const α_a = parameters.variableViscosity ?
                data.sourceBulkViscosityCoefficient[particleIndex] :
                parameters.bulkViscosityCoefficient;
            float α = α_a;
            float maximumCompressionSpeed = 0.0f;    // max_b(−μ_ab) for δt_cv
            int maximumNeighborRung = 0;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.neighbors.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.neighbors[neighborIndex];
                // Particle b. Pairs outside both supports or at zero separation are skipped.
                Vector3<float> const r_ab(distance(positionX, neighbor.positionX),
                    distance(positionY, neighbor.positionY),
                    distance(positionZ, neighbor.positionZ));
                float const separationSquared = sph_geometry::spatialDot(r_ab, r_ab);    // |r_ab|²
                float const H_b = neighbor.smoothingLength;
                flops +=
                    8 + (separationSquared == 0.0f ? 0 : 2);    // Distances, dot, support squares.
                bool const supportsIntersect = separationSquared < fmaxf(H_a * H_a, H_b * H_b);
                if (parameters.midpointHydro && parameters.phase == 1 && supportsIntersect) {
                    maximumNeighborRung = max(maximumNeighborRung, neighbor.rung);
                }
                if (separationSquared == 0.0f || !supportsIntersect) {
                    continue;
                }
                float const neighborDensity = neighbor.density;
                float const neighborInternalEnergy = neighbor.internalEnergy;
                float const neighborPressure =
                    pressureEnergyRatio * neighborDensity * neighborInternalEnergy;
                float const neighborSoundSpeed =
                    sqrtf(γ * pressureEnergyRatio * neighborInternalEnergy);
                float const abs_r_ab = sqrtf(separationSquared);    // |r_ab|
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
                float const inverseNeighborSmoothingLength = 1.0f / H_b;
                float const normalizedSinkRadius = abs_r_ab * inverseSmoothingLength;
                float const normalizedNeighborRadius = abs_r_ab * inverseNeighborSmoothingLength;
                float const F_a = dkernelWDq(normalizedSinkRadius) *
                    inverseSmoothingLengthGradientScale / abs_r_ab;
                float const F_b = dkernelWDq(normalizedNeighborRadius) *
                    sph_geometry::inverseGradientScale(inverseNeighborSmoothingLength) / abs_r_ab;
                float const F_ab = 0.5f * (F_a + F_b);
                // v_ab = v_a − v_b from the stored peculiar velocities. The physical relative
                // velocity adds the Hubble flow (a da/dt) r_ab and enters only the viscosity.
                Vector3<float> const v_ab = velocity -
                    Vector3<float>(neighbor.velocityX, neighbor.velocityY, neighbor.velocityZ);
                float const velocityDotSeparation = sph_geometry::spatialDot(v_ab, r_ab);
                float const physicalVelocityDotSeparation =
                    velocityDotSeparation + expansionRate * separationSquared;
                // Phantom uses radial approach speed, without Monaghan's h/eta regularization.
                float const μ_ab = parameters.priceEtAl ?
                    fminf(physicalVelocityDotSeparation / abs_r_ab, 0.0f) :
                    monaghan::viscosityMu(separationSquared, physicalVelocityDotSeparation, H_a,
                        H_b, viscosityRegularization, smoothingLengthPerEffectiveRadius,
                        float(monaghan::kViscosityWidthScale));
                float const α_b =
                    parameters.variableViscosity ? neighbor.bulkViscosityCoefficient : α_a;
                float const α_ab = parameters.variableViscosity ? 0.5f * α_a + 0.5f * α_b : α_a;
                float const β_ab = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                    parameters.vonNeumannRichtmyerCoefficient * α_ab :
                    parameters.vonNeumannRichtmyerCoefficient;
                α = fmaxf(α, parameters.priceEtAl ? fmaxf(α_a, α_b) : α_ab);
                flops += 3 * int(parameters.variableViscosity) +
                    int(parameters.vonNeumannRichtmyerCoefficientFollowsBulk());
                float shearFactor = 1.0f;
                if (parameters.balsaraSwitch) {
                    shearFactor =
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
                float const Q_ba_over_m_a = variational::directedCoefficient(
                    variational::pressureVolumeFactor(neighborPressure, 1.0f, neighborDensity), m_b,
                    neighbor.gradHCorrection);
                float const pressureForce = inverseScaleFactor *
                    variational::pairForceFactor(Q_ab_over_m_a, Q_ba_over_m_a, F_a, F_b);
                float viscousForce;
                float viscousHeating;
                float neighborViscousHeating = 0.0f;
                if (parameters.priceEtAl) {
                    /*
                     * Phantom (2018), equations (39)-(42), including each side's grad-H factor.
                     */
                    float const q_abᵃ =
                        price_etal::artificialPressure(ρ_a, soundSpeed, μ_ab, α_a, β_ab);
                    float const q_abᵇ = price_etal::artificialPressure(
                        neighborDensity, neighborSoundSpeed, μ_ab, α_b, β_ab);
                    float C_ab = price_etal::directedViscosityCoefficient(q_abᵃ, ρ_a, m_b, ζ_a);
                    float C_ba = price_etal::directedViscosityCoefficient(
                        q_abᵇ, neighborDensity, m_b, neighbor.gradHCorrection);
                    if (parameters.balsaraSwitch) {
                        C_ab *= shearFactor;
                        C_ba *= shearFactor;
                        flops += 2;
                    }
                    viscousForce =
                        inverseScaleFactor * variational::pairForceFactor(C_ab, C_ba, F_a, F_b);
                    viscousHeating = inverseScaleFactor *
                        variational::pairWork(C_ab, F_a, physicalVelocityDotSeparation);
                    if (applyPairKicks) {
                        // Negating both separation and relative velocity leaves
                        // their dot product unchanged in the reverse pair.
                        neighborViscousHeating = inverseScaleFactor *
                            variational::pairWork(C_ba, F_b, physicalVelocityDotSeparation);
                        flops += 3;
                    }
                    flops += 28;    // radial division, two q/C terms, force/work
                } else {
                    float Π_ab = monaghan::artificialViscosity(
                        ρ_a, neighborDensity, soundSpeed, neighborSoundSpeed, μ_ab, α_ab, β_ab);
                    if (parameters.balsaraSwitch) {
                        Π_ab *= shearFactor;
                        flops += 1;
                    }
                    viscousForce = scaledGasMass * Π_ab * F_ab;
                    viscousHeating =
                        scaledGasMass * 0.5f * Π_ab * F_ab * physicalVelocityDotSeparation;
                    neighborViscousHeating = viscousHeating;
                    flops += sph_flops::kViscosityMu + sph_flops::kArtificialViscosity + 6;
                }
                if constexpr (retainHydroDiagnostics) {
                    viscosityEnergyRate += viscousHeating;
                    flops += 1;
                }
                float targetDissipativeEnergyRate = viscousHeating;
                float neighborDissipativeEnergyRate = neighborViscousHeating;
                if (parameters.conductionCoefficient > 0.0f) {
                    float const exchangeRate = sph_conduction::exchangeRate(ρ_a, neighborDensity,
                        soundSpeed, neighborSoundSpeed, H_a, H_b, separationSquared,
                        physicalVelocityDotSeparation, F_ab, parameters.conductionCoefficient,
                        viscosityRegularization, smoothingLengthPerEffectiveRadius, scaledGasMass);
                    float const conductiveEnergyRate =
                        exchangeRate * (neighborInternalEnergy - u_a);
                    if constexpr (retainHydroDiagnostics) {
                        conductionEnergyRate += conductiveEnergyRate;
                        flops += 1;
                    }
                    targetDissipativeEnergyRate += conductiveEnergyRate;
                    neighborDissipativeEnergyRate -= conductiveEnergyRate;
                    conductionExchangeRate += exchangeRate;
                    flops += 27;    // 22 exchange + 5 heating/row-sum operations.
                }
                Vector3<float> const pairForce = (pressureForce + viscousForce) * r_ab;
                dv_a_dτ -= pairForce;
                if constexpr (retainHydroDiagnostics) {
                    // Decompose the existing pair operator: a_s = −Σ_b F_s,ab r_ab,
                    // s ∈ {pressure, viscosity}. F_s,ab is pressureForce/viscousForce;
                    // r_ab is separation. Independent sums preserve small residuals.
                    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                        pressureAcceleration[dim] -= pressureForce * r_ab[dim];
                        viscosityAcceleration[dim] -= viscousForce * r_ab[dim];
                    }
                    flops += 4 * kSpatialDimensions;
                }
                if (applyPairKicks &&
                    parameters.isPairRungDue(max(previousRung, int(neighbor.rung))) &&
                    sphOwnsPairKick(data.sourceIds[particleIndex],
                        data.sourceIds[neighbor.sourceIndex], previousRung, neighbor.rung)) {
                    Vector3<float> const pairAcceleration(
                        -pairForce[kXi], -pairForce[kYi], -pairForce[kZi]);
                    float const targetEntropyRate = sph_entropy::fromStoredEnergy(
                        targetDissipativeEnergyRate, ρ_a, γ, parameters.scaleFactor);
                    float const neighborEntropyRate = sph_entropy::fromStoredEnergy(
                        neighborDissipativeEnergyRate, neighborDensity, γ, parameters.scaleFactor);
                    flops += sphAccumulatePairKick(data.sourcePairUpdates[particleIndex],
                        data.sourcePairUpdates[neighbor.sourceIndex], pairAcceleration,
                        targetEntropyRate, neighborEntropyRate, kickTimestep);
                    flops += 2 * sph_flops::entropyConversionFlops(γ);
                }
                du_a_dτ += inverseScaleFactor *
                    variational::pairWork(Q_ab_over_m_a, F_a, velocityDotSeparation);
                dissipativeEnergyRate += targetDissipativeEnergyRate;
                flops += 54;    // common geometry, pressure, and accumulation
                maximumCompressionSpeed = fmaxf(maximumCompressionSpeed, -μ_ab);
                flops += sph_flops::densityKernelOperations(
                             selectedHydroKernel(), normalizedSinkRadius, true) +
                    sph_flops::densityKernelOperations(
                        selectedHydroKernel(), normalizedNeighborRadius, true);
            }
            sharedReduceAdd<float, kHydroBlockSize>(dv_a_dτ[kXi]);
            sharedReduceAdd<float, kHydroBlockSize>(dv_a_dτ[kYi]);
            sharedReduceAdd<float, kHydroBlockSize>(dv_a_dτ[kZi]);
            sharedReduceAdd<float, kHydroBlockSize>(du_a_dτ);
            sharedReduceAdd<float, kHydroBlockSize>(dissipativeEnergyRate);
            if constexpr (retainHydroDiagnostics) {
                for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                    sharedReduceAdd<float, kHydroBlockSize>(pressureAcceleration[dim]);
                    sharedReduceAdd<float, kHydroBlockSize>(viscosityAcceleration[dim]);
                }
                flops += 2 * kSpatialDimensions * detail::sphReductionFlops<kHydroBlockSize>();
                sharedReduceAdd<float, kHydroBlockSize>(viscosityEnergyRate);
                sharedReduceAdd<float, kHydroBlockSize>(conductionEnergyRate);
                flops += 2 * detail::sphReductionFlops<kHydroBlockSize>();
            }
            if (parameters.conductionCoefficient > 0.0f) {
                sharedReduceAdd<float, kHydroBlockSize>(conductionExchangeRate);
                flops += detail::sphReductionFlops<kHydroBlockSize>();
            }
            sharedReduceMax<kHydroBlockSize>(maximumCompressionSpeed);
            sharedReduceMax<kHydroBlockSize>(α);
            if (parameters.midpointHydro && parameters.phase == 1) {
                sharedReduceMax<kHydroBlockSize>(maximumNeighborRung);
            }
            flops += 5 * detail::sphReductionFlops<kHydroBlockSize>();
            if (threadIndex == 0) {
                // Corrections are derived, never checkpointed or migrated; slots that the
                // density pass has not rebuilt hold NaN, so a stale source cannot pass silently.
                if (!(sph_volume::isFiniteValue(dv_a_dτ.dot(dv_a_dτ)) &&
                        sph_volume::isFiniteValue(du_a_dτ) &&
                        sph_volume::isFiniteValue(dissipativeEnergyRate) &&
                        sph_volume::isFiniteValue(conductionExchangeRate))) {
                    printf("SPH variational rates are not finite: smoothingLength=%e density=%e "
                           "ζ=%e\n",
                        H_a, ρ_a, ζ_a);
                    __trap();
                }
                if constexpr (retainHydroDiagnostics) {
                    auto& diagnostic = data.sinkHydroDiagnostics[sinkIndex];
                    // Hydro pair force only, before adding external sources.
                    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                        diagnostic.acceleration[dim] = dv_a_dτ[dim];
                        diagnostic.pressureAcceleration[dim] = pressureAcceleration[dim];
                        diagnostic.viscosityAcceleration[dim] = viscosityAcceleration[dim];
                    }
                    // Derived from the A↔E conversion in thermodynamics.hpp:
                    // (dA/dτ)_s = (γ−1) ρ_c^(1−γ) a^(3γ−5) (dE/dτ)_s,
                    // s ∈ {viscosity, conduction}; their sum is the hydro entropy rate.
                    // Here ρ_a=ρ_c, parameters.scaleFactor=a, and each accumulated
                    // dissipative energy rate is (dE/dτ)_s. Reversible work contributes zero.
                    diagnostic.entropyRate = sph_entropy::fromStoredEnergy(
                        dissipativeEnergyRate, ρ_a, γ, parameters.scaleFactor);
                    diagnostic.viscosityEntropyRate = sph_entropy::fromStoredEnergy(
                        viscosityEnergyRate, ρ_a, γ, parameters.scaleFactor);
                    diagnostic.conductionEntropyRate = sph_entropy::fromStoredEnergy(
                        conductionEnergyRate, ρ_a, γ, parameters.scaleFactor);
                    flops += 3 * sph_flops::entropyConversionFlops(γ);
                }
                Vector3<float> gravityAcceleration(0.0f);
#if COSMICTIGER_WITH_GRAVITY
                if (!parameters.midpointHydro && !parameters.deferAccelerationSources &&
                    data.selfGravityEnabled) {
                    gravityAcceleration = Vector3<float>(data.sinkGravityAccelerationX[sinkIndex],
                        data.sinkGravityAccelerationY[sinkIndex],
                        data.sinkGravityAccelerationZ[sinkIndex]);
                }
#endif
                if (!parameters.deferAccelerationSources) {
                    dv_a_dτ = sphCompleteAcceleration(parameters,
                        Vector3<Position>(positionX, positionY, positionZ), velocity, dv_a_dτ,
                        gravityAcceleration, data.centralAcceleration, data.centralSmoothingLength,
                        flops);
                } else {
                    ALWAYS_ASSERT(parameters.phase == 0);
                }
                if (applyPairKicks && parameters.isPairRungDue(previousRung)) {
                    Vector3<float> const externalAcceleration = sphCompleteAcceleration(parameters,
                        Vector3<Position>(positionX, positionY, positionZ), velocity,
                        Vector3<float>(0.0f), Vector3<float>(0.0f), data.centralAcceleration,
                        data.centralSmoothingLength, flops);
                    flops += sphAccumulateExternalKick(
                        data.sourcePairUpdates[particleIndex], externalAcceleration, kickTimestep);
                }
                if (data.sinkAccelerations) {
                    for (int dim = 0; dim < kSpatialDimensions; dim++) {
                        if (parameters.accelerationGradientViscosity) {
                            data.sinkAccelerations[sinkIndex][dim] += dv_a_dτ[dim];
                        } else {
                            data.sinkAccelerations[sinkIndex][dim] = dv_a_dτ[dim];
                        }
                    }
                }
                // E=a^2 u and homogeneous work du/dt=-3 H P/rho give
                // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E = (5-3 gamma) (da/dtau)/a E.
                float const expansionCoefficient = 5.0f - 3.0f * γ;
                du_a_dτ += expansionCoefficient * expansionRate * inverseScaleFactor * u_a;
                flops += 6;    // Homogeneous thermal expansion; force sources count separately.
                if (data.sinkInternalEnergyRate) {
                    data.sinkInternalEnergyRate[sinkIndex] = du_a_dτ + dissipativeEnergyRate;
                    flops += 1;
                }
                if (data.sinkEntropyRate) {
                    // Reversible PdV and homogeneous expansion leave A unchanged.
                    data.sinkEntropyRate[sinkIndex] = sph_entropy::fromStoredEnergy(
                        dissipativeEnergyRate, ρ_a, γ, parameters.scaleFactor);
                    flops += sph_flops::entropyConversionFlops(γ);
                }
                if (parameters.phase == 1) {
                    // clang-format off
                    /*
                     * Section 10.3, p. 563:
                     *   δt_f = min_a (smoothingLength/|f_a|),
                     *   δt_cv = min_a smoothingLength/(soundSpeed + 0.6 (α soundSpeed + β max_b
                     *   μ_ab)), δt = 0.25 min(δt_f, δt_cv).
                     * smoothingLength is the common effective radius 2σ (docs/monaghan_sph.md). The
                     * printed δt_f lacks a square root and is used as sqrt(smoothingLength/|f_a|);
                     * max_b μ_ab is read as max_b(−μ_ab), or radial approach speed for Phantom; the
                     * Courant factor and the calibrated acceleration coefficient replace
                     * the fixed 0.25. The reciprocals are kept apart for the diagnostics.
                     *
                     * Symbol directory (in the code below):
                     *   μ_ab → viscosity velocity
                     *   |f_a| → acceleration magnitude
                     *   Code names: abs_f_a = |f_a|.
                     *   δt → time step
                     *   α → maximum bulk viscosity coefficient
                     *   β → maximum von Neumann–Richtmyer coefficient
                     */
                    // clang-format on
                    float const smoothingLength = H_a / smoothingLengthPerEffectiveRadius;
                    float const inverseSoundTimestep = soundSpeed / smoothingLength;
                    float const β = parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                        parameters.vonNeumannRichtmyerCoefficient * α :
                        parameters.vonNeumannRichtmyerCoefficient;
                    float const inverseViscosityTimestep =
                        0.6f * (α * soundSpeed + β * maximumCompressionSpeed) / smoothingLength;
                    float const inverseCourantTimestep =
                        inverseSoundTimestep + inverseViscosityTimestep;
                    float const abs_f_a = sqrtf(dv_a_dτ.dot(dv_a_dτ));
                    float const inverseAccelerationTimestep = parameters.courantFactor *
                        sqrtf(parameters.scaleFactor * abs_f_a / smoothingLength) /
                        (data.accelerationTimestepFactor * kHydroAccelerationScale);
                    float δt = fminf(parameters.maximumTimestep,
                        parameters.courantFactor * parameters.scaleFactor /
                            fmaxf(fmaxf(inverseCourantTimestep, inverseAccelerationTimestep),
                                1.0e-30f));
                    // Bound reversible work; the approach-speed CFL term controls viscosity.
                    if (du_a_dτ != 0.0f) {
                        flops += 3;
                        δt = fminf(δt,
                            parameters.courantFactor / float(monaghan::kViscosityWidthScale) * u_a /
                                fabsf(du_a_dτ));
                    }
                    // Derived positivity bound, matching the CPU sum over all neighbors.
                    if (conductionExchangeRate > 0.0f) {
                        δt = fminf(δt, 0.5f / conductionExchangeRate);
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
                    float gravityTimestep = parameters.maximumTimestep;
                    int gravityRung = 0;
                    if (!parameters.midpointHydro) {
                        gravityTimestep = fminf(parameters.maximumTimestep,
                            data.accelerationTimestepFactor *
                                sqrtf(parameters.scaleFactor * data.gravityEffectiveRadius /
                                    fmaxf(sqrtf(gravityAcceleration.dot(gravityAcceleration)),
                                        1.0e-30f)));
                        if (data.selfGravityEnabled) {
                            atomicMin(&reduction->minimumTimestep, gravityTimestep);
                        }
                        gravityRung = parameters.globalTimestep ?
                            0 :
                            max(1, int(ceilf(log2f(parameters.baseTimestep / gravityTimestep))));
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
                                fmaxf(inverseCourantTimestep, 1.0e-30f),
                            parameters.courantFactor * parameters.scaleFactor /
                                fmaxf(inverseAccelerationTimestep, 1.0e-30f),
                            du_a_dτ != 0.0f ? parameters.courantFactor /
                                    float(monaghan::kViscosityWidthScale) * u_a / fabsf(du_a_dτ) :
                                              parameters.maximumTimestep,
                            H_a, ρ_a, u_a, du_a_dτ, abs_f_a, soundSpeed, maximumCompressionSpeed, α,
                            m_b / (m_b - ζ_a), parameters.scaleFactor);
#if COSMICTIGER_WITH_GRAVITY
                        printf("SPH gravity timestep: enabled=%i gravityRung=%i gravityDt=%e\n",
                            int(data.selfGravityEnabled), gravityRung, gravityTimestep);
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
                    maximumSignalSpeed =
                        fmaxf(maximumSignalSpeed, inverseCourantTimestep * smoothingLength);
                    // Exclude the signal-speed report.
                    flops += parameters.globalTimestep ? 22 : 25;
                    atomicMax(&reduction->inverseSoundTimestep, inverseSoundTimestep);
                    atomicMax(&reduction->inverseViscosityTimestep, inverseViscosityTimestep);
                    atomicMax(&reduction->inverseAccelerationTimestep, inverseAccelerationTimestep);
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

}    // namespace detail

// Strong entry points give each variant its own shared-memory layout.
template <>
__global__ void sphCudaHydro<false>(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    __shared__ int targetNodeIndex;
    __shared__ detail::HydroWorkspace workspace;
    detail::sphCudaHydroImpl<false>(parameters, data, reduction, workspace, targetNodeIndex);
}
template <>
__global__ void sphCudaHydro<true>(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    __shared__ int targetNodeIndex;
    __shared__ detail::HydroWorkspace workspace;
    detail::sphCudaHydroImpl<true>(parameters, data, reduction, workspace, targetNodeIndex);
}
