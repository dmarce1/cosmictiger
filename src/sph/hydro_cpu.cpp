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
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/conduction.hpp>
#include <cosmictiger/sph/cpu.hpp>
#include <cosmictiger/sph/variable_smoothing_lengths.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/viscosity/price_etal.hpp>
#include <cosmictiger/sph/volume.hpp>

/*
 * Variational pressure force and compatible work use the equations documented in
 * cosmictiger/sph/variable_smoothing_lengths.hpp. Artificial viscosity uses Monaghan (1992),
 * equations 4.1-4.2 and the symmetric kernel of p. 554; its heating follows equation
 * 6.5 and its time step section 10.3, p. 563. The viscosity header gives the full
 * citation. Price mode instead uses Phantom (2018), equations (39)-(42), below.
 * Particle a is the sink and particle b a staged neighbor; the computation
 * uses descriptive names mapped to the papers' symbols below. Positions and density are comoving,
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
 *   a → parameters.scaleFactor; m_b → gasMass; γ → gasAdiabaticIndex (CPU), adiabaticIndex (CUDA)
 *   ρ_a, P_a, u_a → density, pressure, internalEnergy
 *   H_a, H_b → sinkKernelSmoothingLength, neighborSmoothingLength
 *   r_ab, |r_ab| → separation, pairDistance; v_ab → velocityDifference
 *   ζ_a → gradHCorrection; Q_ab/m_a, Q_ba/m_a → sinkPressureCoefficient, neighborPressureCoefficient
 *   F_a, F_b → sinkGradientFactor, neighborGradientFactor
 *   dv_a/dτ, du_a/dτ → acceleration, internalEnergyRate
 *   α_a, α_b, α_ab → bulkViscosityCoefficient, neighborBulkViscosityCoefficient, pairBulkViscosityCoefficient
 *   β_ab → pairVonNeumannRichtmyerCoefficient; Π_ab → pairArtificialViscosity
 *   μ_ab → viscosityVelocity (Phantom instead stores min(v_ab · r̂_ab, 0) here)
 *   q_abᵃ, q_abᵇ → sinkViscousPressure, neighborViscousPressure
 *   C_ab, C_ba → sinkViscosityCoefficient, neighborViscosityCoefficient
 *   SIMD partial sums append Pack; CUDA partial sums are reduced in place.
 */
// clang-format on
namespace detail {

template <bool retainHydroDiagnostics>
SphRunReturn sphCpuHydroImpl(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    sph_flops::Counter flops = 3;    // Scale-factor reciprocal, gas mass scaling and γ − 1.
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
    float const inverseScaleFactor = 1.0f / parameters.scaleFactor;
    float const gasAdiabaticIndex = data.adiabaticIndex;
    float const pressureEnergyRatio = gasAdiabaticIndex - 1.0f;    // P/(ρu) for every gas particle
    SimdFloat const adiabaticIndexPack(gasAdiabaticIndex);
    SimdFloat const pressureEnergyRatioPack(pressureEnergyRatio);
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4 defaults
    // after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    float const smoothingLengthPerEffectiveRadius =
        selectedHydroKernel().smoothingLengthPerEffectiveRadius;
    SimdFloat const zero(0.0f);
    SimdFloat const one(1.0f);
    SimdFloat const half(0.5f);
    float const gasMass = data.gasMass;    // every gas particle has the same mass
    SimdFloat const scaledGasMass(gasMass * inverseScaleFactor);
    float const viscosityRegularization = parameters.viscosityEpsilon;    // η²/h² of p. 551
    float const expansionRate = parameters.expansionRate;    // a·da/dt, the Hubble flow of a pair
    bool const applyPairKicks = data.sourcePairUpdates && parameters.hydroKickTimestep > 0.0;
    float const kickTimestep = applyPairKicks ? float(parameters.hydroKickTimestep) : 0.0f;
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
         *   u_a → internalEnergy
         *   ρ_a → density
         *   γ → gasAdiabaticIndex
         */
        // clang-format on
        float const sinkKernelSmoothingLength =
            sinkSmoothingLength;    // support radius, 2h_a for the cubic spline
        float const density = sinkDensity;
        float const internalEnergy = sinkInternalEnergy;
        float const pressure = pressureEnergyRatio * density * internalEnergy;
        float const soundSpeed =
            std::sqrt(gasAdiabaticIndex * pressureEnergyRatio * internalEnergy);
        Vector3<float> const velocity(data.sourceVelocityX[particleIndex],
            data.sourceVelocityY[particleIndex], data.sourceVelocityZ[particleIndex]);
        Vector3<SimdFloat> const velocityPack(velocity);
        float const bulkViscosityCoefficient = parameters.variableViscosity ?
            data.sourceBulkViscosityCoefficient[particleIndex] :
            parameters.bulkViscosityCoefficient;
        SimdFloat maximumBulkViscosityCoefficientPack(bulkViscosityCoefficient);
        float const inverseSmoothingLength = 1.0f / sinkKernelSmoothingLength;
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
         *   v_ab → velocityDifference
         *   P_a → pressure
         *   g_a → maskedSinkGradientFactor
         *   g_b → maskedNeighborGradientFactor
         *   ζ_a → gradHCorrection
         */
        // clang-format on
        float const gradHCorrection = data.sourceGradHCorrection[particleIndex];
        float const sinkPressureCoefficient = variational::directedCoefficient(
            variational::pressureVolumeFactor(pressure, 1.0f, density), gasMass, gradHCorrection);
        flops += 5;
        float const inverseSmoothingLengthGradientScale =
            sph_geometry::inverseGradientScale(inverseSmoothingLength);    // H_a^-(D+1)
        Vector3<SimdFloat> accelerationPack(zero);    // lane partial sums of acceleration
        Vector3<SimdFloat> pressureAccelerationPack(zero);
        Vector3<SimdFloat> viscosityAccelerationPack(zero);
        SimdFloat dissipativeEnergyRatePack(zero);
        SimdFloat viscosityEnergyRatePack(zero);
        SimdFloat conductionEnergyRatePack(zero);
        SimdFloat conductionExchangeRatePack(zero);
        SimdFloat internalEnergyRatePack(zero);         // lane partial sums of thermal work
        SimdFloat maximumCompressionSpeedPack(zero);    // lane maxima of −μ_ab for δt_cv
        SimdFloat maximumNeighborRungPack(zero);
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            // Particle b: one SIMD pack of staged neighbors. Lanes beyond the list, outside
            // both supports, or at zero separation are excluded. Their inputs are replaced
            // before any division or square root; masking only the result can still trap.
            Vector3<SimdFloat> const separation =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);    // r_a − r_b
            SimdFloat const stagedSeparationSquared =
                sph_geometry::spatialDot(separation, separation);
            SimdFloat const stagedNeighborSmoothingLength =
                detail::sphCpuLoad(data.sourceSmoothingLength, neighbors, first, 1.0f);
            auto const supportsIntersect = detail::sphCpuValidLanes(neighbors, first) &
                (stagedSeparationSquared <
                    simdMax(SimdFloat(sinkKernelSmoothingLength * sinkKernelSmoothingLength),
                        sqr(stagedNeighborSmoothingLength)));
            if (parameters.midpointHydro && parameters.phase == 1) {
                SimdFloat const neighborRungs([&](auto lane) {
                    size_t const index = first + size_t(lane);
                    return index < neighbors.size() ? float(data.sourceRungs[neighbors[index]]) :
                                                      0.0f;
                });
                maximumNeighborRungPack = simdMax(
                    maximumNeighborRungPack, simdSelect(supportsIntersect, neighborRungs, zero));
            }
            auto const interacts = supportsIntersect & (stagedSeparationSquared > zero);
            // Three fixed scales, five dot operations and H_b² per lane; H_a² is scalar.
            flops += 9 * SimdFloat::size() + 1;
            if (!simdAnyOf(interacts)) {
                continue;
            }
            SimdFloat const neighborSmoothingLength =
                simdSelect(interacts, stagedNeighborSmoothingLength, one);
            SimdFloat const neighborDensity = simdSelect(
                interacts, detail::sphCpuLoad(data.sourceDensity, neighbors, first, 1.0f), one);
            SimdFloat const neighborInternalEnergy = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceInternalEnergy, neighbors, first, 1.0f), one);
            SimdFloat const neighborPressure =
                pressureEnergyRatioPack * neighborDensity * neighborInternalEnergy;
            SimdFloat const neighborSoundSpeed =
                simdSqrt(adiabaticIndexPack * pressureEnergyRatioPack * neighborInternalEnergy);
            SimdFloat const separationSquared =
                simdSelect(interacts, stagedSeparationSquared, one);       // |r_ab|²
            SimdFloat const pairDistance = simdSqrt(separationSquared);    // |r_ab|
            // clang-format off
            /*
             * Symmetric kernel of p. 554: ∇_a W̃_ab = ½[∇_a W(r_ab, smoothingLength) + ∇_a W(r_ab,
             * neighborSmoothingLength)]. The selected kernel is W(r, H) = w(r/H)/H³, so each
             * gradient equals w'(r/H) r_ab/(H⁴ r) and ∇_a W̃_ab = F_ab r_ab with the scalar F_ab.
             *
             * Symbol directory (in the code below):
             *   F_ab → symmetricGradientFactor
             *   r_ab → separation
             */
            // clang-format on
            SimdFloat const inverseNeighborSmoothingLength = one / neighborSmoothingLength;
            SimdFloat const normalizedSinkRadius = pairDistance * SimdFloat(inverseSmoothingLength);
            SimdFloat const normalizedNeighborRadius =
                pairDistance * inverseNeighborSmoothingLength;
            SimdFloat const sinkGradientFactor = dkernelWDq(normalizedSinkRadius) *
                SimdFloat(inverseSmoothingLengthGradientScale) / pairDistance;
            SimdFloat const neighborGradientFactor = dkernelWDq(normalizedNeighborRadius) *
                sph_geometry::inverseGradientScale(inverseNeighborSmoothingLength) / pairDistance;
            flops += sph_flops::densityKernelOperations(
                         selectedHydroKernel(), normalizedSinkRadius, true) +
                sph_flops::densityKernelOperations(
                    selectedHydroKernel(), normalizedNeighborRadius, true);
            SimdFloat const symmetricGradientFactor =
                simdSelect(interacts, half * (sinkGradientFactor + neighborGradientFactor), zero);
            // v_ab = v_a − v_b from the stored peculiar velocities. The physical relative
            // velocity adds the Hubble flow (a da/dt) r_ab and enters only the viscosity.
            Vector3<SimdFloat> const velocityDifference = simdSelect(interacts,
                velocityPack -
                    detail::sphCpuLoad3(data.sourceVelocityX, data.sourceVelocityY,
                        data.sourceVelocityZ, neighbors, first, 0.0f),
                Vector3<SimdFloat>(zero));
            SimdFloat const velocityDotSeparation =
                sph_geometry::spatialDot(velocityDifference, separation);
            SimdFloat const physicalVelocityDotSeparation = velocityDotSeparation +
                SimdFloat(expansionRate) * simdSelect(interacts, separationSquared, zero);
            /*
             * Phantom uses the unregularized radial approach speed in equations (40)-(41).
             * Retain its maximum in the existing Courant bound; other modes use -mu_ab.
             */
            SimdFloat const viscosityVelocity = parameters.priceEtAl ?
                simdMin(physicalVelocityDotSeparation / pairDistance, zero) :
                monaghan::viscosityMu(separationSquared, physicalVelocityDotSeparation,
                    SimdFloat(sinkKernelSmoothingLength), neighborSmoothingLength,
                    SimdFloat(viscosityRegularization),
                    SimdFloat(smoothingLengthPerEffectiveRadius),
                    SimdFloat(float(monaghan::kViscosityWidthScale)));
            SimdFloat const neighborBulkViscosityCoefficient = parameters.variableViscosity ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourceBulkViscosityCoefficient, neighbors, first,
                        bulkViscosityCoefficient),
                    SimdFloat(bulkViscosityCoefficient)) :
                SimdFloat(bulkViscosityCoefficient);
            SimdFloat const pairBulkViscosityCoefficient = parameters.variableViscosity ?
                half * SimdFloat(bulkViscosityCoefficient) +
                    half * neighborBulkViscosityCoefficient :
                SimdFloat(bulkViscosityCoefficient);
            SimdFloat const pairVonNeumannRichtmyerCoefficient =
                parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                SimdFloat(parameters.vonNeumannRichtmyerCoefficient) *
                    pairBulkViscosityCoefficient :
                SimdFloat(parameters.vonNeumannRichtmyerCoefficient);
            maximumBulkViscosityCoefficientPack = simdMax(maximumBulkViscosityCoefficientPack,
                simdSelect(interacts,
                    parameters.priceEtAl ? simdMax(SimdFloat(bulkViscosityCoefficient),
                                               neighborBulkViscosityCoefficient) :
                                           pairBulkViscosityCoefficient,
                    SimdFloat(bulkViscosityCoefficient)));
            flops += (3 * int(parameters.variableViscosity) +
                         int(parameters.vonNeumannRichtmyerCoefficientFollowsBulk())) *
                SimdFloat::size();
            SimdFloat shearFactor(one);
            if (parameters.balsaraSwitch) {
                // Preserve the optional symmetric Balsara multiplier for both directed terms.
                shearFactor = half *
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
             *   F_a → sinkGradientFactor
             *   F_b → neighborGradientFactor
             *   g_a → maskedSinkGradientFactor
             *   g_b → maskedNeighborGradientFactor
             */
            // clang-format on
            SimdFloat const neighborGradHCorrection = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceGradHCorrection, neighbors, first, 0.0f), zero);
            SimdFloat const neighborPressureCoefficient = variational::directedCoefficient(
                variational::pressureVolumeFactor(neighborPressure, one, neighborDensity),
                SimdFloat(gasMass), neighborGradHCorrection);
            SimdFloat const maskedSinkGradientFactor =
                simdSelect(interacts, sinkGradientFactor, zero);
            SimdFloat const maskedNeighborGradientFactor =
                simdSelect(interacts, neighborGradientFactor, zero);
            SimdFloat const pressureForce = SimdFloat(inverseScaleFactor) *
                variational::pairForceFactor(SimdFloat(sinkPressureCoefficient),
                    neighborPressureCoefficient, maskedSinkGradientFactor,
                    maskedNeighborGradientFactor);
            SimdFloat viscousForce;
            SimdFloat viscousHeating;
            SimdFloat neighborViscousHeating(zero);
            if (parameters.priceEtAl) {
                // clang-format off
                /*
                 * Phantom (2018), equations (39)-(42): two directed q terms, each with
                 * its own kernel gradient and grad-H factor; local heating uses only q^a_ab.
                 *
                 * Symbol directory (in the code below):
                 *   a_ab → accelerationDifference
                 */
                // clang-format on
                SimdFloat const sinkViscousPressure = price_etal::artificialPressure(
                    SimdFloat(density), SimdFloat(soundSpeed), viscosityVelocity,
                    SimdFloat(bulkViscosityCoefficient), pairVonNeumannRichtmyerCoefficient);
                SimdFloat const neighborViscousPressure = price_etal::artificialPressure(
                    neighborDensity, neighborSoundSpeed, viscosityVelocity,
                    neighborBulkViscosityCoefficient, pairVonNeumannRichtmyerCoefficient);
                SimdFloat sinkViscosityCoefficient =
                    price_etal::directedViscosityCoefficient(sinkViscousPressure,
                        SimdFloat(density), SimdFloat(gasMass), SimdFloat(gradHCorrection));
                SimdFloat neighborViscosityCoefficient =
                    price_etal::directedViscosityCoefficient(neighborViscousPressure,
                        neighborDensity, SimdFloat(gasMass), neighborGradHCorrection);
                if (parameters.balsaraSwitch) {
                    sinkViscosityCoefficient *= shearFactor;
                    neighborViscosityCoefficient *= shearFactor;
                    flops += 2 * SimdFloat::size();
                }
                viscousForce = SimdFloat(inverseScaleFactor) *
                    variational::pairForceFactor(sinkViscosityCoefficient,
                        neighborViscosityCoefficient, maskedSinkGradientFactor,
                        maskedNeighborGradientFactor);
                viscousHeating = SimdFloat(inverseScaleFactor) *
                    variational::pairWork(sinkViscosityCoefficient, maskedSinkGradientFactor,
                        physicalVelocityDotSeparation);
                if (applyPairKicks) {
                    // The reverse pair has both r_ba and v_ba negated, leaving
                    // their dot product unchanged; its directed q term differs.
                    neighborViscousHeating = SimdFloat(inverseScaleFactor) *
                        variational::pairWork(neighborViscosityCoefficient,
                            maskedNeighborGradientFactor, physicalVelocityDotSeparation);
                    flops += 3 * SimdFloat::size();
                }
                flops += 28 * SimdFloat::size();    // radial division, two q/C terms, force/work
            } else {
                SimdFloat pairArtificialViscosity =
                    monaghan::artificialViscosity(SimdFloat(density), neighborDensity,
                        SimdFloat(soundSpeed), neighborSoundSpeed, viscosityVelocity,
                        pairBulkViscosityCoefficient, pairVonNeumannRichtmyerCoefficient);
                if (parameters.balsaraSwitch) {
                    pairArtificialViscosity *= shearFactor;
                    flops += SimdFloat::size();
                }
                viscousForce = scaledGasMass * pairArtificialViscosity * symmetricGradientFactor;
                viscousHeating = scaledGasMass * half * pairArtificialViscosity *
                    symmetricGradientFactor * physicalVelocityDotSeparation;
                neighborViscousHeating = viscousHeating;
                flops += (sph_flops::kViscosityMu + sph_flops::kArtificialViscosity + 6) *
                    SimdFloat::size();
            }
            if constexpr (retainHydroDiagnostics) {
                viscosityEnergyRatePack += viscousHeating;
                flops += SimdFloat::size();
            }
            SimdFloat targetDissipativeEnergyRate = viscousHeating;
            SimdFloat neighborDissipativeEnergyRate = neighborViscousHeating;
            if (parameters.conductionCoefficient > 0.0f) {
                SimdFloat const exchangeRate = sph_conduction::exchangeRate(SimdFloat(density),
                    neighborDensity, SimdFloat(soundSpeed), neighborSoundSpeed,
                    SimdFloat(sinkKernelSmoothingLength), neighborSmoothingLength,
                    separationSquared, physicalVelocityDotSeparation, symmetricGradientFactor,
                    SimdFloat(parameters.conductionCoefficient), SimdFloat(viscosityRegularization),
                    SimdFloat(smoothingLengthPerEffectiveRadius), scaledGasMass);
                SimdFloat const conductiveEnergyRate =
                    exchangeRate * (neighborInternalEnergy - SimdFloat(internalEnergy));
                if constexpr (retainHydroDiagnostics) {
                    conductionEnergyRatePack += conductiveEnergyRate;
                    flops += SimdFloat::size();
                }
                targetDissipativeEnergyRate += conductiveEnergyRate;
                neighborDissipativeEnergyRate -= conductiveEnergyRate;
                conductionExchangeRatePack += exchangeRate;
                flops += 27 * SimdFloat::size();    // 22 exchange + 5 heating/row-sum operations.
            }
            Vector3<SimdFloat> const pairForcePack = (pressureForce + viscousForce) * separation;
            accelerationPack -= pairForcePack;
            if constexpr (retainHydroDiagnostics) {
                // Decompose the existing pair operator: a_s = −Σ_b F_s,ab r_ab,
                // s ∈ {pressure, viscosity}. F_s,ab is pressureForce/viscousForce;
                // r_ab is separation. Accumulate each term independently to avoid
                // subtracting large forces when the pressure residual is small.
                for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                    pressureAccelerationPack[dim] -= pressureForce * separation[dim];
                    viscosityAccelerationPack[dim] -= viscousForce * separation[dim];
                }
                flops += 4 * kSpatialDimensions * SimdFloat::size();
            }
            if (applyPairKicks) {
                for (size_t lane = 0; lane < SimdFloat::size(); ++lane) {
                    if (!interacts[lane]) {
                        continue;
                    }
                    int const neighborIndex = neighbors[first + lane];
                    if (!parameters.isPairRungDue(
                            std::max(previousRung, int(data.sourceRungs[neighborIndex]))) ||
                        !sphOwnsPairKick(data.sourceIds[particleIndex],
                            data.sourceIds[neighborIndex], previousRung,
                            data.sourceRungs[neighborIndex])) {
                        continue;
                    }
                    Vector3<float> const pairAcceleration(-float(pairForcePack[kXi][lane]),
                        -float(pairForcePack[kYi][lane]),
                        -float(pairForcePack[kZi][lane]));
                    float const targetEntropyRate =
                        sph_entropy::fromStoredEnergy(float(targetDissipativeEnergyRate[lane]),
                            density, gasAdiabaticIndex, parameters.scaleFactor);
                    float const neighborEntropyRate = sph_entropy::fromStoredEnergy(
                        float(neighborDissipativeEnergyRate[lane]), float(neighborDensity[lane]),
                        gasAdiabaticIndex, parameters.scaleFactor);
                    flops += sphAccumulatePairKick(data.sourcePairUpdates[particleIndex],
                        data.sourcePairUpdates[neighborIndex], pairAcceleration, targetEntropyRate,
                        neighborEntropyRate, kickTimestep);
                    flops += 2 * sph_flops::entropyConversionFlops(gasAdiabaticIndex);
                }
            }
            internalEnergyRatePack += SimdFloat(inverseScaleFactor) *
                variational::pairWork(SimdFloat(sinkPressureCoefficient), maskedSinkGradientFactor,
                    velocityDotSeparation);
            dissipativeEnergyRatePack += targetDissipativeEnergyRate;
            flops += 54 * SimdFloat::size();    // common geometry, pressure, and accumulation
            maximumCompressionSpeedPack = simdMax(maximumCompressionSpeedPack, -viscosityVelocity);
        }
        Vector3<float> acceleration(simdSum(accelerationPack[kXi]),
            simdSum(accelerationPack[kYi]), simdSum(accelerationPack[kZi]));
        float internalEnergyRate = simdSum(internalEnergyRatePack);
        float const conductionExchangeRate =
            parameters.conductionCoefficient > 0.0f ? simdSum(conductionExchangeRatePack) : 0.0f;
        if (parameters.conductionCoefficient > 0.0f) {
            flops += SimdFloat::size() - 1;
        }
        flops +=
            4 * (SimdFloat::size() - 1);    // Four additive reductions; maxima are comparisons.
        // Corrections are derived, never checkpointed or migrated; slots that the density
        // pass has not rebuilt hold NaN, so a stale source cannot pass silently.
        if (!(sph_volume::isFiniteValue(acceleration.dot(acceleration)) &&
                sph_volume::isFiniteValue(internalEnergyRate) &&
                sph_volume::isFiniteValue(simdSum(dissipativeEnergyRatePack)) &&
                sph_volume::isFiniteValue(conductionExchangeRate))) {
            std::cout << std::format("SPH variational rates are not finite: smoothingLength={:e} "
                                     "density={:e} \u03b6={:e}\n",
                             sinkKernelSmoothingLength, density, gradHCorrection)
                      << std::flush;
            ALWAYS_ASSERT(false);
        }
        float const maximumBulkViscosityCoefficient =
            simdMaximum(maximumBulkViscosityCoefficientPack);
        float const maximumVonNeumannRichtmyerCoefficient =
            parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
            parameters.vonNeumannRichtmyerCoefficient * maximumBulkViscosityCoefficient :
            parameters.vonNeumannRichtmyerCoefficient;
        float const maximumCompressionSpeed =
            simdMaximum(maximumCompressionSpeedPack);    // max_b(−μ_ab) over approaching pairs
        if constexpr (retainHydroDiagnostics) {
            auto& diagnostic = data.sinkHydroDiagnostics[sinkIndex];
            // Hydro pair force only: capture before gravity, imposed forces, and damping.
            for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                diagnostic.acceleration[dim] = acceleration[dim];
                diagnostic.pressureAcceleration[dim] = simdSum(pressureAccelerationPack[dim]);
                diagnostic.viscosityAcceleration[dim] = simdSum(viscosityAccelerationPack[dim]);
            }
            flops += 2 * kSpatialDimensions * (SimdFloat::size() - 1);
            // Derived from the A↔E conversion in thermodynamics.hpp:
            // (dA/dτ)_s = (γ−1) ρ_c^(1−γ) a^(3γ−5) (dE/dτ)_s,
            // s ∈ {viscosity, conduction}; their sum is the hydro entropy rate.
            // Here density=ρ_c, parameters.scaleFactor=a, and each accumulated
            // dissipative energy rate is (dE/dτ)_s. Reversible work contributes zero.
            diagnostic.entropyRate =
                sph_entropy::fromStoredEnergy(simdSum(dissipativeEnergyRatePack), density,
                    gasAdiabaticIndex, parameters.scaleFactor);
            diagnostic.viscosityEntropyRate =
                sph_entropy::fromStoredEnergy(simdSum(viscosityEnergyRatePack), density,
                    gasAdiabaticIndex, parameters.scaleFactor);
            diagnostic.conductionEntropyRate =
                sph_entropy::fromStoredEnergy(simdSum(conductionEnergyRatePack), density,
                    gasAdiabaticIndex, parameters.scaleFactor);
            flops +=
                3 * (SimdFloat::size() - 1 + sph_flops::entropyConversionFlops(gasAdiabaticIndex));
        }
        // The pair calculation can overlap gravity. Only source assembly and
        // the following corrector require the fresh endpoint acceleration.
        Vector3<float> gravityAcceleration(0.0f);
#if COSMICTIGER_WITH_GRAVITY
        if (!parameters.midpointHydro && !parameters.deferAccelerationSources &&
            data.selfGravityEnabled) {
            gravityAcceleration = Vector3<float>(data.sinkGravityAccelerationX[sinkIndex],
                data.sinkGravityAccelerationY[sinkIndex], data.sinkGravityAccelerationZ[sinkIndex]);
        }
#endif
        if (!parameters.deferAccelerationSources) {
            acceleration = sphCompleteAcceleration(parameters,
                Vector3<Position>(data.sourcePositionX[particleIndex],
                    data.sourcePositionY[particleIndex], data.sourcePositionZ[particleIndex]),
                velocity, acceleration, gravityAcceleration, data.centralAcceleration,
                data.centralSmoothingLength, flops);
        } else {
            ALWAYS_ASSERT(parameters.phase == 0);
        }
        if (applyPairKicks && parameters.isPairRungDue(previousRung)) {
            // External forces have no reciprocal gas endpoint. Integrate them
            // once on the target's own rung, separately from pair reductions.
            Vector3<float> const externalAcceleration = sphCompleteAcceleration(parameters,
                Vector3<Position>(data.sourcePositionX[particleIndex],
                    data.sourcePositionY[particleIndex], data.sourcePositionZ[particleIndex]),
                velocity, Vector3<float>(0.0f), Vector3<float>(0.0f), data.centralAcceleration,
                data.centralSmoothingLength, flops);
            flops += sphAccumulateExternalKick(
                data.sourcePairUpdates[particleIndex], externalAcceleration, kickTimestep);
        }
        if (data.sinkAccelerations) {
            for (int dim = 0; dim < kSpatialDimensions; dim++) {
                if (parameters.accelerationGradientViscosity) {
                    data.sinkAccelerations[sinkIndex][dim] += acceleration[dim];
                } else {
                    data.sinkAccelerations[sinkIndex][dim] = acceleration[dim];
                }
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
        float const expansionCoefficient = 5.0f - 3.0f * gasAdiabaticIndex;
        internalEnergyRate +=
            expansionCoefficient * expansionRate * inverseScaleFactor * internalEnergy;
        flops += 6;    // Homogeneous thermal expansion; force sources count separately.
        if (data.sinkInternalEnergyRate) {
            data.sinkInternalEnergyRate[sinkIndex] =
                internalEnergyRate + simdSum(dissipativeEnergyRatePack);
            flops += SimdFloat::size();
        }
        if (data.sinkEntropyRate) {
            // Reversible PdV and homogeneous expansion leave A unchanged.
            data.sinkEntropyRate[sinkIndex] =
                sph_entropy::fromStoredEnergy(simdSum(dissipativeEnergyRatePack), density,
                    gasAdiabaticIndex, parameters.scaleFactor);
            flops += sph_flops::entropyConversionFlops(gasAdiabaticIndex);
            flops += SimdFloat::size() - 1;
        }
        if (parameters.phase == 1) {
            // clang-format off
            /*
             * Section 10.3, p. 563:
             *   δt_f = min_a (smoothingLength/|f_a|),   δt_cv = min_a smoothingLength/(soundSpeed +
             *   0.6 (α soundSpeed + β max_b μ_ab)), δt = 0.25 min(δt_f, δt_cv).
             * smoothingLength is the common effective radius 2σ (docs/monaghan_sph.md). The printed
             * δt_f lacks a square root and is used as sqrt(smoothingLength/|f_a|); max_b μ_ab is
             * read as the compression magnitude max_b(−μ_ab), or max_b(-v_ab.rhat) for
             * Phantom; the Courant factor and the calibrated
             * acceleration coefficient replace the fixed 0.25. The reciprocals are kept
             * apart for the diagnostics in SphRunReturn.
             *
             * Symbol directory (in the code below):
             *   v_ab → velocityDifference
             *   μ_ab → viscosityVelocity
             *   f_a → accelerationMagnitude
             *   δt → timeStep
             *   α → linearViscosityCoefficient
             *   β → quadraticViscosityCoefficient
             */
            // clang-format on
            float const smoothingLength =
                sinkKernelSmoothingLength / smoothingLengthPerEffectiveRadius;
            float const inverseSoundTimestep = soundSpeed / smoothingLength;
            float const inverseViscosityTimestep = 0.6f *
                (maximumBulkViscosityCoefficient * soundSpeed +
                    maximumVonNeumannRichtmyerCoefficient * maximumCompressionSpeed) /
                smoothingLength;
            float const inverseCourantTimestep = inverseSoundTimestep + inverseViscosityTimestep;
            float const accelerationMagnitude = acceleration.norm();
            float const inverseAccelerationTimestep = parameters.courantFactor *
                std::sqrt(parameters.scaleFactor * accelerationMagnitude / smoothingLength) /
                (data.accelerationTimestepFactor * kHydroAccelerationScale);
            float timeStep = std::fmin(parameters.maximumTimestep,
                parameters.courantFactor * parameters.scaleFactor /
                    std::fmax(
                        std::fmax(inverseCourantTimestep, inverseAccelerationTimestep), 1.0e-30f));
            // Bound reversible work; the approach-speed CFL term controls viscous heating.
            if (internalEnergyRate != 0.0f) {
                timeStep = std::fmin(timeStep,
                    parameters.courantFactor / float(monaghan::kViscosityWidthScale) *
                        internalEnergy / std::fabs(internalEnergyRate));
                flops += 3;    // Two divisions and a product; abs is a sign operation.
            }
            // Derived forward-Euler diffusion bound: Δτ Σ_b C_ab/a ≤ 1/2.
            // It uses the sum of positive exchange rates, even when net heating cancels.
            if (conductionExchangeRate > 0.0f) {
                timeStep = std::fmin(timeStep, 0.5f / conductionExchangeRate);
                flops += 1;
            }
            ALWAYS_ASSERT(std::isfinite(timeStep) && timeStep > 0.0f);
            result.minimumTimestep = std::min(result.minimumTimestep, double(timeStep));
            flops.doublePrecision += 1;
            int const hydroRung = parameters.globalTimestep ?
                0 :
                std::max(parameters.midpointHydro ? 0 : 1,
                    int(std::ceil(std::log2(parameters.baseTimestep / timeStep))));
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
            float gravityTimestep = parameters.maximumTimestep;
            int gravityRung = 0;
            if (!parameters.midpointHydro) {
                gravityTimestep = std::fmin(parameters.maximumTimestep,
                    data.accelerationTimestepFactor *
                        std::sqrt(parameters.scaleFactor * data.gravityEffectiveRadius /
                            std::fmax(gravityAcceleration.norm(), 1.0e-30f)));
                if (data.selfGravityEnabled) {
                    result.minimumTimestep =
                        std::min(result.minimumTimestep, double(gravityTimestep));
                    flops.doublePrecision += 1;
                }
                gravityRung = parameters.globalTimestep ?
                    0 :
                    std::max(
                        1, int(std::ceil(std::log2(parameters.baseTimestep / gravityTimestep))));
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
                                 parameters.baseTimestep, timeStep, parameters.maximumTimestep,
                                 parameters.courantFactor * parameters.scaleFactor /
                                     std::fmax(inverseCourantTimestep, 1.0e-30f),
                                 parameters.courantFactor * parameters.scaleFactor /
                                     std::fmax(inverseAccelerationTimestep, 1.0e-30f),
                                 internalEnergyRate != 0.0f ? parameters.courantFactor /
                                         float(monaghan::kViscosityWidthScale) * internalEnergy /
                                         std::fabs(internalEnergyRate) :
                                                              parameters.maximumTimestep,
                                 sinkKernelSmoothingLength, density, internalEnergy,
                                 internalEnergyRate, accelerationMagnitude, soundSpeed,
                                 maximumCompressionSpeed, maximumBulkViscosityCoefficient,
                                 gasMass / (gasMass - gradHCorrection), parameters.scaleFactor)
                          << std::flush;
#if COSMICTIGER_WITH_GRAVITY
                std::cout
                    << std::format(
                           "SPH gravity timestep: enabled={:d} gravityRung={:d} gravityDt={:e}\n",
                           int(data.selfGravityEnabled), gravityRung, gravityTimestep)
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
            result.maximumSignalSpeed =
                std::fmax(result.maximumSignalSpeed, inverseCourantTimestep * smoothingLength);
            result.inverseSoundTimestep =
                std::fmax(result.inverseSoundTimestep, inverseSoundTimestep);
            result.inverseViscosityTimestep =
                std::fmax(result.inverseViscosityTimestep, inverseViscosityTimestep);
            result.inverseAccelerationTimestep =
                std::fmax(result.inverseAccelerationTimestep, inverseAccelerationTimestep);
            // h/c/visc (8), acceleration norm/limit (12), timestep (2), hydro rung (3).
            // Exclude the signal-speed diagnostic.
            flops += parameters.globalTimestep ? 22 : 25;
        }
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops.counts());
    return result;
}

}    // namespace detail

SphRunReturn sphCpuHydro(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    // One dispatch per target node; the disabled specialization contains no diagnostic work.
    return data.sinkHydroDiagnostics ?
        detail::sphCpuHydroImpl<true>(parameters, data, targetNodeIndex) :
        detail::sphCpuHydroImpl<false>(parameters, data, targetNodeIndex);
}
