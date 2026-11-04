/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
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
    float const inverseScaleFactor = 1.0f / parameters.scaleFactor;
    float const adiabaticIndex = data.adiabaticIndex;
    float const pressureEnergyRatio = adiabaticIndex - 1.0f;    // P/(ρu) for every particle.
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4
    // defaults after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    float const smoothingLengthPerEffectiveRadius =
        float(selectedHydroKernel().smoothingLengthPerEffectiveRadius);
    flops.doublePrecision += 1;
    float const gasMass = data.gasMass;    // every gas particle has the same mass
    float const scaledGasMass = gasMass * inverseScaleFactor;
    float const viscosityRegularization = parameters.viscosityEpsilon;    // η²/h² of p. 551
    float const expansionRate = parameters.expansionRate;    // a·da/dt, the Hubble flow of a pair
    bool const applyPairKicks = data.sourcePairUpdates && parameters.hydroKickTimestep > 0.0;
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
                    position[kXDimension] = data.sourcePositionX[sourceParticleIndex];
                    position[kYDimension] = data.sourcePositionY[sourceParticleIndex];
                    position[kZDimension] = data.sourcePositionZ[sourceParticleIndex];
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
                    workspace.neighbors[offset + slot] = {position[kXDimension],
                        position[kYDimension], position[kZDimension],
                        data.sourceSmoothingLength[sourceParticleIndex],
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
             *   u_a → internalEnergy
             *   ρ_a → density
             *   γ → gasAdiabaticIndex
             */
            // clang-format on
            float const sinkKernelSmoothingLength =
                sinkSmoothingLength;    // support radius, 2h_a for the cubic spline
            float const density = sinkDensity;
            float const internalEnergy = sinkInternalEnergy;
            float const gasAdiabaticIndex = adiabaticIndex;
            float const pressure = pressureEnergyRatio * density * internalEnergy;
            float const soundSpeed =
                sqrtf(gasAdiabaticIndex * pressureEnergyRatio * internalEnergy);
            Position const positionX = data.sourcePositionX[particleIndex];
            Position const positionY = data.sourcePositionY[particleIndex];
            Position const positionZ = data.sourcePositionZ[particleIndex];
            Vector3<float> const velocity(data.sourceVelocityX[particleIndex],
                data.sourceVelocityY[particleIndex], data.sourceVelocityZ[particleIndex]);
            float const inverseSmoothingLength = 1.0f / sinkKernelSmoothingLength;
            float const inverseSmoothingLengthFourth =
                sqr(sqr(inverseSmoothingLength));    // 1/H_a⁴
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
                variational::pressureVolumeFactor(pressure, 1.0f, density), gasMass,
                gradHCorrection);
            flops += 5;
            Vector3<float> acceleration(0.0f);    // this thread's acceleration sum
            float internalEnergyRate = 0.0f;
            float viscousHeatingRate = 0.0f;    // this thread's thermal work sum
            float const bulkViscosityCoefficient = parameters.variableViscosity ?
                data.sourceBulkViscosityCoefficient[particleIndex] :
                parameters.bulkViscosityCoefficient;
            float maximumBulkViscosityCoefficient = bulkViscosityCoefficient;
            float maximumCompressionSpeed = 0.0f;    // max_b(−μ_ab) for δt_cv
            int maximumNeighborRung = 0;
            for (int neighborIndex = threadIndex; neighborIndex < workspace.neighbors.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.neighbors[neighborIndex];
                // Particle b. Pairs outside both supports or at zero separation are skipped.
                Vector3<float> const separation(distance(positionX, neighbor.positionX),
                    distance(positionY, neighbor.positionY),
                    distance(positionZ, neighbor.positionZ));
                float const separationSquared = separation.dot(separation);    // |r_ab|²
                float const neighborSmoothingLength = neighbor.smoothingLength;
                flops +=
                    8 + (separationSquared == 0.0f ? 0 : 2);    // Distances, dot, support squares.
                bool const supportsIntersect =
                    separationSquared < fmaxf(sinkKernelSmoothingLength * sinkKernelSmoothingLength,
                                            neighborSmoothingLength * neighborSmoothingLength);
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
                    sqrtf(gasAdiabaticIndex * pressureEnergyRatio * neighborInternalEnergy);
                float const pairDistance = sqrtf(separationSquared);    // |r_ab|
                // clang-format off
                /*
                 * Symmetric kernel of p. 554: ∇_a W̃_ab = ½[∇_a W(r_ab, smoothingLength) + ∇_a
                 * W(r_ab, neighborSmoothingLength)]. The selected kernel is W(r, H) = w(r/H)/H³, so
                 * each gradient equals w'(r/H) r_ab/(H⁴ r) and ∇_a W̃_ab = F_ab r_ab with the scalar
                 * F_ab.
                 *
                 * Symbol directory (in the code below):
                 *   F_ab → symmetricGradientFactor
                 *   r_ab → separation
                 */
                // clang-format on
                float const inverseNeighborSmoothingLength = 1.0f / neighborSmoothingLength;
                float const normalizedSinkRadius = pairDistance * inverseSmoothingLength;
                float const normalizedNeighborRadius =
                    pairDistance * inverseNeighborSmoothingLength;
                float const sinkGradientFactor =
                    dkernelWDq(normalizedSinkRadius) * inverseSmoothingLengthFourth / pairDistance;
                float const neighborGradientFactor = dkernelWDq(normalizedNeighborRadius) *
                    sqr(sqr(inverseNeighborSmoothingLength)) / pairDistance;
                float const symmetricGradientFactor =
                    0.5f * (sinkGradientFactor + neighborGradientFactor);
                // v_ab = v_a − v_b from the stored peculiar velocities. The physical relative
                // velocity adds the Hubble flow (a da/dt) r_ab and enters only the viscosity.
                Vector3<float> const velocityDifference = velocity -
                    Vector3<float>(neighbor.velocityX, neighbor.velocityY, neighbor.velocityZ);
                float const velocityDotSeparation = velocityDifference.dot(separation);
                float const physicalVelocityDotSeparation =
                    velocityDotSeparation + expansionRate * separationSquared;
                // Phantom uses radial approach speed, without Monaghan's h/eta regularization.
                float const viscosityVelocity = parameters.priceEtAl ?
                    fminf(physicalVelocityDotSeparation / pairDistance, 0.0f) :
                    monaghan::viscosityMu(separationSquared, physicalVelocityDotSeparation,
                        sinkKernelSmoothingLength, neighborSmoothingLength, viscosityRegularization,
                        smoothingLengthPerEffectiveRadius, float(monaghan::kViscosityWidthScale));
                float const neighborBulkViscosityCoefficient = parameters.variableViscosity ?
                    neighbor.bulkViscosityCoefficient :
                    bulkViscosityCoefficient;
                float const pairBulkViscosityCoefficient = parameters.variableViscosity ?
                    0.5f * bulkViscosityCoefficient + 0.5f * neighborBulkViscosityCoefficient :
                    bulkViscosityCoefficient;
                float const pairVonNeumannRichtmyerCoefficient =
                    parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                    parameters.vonNeumannRichtmyerCoefficient * pairBulkViscosityCoefficient :
                    parameters.vonNeumannRichtmyerCoefficient;
                maximumBulkViscosityCoefficient = fmaxf(maximumBulkViscosityCoefficient,
                    parameters.priceEtAl ?
                        fmaxf(bulkViscosityCoefficient, neighborBulkViscosityCoefficient) :
                        pairBulkViscosityCoefficient);
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
                 *   F_a → sinkGradientFactor
                 *   F_b → neighborGradientFactor
                 *   g_a → maskedSinkGradientFactor
                 *   g_b → maskedNeighborGradientFactor
                 */
                // clang-format on
                float const neighborPressureCoefficient = variational::directedCoefficient(
                    variational::pressureVolumeFactor(neighborPressure, 1.0f, neighborDensity),
                    gasMass, neighbor.gradHCorrection);
                float const pressureForce = inverseScaleFactor *
                    variational::pairForceFactor(sinkPressureCoefficient,
                        neighborPressureCoefficient, sinkGradientFactor, neighborGradientFactor);
                float viscousForce;
                float viscousHeating;
                float neighborViscousHeating = 0.0f;
                if (parameters.priceEtAl) {
                    /*
                     * Phantom (2018), equations (39)-(42), including each side's grad-H factor.
                     */
                    float const sinkViscousPressure =
                        price_etal::artificialPressure(density, soundSpeed, viscosityVelocity,
                            bulkViscosityCoefficient, pairVonNeumannRichtmyerCoefficient);
                    float const neighborViscousPressure = price_etal::artificialPressure(
                        neighborDensity, neighborSoundSpeed, viscosityVelocity,
                        neighborBulkViscosityCoefficient, pairVonNeumannRichtmyerCoefficient);
                    float sinkViscosityCoefficient = price_etal::directedViscosityCoefficient(
                        sinkViscousPressure, density, gasMass, gradHCorrection);
                    float neighborViscosityCoefficient =
                        price_etal::directedViscosityCoefficient(neighborViscousPressure,
                            neighborDensity, gasMass, neighbor.gradHCorrection);
                    if (parameters.balsaraSwitch) {
                        sinkViscosityCoefficient *= shearFactor;
                        neighborViscosityCoefficient *= shearFactor;
                        flops += 2;
                    }
                    viscousForce = inverseScaleFactor *
                        variational::pairForceFactor(sinkViscosityCoefficient,
                            neighborViscosityCoefficient, sinkGradientFactor,
                            neighborGradientFactor);
                    viscousHeating = inverseScaleFactor *
                        variational::pairWork(sinkViscosityCoefficient, sinkGradientFactor,
                            physicalVelocityDotSeparation);
                    if (applyPairKicks) {
                        // Negating both separation and relative velocity leaves
                        // their dot product unchanged in the reverse pair.
                        neighborViscousHeating = inverseScaleFactor *
                            variational::pairWork(neighborViscosityCoefficient,
                                neighborGradientFactor, physicalVelocityDotSeparation);
                        flops += 3;
                    }
                    flops += 28;    // radial division, two q/C terms, force/work
                } else {
                    float pairArtificialViscosity = monaghan::artificialViscosity(density,
                        neighborDensity, soundSpeed, neighborSoundSpeed, viscosityVelocity,
                        pairBulkViscosityCoefficient, pairVonNeumannRichtmyerCoefficient);
                    if (parameters.balsaraSwitch) {
                        pairArtificialViscosity *= shearFactor;
                        flops += 1;
                    }
                    viscousForce =
                        scaledGasMass * pairArtificialViscosity * symmetricGradientFactor;
                    viscousHeating = scaledGasMass * 0.5f * pairArtificialViscosity *
                        symmetricGradientFactor * physicalVelocityDotSeparation;
                    neighborViscousHeating = viscousHeating;
                    flops += sph_flops::kViscosityMu + sph_flops::kArtificialViscosity + 6;
                }
                Vector3<float> const pairForce = (pressureForce + viscousForce) * separation;
                acceleration -= pairForce;
                if (applyPairKicks &&
                    sphOwnsPairKick(data.sourceIds[particleIndex],
                        data.sourceIds[neighbor.sourceIndex], previousRung, neighbor.rung)) {
                    Vector3<float> const pairAcceleration(
                        -pairForce[kXDimension], -pairForce[kYDimension], -pairForce[kZDimension]);
                    double const targetEntropyRate = sph_entropy::fromStoredEnergy(
                        viscousHeating, density, adiabaticIndex, parameters.scaleFactor);
                    double const neighborEntropyRate =
                        sph_entropy::fromStoredEnergy(neighborViscousHeating, neighborDensity,
                            adiabaticIndex, parameters.scaleFactor);
                    sphAccumulatePairKick(data.sourcePairUpdates[particleIndex],
                        data.sourcePairUpdates[neighbor.sourceIndex], pairAcceleration,
                        targetEntropyRate, neighborEntropyRate, parameters.hydroKickTimestep);
                    flops.doublePrecision +=
                        13 + 2 * (adiabaticIndex == float(5.0 / 3.0) ? 12 : 14);
                }
                internalEnergyRate += inverseScaleFactor *
                    variational::pairWork(
                        sinkPressureCoefficient, sinkGradientFactor, velocityDotSeparation);
                viscousHeatingRate += viscousHeating;
                flops += 54;    // common geometry, pressure, and accumulation
                maximumCompressionSpeed = fmaxf(maximumCompressionSpeed, -viscosityVelocity);
                flops += sph_flops::densityKernelOperations(
                             selectedHydroKernel(), normalizedSinkRadius, true) +
                    sph_flops::densityKernelOperations(
                        selectedHydroKernel(), normalizedNeighborRadius, true);
            }
            sharedReduceAdd<float, kHydroBlockSize>(acceleration[kXDimension]);
            sharedReduceAdd<float, kHydroBlockSize>(acceleration[kYDimension]);
            sharedReduceAdd<float, kHydroBlockSize>(acceleration[kZDimension]);
            sharedReduceAdd<float, kHydroBlockSize>(internalEnergyRate);
            sharedReduceAdd<float, kHydroBlockSize>(viscousHeatingRate);
            sharedReduceMax<kHydroBlockSize>(maximumCompressionSpeed);
            sharedReduceMax<kHydroBlockSize>(maximumBulkViscosityCoefficient);
            if (parameters.midpointHydro && parameters.phase == 1) {
                sharedReduceMax<kHydroBlockSize>(maximumNeighborRung);
            }
            flops += 5 * detail::sphReductionFlops<kHydroBlockSize>();
            if (threadIndex == 0) {
                // Corrections are derived, never checkpointed or migrated; slots that the
                // density pass has not rebuilt hold NaN, so a stale source cannot pass silently.
                if (!(sph_volume::isFiniteValue(acceleration.dot(acceleration)) &&
                        sph_volume::isFiniteValue(internalEnergyRate) &&
                        sph_volume::isFiniteValue(viscousHeatingRate))) {
                    printf("SPH variational rates are not finite: smoothingLength=%e density=%e "
                           "ζ=%e\n",
                        sinkKernelSmoothingLength, density, gradHCorrection);
                    __trap();
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
                    acceleration = sphCompleteAcceleration(parameters,
                        Vector3<Position>(positionX, positionY, positionZ), velocity, acceleration,
                        gravityAcceleration, data.centralAcceleration, data.centralSmoothingLength,
                        flops);
                } else {
                    ALWAYS_ASSERT(parameters.phase == 0);
                }
                if (applyPairKicks) {
                    Vector3<float> const externalAcceleration = sphCompleteAcceleration(parameters,
                        Vector3<Position>(positionX, positionY, positionZ), velocity,
                        Vector3<float>(0.0f), Vector3<float>(0.0f), data.centralAcceleration,
                        data.centralSmoothingLength, flops);
                    sphAccumulateExternalKick(data.sourcePairUpdates[particleIndex],
                        externalAcceleration, parameters.hydroKickTimestep);
                    flops.doublePrecision += 6;
                }
                if (data.sinkAccelerations) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        data.sinkAccelerations[sinkIndex][dim] = acceleration[dim];
                    }
                }
                // E=a^2 u and homogeneous work du/dt=-3 H P/rho give
                // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E = (5-3 gamma) (da/dtau)/a E.
                float const expansionCoefficient = 5.0f - 3.0f * adiabaticIndex;
                internalEnergyRate +=
                    expansionCoefficient * expansionRate * inverseScaleFactor * internalEnergy;
                flops += 6;    // Homogeneous thermal expansion; force sources count separately.
                if (data.sinkInternalEnergyRate) {
                    data.sinkInternalEnergyRate[sinkIndex] =
                        internalEnergyRate + viscousHeatingRate;
                    flops += 1;
                }
                if (data.sinkEntropyRate) {
                    // Reversible PdV and homogeneous expansion leave A unchanged.
                    data.sinkEntropyRate[sinkIndex] = sph_entropy::fromStoredEnergy(
                        viscousHeatingRate, density, adiabaticIndex, parameters.scaleFactor);
                    flops.doublePrecision += adiabaticIndex == float(5.0 / 3.0) ? 12 : 14;
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
                    float const maximumVonNeumannRichtmyerCoefficient =
                        parameters.vonNeumannRichtmyerCoefficientFollowsBulk() ?
                        parameters.vonNeumannRichtmyerCoefficient *
                            maximumBulkViscosityCoefficient :
                        parameters.vonNeumannRichtmyerCoefficient;
                    float const inverseViscosityTimestep = 0.6f *
                        (maximumBulkViscosityCoefficient * soundSpeed +
                            maximumVonNeumannRichtmyerCoefficient * maximumCompressionSpeed) /
                        smoothingLength;
                    float const inverseCourantTimestep =
                        inverseSoundTimestep + inverseViscosityTimestep;
                    float const accelerationMagnitude = sqrtf(acceleration.dot(acceleration));
                    float const inverseAccelerationTimestep = parameters.courantFactor *
                        sqrtf(parameters.scaleFactor * accelerationMagnitude / smoothingLength) /
                        (data.accelerationTimestepFactor * kHydroAccelerationScale);
                    float timeStep = fminf(parameters.maximumTimestep,
                        parameters.courantFactor * parameters.scaleFactor /
                            fmaxf(fmaxf(inverseCourantTimestep, inverseAccelerationTimestep),
                                1.0e-30f));
                    // Bound reversible work; the approach-speed CFL term controls viscosity.
                    if (internalEnergyRate != 0.0f) {
                        flops += 3;
                        timeStep = fminf(timeStep,
                            parameters.courantFactor / float(monaghan::kViscosityWidthScale) *
                                internalEnergy / fabsf(internalEnergyRate));
                    }
                    ALWAYS_ASSERT(isfinite(timeStep) && timeStep > 0.0f);
                    atomicMin(&reduction->minimumTimestep, timeStep);
                    int const hydroRung = parameters.globalTimestep ?
                        0 :
                        max(parameters.midpointHydro ? 0 : 1,
                            int(ceilf(log2f(parameters.baseTimestep / timeStep))));
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
                        atomicCAS(&reduction->timestepFailureReported, 0, 1) == 0) {
                        printf(
                            "SPH timestep exceeds rung range: particle=%lli rung=%i hydroRung=%i "
                            "previousRung=%i minimumRung=%i baseDt=%e selectedDt=%e capDt=%e "
                            "courantDt=%e accelerationDt=%e thermalDt=%e H=%e rho=%e u=%e "
                            "duDt=%e acceleration=%e soundSpeed=%e compression=%e alpha=%e "
                            "omegaFromZeta=%e scaleFactor=%e\n",
                            static_cast<long long>(data.sinkParticleIndices[sinkIndex]), nextRung,
                            hydroRung, previousRung, parameters.minimumRung,
                            parameters.baseTimestep, timeStep, parameters.maximumTimestep,
                            parameters.courantFactor * parameters.scaleFactor /
                                fmaxf(inverseCourantTimestep, 1.0e-30f),
                            parameters.courantFactor * parameters.scaleFactor /
                                fmaxf(inverseAccelerationTimestep, 1.0e-30f),
                            internalEnergyRate != 0.0f ?
                                parameters.courantFactor / float(monaghan::kViscosityWidthScale) *
                                    internalEnergy / fabsf(internalEnergyRate) :
                                parameters.maximumTimestep,
                            sinkKernelSmoothingLength, density, internalEnergy, internalEnergyRate,
                            accelerationMagnitude, soundSpeed, maximumCompressionSpeed,
                            maximumBulkViscosityCoefficient, gasMass / (gasMass - gradHCorrection),
                            parameters.scaleFactor);
#if COSMICTIGER_WITH_GRAVITY
                        printf("SPH gravity timestep: enabled=%i gravityRung=%i gravityDt=%e\n",
                            int(data.selfGravityEnabled), gravityRung, gravityTimestep);
#endif
                    }
                    ALWAYS_ASSERT(nextRung < kMaxRung);
                    if (data.proposeRungs) {
                        data.sinkFlags[sinkIndex].setProposedRung(nextRung);
                    } else {
                        data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
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
            targetNodeIndex = atomicAdd(&reduction->nextNodeIndex, 1);
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
