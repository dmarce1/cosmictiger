/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/monaghan_sph.hpp>
#include <cosmictiger/sph/sph_cuda.hpp>

namespace detail {

struct HydroNeighbor {
    Position positionX;
    Position positionY;
    Position positionZ;
    float supportRadius;
    float velocityX;
    float velocityY;
    float velocityZ;
    float internalEnergy;
    float density;
    float pressureEnergyRatio;
    float adiabaticIndex;
    float balsaraFactor;
};

struct HydroWorkspace {
    DeviceVector<HydroNeighbor> neighbors;
};

}    // namespace detail

// Monaghan (1992), ARA&A 30, 543; cosmictiger/monaghan_sph.hpp gives the full citation
// and the notation. The pair loop implements equations 6.4-6.5 with the artificial
// viscosity of equations 4.1-4.2 and the symmetric kernel of p. 554; the time step
// follows section 10.3, p. 563. Particle a is the sink and particle b a staged neighbor,
// and the computation uses the paper's symbols. Positions and density are comoving; the
// stored velocities and internal energies carry the factors a and a² of
// docs/monaghan_sph.md, so every pair term is multiplied by m_b/a. The stored radius is
// the compact-support radius H; the effective smoothing width is 2σ.
// NOLINTBEGIN(readability-identifier-naming)
__global__ void sphCudaHydro(
    SphRunParameters parameters, SphRunCudaData data, SphReduction* reduction) {
    FlopCounter<> flops = 2;    // Inverse scale factor and gas mass scaling.
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
    // Keep the historical cubic viscosity scale while measuring every kernel
    // with the same effective width of 2 sigma: the paper smoothing length is
    // sqrt(5/6) times this width for the cubic spline.
    constexpr float kViscosityWidthScale = 0.9128709291752769f;
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4
    // defaults after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    float const supportRadiusPerEffectiveRadius =
        float(selectedHydroKernel().supportRadiusPerEffectiveRadius);
    float const m_b = data.gasMass;    // every gas particle has the same mass
    float const m_b_over_a = m_b * inverseScaleFactor;
    float const α = parameters.viscosityAlpha;
    float const β = parameters.viscosityBeta;
    float const ε = parameters.viscosityEpsilon;       // η²/h² of p. 551
    float const a_da_dt = parameters.expansionRate;    // a·da/dt, the Hubble flow of a pair
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
                    blockSize);
            for (int sourceParticleIndex = neighborNode.particleRange.first + threadIndex;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                Array<Position, NDIM> position;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    position[XDIM] = data.sourcePositionX[sourceParticleIndex];
                    position[YDIM] = data.sourcePositionY[sourceParticleIndex];
                    position[ZDIM] = data.sourcePositionZ[sourceParticleIndex];
                    contains = targetNode.outerBox.contains(position, &flops);
                    if (!contains) {
                        contains = sphSupportIntersects(targetNode.innerBox, position,
                            data.sourceSupportRadius[sourceParticleIndex], &flops);
                    }
                }
                int slot = contains;
                int total;
                computeIndices<HYDRO_BLOCK_SIZE>(slot, total);
                int const offset = workspace.neighbors.size();
                __syncthreads();
                workspace.neighbors.resize(offset + total);
                if (contains) {
                    flops += data.sourcePressureEnergyRatio ? 0 : 1;
                    workspace.neighbors[offset + slot] = {position[XDIM], position[YDIM],
                        position[ZDIM], data.sourceSupportRadius[sourceParticleIndex],
                        data.sourceVelocityX[sourceParticleIndex],
                        data.sourceVelocityY[sourceParticleIndex],
                        data.sourceVelocityZ[sourceParticleIndex],
                        data.sourceInternalEnergy[sourceParticleIndex],
                        data.sourceDensity[sourceParticleIndex],
                        data.sourcePressureEnergyRatio ?
                            data.sourcePressureEnergyRatio[sourceParticleIndex] :
                            adiabaticIndex - 1.0f,
                        data.sourceAdiabaticIndex ? data.sourceAdiabaticIndex[sourceParticleIndex] :
                                                    adiabaticIndex,
                        parameters.balsaraSwitch ? data.sourceBalsaraFactor[sourceParticleIndex] :
                                                   1.0f};
                }
            }
        }
        for (int particleIndex = targetNode.particleRange.first;
            particleIndex < targetNode.particleRange.second; particleIndex++) {
            __syncthreads();
            int const sinkIndex =
                targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
            int const previousRung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
            if (previousRung < parameters.minimumRung) {
                continue;
            }
            float const sinkSupportRadius = data.sourceSupportRadius[particleIndex];
            float const sinkDensity = data.sourceDensity[particleIndex];
            float const sinkInternalEnergy = data.sourceInternalEnergy[particleIndex];
            ALWAYS_ASSERT(
                sinkSupportRadius > 0.0f && sinkDensity > 0.0f && sinkInternalEnergy > 0.0f);
            float const sinkPressureEnergyRatio = data.sourcePressureEnergyRatio ?
                data.sourcePressureEnergyRatio[particleIndex] :
                adiabaticIndex - 1.0f;
            float const sinkAdiabaticIndex = data.sourceAdiabaticIndex ?
                data.sourceAdiabaticIndex[particleIndex] :
                adiabaticIndex;
            // Particle a. Section 3.5 leaves the equation of state free; the ideal gas gives
            // P_a = (γ_a − 1) ρ_a u_a and c_a² = γ_a P_a/ρ_a, where the optional composition
            // arrays replace γ_a − 1 and γ_a.
            float const H_a = sinkSupportRadius;    // support radius, 2h_a for the cubic spline
            float const ρ_a = sinkDensity;
            float const u_a = sinkInternalEnergy;
            float const γ_a = sinkAdiabaticIndex;
            float const P_a = sinkPressureEnergyRatio * ρ_a * u_a;
            float const c_a = sqrtf(γ_a * sinkPressureEnergyRatio * u_a);
            Position const x_a = data.sourcePositionX[particleIndex];
            Position const y_a = data.sourcePositionY[particleIndex];
            Position const z_a = data.sourcePositionZ[particleIndex];
            Vector3<float> const v_a(data.sourceVelocityX[particleIndex],
                data.sourceVelocityY[particleIndex], data.sourceVelocityZ[particleIndex]);
            float const one_over_H_a = 1.0f / H_a;
            float const one_over_H_a4 = sqr(sqr(one_over_H_a));    // 1/H_a⁴
            flops += 8 + (data.sourcePressureEnergyRatio ? 0 : 1);
            Vector3<float> dv_a_dt(0.0f);    // this thread's part of equation 6.4
            float du_a_dt = 0.0f;            // this thread's part of equation 6.5
            float μ_max = 0.0f;              // max_b(−μ_ab) for δt_cv
            for (int neighborIndex = threadIndex; neighborIndex < workspace.neighbors.size();
                neighborIndex += blockSize) {
                auto const& neighbor = workspace.neighbors[neighborIndex];
                // Particle b. Pairs outside both supports or at zero separation are skipped.
                Vector3<float> const r_ab(distance(x_a, neighbor.positionX),
                    distance(y_a, neighbor.positionY), distance(z_a, neighbor.positionZ));
                float const r_ab2 = r_ab.dot(r_ab);    // |r_ab|²
                float const H_b = neighbor.supportRadius;
                flops += 8 + (r_ab2 == 0.0f ? 0 : 2);    // Distances, dot, support squares.
                if (r_ab2 == 0.0f || r_ab2 >= fmaxf(H_a * H_a, H_b * H_b)) {
                    continue;
                }
                float const ρ_b = neighbor.density;
                float const u_b = neighbor.internalEnergy;
                float const γ_b = neighbor.adiabaticIndex;
                float const P_b = neighbor.pressureEnergyRatio * ρ_b * u_b;
                float const c_b = sqrtf(γ_b * neighbor.pressureEnergyRatio * u_b);
                float const r = sqrtf(r_ab2);    // |r_ab|
                // Symmetric kernel of p. 554: ∇_a W̃_ab = ½[∇_a W(r_ab, h_a) + ∇_a W(r_ab, h_b)].
                // The selected kernel is W(r, H) = w(r/H)/H³, so each gradient equals
                // w'(r/H) r_ab/(H⁴ r) and ∇_a W̃_ab = F_ab r_ab with the scalar F_ab.
                float const one_over_H_b = 1.0f / H_b;
                float const q_a = r * one_over_H_a;
                float const q_b = r * one_over_H_b;
                float const F_a = dkernelWDq(q_a) * one_over_H_a4 / r;
                float const F_b = dkernelWDq(q_b) * sqr(sqr(one_over_H_b)) / r;
                float const F_ab = 0.5f * (F_a + F_b);
                Vector3<float> const gradW̃_ab = F_ab * r_ab;    // ∇_a W̃_ab
                // v_ab = v_a − v_b from the stored peculiar velocities. The physical relative
                // velocity adds the Hubble flow (a da/dt) r_ab and enters only the viscosity.
                Vector3<float> const v_ab = v_a -
                    Vector3<float>(neighbor.velocityX, neighbor.velocityY, neighbor.velocityZ);
                float const v_ab_dot_r_ab = v_ab.dot(r_ab);
                float const v_ab_dot_r_ab_physical = v_ab_dot_r_ab + a_da_dt * r_ab2;
                // Equation 4.2 and the Π_ab of p. 550.
                float const μ_ab = monaghan_sph::viscosityMu(r_ab2, v_ab_dot_r_ab_physical, H_a,
                    H_b, ε, supportRadiusPerEffectiveRadius, kViscosityWidthScale);
                float Π_ab = monaghan_sph::artificialViscosity(ρ_a, ρ_b, c_a, c_b, μ_ab, α, β);
                if (parameters.balsaraSwitch) {
                    // GADGET-4 equation (72): Π_ab <- ½(f_a^AV + f_b^AV) Π_ab.
                    float const f_ab_AV =
                        0.5f * (data.sourceBalsaraFactor[particleIndex] + neighbor.balsaraFactor);
                    Π_ab *= f_ab_AV;
                    flops += 3;
                }
                float const P_over_ρ2_ab =
                    monaghan_sph::pressureCoefficient(ρ_a, ρ_b, P_a, P_b);    // P_b/ρ_b² + P_a/ρ_a²
                // Equation 6.4: dv_a/dt = −Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) ∇_a W̃_ab.
                dv_a_dt -= (m_b_over_a * (P_over_ρ2_ab + Π_ab)) * gradW̃_ab;
                // Equation 6.5: du_a/dt = ½ Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) v_ab·∇_a W̃_ab,
                // with v_ab·∇_a W̃_ab = F_ab v_ab·r_ab. The pressure work takes the peculiar
                // velocity and the viscous heating the physical one.
                du_a_dt += m_b_over_a *
                    monaghan_sph::pairEnergyRate(
                        P_over_ρ2_ab, Π_ab, v_ab_dot_r_ab * F_ab, v_ab_dot_r_ab_physical * F_ab);
                μ_max = fmaxf(μ_max, -μ_ab);
                // EOS/root 6; inverse support 1; gradients 13; relative velocity
                // and dot products 10; force accumulation 8; energy factors 4.
                flops += 42 + sph_flops::kViscosityMu + sph_flops::kArtificialViscosity +
                    sph_flops::kPressureCoefficient + sph_flops::kPairEnergyRate +
                    sph_flops::densityKernelFlops(selectedHydroKernel(), q_a, true) +
                    sph_flops::densityKernelFlops(selectedHydroKernel(), q_b, true);
            }
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dv_a_dt[XDIM]);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dv_a_dt[YDIM]);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dv_a_dt[ZDIM]);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(du_a_dt);
            sharedReduceMax<HYDRO_BLOCK_SIZE>(μ_max);
            flops += 4 * detail::sphReductionFlops<HYDRO_BLOCK_SIZE>();
            if (threadIndex == 0) {
                // Accelerations outside the paper: self-gravity, the optional uniform, midplane
                // and central forces of the test problems, and velocity damping while relaxing
                // the gas sphere.
                Vector3<float> const gravityAcceleration = data.selfGravityEnabled ?
                    Vector3<float>(data.sinkGravityAccelerationX[sinkIndex],
                        data.sinkGravityAccelerationY[sinkIndex],
                        data.sinkGravityAccelerationZ[sinkIndex]) :
                    Vector3<float>(0.0f);
                dv_a_dt += gravityAcceleration;
                dv_a_dt[YDIM] += parameters.uniformAccelerationY;
                dv_a_dt[ZDIM] += parameters.midplaneAccelerationZ(z_a);
                flops += 6;    // Force sums and the midplane fixed-point conversion.
                if (data.centralAcceleration != 0.0f) {
                    Vector3<float> const separation(distance(x_a, Position(0.5)),
                        distance(y_a, Position(0.5)), distance(z_a, Position(0.5)));
                    float const radius = sqrtf(separation.dot(separation));
                    float const inverseSupportRadius = 1.0f / data.centralSupportRadius;
                    float const normalizedRadius = radius * inverseSupportRadius;
                    float const force = data.centralAcceleration *
                        radial_kernel::forceFactor(selectedGravityKernel(), normalizedRadius) *
                        inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
                    dv_a_dt -= force * separation;
                    flops +=
                        24 + sph_flops::forceKernelFlops(selectedGravityKernel(), normalizedRadius);
                }
                dv_a_dt -= parameters.velocityDampingRate * v_a;
                for (int dim = 0; dim < NDIM; dim++) {
                    data.sinkAccelerations[sinkIndex].acceleration[dim] = dv_a_dt[dim];
                }
                // E=a^2 u and homogeneous work du/dt=-3 H P/rho give
                // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E.
                float const expansionCoefficient = data.sourcePressureEnergyRatio ?
                    2.0f - 3.0f * sinkPressureEnergyRatio :
                    5.0f - 3.0f * adiabaticIndex;
                du_a_dt += expansionCoefficient * a_da_dt * inverseScaleFactor * u_a;
                flops += 12;    // Damping and homogeneous thermal expansion.
                data.sinkInternalEnergyRate[sinkIndex] = du_a_dt;
                if (parameters.phase == 1) {
                    // Section 10.3, p. 563:
                    //   δt_f = min_a (h_a/|f_a|),
                    //   δt_cv = min_a h_a/(c_a + 0.6 (α c_a + β max_b μ_ab)),
                    //   δt = 0.25 min(δt_f, δt_cv).
                    // h_a is the common effective radius 2σ (docs/monaghan_sph.md). The
                    // printed δt_f lacks a square root and is used as sqrt(h_a/|f_a|);
                    // max_b μ_ab is read as the compression magnitude max_b(−μ_ab); the
                    // Courant factor and the calibrated acceleration coefficient replace
                    // the fixed 0.25. The reciprocals are kept apart for the diagnostics.
                    float const h_a = H_a / supportRadiusPerEffectiveRadius;
                    float const one_over_δt_c = c_a / h_a;
                    float const one_over_δt_visc = 0.6f * (α * c_a + β * μ_max) / h_a;
                    float const one_over_δt_cv = one_over_δt_c + one_over_δt_visc;
                    float const f_a = sqrtf(dv_a_dt.dot(dv_a_dt));
                    float const one_over_δt_f = parameters.courantFactor *
                        sqrtf(parameters.scaleFactor * f_a / h_a) /
                        (data.accelerationTimestepFactor * kHydroAccelerationScale);
                    float δt = fminf(parameters.maximumTimestep,
                        parameters.courantFactor * parameters.scaleFactor /
                            fmaxf(fmaxf(one_over_δt_cv, one_over_δt_f), 1.0e-30f));
                    // Bound fractional thermal change in the existing explicit integrator.
                    if (du_a_dt != 0.0f) {
                        flops += 3;
                        δt = fminf(δt,
                            parameters.courantFactor / kViscosityWidthScale * u_a / fabsf(du_a_dt));
                    }
                    float const gravityTimestep = fminf(parameters.maximumTimestep,
                        data.accelerationTimestepFactor *
                            sqrtf(parameters.scaleFactor * data.gravityEffectiveRadius /
                                fmaxf(sqrtf(gravityAcceleration.dot(gravityAcceleration)),
                                    1.0e-30f)));
                    ALWAYS_ASSERT(isfinite(δt) && δt > 0.0f);
                    int const hydroRung = max(1, int(ceilf(log2f(parameters.baseTimestep / δt))));
                    int const gravityRung =
                        max(1, int(ceilf(log2f(parameters.baseTimestep / gravityTimestep))));
                    int const nextRung = max(max(parameters.minimumRung, previousRung - 1),
                        max(hydroRung, data.selfGravityEnabled ? gravityRung : 1));
                    ALWAYS_ASSERT(nextRung < MAX_RUNG);
                    data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
                    maximumHydroRung = max(maximumHydroRung, hydroRung);
                    maximumGravityRung = max(maximumGravityRung, gravityRung);
                    maximumRung = max(maximumRung, nextRung);
                    maximumSignalSpeed = fmaxf(maximumSignalSpeed, one_over_δt_cv * h_a);
                    // Exclude the signal-speed report and the gravity timestep/rung
                    // computed only for diagnostics when self-gravity is disabled.
                    flops += 25 + (data.selfGravityEnabled ? 13 : 0);
                    atomicMax(&reduction->inverseSoundTimestep, one_over_δt_c);
                    atomicMax(&reduction->inverseViscosityTimestep, one_over_δt_visc);
                    atomicMax(&reduction->inverseAccelerationTimestep, one_over_δt_f);
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
// NOLINTEND(readability-identifier-naming)
