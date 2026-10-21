/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/monaghan_sph.hpp>
#include <cosmictiger/sph/sph_cpu.hpp>

// Monaghan (1992), ARA&A 30, 543; cosmictiger/monaghan_sph.hpp gives the full citation
// and the notation. The pair loop implements equations 6.4-6.5 with the artificial
// viscosity of equations 4.1-4.2 and the symmetric kernel of p. 554; the time step
// follows section 10.3, p. 563. Particle a is the sink and particle b a pack of staged
// neighbors, and the computation uses the paper's symbols. Positions and density are
// comoving; the stored velocities and internal energies carry the factors a and a² of
// docs/monaghan_sph.md, so every pair term is multiplied by m_b/a.
// NOLINTBEGIN(readability-identifier-naming)
SphRunReturn sphCpuHydro(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    SphRunReturn result;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true);
    float const inverseScaleFactor = 1.0f / parameters.scaleFactor;
    float const adiabaticIndex = data.adiabaticIndex;
    // Keep the historical cubic viscosity scale while measuring every kernel with the
    // same effective width of 2σ: the paper's h is sqrt(5/6) times this width for the
    // cubic spline.
    constexpr float kViscosityWidthScale = 0.9128709291752769f;
    // Calibrate the hydro acceleration limit to the former cubic/Ferrers F4 defaults
    // after gravity eta and smoothing lengths adopt the common width.
    constexpr float kHydroAccelerationScale = 0.7115965735877551f;
    float const supportRadiusPerEffectiveRadius =
        float(selectedHydroKernel().supportRadiusPerEffectiveRadius);
    SimdFloat const zero(0.0f);
    SimdFloat const one(1.0f);
    SimdFloat const half(0.5f);
    float const m_b = data.gasMass;    // every gas particle has the same mass
    SimdFloat const m_b_over_a(m_b * inverseScaleFactor);
    float const α = parameters.viscosityAlpha;
    float const β = parameters.viscosityBeta;
    float const ε = parameters.viscosityEpsilon;       // η²/h² of p. 551
    float const a_da_dt = parameters.expansionRate;    // a·da/dt, the Hubble flow of a pair
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; particleIndex++) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        int const previousRung = data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]];
        if (previousRung < parameters.minimumRung) {
            continue;
        }
        float const sinkSupportRadius = data.sourceSupportRadius[particleIndex];
        float const sinkDensity = data.sourceDensity[particleIndex];
        float const sinkInternalEnergy = data.sourceInternalEnergy[particleIndex];
        ALWAYS_ASSERT(sinkSupportRadius > 0.0f && sinkDensity > 0.0f && sinkInternalEnergy > 0.0f);
        float const sinkPressureEnergyRatio = data.sourcePressureEnergyRatio ?
            data.sourcePressureEnergyRatio[particleIndex] :
            adiabaticIndex - 1.0f;
        float const sinkAdiabaticIndex =
            data.sourceAdiabaticIndex ? data.sourceAdiabaticIndex[particleIndex] : adiabaticIndex;
        // Particle a. Section 3.5 leaves the equation of state free; the ideal gas gives
        // P_a = (γ_a − 1) ρ_a u_a and c_a² = γ_a P_a/ρ_a, where the optional composition
        // arrays replace γ_a − 1 and γ_a.
        float const H_a = sinkSupportRadius;    // support radius, 2h_a for the cubic spline
        float const ρ_a = sinkDensity;
        float const u_a = sinkInternalEnergy;
        float const γ_a = sinkAdiabaticIndex;
        float const P_a = sinkPressureEnergyRatio * ρ_a * u_a;
        float const c_a = std::sqrt(γ_a * sinkPressureEnergyRatio * u_a);
        Vector3<float> const v_a(data.sourceVelocityX[particleIndex],
            data.sourceVelocityY[particleIndex], data.sourceVelocityZ[particleIndex]);
        Vector3<SimdFloat> const v_a_pack(v_a);
        float const one_over_H_a = 1.0f / H_a;
        float const one_over_H_a4 = sqr(sqr(one_over_H_a));    // 1/H_a⁴
        Vector3<SimdFloat> dv_a_dt_pack(zero);                 // lane partial sums of equation 6.4
        SimdFloat du_a_dt_pack(zero);                          // lane partial sums of equation 6.5
        SimdFloat μ_max_pack(zero);                            // lane maxima of −μ_ab for δt_cv
        for (size_t first = 0; first < neighbors.size(); first += SimdFloat::size()) {
            // Particle b: one SIMD pack of staged neighbors. Lanes beyond the list, outside
            // both supports, or at zero separation are excluded. Their inputs are replaced
            // before any division or square root; masking only the result can still trap.
            Vector3<SimdFloat> const r_ab =
                detail::sphCpuSeparations(data, particleIndex, neighbors, first);    // r_a − r_b
            SimdFloat const r_ab2_staged = r_ab.dot(r_ab);
            SimdFloat const H_b_staged =
                detail::sphCpuLoad(data.sourceSupportRadius, neighbors, first, 1.0f);
            auto const interacts = detail::sphCpuValidLanes(neighbors, first) &
                (r_ab2_staged > zero) &
                (r_ab2_staged < simdMax(SimdFloat(H_a * H_a), sqr(H_b_staged)));
            if (!simdAnyOf(interacts)) {
                continue;
            }
            SimdFloat const H_b = simdSelect(interacts, H_b_staged, one);
            SimdFloat const ρ_b = simdSelect(
                interacts, detail::sphCpuLoad(data.sourceDensity, neighbors, first, 1.0f), one);
            SimdFloat const u_b = simdSelect(interacts,
                detail::sphCpuLoad(data.sourceInternalEnergy, neighbors, first, 1.0f), one);
            SimdFloat const pressureEnergyRatio_b = data.sourcePressureEnergyRatio ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourcePressureEnergyRatio, neighbors, first, 1.0f),
                    one) :
                SimdFloat(adiabaticIndex - 1.0f);
            SimdFloat const γ_b = data.sourceAdiabaticIndex ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourceAdiabaticIndex, neighbors, first, 1.0f), one) :
                SimdFloat(adiabaticIndex);
            SimdFloat const P_b = pressureEnergyRatio_b * ρ_b * u_b;
            SimdFloat const c_b = simdSqrt(γ_b * pressureEnergyRatio_b * u_b);
            SimdFloat const r_ab2 = simdSelect(interacts, r_ab2_staged, one);    // |r_ab|²
            SimdFloat const r = simdSqrt(r_ab2);                                 // |r_ab|
            // Symmetric kernel of p. 554: ∇_a W̃_ab = ½[∇_a W(r_ab, h_a) + ∇_a W(r_ab, h_b)].
            // The selected kernel is W(r, H) = w(r/H)/H³, so each gradient equals
            // w'(r/H) r_ab/(H⁴ r) and ∇_a W̃_ab = F_ab r_ab with the scalar F_ab.
            SimdFloat const one_over_H_b = one / H_b;
            SimdFloat const F_a =
                dkernelWDq(r * SimdFloat(one_over_H_a)) * SimdFloat(one_over_H_a4) / r;
            SimdFloat const F_b = dkernelWDq(r * one_over_H_b) * sqr(sqr(one_over_H_b)) / r;
            SimdFloat const F_ab = simdSelect(interacts, half * (F_a + F_b), zero);
            Vector3<SimdFloat> const gradW̃_ab = F_ab * r_ab;    // ∇_a W̃_ab
            // v_ab = v_a − v_b from the stored peculiar velocities. The physical relative
            // velocity adds the Hubble flow (a da/dt) r_ab and enters only the viscosity.
            Vector3<SimdFloat> const v_ab = simdSelect(interacts,
                v_a_pack -
                    detail::sphCpuLoad3(data.sourceVelocityX, data.sourceVelocityY,
                        data.sourceVelocityZ, neighbors, first, 0.0f),
                Vector3<SimdFloat>(zero));
            SimdFloat const v_ab_dot_r_ab = v_ab.dot(r_ab);
            SimdFloat const v_ab_dot_r_ab_physical =
                v_ab_dot_r_ab + SimdFloat(a_da_dt) * simdSelect(interacts, r_ab2, zero);
            // Equation 4.2 and the Π_ab of p. 550.
            SimdFloat const μ_ab = monaghan_sph::viscosityMu(r_ab2, v_ab_dot_r_ab_physical,
                SimdFloat(H_a), H_b, SimdFloat(ε), SimdFloat(supportRadiusPerEffectiveRadius),
                SimdFloat(kViscosityWidthScale));
            SimdFloat const Π_ab = monaghan_sph::artificialViscosity(
                SimdFloat(ρ_a), ρ_b, SimdFloat(c_a), c_b, μ_ab, SimdFloat(α), SimdFloat(β));
            SimdFloat const P_over_ρ2_ab = monaghan_sph::pressureCoefficient(
                SimdFloat(ρ_a), ρ_b, SimdFloat(P_a), P_b);    // P_b/ρ_b² + P_a/ρ_a²
            // Equation 6.4: dv_a/dt = −Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) ∇_a W̃_ab.
            dv_a_dt_pack -= (m_b_over_a * (P_over_ρ2_ab + Π_ab)) * gradW̃_ab;
            // Equation 6.5: du_a/dt = ½ Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) v_ab·∇_a W̃_ab,
            // with v_ab·∇_a W̃_ab = F_ab v_ab·r_ab. The pressure work takes the peculiar
            // velocity and the viscous heating the physical one.
            du_a_dt_pack += m_b_over_a *
                monaghan_sph::pairEnergyRate(
                    P_over_ρ2_ab, Π_ab, v_ab_dot_r_ab * F_ab, v_ab_dot_r_ab_physical * F_ab);
            μ_max_pack = simdMax(μ_max_pack, -μ_ab);
        }
        Vector3<float> dv_a_dt(
            simdSum(dv_a_dt_pack[XDIM]), simdSum(dv_a_dt_pack[YDIM]), simdSum(dv_a_dt_pack[ZDIM]));
        float du_a_dt = simdSum(du_a_dt_pack);
        float const μ_max = simdMaximum(μ_max_pack);    // max_b(−μ_ab) over approaching pairs
        // Accelerations outside the paper: self-gravity, the optional uniform, midplane and
        // central forces of the test problems, and velocity damping while relaxing the gas sphere.
        Vector3<float> const gravityAcceleration = data.selfGravityEnabled ?
            Vector3<float>(data.sinkGravityAccelerationX[sinkIndex],
                data.sinkGravityAccelerationY[sinkIndex],
                data.sinkGravityAccelerationZ[sinkIndex]) :
            Vector3<float>(0.0f);
        dv_a_dt += gravityAcceleration;
        dv_a_dt[YDIM] += parameters.uniformAccelerationY;
        dv_a_dt[ZDIM] += parameters.midplaneAccelerationZ(data.sourcePositionZ[particleIndex]);
        if (data.centralAcceleration != 0.0f) {
            Vector3<float> const separation(
                distance(data.sourcePositionX[particleIndex], Fixed32(0.5)),
                distance(data.sourcePositionY[particleIndex], Fixed32(0.5)),
                distance(data.sourcePositionZ[particleIndex], Fixed32(0.5)));
            float const radius = separation.norm();
            float const inverseSupportRadius = 1.0f / data.centralSupportRadius;
            float const force = data.centralAcceleration *
                radial_kernel::forceFactor(selectedGravityKernel(), radius * inverseSupportRadius) *
                inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
            dv_a_dt -= force * separation;
        }
        dv_a_dt -= parameters.velocityDampingRate * v_a;
        for (int dim = 0; dim < NDIM; dim++) {
            data.sinkAccelerations[sinkIndex].acceleration[dim] = dv_a_dt[dim];
        }
        // E=a^2 u and homogeneous adiabatic work du/dt=-3 H P/rho give
        // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E for a composition-dependent EOS.
        float const expansionCoefficient = data.sourcePressureEnergyRatio ?
            2.0f - 3.0f * sinkPressureEnergyRatio :
            5.0f - 3.0f * adiabaticIndex;
        du_a_dt += expansionCoefficient * a_da_dt * inverseScaleFactor * u_a;
        data.sinkInternalEnergyRate[sinkIndex] = du_a_dt;
        if (parameters.phase == 1) {
            // Section 10.3, p. 563:
            //   δt_f = min_a (h_a/|f_a|),   δt_cv = min_a h_a/(c_a + 0.6 (α c_a + β max_b μ_ab)),
            //   δt = 0.25 min(δt_f, δt_cv).
            // h_a is the common effective radius 2σ (docs/monaghan_sph.md). The printed δt_f
            // lacks a square root and is used as sqrt(h_a/|f_a|); max_b μ_ab is read as the
            // compression magnitude max_b(−μ_ab); the Courant factor and the calibrated
            // acceleration coefficient replace the fixed 0.25. The reciprocals are kept
            // apart for the diagnostics in SphRunReturn.
            float const h_a = H_a / supportRadiusPerEffectiveRadius;
            float const one_over_δt_c = c_a / h_a;
            float const one_over_δt_visc = 0.6f * (α * c_a + β * μ_max) / h_a;
            float const one_over_δt_cv = one_over_δt_c + one_over_δt_visc;
            float const f_a = dv_a_dt.norm();
            float const one_over_δt_f = parameters.courantFactor *
                std::sqrt(parameters.scaleFactor * f_a / h_a) /
                (data.accelerationTimestepFactor * kHydroAccelerationScale);
            float δt = std::fmin(parameters.maximumTimestep,
                parameters.courantFactor * parameters.scaleFactor /
                    std::fmax(std::fmax(one_over_δt_cv, one_over_δt_f), 1.0e-30f));
            // Bound the fractional thermal change in the existing explicit integrator.
            if (du_a_dt != 0.0f) {
                δt = std::fmin(
                    δt, parameters.courantFactor / kViscosityWidthScale * u_a / std::fabs(du_a_dt));
            }
            float const gravityTimestep = std::fmin(parameters.maximumTimestep,
                data.accelerationTimestepFactor *
                    std::sqrt(parameters.scaleFactor * data.gravityEffectiveRadius /
                        std::fmax(gravityAcceleration.norm(), 1.0e-30f)));
            ALWAYS_ASSERT(std::isfinite(δt) && δt > 0.0f);
            int const hydroRung =
                std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / δt))));
            int const gravityRung =
                std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / gravityTimestep))));
            int const nextRung = std::max(std::max(parameters.minimumRung, previousRung - 1),
                std::max(hydroRung, data.selfGravityEnabled ? gravityRung : 1));
            ALWAYS_ASSERT(nextRung < MAX_RUNG);
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
            result.maximumHydroRung = std::max(result.maximumHydroRung, hydroRung);
            result.maximumGravityRung = std::max(result.maximumGravityRung, gravityRung);
            result.maximumRung = std::max(result.maximumRung, nextRung);
            result.maximumSignalSpeed = std::fmax(result.maximumSignalSpeed, one_over_δt_cv * h_a);
            result.inverseSoundTimestep = std::fmax(result.inverseSoundTimestep, one_over_δt_c);
            result.inverseViscosityTimestep =
                std::fmax(result.inverseViscosityTimestep, one_over_δt_visc);
            result.inverseAccelerationTimestep =
                std::fmax(result.inverseAccelerationTimestep, one_over_δt_f);
        }
    }
    return result;
}
// NOLINTEND(readability-identifier-naming)
