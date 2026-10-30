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
#include <cosmictiger/sph/monaghan/equations.hpp>
#include <cosmictiger/sph/monaghan/viscosity.hpp>
#include <cosmictiger/sph/sph_cpu.hpp>

// Monaghan (1992), ARA&A 30, 543; cosmictiger/sph/monaghan/equations.hpp gives the full citation
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
    FlopCounter<> flops = 2;    // Scale-factor reciprocal and gas mass scaling.
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const neighbors = detail::sphCpuNeighbors(data, targetNode, true, flops);
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
        float const α_a = parameters.variableViscosity ? data.sourceViscosityAlpha[particleIndex] :
                                                         parameters.viscosityAlpha;
        SimdFloat α_max_pack(α_a);
        float const one_over_H_a = 1.0f / H_a;
        flops += 8 + int(data.sourcePressureEnergyRatio == nullptr);
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
            auto const coupled = data.sourceHydroCoupling ?
                (detail::sphCpuLoad(data.sourceHydroCoupling, neighbors, first, 0.0f) > zero) &
                    (SimdFloat(data.sourceHydroCoupling[particleIndex]) > zero) :
                (one > zero);
            auto const interacts = coupled & detail::sphCpuValidLanes(neighbors, first) &
                (r_ab2_staged > zero) &
                (r_ab2_staged < simdMax(SimdFloat(H_a * H_a), sqr(H_b_staged)));
            // Three fixed scales, five dot operations and H_b² per lane; H_a² is scalar.
            flops += 9 * SimdFloat::size() + 1;
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
            SimdFloat const q_a = r * SimdFloat(one_over_H_a);
            SimdFloat const q_b = r * one_over_H_b;
            SimdFloat const F_a = dkernelWDq(q_a) * SimdFloat(one_over_H_a4) / r;
            SimdFloat const F_b = dkernelWDq(q_b) * sqr(sqr(one_over_H_b)) / r;
            flops += sph_flops::densityKernelFlops(selectedHydroKernel(), q_a, true) +
                sph_flops::densityKernelFlops(selectedHydroKernel(), q_b, true);
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
            SimdFloat const μ_ab = monaghan::viscosityMu(r_ab2, v_ab_dot_r_ab_physical,
                SimdFloat(H_a), H_b, SimdFloat(ε), SimdFloat(supportRadiusPerEffectiveRadius),
                SimdFloat(kViscosityWidthScale));
            // Morris & Monaghan (1997), equation (3) and p. 44: symmetric α_ab.
            SimdFloat const α_b = parameters.variableViscosity ?
                simdSelect(interacts,
                    detail::sphCpuLoad(data.sourceViscosityAlpha, neighbors, first, α_a),
                    SimdFloat(α_a)) :
                SimdFloat(α_a);
            SimdFloat const α_ab =
                parameters.variableViscosity ? half * SimdFloat(α_a) + half * α_b : SimdFloat(α_a);
            // Price et al. (2018), equation (41): β stays constant under their switch.
            SimdFloat const β_ab = parameters.betaFollowsAlpha() ?
                SimdFloat(parameters.viscosityBeta) * α_ab :
                SimdFloat(parameters.viscosityBeta);
            α_max_pack = simdMax(α_max_pack, simdSelect(interacts, α_ab, SimdFloat(α_a)));
            SimdFloat Π_ab = monaghan::artificialViscosity(
                SimdFloat(ρ_a), ρ_b, SimdFloat(c_a), c_b, μ_ab, α_ab, β_ab);
            flops += (3 * int(parameters.variableViscosity) + int(parameters.betaFollowsAlpha())) *
                SimdFloat::size();
            if (parameters.balsaraSwitch) {
                // GADGET-4 equation (72): Π_ab <- ½(f_a^AV + f_b^AV) Π_ab.
                SimdFloat const f_ab_AV = half *
                    (SimdFloat(data.sourceBalsaraFactor[particleIndex]) +
                        detail::sphCpuLoad(data.sourceBalsaraFactor, neighbors, first, 0.0f));
                Π_ab *= f_ab_AV;
                flops += 3 * SimdFloat::size();
            }
            SimdFloat const P_over_ρ2_ab = monaghan::pressureCoefficient(
                SimdFloat(ρ_a), ρ_b, SimdFloat(P_a), P_b);    // P_b/ρ_b² + P_a/ρ_a²
            // Equation 6.4: dv_a/dt = −Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) ∇_a W̃_ab.
            dv_a_dt_pack -= (m_b_over_a * (P_over_ρ2_ab + Π_ab)) * gradW̃_ab;
            // Equation 6.5: du_a/dt = ½ Σ_b m_b (P_b/ρ_b² + P_a/ρ_a² + Π_ab) v_ab·∇_a W̃_ab,
            // with v_ab·∇_a W̃_ab = F_ab v_ab·r_ab. The pressure work takes the peculiar
            // velocity and the viscous heating the physical one.
            du_a_dt_pack += m_b_over_a *
                monaghan::pairEnergyRate(
                    P_over_ρ2_ab, Π_ab, v_ab_dot_r_ab * F_ab, v_ab_dot_r_ab_physical * F_ab);
            μ_max_pack = simdMax(μ_max_pack, -μ_ab);
            // EOS/radius (6), inverse H (1), gradients (13), velocity products (10),
            // Monaghan helpers (28), and momentum/thermal accumulation (12).
            flops += SimdFloat::size() *
                    (42 + sph_flops::kViscosityMu + sph_flops::kArtificialViscosity +
                        sph_flops::kPressureCoefficient + sph_flops::kPairEnergyRate) +
                int(data.sourcePressureEnergyRatio == nullptr);
        }
        Vector3<float> dv_a_dt(simdSum(dv_a_dt_pack[kXDimension]),
            simdSum(dv_a_dt_pack[kYDimension]), simdSum(dv_a_dt_pack[kZDimension]));
        float du_a_dt = simdSum(du_a_dt_pack);
        flops +=
            4 * (SimdFloat::size() - 1);    // Four additive reductions; maxima are comparisons.
        float const α_max = simdMaximum(α_max_pack);
        float const β_max = parameters.betaFollowsAlpha() ? parameters.viscosityBeta * α_max :
                                                            parameters.viscosityBeta;
        float const μ_max = simdMaximum(μ_max_pack);    // max_b(−μ_ab) over approaching pairs
        // Accelerations outside the paper: self-gravity, the optional uniform, midplane and
        // central forces of the test problems, and velocity damping while relaxing the gas sphere.
#if COSMICTIGER_WITH_GRAVITY
        Vector3<float> const gravityAcceleration = data.selfGravityEnabled ?
            Vector3<float>(data.sinkGravityAccelerationX[sinkIndex],
                data.sinkGravityAccelerationY[sinkIndex],
                data.sinkGravityAccelerationZ[sinkIndex]) :
            Vector3<float>(0.0f);
        dv_a_dt += gravityAcceleration;
        flops += 3;
#endif
        dv_a_dt[kYDimension] += parameters.uniformAccelerationY;
        dv_a_dt[kZDimension] +=
            parameters.midplaneAccelerationZ(data.sourcePositionZ[particleIndex]);
        flops += 3;    // Two acceleration additions and fixed-point midpoint conversion.
        if (data.centralAcceleration != 0.0f) {
            Vector3<float> const separation(
                distance(data.sourcePositionX[particleIndex], Position(0.5)),
                distance(data.sourcePositionY[particleIndex], Position(0.5)),
                distance(data.sourcePositionZ[particleIndex], Position(0.5)));
            float const radius = separation.norm();
            float const inverseSupportRadius = 1.0f / data.centralSupportRadius;
            float const q = radius * inverseSupportRadius;
            float const force = data.centralAcceleration *
                radial_kernel::forceFactor(selectedGravityKernel(), q) * inverseSupportRadius *
                inverseSupportRadius * inverseSupportRadius;
            dv_a_dt -= force * separation;
            flops += 24 + sph_flops::forceKernelFlops(selectedGravityKernel(), q);
        }
        dv_a_dt -= parameters.velocityDampingRate * v_a;
        for (int dim = 0; dim < kDimensions; dim++) {
            data.sinkAccelerations[sinkIndex].acceleration[dim] = dv_a_dt[dim];
        }
        // E=a^2 u and homogeneous adiabatic work du/dt=-3 H P/rho give
        // dE/dtau=(2-3 P/(rho*u)) (da/dtau)/a E for a composition-dependent EOS.
        float const expansionCoefficient = data.sourcePressureEnergyRatio ?
            2.0f - 3.0f * sinkPressureEnergyRatio :
            5.0f - 3.0f * adiabaticIndex;
        du_a_dt += expansionCoefficient * a_da_dt * inverseScaleFactor * u_a;
        flops += 12;    // Six damping operations and six cosmological energy operations.
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
            float const one_over_δt_visc = 0.6f * (α_max * c_a + β_max * μ_max) / h_a;
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
                flops += 3;    // Two divisions and a product; abs is a sign operation.
            }
            ALWAYS_ASSERT(std::isfinite(δt) && δt > 0.0f);
            result.minimumTimestep = std::min(result.minimumTimestep, double(δt));
            int const hydroRung = parameters.globalTimestep ?
                0 :
                std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / δt))));
            int nextRung = std::max(std::max(parameters.minimumRung, previousRung - 1), hydroRung);
#if COSMICTIGER_WITH_GRAVITY
            float const gravityTimestep = std::fmin(parameters.maximumTimestep,
                data.accelerationTimestepFactor *
                    std::sqrt(parameters.scaleFactor * data.gravityEffectiveRadius /
                        std::fmax(gravityAcceleration.norm(), 1.0e-30f)));
            if (data.selfGravityEnabled) {
                result.minimumTimestep = std::min(result.minimumTimestep, double(gravityTimestep));
            }
            int const gravityRung = parameters.globalTimestep ?
                0 :
                std::max(1, int(std::ceil(std::log2(parameters.baseTimestep / gravityTimestep))));
            nextRung = std::max(nextRung, data.selfGravityEnabled ? gravityRung : 1);
            result.maximumGravityRung = std::max(result.maximumGravityRung, gravityRung);
            // Exclude the bound computed only for disabled-gravity diagnostics.
            flops += data.selfGravityEnabled ? (parameters.globalTimestep ? 10 : 13) : 0;
#endif
            if (parameters.globalTimestep) {
                nextRung = 0;
            }
            ALWAYS_ASSERT(nextRung < kMaxRung);
            data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]] = nextRung;
            result.maximumHydroRung = std::max(result.maximumHydroRung, hydroRung);
            result.maximumRung = std::max(result.maximumRung, nextRung);
            result.maximumSignalSpeed = std::fmax(result.maximumSignalSpeed, one_over_δt_cv * h_a);
            result.inverseSoundTimestep = std::fmax(result.inverseSoundTimestep, one_over_δt_c);
            result.inverseViscosityTimestep =
                std::fmax(result.inverseViscosityTimestep, one_over_δt_visc);
            result.inverseAccelerationTimestep =
                std::fmax(result.inverseAccelerationTimestep, one_over_δt_f);
            // h/c/visc (8), acceleration norm/limit (12), timestep (2), hydro rung (3).
            // Exclude the signal-speed diagnostic.
            flops += parameters.globalTimestep ? 22 : 25;
        }
    }
    result.floatingPointOperations = FlopCount(flops);
    addCpuFlops(flops);
    return result;
}
// NOLINTEND(readability-identifier-naming)
