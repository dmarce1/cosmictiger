/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstdint>
#include <numbers>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/pair_kick.hpp>
#include <cosmictiger/sph/particle_flags.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/sph/volume.hpp>

struct SphRunData {
    // Staged source arrays, including the sources that are also local sinks.
    Position* sourcePositionX;
    Position* sourcePositionY;
    Position* sourcePositionZ;
    float* sourceVelocityX;
    float* sourceVelocityY;
    float* sourceVelocityZ;
    float* sourceInternalEnergy;
    float* sourceDensity;
    float* sourceBalsaraFactor;
    float* sourceBulkViscosityCoefficient;
    float* sourceSmoothingLength;    // Full compact-support H, as in particle storage.
    // Variational operator: zeta_i of cosmictiger/sph/volume.hpp from this step's density
    // pass.
    float* sourceGradHCorrection;
    // Frozen before sink writes; required by prehydro and rung limiting.
    char* sourceRungs;
    // Optional conservative midpoint output. Source identities and compensated float
    // increments live only for the frozen pass and its reverse MPI exchange.
    SphPairSourceId const* sourceIds = nullptr;
    SphPairUpdate* sourcePairUpdates = nullptr;

    // Local sink arrays indexed by the SPH particle index.
    // Optional total dE/dtau diagnostic; it is not integrated by production kicks.
    float* sinkInternalEnergyRate = nullptr;
    float* sinkEntropyRate = nullptr;    // dA/dtau from nonadiabatic heating and conduction.
    float* sinkDensity;
    float* sinkGradHCorrection;    // Rebuilt with the density on every preparation pass.
    uint8_t* sinkBalsaraFactor;
    Float16Storage* sinkBulkViscosityCoefficient;
    SphParticleFlags* sinkFlags;
    float* sinkSmoothingLength;    // Full compact-support H, as in particle storage.
    SphAcceleration* sinkAccelerations;
    float* sinkGravityAccelerationX;
    float* sinkGravityAccelerationY;
    float* sinkGravityAccelerationZ;
    ParticleInteger* sinkParticleIndices;
    char* sinkParticleRungs;    // Indexed through sinkParticleIndices into the full particle array.
    // A phase-1 hydro pass can store its rung proposals in sinkFlags instead of
    // sinkParticleRungs, preserving the closing rung until the driver commits it.
    bool proposeRungs = false;
    // Optional globally closed rung candidates, indexed by full particle index.
    char* sinkRungCandidates = nullptr;

    int numberOfTargetNodes;
    SphTreeNode* treeNodes;
    int* targetNodeIndices;
    int* neighborNodeIndices;
    float targetNeighborCount;    // Weighted neighbor target for the support-radius solve.
    float gasMass;
    float adiabaticIndex;    // Ideal-gas index shared by all gas particles.
    bool selfGravityEnabled;
    float gravityEffectiveRadius;        // Twice the gravity kernel standard deviation.
    float accelerationTimestepFactor;    // Coefficient of the acceleration timestep bound.
    // Optional external central potential, used by the orbiting-disc problem.
    float centralAcceleration;       // Zero disables this force.
    float centralSmoothingLength;    // Support radius of the selected gravity kernel.

    // One particle's vectors, read in one call from the component arrays above.
    CUDA_EXPORT sph_geometry::SpatialVector<Position> sourcePosition(ParticleInteger index) const {
        return sph_geometry::loadSpatialVector(
            sourcePositionX, sourcePositionY, sourcePositionZ, index);
    }
    CUDA_EXPORT sph_geometry::SpatialVector<float> sourceVelocity(ParticleInteger index) const {
        return sph_geometry::loadSpatialVector(
            sourceVelocityX, sourceVelocityY, sourceVelocityZ, index);
    }
    CUDA_EXPORT sph_geometry::SpatialVector<float> sinkGravityAcceleration(
        ParticleInteger index) const {
        return sph_geometry::loadSpatialVector(
            sinkGravityAccelerationX, sinkGravityAccelerationY, sinkGravityAccelerationZ, index);
    }
    // Box tests and source assembly take padded records, with the stored inactive-axis padding.
    CUDA_EXPORT Vector3<Position> sourcePaddedPosition(ParticleInteger index) const {
        return Vector3<Position>(
            sourcePositionX[index], sourcePositionY[index], sourcePositionZ[index]);
    }
};

// clang-format off
/*
 * Preserve the force-source arithmetic order for both the unsplit kernels and
 * the post-gravity completion of a phase-zero pair pass. Locally derived sum of the sources:
 *   dv_a/dτ ← dv_a/dτ + (dv_a/dτ)_grav + g_y ŷ + g_z ẑ − g_c f(q_ac) H_c⁻ᴰ r_ac − λ v_a,
 *   r_ac = x_a − x_c,  q_ac = |r_ac|/H_c.
 * Symbol directory:
 *   x_a, v_a → position, velocity; dv_a/dτ → acceleration; (dv_a/dτ)_grav → gravity acceleration
 *   g_y, g_z → uniform y acceleration, midplane z acceleration; λ → velocity damping rate
 *   g_c → central acceleration; H_c → central smoothing length; H_c⁻¹ → its inverse
 *   x_c → box center; r_ac, |r_ac| → separation from the center, its magnitude
 *   q_ac → normalized radius; f(q_ac) → gravity kernel force factor; f_ac → central force factor
 *   Code names: dv_a_dτ = dv_a/dτ; dv_a_dτ_grav = (dv_a/dτ)_grav; H_inv_c = H_c⁻¹;
 *     abs_r_ac = |r_ac|.
 */
// clang-format on
CUDA_EXPORT inline Vector3<float> sphCompleteAcceleration(SphRunParameters const& parameters,
    Vector3<Position> const& x_a, Vector3<float> const& v_a, Vector3<float> dv_a_dτ,
    Vector3<float> const& dv_a_dτ_grav, float g_c, float H_c, sph_flops::Counter& flops) {
#if COSMICTIGER_WITH_GRAVITY
    dv_a_dτ += dv_a_dτ_grav;
    flops += 3;
#endif
    dv_a_dτ[kYi] += parameters.uniformAccelerationY;
    dv_a_dτ[kZi] += parameters.midplaneAccelerationZ(x_a[kZi]);
    flops += 2;
    flops.doublePrecision += sph_flops::positionConstructionFlops();
    if (g_c != 0.0f) {
        Vector3<float> const r_ac(distance(x_a[kXi], Position(0.5)),
            distance(x_a[kYi], Position(0.5)), distance(x_a[kZi], Position(0.5)));
        float const abs_r_ac = r_ac.norm();
        float const H_inv_c = 1.0f / H_c;
        float const q_ac = abs_r_ac * H_inv_c;
        float const f_ac = g_c * radial_kernel::forceFactor(selectedGravityKernel<float>(), q_ac) *
            sph_geometry::inverseVolume(H_inv_c);
        dv_a_dτ -= f_ac * r_ac;
        flops += 21 + sph_flops::forceKernelOperations(selectedGravityKernel<float>(), q_ac);
        flops.doublePrecision += 3 * sph_flops::positionConstructionFlops();
    }
    dv_a_dτ -= parameters.velocityDampingRate * v_a;
    flops += 6;
    return dv_a_dτ;
}

// The one boundary between the spatial-vector kernels and the padded assembly above, which
// also takes the padded position x_a. The result keeps the active axes.
CUDA_EXPORT inline sph_geometry::SpatialVector<float> sphCompleteAcceleration(
    SphRunParameters const& parameters, Vector3<Position> const& x_a,
    sph_geometry::SpatialVector<float> const& v_a,
    sph_geometry::SpatialVector<float> const& dv_a_dτ,
    sph_geometry::SpatialVector<float> const& dv_a_dτ_grav, float g_c, float H_c,
    sph_flops::Counter& flops) {
    Vector3<float> const padded = sphCompleteAcceleration(parameters, x_a,
        sph_geometry::paddedVector(v_a), sph_geometry::paddedVector(dv_a_dτ),
        sph_geometry::paddedVector(dv_a_dτ_grav), g_c, H_c, flops);
    sph_geometry::SpatialVector<float> result;
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        result[dim] = padded[dim];
    }
    return result;
}

inline constexpr int kMaxNeighborRungDifference = 1;
inline constexpr float kSphSmoothingLengthTolerance = float(1.0e-4);

// Target of the support solves: ΣW = Σ_b w(q_ab) = 3 N_target/(4π).
// N_target → target neighbor count.
CUDA_EXPORT inline float sphTargetKernelSum(float N_target) {
    return N_target * float(1.0 / sph_geometry::kUnitBallVolume);
}

struct SphGradHEvaluation {
    sph_volume::GradHCorrection<float> correction;
    sph_volume::SupportConstraint constraint;
    float relativeResidual;    // (ΣW − target)/target at the stored support radius
};

// The variational density pass's correction for one sink, from its unweighted sums
// ΣW = Σ_b w(q_ab) and Σ_b q_ab w′(q_ab) at the converged support.
// Every gas particle has the mass m and the volume weight x = m.
// Symbol directory: H_a, H_min, H_max → support radius and its bounds; N_target → target neighbor
// count; ΣW_target → target kernel sum; m → gas mass; Σ_b q_ab w′ → kernel derivative moment.
// Code names: Σqwʹ = Σ_b q_ab w′.
CUDA_EXPORT inline SphGradHEvaluation sphGradHEvaluation(
    float H_a, float H_min, float H_max, float N_target, float m, float ΣW, float Σqwʹ) {
    float const ΣW_target = sphTargetKernelSum(N_target);
    SphGradHEvaluation result;
    result.relativeResidual = (ΣW - ΣW_target) / ΣW_target;
    result.constraint = sph_volume::classifySupport(
        H_a, H_min, H_max, result.relativeResidual, kSphSmoothingLengthTolerance);
    result.correction = sph_volume::gradHCorrection(
        sph_volume::equalWeightSupportSums(ΣW, Σqwʹ, m), result.constraint);
    return result;
}

// A source can reach any sink in this box through a periodic image. Expanding
// the unwrapped interval before testing avoids inconsistent endpoint wrapping.
// x_b → padded source position; H_b → source support radius.
CUDA_EXPORT inline bool sphSupportIntersects(Fixed32Range const& sinkBox,
    Vector3<Position> const& x_b, float H_b, sph_flops::Counter* flops = nullptr) {
    Fixed32Range expanded = sinkBox;
    for (int dim = 0; dim < kDimensions; dim++) {
        expanded.begin[dim] -= RangeFixed(H_b);
        expanded.end[dim] += RangeFixed(H_b);
    }
    if (flops) {
        // Each float support needs one single-precision fixed-point scale.
        flops->singlePrecision += 2 * kDimensions;
    }
    return expanded.contains(x_b, flops ? &flops->doublePrecision : nullptr);
}
