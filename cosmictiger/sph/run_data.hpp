/*
 CosmicTiger - A cosmological N-Body code
 Shared CPU/CUDA SPH source data, force completion, and support calculations.
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
    Position* sourcePositionX;      // Comoving source x coordinate in box units.
    Position* sourcePositionY;      // Comoving source y coordinate in box units.
    Position* sourcePositionZ;      // Comoving source z coordinate in box units.
    float* sourceVelocityX;         // Stored peculiar x velocity, a times physical velocity.
    float* sourceVelocityY;         // Stored peculiar y velocity, a times physical velocity.
    float* sourceVelocityZ;         // Stored peculiar z velocity, a times physical velocity.
    float* sourceInternalEnergy;    // Stored specific energy E=a²u in both formulations.
    float* sourceDensity;           // Comoving mass density from the current density pass.
    float* sourceBalsaraFactor;     // Dimensionless shear suppression of artificial viscosity.
    float* sourceBulkViscosityCoefficient;    // Dimensionless bulk viscosity coefficient α.
    float* sourceSmoothingLength;             // Full compact-support h, as in particle storage.
    // Variational operator: zeta_i of cosmictiger/sph/volume.hpp from this step's density
    // pass.
    float* sourceGradHCorrection;    // ζᵢ, with the same units as gasMass.
    // Frozen before sink writes; required by prehydro and rung limiting.
    char* sourceRungs;    // Frozen source timestep rung in [0,kMaxRung).
    // Optional conservative midpoint output. Source identities and compensated float
    // increments live only for the frozen pass and its reverse MPI exchange.
    SphPairSourceId const* sourceIds = nullptr;    // Owner rank/index for unique pair scheduling.
    SphPairUpdate* sourcePairUpdates = nullptr;    // Event-local velocity and thermal increments.

    // Local sink arrays indexed by the SPH particle index.
    float* sinkThermalRate = nullptr;    // dA/dτ or dE/dτ, selected by energyEvolution.
    float* sinkDensity;            // Output comoving density, indexed by local SPH particle index.
    float* sinkGradHCorrection;    // Rebuilt with the density on every preparation pass.
    uint8_t* sinkBalsaraFactor;    // Encoded dimensionless Balsara factor.
    Float16Storage* sinkBulkViscosityCoefficient;    // Persistent dimensionless viscosity α.
    SphParticleFlags* sinkFlags;             // Convergence, semiactive, and proposed-rung bits.
    float* sinkSmoothingLength;              // Full compact-support h, as in particle storage.
    float* sinkGravityAccelerationX;         // Completed x gravity acceleration dv/dτ.
    float* sinkGravityAccelerationY;         // Completed y gravity acceleration dv/dτ.
    float* sinkGravityAccelerationZ;         // Completed z gravity acceleration dv/dτ.
    ParticleInteger* sinkParticleIndices;    // Map local SPH index to full particle index.
    char* sinkParticleRungs;    // Indexed through sinkParticleIndices into the full particle array.
    // A phase-1 hydro pass can store its rung proposals in sinkFlags instead of
    // sinkParticleRungs, preserving the closing rung until the driver commits it.
    bool proposeRungs = false;    // Store proposals without changing the closing rung.
    // Optional globally closed rung candidates, indexed by full particle index.
    char* sinkRungCandidates = nullptr;    // Globally closed proposals, or null when unused.

    int numberOfTargetNodes;             // Number of entries in targetNodeIndices.
    SphTreeNode* treeNodes;              // Staged source and local target tree nodes.
    int* targetNodeIndices;              // Target node indices in treeNodes.
    int* neighborNodeIndices;            // Flattened neighbor node indices in treeNodes.
    float targetNeighborCount;           // Weighted neighbor target for the smoothing-length solve.
    float gasMass;                       // Equal mass of every gas particle, in code mass units.
    float adiabaticIndex;                // Ideal-gas index shared by all gas particles.
    bool energyEvolution = false;        // E=a²u evolves directly; false evolves entropy A.
    bool selfGravityEnabled;             // Include gas self-gravity and its timestep bound.
    float gravityEffectiveRadius;        // Twice the gravity kernel standard deviation.
    float accelerationTimestepFactor;    // Coefficient of the acceleration timestep bound.
    // Optional external central potential, used by the orbiting-disc problem.
    float centralAcceleration;       // Zero disables this force.
    float centralSmoothingLength;    // Support radius of the selected gravity kernel.

    // One particle's vectors, read in one call from the component arrays above.
    template <int dimension = 0>
    CUDA_EXPORT sph_geometry::SpatialVector<Position> sourcePosition(ParticleInteger index) const {
        return sph_geometry::loadSpatialVector<dimension>(
            sourcePositionX, sourcePositionY, sourcePositionZ, index);
    }
    /// Gather the source's active velocity components into one spatial vector.
    template <int dimension = 0>
    CUDA_EXPORT sph_geometry::SpatialVector<float> sourceVelocity(ParticleInteger index) const {
        return sph_geometry::loadSpatialVector<dimension>(
            sourceVelocityX, sourceVelocityY, sourceVelocityZ, index);
    }
    /// Gather the sink's active gravity-acceleration components into one spatial
    /// vector.
    template <int dimension = 0>
    CUDA_EXPORT sph_geometry::SpatialVector<float> sinkGravityAcceleration(
        ParticleInteger index) const {
        return sph_geometry::loadSpatialVector<dimension>(
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
 *   dv𞁑/dτ ← dv𞁑/dτ + (dv𞁑/dτ)_G + g_y ŷ + gₘ n̂ − g𞁞 f(q𞁑𞁞) h𞁞⁻ᴰ r𞁑𞁞 − λ v𞁑,
 *   r𞁑𞁞 = x𞁑 − x𞁞,  q𞁑𞁞 = |r𞁑𞁞|/h𞁞.
 * Symbol directory:
 *   x𞁑, v𞁑 → position, velocity; dv𞁑/dτ → acceleration; (dv𞁑/dτ)_G → gravity acceleration
 *   g_y, gₘ → uniform y acceleration, signed midplane acceleration; λ → velocity damping rate
 *   n̂ → last active coordinate direction: ŷ in 2D, ẑ in 3D
 *   g𞁞 → central acceleration; h𞁞 → central smoothing length; h𞁞⁻¹ → its inverse
 *   x𞁞 → box center; r𞁑𞁞, |r𞁑𞁞| → separation from the center, its magnitude
 *   q𞁑𞁞 → normalized radius; f(q𞁑𞁞) → gravity kernel force factor; f𞁑𞁞 → central force factor
 *   Code names: dv𞁑_dτ = dv𞁑/dτ; dv𞁑_dτ_G = (dv𞁑/dτ)_G; ih𞁞 = h𞁞⁻¹
 *     abs_r𞁑𞁞 = |r𞁑𞁞|.
 */
// clang-format on
template <int dimension = 0>
CUDA_EXPORT inline Vector3<float> sphCompleteAcceleration(SphRunParameters const& parameters,
    Vector3<Position> const& x𞁑, Vector3<float> const& v𞁑, Vector3<float> dv𞁑_dτ,
    Vector3<float> const& dv𞁑_dτ_G, float g𞁞, float h𞁞, sph_flops::Counter& flops) {
#if COSMICTIGER_WITH_GRAVITY
    dv𞁑_dτ += dv𞁑_dτ_G;
    flops += 3;
#endif
    dv𞁑_dτ[kYi] += parameters.uniformAccelerationY;
    int const midplaneAxis = spatialDimensions<dimension>() - 1;
    dv𞁑_dτ[midplaneAxis] += parameters.midplaneAcceleration(x𞁑[midplaneAxis]);
    flops += 2;
    flops.doublePrecision += sph_flops::positionConstructionFlops();
    if (g𞁞 != 0.0f) {
        Vector3<float> const r𞁑𞁞(distance(x𞁑[kXi], Position(0.5)), distance(x𞁑[kYi], Position(0.5)),
            distance(x𞁑[kZi], Position(0.5)));
        float const abs_r𞁑𞁞 = r𞁑𞁞.norm();
        float const ih𞁞 = 1.0f / h𞁞;
        float const q𞁑𞁞 = abs_r𞁑𞁞 * ih𞁞;
        float const f𞁑𞁞 = g𞁞 * radial_kernel::forceFactor(selectedGravityKernel<float>(), q𞁑𞁞) *
            sph_geometry::inverseVolume<dimension>(ih𞁞);
        dv𞁑_dτ -= f𞁑𞁞 * r𞁑𞁞;
        flops += 21 + sph_flops::forceKernelOperations(selectedGravityKernel<float>(), q𞁑𞁞);
        flops.doublePrecision += 3 * sph_flops::positionConstructionFlops();
    }
    dv𞁑_dτ -= parameters.velocityDampingRate * v𞁑;
    flops += 6;
    return dv𞁑_dτ;
}

// The one boundary between the spatial-vector kernels and the padded assembly above, which
// also takes the padded position x𞁑. The result keeps the active axes.
template <int dimension = 0>
CUDA_EXPORT inline sph_geometry::SpatialVector<float> sphCompleteAcceleration(
    SphRunParameters const& parameters, Vector3<Position> const& x𞁑,
    sph_geometry::SpatialVector<float> const& v𞁑, sph_geometry::SpatialVector<float> const& dv𞁑_dτ,
    sph_geometry::SpatialVector<float> const& dv𞁑_dτ_G, float g𞁞, float h𞁞,
    sph_flops::Counter& flops) {
    Vector3<float> const padded = sphCompleteAcceleration<dimension>(parameters, x𞁑,
        sph_geometry::paddedVector<dimension>(v𞁑), sph_geometry::paddedVector<dimension>(dv𞁑_dτ),
        sph_geometry::paddedVector<dimension>(dv𞁑_dτ_G), g𞁞, h𞁞, flops);
    sph_geometry::SpatialVector<float> result(0.0f);
    for (int dim = 0; dim < spatialDimensions<dimension>(); ++dim) {
        result[dim] = padded[dim];
    }
    return result;
}

inline constexpr int kMaxNeighborRungDifference = 1;
inline constexpr float kSphSmoothingLengthTolerance = float(1.0e-4);
// Half the periodic box in box units: the nearest-image limit of the neighbor search. A
// smoothing length that would have to exceed it stops the run.
inline constexpr float kSphHalfBoxSmoothingLength = 0.5f;

// Target of the smoothing-length solves: ΣW = Σ𞁓 w(q𞁑𞁓) = 3 Nₛₑₜ/(4π).
// Nₛₑₜ → target neighbor count.
template <int dimension = 0>
CUDA_EXPORT inline float sphTargetKernelSum(float Nₛₑₜ) {
    return Nₛₑₜ * float(1.0 / sph_geometry::unitBallVolume<dimension>());
}

struct SphGradHEvaluation {
    sph_volume::GradHCorrection<float> correction;
    sph_volume::SupportConstraint constraint;
    float relativeResidual;    // (ΣW − target)/target at the stored smoothing length
};

// The variational density pass's correction for one sink, from its unweighted sums
// ΣW = Σ𞁓 w(q𞁑𞁓) and Σ𞁓 q𞁑𞁓 w′(q𞁑𞁓) at the converged smoothing length.
// Every gas particle has the mass m and the volume weight x = m.
// Symbol directory: h𞁑, hₘᵢₙ, hₘₐₓ → smoothing length and its bounds; Nₛₑₜ → target neighbor
// count; ΣWₛₑₜ → target kernel sum; m → gas mass; Σ𞁓 q𞁑𞁓 w′ → kernel derivative moment.
// Code names: Σqwʹ = Σ𞁓 q𞁑𞁓 w′.
template <int dimension = 0>
CUDA_EXPORT inline SphGradHEvaluation sphGradHEvaluation(
    float h𞁑, float hₘᵢₙ, float hₘₐₓ, float Nₛₑₜ, float m, float ΣW, float Σqwʹ) {
    float const ΣWₛₑₜ = sphTargetKernelSum<dimension>(Nₛₑₜ);
    SphGradHEvaluation result;
    result.relativeResidual = (ΣW - ΣWₛₑₜ) / ΣWₛₑₜ;
    result.constraint = sph_volume::classifySupport(
        h𞁑, hₘᵢₙ, hₘₐₓ, result.relativeResidual, kSphSmoothingLengthTolerance);
    result.correction = sph_volume::gradHCorrection<dimension>(
        sph_volume::equalWeightSupportSums(ΣW, Σqwʹ, m), result.constraint);
    return result;
}

// A source can reach any sink in this box through a periodic image. Expanding
// the unwrapped interval before testing avoids inconsistent endpoint wrapping.
// x𞁓 → padded source position; h𞁓 → source smoothing length.
CUDA_EXPORT inline bool sphSupportIntersects(SphBox const& sinkBox, Vector3<Position> const& x𞁓,
    float h𞁓, sph_flops::Counter* flops = nullptr) {
    SphBox expanded = sinkBox;
    for (int dim = 0; dim < kDimensions; dim++) {
        expanded.begin[dim] -= RangeFixed(h𞁓);
        expanded.end[dim] += RangeFixed(h𞁓);
    }
    if (flops) {
        // Each float support needs one single-precision fixed-point scale.
        flops->singlePrecision += 2 * kDimensions;
    }
    return expanded.contains(x𞁓, flops ? &flops->doublePrecision : nullptr);
}
