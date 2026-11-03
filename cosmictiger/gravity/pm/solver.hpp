/* CosmicTiger - scalar particle-mesh (PM) solve; BSL-1.0. */
#pragma once

#include <cstdint>

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

struct PmDiagnostics {
    int meshSize = 0;
    int cloudOrder = 0;
    // Box-unit Gaussian split radius; zero for pure PM.
    double splitRadius = 0;
    double sourceMass = 0;
    double depositedMass = 0;
    uint64_t sourceParticles = 0;
    // Cumulative per-locality generations, retained across explicit destruction.
    uint64_t planBuilds = 0;
    uint64_t solveCount = 0;
    // Forward and inverse transforms executed by the last solve on this mesh.
    uint64_t forwardTransforms = 0;
    uint64_t inverseTransforms = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & meshSize & cloudOrder & splitRadius & sourceMass & depositedMass &
            sourceParticles & planBuilds & solveCount & forwardTransforms & inverseTransforms;
    }
};

struct PmSolveRequest {
    bool doPotential = true;
    // Retain per-particle result arrays for diagnostics (pmAcceleration/pmPotential).
    bool storeFields = false;
    // Write the gm-scaled long acceleration into the particles' long-acceleration field.
    bool applyToParticles = true;
    // Timestep proposal: dt = eta * sqrt(scale * effectiveRadius / |acceleration|),
    // bounded by maximumTimestep. A nonpositive eta disables the proposal.
    double scale = 1.0;
    double eta = 0.0;
    double maximumTimestep = 1.0e30;
};

struct PmSolveResult {
    // Global sum of 0.5 * mass * phi with the discrete self potential removed,
    // scaled by gm; zero unless the potential was requested.
    double potentialEnergy = 0.0;
    // Global minimum of the PM acceleration timestep bound (1e30 when disabled).
    double minimumTimestep = 1.0e30;
    double sourceMass = 0.0;
    double depositedMass = 0.0;
    uint64_t sourceParticles = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & potentialEnergy & minimumTimestep & sourceMass & depositedMass & sourceParticles;
    }
};

// Collective root entry points: call once on the controlling HPX locality with
// every owned particle synchronized. Sorting permutes the local particle arrays
// in place; plans, buffers and spectral tables persist between solves.
PmSolveResult pmSolve(PmSolveRequest const& request);
void pmDestroy();
PmDiagnostics pmDiagnostics();
// Count of transforms executed since the last reset, on this locality.
uint64_t pmTransformCount();
void pmResetTransformCount();

// Retained results (storeFields) remain valid until the next particle
// permutation or PM solve; both are gm-scaled.
Vector3<float> const& pmAcceleration(ParticleInteger index);
float pmPotential(ParticleInteger index);
// Effective smoothing length of the long force for the PM timestep criterion.
double pmEffectiveRadius();
