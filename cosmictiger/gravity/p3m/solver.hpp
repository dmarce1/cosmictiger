/* CosmicTiger - P3M short-range solver (PP-PM and FMM-PM); BSL-1.0. */
#pragma once

#include <cstdint>

#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

struct P3mDiagnostics {
    // Short-range search mesh and the populations of the most recent solve.
    int meshSize = 0;
    uint64_t activeParticles = 0;
    uint64_t totalParticles = 0;
    uint64_t evaluatedParticles = 0;
    // Remote source cells and particle records fetched for this solve.
    uint64_t ghostCells = 0;
    uint64_t ghostParticles = 0;
    // CPU interaction counts (CUDA pair work is not counted here).
    uint64_t directPairs = 0;
    uint64_t acceptedMultipoles = 0;
    uint64_t cutoffRejections = 0;
    uint64_t sourceTreeNodes = 0;
    uint64_t batches = 0;
    // PP-PM pair batches that ran on the CUDA device.
    uint64_t cudaBatches = 0;
    uint64_t storedAccelerationParticles = 0;
    uint64_t solveCount = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & meshSize & activeParticles & totalParticles & evaluatedParticles & ghostCells &
            ghostParticles & directPairs & acceptedMultipoles & cutoffRejections & sourceTreeNodes &
            batches & cudaBatches & storedAccelerationParticles & solveCount;
    }
};

// Root entry points. The short field of every active sink is streamed through
// particleMeshApplyField together with the long half-kicks requested in the
// parameters; hierarchical stages use the current rung-selected range as both
// sources and sinks, traditional steps use every owned particle as a source.
KickReturn p3mKick(KickParameters const& parameters);
// Diagnostic solve retaining the gm-scaled short field of every owned particle.
void p3mSolve(bool doPotential);
void p3mDestroy();
P3mDiagnostics p3mDiagnostics();

// Results of p3mSolve remain valid until the next particle permutation or solve.
Vector3<float> const& p3mAcceleration(ParticleInteger index);
float p3mPotential(ParticleInteger index);
