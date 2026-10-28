/* CosmicTiger - periodic particle mesh multipoles; BSL-1.0. */
#pragma once

#include <cstdint>
#include <functional>

#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/particles/particles.hpp>

struct PmmmDiagnostics {
    // Effective mesh and global owned population for the most recent solve.
    int meshSize = 0;
    uint64_t activeParticles = 0;
    uint64_t totalParticles = 0;
    double sourceMass = 0;
    double depositedMass = 0;
    // Current solve CPU traversal counts (CUDA work is not counted).
    uint64_t nearCpuAcceptedMultipoles = 0;
    uint64_t nearCpuDirectPairs = 0;
    // Current solve incoming halo counts, summed over localities. Bytes count
    // position/type/rung payloads only, excluding transport headers.
    uint64_t ghostParticles = 0;
    uint64_t ghostCells = 0;
    uint64_t ghostBytes = 0;
    uint64_t nearRemoteTreeFetches = 0;
    uint64_t nearRemoteParticleFetches = 0;
    uint64_t nearBatches = 0;
    uint64_t peakBatchParticles = 0;
    uint64_t completedWhileGhostsPending = 0;
    uint64_t storedAccelerationParticles = 0;
    uint64_t storedPotentialParticles = 0;
    // Cumulative per-locality generations, retained across explicit destruction.
    uint64_t planBuilds = 0;
    uint64_t greenBuilds = 0;
    uint64_t solveCount = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & meshSize & activeParticles & totalParticles & sourceMass & depositedMass &
            nearCpuAcceptedMultipoles & nearCpuDirectPairs & ghostParticles & ghostCells &
            ghostBytes & nearRemoteTreeFetches & nearRemoteParticleFetches & nearBatches &
            peakBatchParticles & completedWhileGhostsPending & storedAccelerationParticles &
            storedPotentialParticles & planBuilds & greenBuilds & solveCount;
    }
};

// Root entry points. Plans, buffers and Green spectra persist between solves.
void pmmmSolve(bool doPotential);
// Production entry: the current rung-selected range supplies both sources and
// sinks. Fields are consumed by the kick without full-particle result buffers.
KickReturn pmmmKick(KickParameters const& parameters);
void pmmmDestroy();
PmmmDiagnostics pmmmDiagnostics();

// Results remain valid until the next particle permutation or solve.
Array<float, kDimensions> const& pmmmAcceleration(ParticleInteger index);
float pmmmPotential(ParticleInteger index);
double pmmmEffectiveRadius();

// Local instrumentation hook. Set/clear only between solves; callbacks execute
// on HPX tasks and may suspend for deterministic readiness tests.
enum class PmmmCellEvent { GhostRequested, GhostReady, NearCompleted };
void pmmmSetCellObserver(std::function<void(PmmmCellEvent, uint64_t)> observer);
