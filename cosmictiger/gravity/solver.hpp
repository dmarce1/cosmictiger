/* CosmicTiger - periodic particle mesh multipoles; BSL-1.0. */
#pragma once

#include <cstdint>
#include <functional>

#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/outer_interval.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

struct PmmmDiagnostics {
    // Effective mesh and global owned population for the most recent solve.
    int meshSize = 0;
    uint64_t activeParticles = 0;
    uint64_t totalParticles = 0;
    double sourceMass = 0;
    double depositedMass = 0;
    // Sinks that actually received near-field and local-expansion evaluation.
    uint64_t evaluatedParticles = 0;
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
    // Mesh transforms (deposit, forward FFT, contraction, inverse FFT) executed
    // on this locality; frozen-membership near passes never add to it.
    uint64_t transformCount = 0;

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & meshSize & activeParticles & totalParticles & sourceMass & depositedMass &
            evaluatedParticles & nearCpuAcceptedMultipoles & nearCpuDirectPairs & ghostParticles &
            ghostCells & ghostBytes & nearRemoteTreeFetches & nearRemoteParticleFetches &
            nearBatches & peakBatchParticles & completedWhileGhostsPending &
            storedAccelerationParticles & storedPotentialParticles & planBuilds & greenBuilds &
            solveCount & transformCount;
    }
};

// Root entry points. Plans, buffers and Green spectra persist between solves.
void pmmmSolve(bool doPotential);
// Production entry: hierarchical stages use the current rung-selected range
// for sources and sinks. Traditional steps retain every owned source and the
// full mesh, evaluating only sinks at or above parameters.minRung. Fields are
// consumed by the kick without full-particle result buffers.
// With parameters.frozenNear (PMMM outer interval, docs/pmmm.md) the pass is a
// near-only evaluation on the stored frozen cell membership: no deposit, FFT
// or far expansion; parameters.refreshMembership first re-sorts every owned
// particle by position and stores the new membership for the coming interval.
KickReturn pmmmKick(KickParameters const& parameters);
// PMMM outer interval boundary: with every owned particle migrated to its
// domain and at the synchronized position, sort by position, solve the far
// field for every particle into particlesLongAcceleration (gm-scaled), and
// choose the next interval from the far acceleration at the mesh-cell scale,
// the expansion bound, the coarse output clock and the end time. No velocity,
// rung or membership changes; the following refreshMembership pass stores them.
PmBoundaryResult pmmmOuterBoundary(
    double scale, double scaleDerivative, double time, double coarseInterval, double endTime);
void pmmmDestroy();
PmmmDiagnostics pmmmDiagnostics();

// Results remain valid until the next particle permutation or solve.
Vector3<float> const& pmmmAcceleration(ParticleInteger index);
float pmmmPotential(ParticleInteger index);

// Local instrumentation hook. Set/clear only between solves; callbacks execute
// on HPX tasks and may suspend for deterministic readiness tests.
enum class PmmmCellEvent {
    GhostRequested,
    GhostReady,
    NearCompleted,
    OwnedRootReady,
    MomentsDeposited,
    FarLocalReady
};
void pmmmSetCellObserver(std::function<void(PmmmCellEvent, uint64_t)> observer);
