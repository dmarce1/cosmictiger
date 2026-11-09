/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_GHOSTS_HPP_
#define SPH_GHOSTS_HPP_

#include <cstdint>

#include <cosmictiger/core/tree_id.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/tree.hpp>

/*
 * SPH ghost leaves (docs/sph_parent_tree_plan.md, sections 3.2 to 3.6).
 *
 * Every locality keeps copies of the remote leaves its neighbor search can reach. A copy is
 * the owner's leaf record verbatim, so TreeId{owner rank, owner node index} still names it
 * and SphPairSourceId{owner rank, owner gas index} still names its particles: neighbor lists
 * and the reverse pair updates are unchanged. The particles' fields live in a
 * structure-of-arrays store in the owner's leaf order.
 *
 * Sync, before every neighbor walk: each locality publishes the union of its leaves' stored
 * inner and outer boxes, and each owner B exports to locality A every leaf l with
 *     inner(l) ∩ outer(A) ≠ ∅   or   outer(l) ∩ inner(A) ≠ ∅,
 * the walk's own opening test against A's union. A leaf pair (s on A, l on B) that the walk
 * would open has inner(s) ⊆ inner(A) and outer(s) ⊆ outer(A), so l is exported: the ghost
 * set is complete for that walk by construction, and the walk need only visit the local
 * root and the ghost leaves.
 *
 * Refresh, before every pass: each owner pushes the current fields of its exported particles,
 * converted exactly as the gas cache-line fetches convert them, so the readers see the same
 * values that the fetch path would read.
 *
 * Messages are pushes acknowledged inside one collective; no pass issues a request and waits
 * for a reply.
 */

// Whether the ghost path serves remote leaves: enabled and more than one locality.
bool sphGhostsActive();
bool sphGhostsEnabled();
// From rank 0: set compute.sphGhosts on every locality, selecting the ghost path (true) or
// the remote fetch path (false), and drop every ghost.
void sphGhostsSetEnabled(bool enabled);

// Collective from rank 0: recompute every locality's exports from the current leaf boxes and
// install the ghost leaf records. Drops any previous ghosts.
void sphGhostsSync();
// Collective from rank 0: push the current fields of every exported particle.
void sphGhostsRefresh();
// Collective from rank 0: drop every ghost and export list.
void sphGhostsClear();

// Local lookups. A null result means the leaf is not a ghost here.
SphTreeNode const* sphGhostsFindNode(TreeId id);
Vector<TreeId> const& sphGhostsLeafIds();

// While a pass stages its sources, every remote leaf must be a ghost: a missing one means the
// export was incomplete. sphGhostsBeginStaging and sphGhostsEndStaging bracket that phase on
// this locality.
void sphGhostsBeginStaging();
void sphGhostsEndStaging();
// Whether a remote read is served by the ghost store: only while a pass stages, when the
// pushed fields are current. Other remote reads (test oracles, diagnostics) use the fetch path.
bool sphGhostsServe(ParticleGlobalRange const& range);

// Local reads of a whole remote leaf range, with the conventions of the gas readers; each
// throws if the range is not a ghost leaf. A null output pointer skips that field.
void sphGhostsReadPositions(
    ParticleGlobalRange range, Position* x, Position* y, Position* z, ParticleInteger offset);
void sphGhostsReadInternalEnergyAndSmoothingLength(ParticleGlobalRange range, float* internalEnergy,
    float* smoothingLength, ParticleInteger offset);
void sphGhostsReadRungs(ParticleGlobalRange range, char* rungs, ParticleInteger offset);
void sphGhostsReadVelocities(
    ParticleGlobalRange range, float* vx, float* vy, float* vz, ParticleInteger offset);
void sphGhostsReadDensity(ParticleGlobalRange range, float* density, ParticleInteger offset,
    float* balsaraFactor, float* bulkViscosityCoefficient, float* gradHCorrection);
void sphGhostsReadRungCandidates(ParticleGlobalRange range, char* rungs, ParticleInteger offset);

struct SphGhostStatistics {
    std::int64_t leaves = 0;            // Ghost leaves held, summed over localities.
    std::int64_t particles = 0;         // Ghost particles held.
    std::int64_t exportedLeaves = 0;    // Leaf exports, one per (leaf, importer).
    std::int64_t syncs = 0;             // Syncs since the last reset (each is collective).
    std::int64_t refreshes = 0;         // Refreshes since the last reset (each is collective).
    std::int64_t recordBytes = 0;       // Bytes of leaf records sent by syncs.
    std::int64_t fieldBytes = 0;        // Bytes of particle fields sent by refreshes.

    SphGhostStatistics& operator+=(SphGhostStatistics const& other);

    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & leaves;
        archive & particles;
        archive & exportedLeaves;
        archive & syncs;
        archive & refreshes;
        archive & recordBytes;
        archive & fieldBytes;
    }
};

// Collective from rank 0: the counters summed over localities.
SphGhostStatistics sphGhostsStatistics();
// Collective from rank 0: zero the cumulative counters.
void sphGhostsResetStatistics();

#endif /* SPH_GHOSTS_HPP_ */
