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
 * Ghost leaves are the only distributed SPH path (M5): no remote node or gas is ever fetched.
 *
 * Sync, before every neighbor walk: each locality publishes the stored inner and outer boxes
 * of a cut of at most 64 nodes through its tree (their union first, for a quick rejection),
 * and each owner B exports to locality A every leaf l with
 *     inner(l) ∩ outer(n) ≠ ∅   or   outer(l) ∩ inner(n) ≠ ∅   for a node n of A's cut,
 * the walk's own opening test. Every leaf s of A lies under a node n of the cut with
 * inner(s) ⊆ inner(n) and outer(s) ⊆ outer(n), so a leaf l that the walk would open for s is
 * exported: the ghost set is complete for that walk by construction, and the walk need only
 * visit the local root and the ghost leaves.
 *
 * Refresh, before every pass: each owner pushes the current fields of its exported particles,
 * tagged with the refresh generation; a read of fields older than the current refresh throws.
 *
 * Messages are pushes acknowledged inside one collective; no pass issues a request and waits
 * for a reply.
 */

// Whether there are remote leaves to mirror: more than one locality.
bool sphGhostsActive();

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
// Throws unless a pass is staging on this locality: remote gas is readable only then, when the
// pushed fields are current. Tests read remote state from its owner instead.
void sphGhostsRequireStaging(ParticleGlobalRange const& range);

// Local reads of a whole remote leaf range, with the conventions of the gas readers; each
// throws if the range is not a ghost leaf or its owner's fields predate the current refresh.
// A null output pointer skips that field.
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
