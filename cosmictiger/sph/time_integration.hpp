/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstddef>

#include <cosmictiger/core/tree_id.hpp>
#include <cosmictiger/particles/particles.hpp>

// The midpoint split uses live operator-stage velocities and entropies. Its
// driver drifts all particles chronologically; no per-particle predictor clocks
// or future-position restoration are needed.
bool sphMidpointEnabled();
bool sphMidpointEventActive();
void sphMidpointBeginEvent();
void sphMidpointEndEvent();

// Count exact-rung targets collectively; a negative rung selects all gas.
size_t sphMidpointTargetCount(int exactRung);

// Gravity-disabled traditional steps choose a new rung only for particles whose
// interval ends at this boundary. Return the maximum occupied rung globally.
int sphParticlesSelectMidpointRungs(int minimumRung);

struct SphRungBalanceReturn {
    int maximumRung = 0;
    int passes = 0;
    size_t changedParticles = 0;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & maximumRung & passes & changedParticles;
    }
};

// Collective, called on rank zero only at a common gas synchronization boundary.
// The caller seeds each proposal byte with its complete planned rung, including
// gravity. Close those exact seeds over all touching kernel pairs and publish
// balanced proposal bytes. Live rungs never change here. The caller commits them
// before opening the next intervals; new contacts between boundaries are not
// guaranteed balanced. The supplied retained tree must contain every owned gas.
SphRungBalanceReturn sphBalanceRungs(TreeId root, float scaleFactor, bool cuda = true);

// Collective boundary scheduling helpers. Seed only after gravity has combined
// its fresh criterion with the hydro floor; commit only when all gas intervals
// are closed. The empty population has maximum zero.
void sphSeedRungProposals();
int sphCommitBalancedRungs();

// Transient rung-pass storage: full-particle-indexed sink candidates and frozen
// gas-indexed source snapshots. No particle migration or sorting may intervene.
bool sphRungBalanceActive();
char* sphRungCandidatesData();
void sphReadRungCandidates(ParticleGlobalRange range, char* rungs, ParticleInteger offset);
// This locality's rung candidates of local gas range, valid while the rung balance runs.
Vector<char> sphFetchRungCandidates(Pair<ParticleInteger> range);
bool sphRungCandidatesActive();
