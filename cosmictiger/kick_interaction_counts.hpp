/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

// Rank-local counters of the CPU tree walk, summed over every sink node since
// the last reset. Cell pairs count once per sink: the walk evaluates each
// reciprocal pair from both sides. Sources without particles are not counted.
// The CUDA walk does not report counts.
struct KickInteractionCounts {
    long long acceptanceTests;                // acceptance evaluations, both passes
    long long cellCellDirect;                 // M2L, direct pass
    long long cellCellEwald;                  // M2L, Ewald pass
    long long cellParticle;                   // P2L: source leaf particles into a sink expansion
    long long cellParticleAtLeafSinks;        // the subset whose sink is a leaf
    long long particleCell;                   // M2P: source multipole onto sink leaf particles
    long long particleCellFromLeafSources;    // the subset whose source is a leaf
    long long particleParticleLeafPairs;      // leaf pairs evaluated by direct summation
    long long particleParticlePairs;          // particle pairs within those leaf pairs
    long long sourceOpenings;                 // rejected sources replaced by their children
    long long deferredSources;                // rejected sources handed down unopened
};

void kickResetInteractionCounts();
void kickAddInteractionCounts(KickInteractionCounts const& counts);
KickInteractionCounts kickInteractionCounts();
