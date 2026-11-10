/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

// Per-batch counters of the CPU tree walk. Cell pairs count once per sink:
// the walk evaluates each reciprocal pair from both sides. Sources without
// particles never enter the walk's lists, so they are neither tested nor
// counted. The CUDA walk does not report counts.
struct KickInteractionCounts {
    long long acceptanceTests;                // acceptance evaluations, both passes
    long long cellCellDirect;                 // M2L
    long long cellParticle;                   // P2L: source leaf particles into a sink expansion
    long long cellParticleAtLeafSinks;        // the subset whose sink is a leaf
    long long particleCell;                   // M2P: source multipole onto sink leaf particles
    long long particleCellFromLeafSources;    // the subset whose source is a leaf
    long long particleParticleLeafPairs;      // leaf pairs evaluated by direct summation
    long long particleParticlePairs;          // particle pairs within those leaf pairs
    long long sourceOpenings;                 // rejected sources replaced by their children
    long long deferredSources;                // rejected sources handed down unopened
};
