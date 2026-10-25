/* CosmicTiger - gravity execution; BSL-1.0. */
#include <atomic>

#include <cosmictiger/gravity/kick_interaction_counts.hpp>

namespace {

struct KickInteractionCounters {
    std::atomic<long long> acceptanceTests{0};
    std::atomic<long long> cellCellDirect{0};
    std::atomic<long long> cellCellEwald{0};
    std::atomic<long long> cellParticle{0};
    std::atomic<long long> cellParticleAtLeafSinks{0};
    std::atomic<long long> particleCell{0};
    std::atomic<long long> particleCellFromLeafSources{0};
    std::atomic<long long> particleParticleLeafPairs{0};
    std::atomic<long long> particleParticlePairs{0};
    std::atomic<long long> sourceOpenings{0};
    std::atomic<long long> deferredSources{0};
};

KickInteractionCounters interactionCounters;

}    // namespace

void kickResetInteractionCounts() {
    interactionCounters.acceptanceTests = 0;
    interactionCounters.cellCellDirect = 0;
    interactionCounters.cellCellEwald = 0;
    interactionCounters.cellParticle = 0;
    interactionCounters.cellParticleAtLeafSinks = 0;
    interactionCounters.particleCell = 0;
    interactionCounters.particleCellFromLeafSources = 0;
    interactionCounters.particleParticleLeafPairs = 0;
    interactionCounters.particleParticlePairs = 0;
    interactionCounters.sourceOpenings = 0;
    interactionCounters.deferredSources = 0;
}

void kickAddInteractionCounts(KickInteractionCounts const& counts) {
    interactionCounters.acceptanceTests += counts.acceptanceTests;
    interactionCounters.cellCellDirect += counts.cellCellDirect;
    interactionCounters.cellCellEwald += counts.cellCellEwald;
    interactionCounters.cellParticle += counts.cellParticle;
    interactionCounters.cellParticleAtLeafSinks += counts.cellParticleAtLeafSinks;
    interactionCounters.particleCell += counts.particleCell;
    interactionCounters.particleCellFromLeafSources += counts.particleCellFromLeafSources;
    interactionCounters.particleParticleLeafPairs += counts.particleParticleLeafPairs;
    interactionCounters.particleParticlePairs += counts.particleParticlePairs;
    interactionCounters.sourceOpenings += counts.sourceOpenings;
    interactionCounters.deferredSources += counts.deferredSources;
}

KickInteractionCounts kickInteractionCounts() {
    KickInteractionCounts counts;
    counts.acceptanceTests = interactionCounters.acceptanceTests;
    counts.cellCellDirect = interactionCounters.cellCellDirect;
    counts.cellCellEwald = interactionCounters.cellCellEwald;
    counts.cellParticle = interactionCounters.cellParticle;
    counts.cellParticleAtLeafSinks = interactionCounters.cellParticleAtLeafSinks;
    counts.particleCell = interactionCounters.particleCell;
    counts.particleCellFromLeafSources = interactionCounters.particleCellFromLeafSources;
    counts.particleParticleLeafPairs = interactionCounters.particleParticleLeafPairs;
    counts.particleParticlePairs = interactionCounters.particleParticlePairs;
    counts.sourceOpenings = interactionCounters.sourceOpenings;
    counts.deferredSources = interactionCounters.deferredSources;
    return counts;
}
