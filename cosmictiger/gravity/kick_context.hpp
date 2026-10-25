/* CosmicTiger - gravity execution; BSL-1.0. */
#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>

// Solver decisions occur once per task/node/leaf. Pair arithmetic stays in the
// shared inlined operators; no interaction-level virtual dispatch is used.
struct KickContext {
    hpx::future<KickReturn> (*fork)(KickParameters, ExpansionBuffer const&, Array<Position, NDIM>,
        TreeId, Vector<TreeId>, Vector<TreeId>, std::shared_ptr<KickWorkspace>, bool) = nullptr;
    FlopCount (*background)(int, KickParameters const&, ExpansionBuffer&, Array<Position, NDIM>&,
        TreeId, Vector<TreeId> const&) = nullptr;
    FlopCount (*periodic)(int, KickParameters const&, ExpansionBuffer&, TreeId, Vector<TreeId>&,
        KickInteractionCounts&) = nullptr;
    void (*beginReturn)(ParticleInteger) = nullptr;
    void (*coverParticles)(ParticleInteger) = nullptr;
    void (*accumulateReturn)(KickReturn const&, ParticleInteger, bool) = nullptr;
    KickReturn (*waitReturn)() = nullptr;
    void (*publishReturn)(KickReturn) = nullptr;
    KickGpuExecutor gpuExecutor = nullptr;
};
KickReturn kickWalkDispatch(KickParameters, ExpansionBuffer, Array<Position, NDIM>, TreeId,
    Vector<TreeId>, Vector<TreeId>, std::shared_ptr<KickWorkspace>, KickContext const&);
