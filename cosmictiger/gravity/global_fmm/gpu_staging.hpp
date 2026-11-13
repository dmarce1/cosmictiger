/* CosmicTiger - gravity execution; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/kick_workspace.hpp>

KickReturn globalFmmStageGpu(KickParameters params, Vector<KickWorkitem> workitems,
    std::unordered_set<TreeId, KickWorkspaceTreeIdHash> const& treeIds,
    detail::FmmForceOutput output);
