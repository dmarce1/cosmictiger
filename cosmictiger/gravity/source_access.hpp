/* CosmicTiger - shared gravity source access; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/tree.hpp>

using GravityRemoteSourceReader = void (*)(
    TreeNode const&, Position*, Position*, Position*, char*, ParticleInteger);
void gravitySetRemoteSourceReader(GravityRemoteSourceReader reader);
void gravityReadSourcePositions(
    TreeNode const&, Position*, Position*, Position*, char*, ParticleInteger);
