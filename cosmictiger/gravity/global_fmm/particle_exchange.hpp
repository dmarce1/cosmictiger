/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/source_access.hpp>

void globalFmmReadSourcePositions(
    TreeNode const&, Position*, Position*, Position*, char*, float*, ParticleInteger);
