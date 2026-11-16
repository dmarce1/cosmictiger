// Demand-fetched remote gravity-tree cache lifecycle and particle source reads.
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/gravity/sources.hpp>

void treeEnableRemoteCache();
void treeResetRemoteCache();

void globalFmmReadSourcePositions(
    TreeNode const&, Position*, Position*, Position*, char*, ParticleInteger);
