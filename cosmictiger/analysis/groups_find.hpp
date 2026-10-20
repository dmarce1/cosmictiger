/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#pragma once

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/tree.hpp>

hpx::future<size_t> groupsFind(TreeId self, Vector<TreeId> checklist, double linkingLength);
