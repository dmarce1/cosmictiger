/* CosmicTiger - gravity execution; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/kick.hpp>

FmmForceResult kickCellForest(Vector<KickWorkitem> workitems, bool doPotential);
// All source trees are local and immutable; disjoint sink batches may overlap.
FmmForceResult kickCellBatch(Vector<KickWorkitem> workitems, bool doPotential);
