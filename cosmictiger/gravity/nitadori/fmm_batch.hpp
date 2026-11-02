/* CosmicTiger - gravity execution; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/kick.hpp>

FmmForceResult kickCellForest(Vector<KickWorkitem> workitems, bool doPotential);
// All source trees are local and immutable; disjoint sink batches may overlap.
// Select sinks before either CPU or CUDA evaluation, without changing source
// trees or permuting their particles. The arena reserves one extra node per
// possible sink when a positive minimum rung can split a source leaf.
FmmForceResult kickCellBatch(Vector<KickWorkitem> workitems, bool doPotential, int minimumRung = 0);
