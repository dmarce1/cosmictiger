/* CosmicTiger - mandatory outer PM interval of the particle-mesh solvers; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/pm/outer_interval.hpp>

// Run at an outer boundary with every particle at the synchronized position:
// migrate domains, solve the mesh with all gravitating particles as sources,
// store the gm-scaled long acceleration in the particles, and choose the next
// interval from the PM acceleration bound, the cosmological expansion bound,
// the time left to the next coarse output boundary and the end of the run.
// The result is never a power of two by construction.
PmBoundaryResult pmOuterBoundary(double scale, double scaleDerivative, double time,
    double coarseInterval, double endTime, bool doPotential);
