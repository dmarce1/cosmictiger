/* CosmicTiger - mandatory outer PM interval of the particle-mesh solvers; BSL-1.0. */
#pragma once

struct PmBoundaryResult {
    // Global long-range potential energy sum(0.5 m phi) scaled by gm.
    double potentialEnergy = 0.0;
    // Global PM acceleration timestep bound before the end/boundary caps.
    double accelerationTimestep = 1.0e30;
    // Duration selected for the next outer interval (zero at the end of the run).
    double nextInterval = 0.0;
};

// Run at an outer boundary with every particle at the synchronized position:
// migrate domains, solve the mesh with all gravitating particles as sources,
// store the gm-scaled long acceleration in the particles, and choose the next
// interval from the PM acceleration bound, the cosmological expansion bound,
// the time left to the next coarse output boundary and the end of the run.
// The result is never a power of two by construction.
PmBoundaryResult pmOuterBoundary(double scale, double scaleDerivative, double time,
    double coarseInterval, double endTime, bool doPotential);

// The selected next interval given the raw bounds, exposed for its tests.
double pmSelectOuterInterval(double accelerationTimestep, double expansionTimestep, double time,
    double coarseInterval, double endTime);
