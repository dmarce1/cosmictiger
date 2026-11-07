/* CosmicTiger - outer long-force interval shared by pm, p3m and the PMMM outer interval; BSL-1.0. */
#pragma once

struct PmBoundaryResult {
    // Global long-range potential energy sum(0.5 m phi) scaled by gm.
    double potentialEnergy = 0.0;
    // Global long-force acceleration timestep bound before the end/boundary caps.
    double accelerationTimestep = 1.0e30;
    // Duration selected for the next outer interval (zero at the end of the run).
    double nextInterval = 0.0;
};

// The selected next interval given the raw bounds: the long-force acceleration
// bound, the cosmological expansion bound, the time left to the next coarse
// output boundary and the end of the run. Never a power of two by construction.
double pmSelectOuterInterval(double accelerationTimestep, double expansionTimestep, double time,
    double coarseInterval, double endTime);
