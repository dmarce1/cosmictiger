/* CosmicTiger - outer long-force interval of the PMMM outer interval; BSL-1.0. */
#pragma once

struct PmBoundaryResult {
    // Global long-range potential energy sum(0.5 m phi) scaled by gm.
    double potentialEnergy = 0.0;
    // Global long-force acceleration timestep bound before the end/boundary caps.
    double accelerationTimestep = 1.0e30;
    // Duration selected for the next outer interval (zero at the end of the run).
    double nextInterval = 0.0;
};

// Round the raw long-force/expansion proposal upward to R/floor(R/h), where
// R is the time remaining to the next output phase or run end and h is the
// capped proposal. This gives an integer number of remaining long steps.
// Inner rung subdivisions remain dyadic; the long interval need not be.
double pmSelectOuterInterval(double accelerationTimestep, double expansionTimestep, double time,
    double outputInterval, double endTime);
