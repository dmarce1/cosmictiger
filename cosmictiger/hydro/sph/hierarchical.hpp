#pragma once

#include <cstdint>

#include <cosmictiger/hydro/sph/sph.hpp>

struct HierarchicalSphResult {
    SphRunReturn run;
    double kineticDelta = 0.0;
    double xMomentumDelta = 0.0;
    double yMomentumDelta = 0.0;
    double zMomentumDelta = 0.0;
    double momentumNormDelta = 0.0;
    // Absolute synchronized gas totals also cover runs without a gravity kick.
    double kineticEnergy = 0.0;
    double xMomentum = 0.0;
    double yMomentum = 0.0;
    double zMomentum = 0.0;
    double momentumNorm = 0.0;
    double thermalEnergy = 0.0;
    // Energy the GADGET-4 sources removed from the closed gas, and its conversions.
    double heating = 0.0;
    uint64_t converted = 0;
};

// Called once at the top of the gravity ladder, after its closing kick and
// before the deferred gas opening kick. Only the current active suffix is used.
HierarchicalSphResult sphHierarchicalStep(int minimumRung, double scaleFactor, double expansionRate,
    double time, double baseTimestep, bool closingOnly);

// Open hydro and retain the exact drift segment before the rung advances.
// Return the canonical kinetic change for the expansion-work ledger.
double sphHierarchicalOpen(int rung, double time, double timestep, double driftScale);
double sphHierarchicalOpenGravity(int minimumRung, double halfTimestep);
void sphHierarchicalReportWork(int minimumRung, double time);
// Sum of m*u over all gas, including the inactive particles between their events.
double sphHierarchicalThermalEnergy();
// Rank-local lifetime hook; callers have already joined all SPH work.
void sphHierarchicalResetSources();
