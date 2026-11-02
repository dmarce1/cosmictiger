#pragma once

#include <cstdio>

#include <cosmictiger/particles/particles.hpp>

struct HopkinsStepResult {
    int maximumRung = 0;
    double minimumTimestep = 0.0;
};

// Project MFV source masses to the force time without changing conserved anchors.
// Must precede migration and every gravity source/tree construction.
void hopkinsPrepareGravityMasses(double time);

// Create/sort the gas tree before a concurrent gravity evaluation begins.
void hopkinsPrepareTree();
// Support/geometry preparation reads no live gravity force or dark-matter rung.
// Call only after hopkinsPrepareTree; completion is consumed by hopkinsKick.
void hopkinsPrepareGeometry(double scale, double expansionRate);

// Close due/woken own intervals, reconstruct active cells, and open pair intervals.
// Gravity accelerations must be fresh for all gas, including potential wake-ups.
// A terminal closing-only call settles every elapsed pair and closes every cell.
HopkinsStepResult hopkinsKick(int minimumRung, double scale, double expansionRate, double time,
    double baseTimestep, EnergiesT& energies, bool closingOnly = false,
    double previousTimestep = 0.0, double maximumGlobalTimestep = 0.0);
// Recompute the one-dimensional sound-wave density at the final drifted positions.
// This diagnostic does not advance the conserved state or the integration clock.
void hopkinsSoundWaveDiagnostics(double time);
void hopkinsWriteCheckpoint(FILE* file);
void hopkinsReadCheckpoint(FILE* file);
