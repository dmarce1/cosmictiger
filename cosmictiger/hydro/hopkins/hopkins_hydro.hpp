#pragma once

#include <cstdio>

#include <cosmictiger/particles/particles.hpp>

struct HopkinsStepResult {
    int maximumRung = 0;
    double minimumTimestep = 0.0;
};

// Traditional integration uses individual power-of-two gas flux intervals.
// A positive maximumGlobalTimestep instead selects one raw collective step;
// previousTimestep supplies the elapsed source interval, including on restart.
// All state updates occur here; drift only consumes stored mesh velocities.
// A terminal closing-only call corrects elapsed source histories and measures
// synchronized energies without starting another hydrodynamic interval.
HopkinsStepResult hopkinsKick(int minimumRung, double scale, double expansionRate, double time,
    double baseTimestep, EnergiesT& energies, bool closingOnly = false,
    double previousTimestep = 0.0, double maximumGlobalTimestep = 0.0);
// Recompute the one-dimensional sound-wave density at the final drifted positions.
// This diagnostic does not advance the conserved state or the integration clock.
void hopkinsSoundWaveDiagnostics(double time);
void hopkinsWriteCheckpoint(FILE* file);
void hopkinsReadCheckpoint(FILE* file);
