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

// Begin a geometry event; the stable-ID source tree is refitted during gathering.
void hopkinsPrepareTree();
// Support/geometry preparation reads no live gravity force or dark-matter rung.
// It refreshes the retained records and solves only the due cells and their
// halo. Call only after hopkinsPrepareTree; completion is consumed by hopkinsKick.
void hopkinsPrepareGeometry(
    double scale, double expansionRate, double time, double baseTimestep, bool global);

// Close due/woken own intervals, reconstruct active cells, and open pair intervals.
// Gravity accelerations must be fresh for all gas, including potential wake-ups.
// A terminal closing-only call settles every elapsed pair and closes every cell.
HopkinsStepResult hopkinsKick(int minimumRung, double scale, double expansionRate, double time,
    double baseTimestep, EnergiesT& energies, bool closingOnly = false,
    double previousTimestep = 0.0, double maximumGlobalTimestep = 0.0);
// Close cached flux/source intervals before the first signed HOLD gravity kick.
// Shared particle rungs remain gravity-only; the returned rung is hydro-only.
HopkinsStepResult hopkinsHierarchicalBegin(int minimumGravityRung, double scale,
    double expansionRate, double time, double baseTimestep, EnergiesT& energies,
    bool closingOnly = false);
// Reconstruct after all signed gravity kicks, schedule hydro wakes, and project
// the current hydro trajectories to the separately retained gravity endpoints.
HopkinsStepResult hopkinsHierarchicalFinish(int minimumHydroRung, double scale,
    double expansionRate, double time, double baseTimestep, EnergiesT& energies,
    bool closingOnly = false);
// Held thermal sum for cosmological expansion work after a completed ladder;
// synchronized output diagnostics retain the pre-opening thermal snapshot.
double hopkinsHierarchicalDriftThermalEnergy();
// Recompute the one-dimensional sound-wave density at the final drifted positions.
// This diagnostic does not advance the conserved state or the integration clock.
void hopkinsSoundWaveDiagnostics(double time);
// Discard the retained coordinator records and owner mirrors; cached pair
// intervals are kept. Gas teardown and checkpoint loading call this.
void hopkinsResetSources();
void hopkinsWriteCheckpoint(FILE* file);
void hopkinsReadCheckpoint(FILE* file);
