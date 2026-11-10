/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stddef.h>

#include <cosmictiger/core/force_step.hpp>
#include <cosmictiger/core/time.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>
#endif

// Output cadence follows the nominal coarse clock even with raw global steps.
inline int driverOutputStep(double time, double coarseInterval) {
    double const phase = time / coarseInterval;
    double const tolerance = 32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, phase);
    return int(std::floor(phase + tolerance));
}

/// Choose the final output index, accounting for a terminal step that coincides with
/// regular output.
inline int driverFinalOutputStep(double time, double coarseInterval) {
    int const coarseStep = driverOutputStep(time, coarseInterval);
    double const tolerance =
        32.0 * std::numeric_limits<double>::epsilon() * std::max(coarseInterval, std::abs(time));
    return coarseStep + (time > coarseStep * coarseInterval + tolerance);
}

struct DriverParameters {
    double a;
    double adot;
    double dummy;
    double tau;
    double tauMax;
    double energy0;
    EnergiesT energies;
    int maxRung;
    int iter;
    int minrung0;
    int step;
    int bucketSize;
    size_t totalProcessed;
    double flops;
    double runtime;
    double years;
    TimeType itime;
    // Interval of the pending global half-step, independent of the rung clock.
    double globalTimestep{};
    // Latest midpoint hydro bound; gravity selects the next global interval
    // at its boundary without retaining a per-gas gravitational acceleration.
    double globalHydroTimestep{};
    // Particle-mesh solvers: the outer PM interval whose opening long half-kick
    // has been applied (zero before the first), the conformal time at which it
    // started, and the number of completed outer intervals. itime is the
    // interval-local integer phase of the inner rung integration.
    double outerInterval{};
    double outerStart{};
    uint64_t outerIndex{};
    /// Archive `bucketSize`, `maxRung`, `adot`, `minrung0`, `step`, `a` and the
    /// remaining fields in transport order.
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & bucketSize;
        arc & maxRung;
        arc & adot;
        arc & minrung0;
        arc & step;
        arc & a;
        arc & tau;
        arc & tauMax;
        arc & energies;
        arc & energy0;
        arc & iter;
        arc & totalProcessed;
        arc & flops;
        arc & runtime;
        arc & itime;
        arc & years;
        arc & globalTimestep;
        arc & globalHydroTimestep;
        arc & outerInterval;
        arc & outerStart;
        arc & outerIndex;
    }
};

// Particle-mesh solvers: how a force pass at an outer PM boundary closes the
// previous interval and opens the next. closingBaseTimestep is the previous
// interval (zero: the substep base timestep is unchanged); the long half-kicks
// apply the per-particle mesh acceleration; longPotentialEnergy is the
// gm-scaled mesh potential energy added to the synchronized diagnostics.
// endTime is the run's end time for the hierarchical drift (zero: the
// substep base timestep times run.numberOfSteps, which a non-dyadic outer PM
// interval does not reproduce).
// PMMM outer interval (docs/pmmm.md): a force pass at a synchronized boundary
// closes the old frozen membership, migrates, runs the far solve that selects
// the next interval and opens the new membership. The force step reports the
// selected interval and the far potential energy back through this record.
struct PmmmBoundaryRequest {
    double time = 0.0;
    double coarseInterval = 0.0;
    double endTime = 0.0;
    // Outputs of the force step.
    double nextInterval = 0.0;
    double potentialEnergy = 0.0;
};

struct MeshKickSchedule {
    double closingBaseTimestep = 0.0;
    double longCloseTimestep = 0.0;
    double longOpenTimestep = 0.0;
    double longPotentialEnergy = 0.0;
    double endTime = 0.0;
    // Set only for a PMMM outer-interval boundary pass.
    PmmmBoundaryRequest* pmmmBoundary = nullptr;
};

#if COSMICTIGER_WITH_SPH
// Positive duration: midpoint kick. Zero: initial force/timestep estimate.
// Negative: density-only synchronized diagnostics. A negative rung selects all gas.
SphRunReturn sphMidpointStep(int rung, double time, double timestep, double baseTimestep,
    double scale, double scaleDerivative, EnergiesT& energies, bool global = false,
    double maximumTimestep = 0.0);
// Evolve endpoint viscosity coefficients and cooling/star-formation sources once,
// after the final drift and before changing old rungs or writing the endpoint state.
// Global steps pass their actual completed duration; rung steps use the base
// duration and each selected particle's old rung. Return refreshed timestep
// bounds, including all surviving gas after source heating or conversion.
SphRunReturn sphCompleteSources(int minimumClosingRung, double time, double baseTimestep,
    double scale, double scaleDerivative, EnergiesT& energies, double explicitTimestep = -1.0,
    double maximumGlobalTimestep = 0.0);
SphTreeCreateReturn sphStep1(int minrung, bool verbose = true);

#endif
#if COSMICTIGER_WITH_GRAVITY
#if COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
ForceStepResult kickStepHierarchical(int& minimumRung, int maximumRung, double scale,
    double scaleDerivative, double time, double baseTimestep, double theta, EnergiesT* energies,
    int minimumRungFloor, bool doPotential, hpx::future<void>* lightcone, bool keepTop, double drag,
    int gravityErrorStep = -1, bool closingOnly = false, MeshKickSchedule const& schedule = {},
    bool deferHydroOpening = false);
#endif

#endif

ForceStepResult advanceForces(int minRung, double scale, double scaleDerivative, double time,
    double baseTimestep, double theta, bool firstCall, bool fullEvaluation, EnergiesT& energies,
    int gravityErrorStep = -1, bool closingOnly = false, double previousTimestep = 0.0,
    double maximumGlobalTimestep = 0.0, MeshKickSchedule const& schedule = {},
    bool deferHydroOpening = false);

void runFixedScenario();
void driver();
