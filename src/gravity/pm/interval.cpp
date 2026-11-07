/* CosmicTiger - mandatory outer PM interval of the particle-mesh solvers; BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <limits>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include <cosmictiger/gravity/pm/interval.hpp>
#include <cosmictiger/gravity/pm/solver.hpp>
#include <cosmictiger/particles/domain.hpp>

PmBoundaryResult pmOuterBoundary(double scale, double scaleDerivative, double time,
    double coarseInterval, double endTime, bool doPotential) {
    auto const& options = getOptions();
    {
        profile::ScopedPhase phase("pmBoundary.domains");
        domainsBegin(0);
        domainsEnd();
    }
    PmSolveRequest request;
    request.doPotential = doPotential;
    request.applyToParticles = true;
    request.scale = scale;
    request.eta = options.gravity.timestepEta;
    request.maximumTimestep = 1.0e30;
    PmSolveResult solve;
    {
        profile::ScopedPhase phase("pmBoundary.solve");
        solve = pmSolve(request);
    }
    PmBoundaryResult result;
    result.potentialEnergy = solve.potentialEnergy;
    result.accelerationTimestep = solve.minimumTimestep;
    double expansion = 1.0e30;
    if (options.run.scenario == "cosmology") {
        constexpr double kScaleTimestepFraction = 0.01;
        expansion = cosmology::expansionTimestep(
            scale, scaleDerivative / scale, cosmosD2adt2(scale), kScaleTimestepFraction);
    }
    result.nextInterval =
        pmSelectOuterInterval(solve.minimumTimestep, expansion, time, coarseInterval, endTime);
    terminal::detailPrint("PM boundary at tau=%e: acceleration bound %e, expansion bound %e, "
                          "next interval %e, long potential %e\n",
        time, solve.minimumTimestep, expansion, result.nextInterval, solve.potentialEnergy);
    return result;
}
