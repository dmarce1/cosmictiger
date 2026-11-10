/* CosmicTiger - PMMM outer interval boundary; BSL-1.0. */

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include "actions.hpp"

/// Solve the far field at an outer boundary and select the next interval from force and
/// expansion limits.
PmBoundaryResult pmmmOuterBoundary(double scale, double scaleDerivative, double time,
    double coarseInterval, double endTime) {
    auto const& options = getOptions();
    pmmm::detail::SolveRequest request;
    request.farOnly = true;
    request.doPotential = true;
    request.scale = scale;
    request.eta = options.gravity.timestepEta;
    {
        profile::ScopedPhase phase("pmmmBoundary.farSolve");
        pmmm::detail::solve(std::move(request));
    }
    auto const solve = pmmm::detail::globalKickResult;
    PmBoundaryResult result;
    result.potentialEnergy = solve.pot;
    result.accelerationTimestep = solve.minimumTimestep;
    double expansion = 1.0e30;
    if (options.run.scenario == "cosmology") {
        constexpr double kScaleTimestepFraction = 0.01;
        expansion = cosmology::expansionTimestep(
            scale, scaleDerivative / scale, cosmosD2adt2(scale), kScaleTimestepFraction);
    }
    result.nextInterval = pmSelectOuterInterval(
        result.accelerationTimestep, expansion, time, coarseInterval, endTime);
    terminal::detailPrint("PMMM boundary at tau=%e: far acceleration bound %e, expansion bound "
                          "%e, next interval %e, far potential %e\n",
        time, result.accelerationTimestep, expansion, result.nextInterval, result.potentialEnergy);
    return result;
}
