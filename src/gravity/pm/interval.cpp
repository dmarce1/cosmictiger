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

double pmSelectOuterInterval(double accelerationTimestep, double expansionTimestep, double time,
    double coarseInterval, double endTime) {
    if (!(coarseInterval > 0.0) || !std::isfinite(coarseInterval)) {
        THROW_ERROR("The coarse output interval must be finite and positive\n");
    }
    double const remaining = endTime - time;
    if (cosmology::endTimeReached(time, endTime) || !(remaining > 0.0)) {
        return 0.0;
    }
    // Outputs and analysis stay on the nominal coarse clock: an interval never
    // crosses the next coarse boundary, and lands exactly on it when it would.
    double const position = time / coarseInterval;
    double const tolerance =
        32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, position);
    double const nextBoundary = (std::floor(position + tolerance) + 1.0) * coarseInterval;
    double interval = std::min({accelerationTimestep, expansionTimestep, coarseInterval});
    interval = std::min(interval, remaining);
    double const toBoundary = nextBoundary - time;
    if (toBoundary > 0.0 && toBoundary < remaining) {
        interval = std::min(interval, toBoundary);
    }
    if (!(interval > 0.0) || !std::isfinite(interval) || time + interval == time) {
        THROW_ERROR("The selected outer PM interval cannot advance the evolution clock\n");
    }
    // Absorb a sliver that would otherwise leave a tiny final interval.
    double const slack = 1.0e-9 * coarseInterval;
    if (toBoundary > interval && toBoundary - interval < slack) {
        interval = toBoundary;
    } else if (remaining > interval && remaining - interval < slack) {
        interval = remaining;
    }
    return interval;
}

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
