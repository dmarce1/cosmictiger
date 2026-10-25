/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/nitadori/near_pp.hpp>
#include <cosmictiger/math/kernel.hpp>

void pmmmNearField(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources, double cellWidth, bool doPotential,
    std::vector<std::array<double, 4>>& fields) {
    double const supportRadius = getOptions().gravity.supportRadius;
    if (!(cellWidth > 0 && cellWidth <= 1 && supportRadius > 0 && supportRadius <= 1)) {
        throw std::invalid_argument("PMMM near-field cell width/support must be in (0,1]");
    }
    fields.assign(sinks.size(), {});
    if (sinks.empty() || sources.empty()) {
        return;
    }
#ifdef USE_CUDA
    if (getOptions().compute.cuda) {
        pmmm::detail::pmmmNearFieldCuda(
            sinks, sources, cellWidth, supportRadius, doPotential, fields);
        return;
    }
#endif
    double const inverseSupportRadius = 1 / supportRadius;
    double const inverseSupportRadiusCubed =
        inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
    auto const& kernel = selectedGravityKernel();
    for (size_t sink = 0; sink < sinks.size(); ++sink) {
        for (auto const& source : sources) {
            pmmm::detail::accumulateNearPair(sinks[sink], source, cellWidth, inverseSupportRadius,
                inverseSupportRadiusCubed, doPotential, kernel, fields[sink]);
        }
    }
}
