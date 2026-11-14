/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <numbers>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/hydro_wave.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace scenarios {

/// Initialize a periodic gas wave on the configured lattice and return its fixed evolution
/// schedule.
ScenarioPlan initializeHydroWave() {
    auto const& options = getOptions();
    double rho = 1.0;
    double p0 = 0.845154255;
    auto opts = options;
    ParticleInteger const ndim = particle_count::roundLattice(opts.run.gasParticleCount).side;
    ParticleInteger const numberOfParticles = particle_count::spatialPower(ndim);
    opts.run.gasParticleCount = numberOfParticles;
    opts.initial.gasMass = rho / numberOfParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    ParticleInteger i = 0;
    double const dx = 1.0 / ndim;
    for (int ix = 0; ix < ndim; ix++) {
        for (int iy = 0; iy < (kSpatialDimensions >= 2 ? ndim : 1); iy++) {
            for (int iz = 0; iz < (kSpatialDimensions >= 3 ? ndim : 1); iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double eint = p0 / rho / (options.hydro.heatCapacityRatio - 1);
                double h = sph_geometry::supportRadius(m, rho, options.hydro.neighborNumber);

                if (ix == ndim / 2) {
                    x += 0.99 * dx;
                } else if (ix == ndim / 2 - 1) {
                    x -= 0.99 * dx;
                }
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXi, i) = x;
                if constexpr (kSpatialDimensions > 1) {
                    sphParticlesPosition(kYi, i) = kSpatialDimensions >= 2 ? y : 0.5;
                }
                if constexpr (kSpatialDimensions > 2) {
                    sphParticlesPosition(kZi, i) = kSpatialDimensions >= 3 ? z : 0.5;
                }
                sphParticlesVelocity(kXi, i) = 0.f;
                if constexpr (kSpatialDimensions > 1) {
                    sphParticlesVelocity(kYi, i) = 0.f;
                }
                if constexpr (kSpatialDimensions > 2) {
                    sphParticlesVelocity(kZi, i) = 0.f;
                }
                sphParticlesRung(i) = 0;
                sphParticlesSetInternalEnergy(i, eint);
                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The hydro wave scenario produced no gas particles; "
                                 "increase run.gasParticleCount");
    }
    return ScenarioPlan{{{0.5, 64}}};
}

}    // namespace scenarios
