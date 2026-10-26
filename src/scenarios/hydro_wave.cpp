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
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/hydro_wave.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace scenarios {

ScenarioPlan initializeHydroWave() {
    double rho = 1.0;
    double p0 = 0.845154255;
    auto opts = getOptions();
    ParticleInteger const ndim = getOptions().initial.particlesPerDimension;
    ParticleInteger const numberOfParticles = std::pow(ndim, kDimensions);
    opts.initial.gasMass = rho / numberOfParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    ParticleInteger i = 0;
    double const dx = 1.0 / ndim;
    for (int ix = 0; ix < ndim; ix++) {
        for (int iy = 0; iy < ndim; iy++) {
            for (int iz = 0; iz < ndim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double eint = p0 / rho / (getOptions().hydro.gamma - 1);
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * std::numbers::pi / 3.0 * rho),
                    1.0 / 3.0);

                if (ix == ndim / 2) {
                    x += 0.99 * dx;
                } else if (ix == ndim / 2 - 1) {
                    x -= 0.99 * dx;
                }
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(kXDimension, i) = x;
                sphParticlesPosition(kYDimension, i) = y;
                sphParticlesPosition(kZDimension, i) = z;
                sphParticlesVelocity(kXDimension, i) = 0.f;
                sphParticlesVelocity(kYDimension, i) = 0.f;
                sphParticlesVelocity(kZDimension, i) = 0.f;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;
                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The hydro wave scenario produced no gas particles; "
                                 "increase initial.particlesPerDimension");
    }
    return ScenarioPlan{{{0.5, 64}}};
}

}    // namespace scenarios
