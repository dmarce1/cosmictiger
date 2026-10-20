/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cmath>
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
    ParticleInteger const numberOfParticles = std::pow(ndim, NDIM);
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
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);

                if (ix == ndim / 2) {
                    x += 0.99 * dx;
                } else if (ix == ndim / 2 - 1) {
                    x -= 0.99 * dx;
                }
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = 0.f;
                sphParticlesVelocity(YDIM, i) = 0.f;
                sphParticlesVelocity(ZDIM, i) = 0.f;
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
