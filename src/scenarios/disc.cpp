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
#include <cosmictiger/scenarios/disc.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace scenarios {

ScenarioPlan initializeDisc() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double const innerRadius = 0.2;
    double const outerRadius = 0.3;
    double const rho = totalNumberOfParticles;
    // Keep the disc cold while resolving viscous heating with the relative
    // thermal timestep bound. Specific energy is independent of resolution.
    double const internalEnergy = 1.0e-4;
    auto opts = getOptions();
    opts.initial.gasMass = 1.0;
    setOptions(opts);
    double const m = opts.initial.gasMass;
    double h = pow(m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);
    int i = 0;
    int const nz = 3;
    int dim = sqrt(totalNumberOfParticles) / nz;
    if (dim < 1) {
        throw std::runtime_error("The disc scenario needs a resolved lattice; "
                                 "increase initial.particlesPerDimension");
    }
    double dx = 1.0 / dim;
    for (int iz = dim / 2 - nz; iz < dim / 2 + nz; iz++) {
        for (int k = 0; k < dim; k++) {
            for (int l = 0; l < dim; l++) {
                double r, x, y, z;
                x = dx * k;
                y = dx * l;
                z = iz * dx;
                r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
                if (r > innerRadius && r < outerRadius) {
                    sphParticlesResize(sphParticlesSize() + 1);
                    sphParticlesSupportRadius(i) = h;
                    sphParticlesPosition(XDIM, i) = x;
                    sphParticlesPosition(YDIM, i) = y;
                    sphParticlesPosition(ZDIM, i) = z;
                    double const v = 1.00 * sqrt(1 / r);
                    double const vx = -(y - 0.5) / r * v;
                    double const vy = (x - 0.5) / r * v;
                    double const vz = 0.0;
                    sphParticlesVelocity(XDIM, i) = vx;
                    sphParticlesVelocity(YDIM, i) = vy;
                    sphParticlesVelocity(ZDIM, i) = vz;
                    sphParticlesRung(i) = 0;
                    sphParticlesThermodynamicState(i).internalEnergy = internalEnergy;

                    i++;
                }
            }
        }
    }
    double const t = 16.0;
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The disc scenario produced no gas particles; "
                                 "increase initial.particlesPerDimension");
    }
    return ScenarioPlan{{{t, 1024}}};
}

}    // namespace scenarios
