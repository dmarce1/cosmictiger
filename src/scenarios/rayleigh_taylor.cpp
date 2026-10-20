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
#include <cosmictiger/scenarios/rayleigh_taylor.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace scenarios {

ScenarioPlan initializeRayleighTaylor() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    ParticleInteger hiParts = 2 * totalNumberOfParticles / 3;
    ParticleInteger loParts = totalNumberOfParticles / 3;
    ParticleInteger dim = pow(loParts * 0.25, (1. / 3.));
    if (dim < 1) {
        throw std::runtime_error("The Rayleigh-Taylor scenario needs a resolved lattice; "
                                 "increase initial.particlesPerDimension");
    }
    double rho1 = 2.0;
    double rho0 = 1.0;
    loParts = 4 * dim * dim * dim;
    hiParts = 2 * loParts;
    double const yAcceleration = getOptions().hydro.yAcceleration;
    auto opts = getOptions();
    opts.initial.gasMass = rho0 / loParts * 0.5;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    rho1 = hiParts * m / 0.5;
    rho0 = loParts * m / 0.5;

    double dx = .5 / dim;
    int i = 0;
    double v0 = 0.1;
    for (int ix = 0; ix < 2 * dim; ix++) {
        for (int iy = dim / 2; iy <= 3 * dim / 2; iy++) {
            for (int iz = 0; iz < 2 * dim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double vy = v0 / 4.0 *
                    (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
                double p = 1.5 + fabs(rho0 * yAcceleration * (y - 0.5));
                double eint = p / (getOptions().hydro.gamma - 1) / rho0;
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = 0;
                sphParticlesVelocity(YDIM, i) = vy;

                sphParticlesVelocity(ZDIM, i) = 0;

                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
            }
        }
    }
    for (int ix = 0; ix < 2 * dim; ix++) {
        for (int iy = 0; iy < 2 * dim; iy++) {
            for (int iz = 0; iz < 2 * dim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double p;
                double vy = v0 / 4.0 *
                    (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
                if (y > 0.75 || y < 0.25) {
                    if (y >= 0.75) {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.75)) +
                            fabs(rho0 * yAcceleration * .25);
                    } else {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.25)) +
                            fabs(rho0 * yAcceleration * .25);
                    }
                    double eint = p / (getOptions().hydro.gamma - 1) / rho1;
                    sphParticlesResize(sphParticlesSize() + 1);
                    sphParticlesSupportRadius(i) = h;
                    sphParticlesPosition(XDIM, i) = x;
                    sphParticlesPosition(YDIM, i) = y;
                    sphParticlesPosition(ZDIM, i) = z;
                    sphParticlesVelocity(XDIM, i) = 0;

                    sphParticlesVelocity(YDIM, i) = vy;

                    sphParticlesVelocity(ZDIM, i) = 0;

                    sphParticlesRung(i) = 0;
                    sphParticlesThermodynamicState(i).internalEnergy = eint;

                    i++;
                }
                x = (ix + 0.0) * dx;
                y = (iy + 0.0) * dx;
                z = (iz + 0.0) * dx;
                vy = v0 / 4.0 *
                    (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
                if (y > 0.75 || y < 0.25) {
                    if (y >= 0.75) {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.75)) +
                            fabs(rho0 * yAcceleration * .25);
                    } else {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.25)) +
                            fabs(rho0 * yAcceleration * .25);
                    }
                    double eint = p / (getOptions().hydro.gamma - 1) / rho1;
                    sphParticlesResize(sphParticlesSize() + 1);
                    sphParticlesSupportRadius(i) = h;
                    sphParticlesPosition(XDIM, i) = x;
                    sphParticlesPosition(YDIM, i) = y;
                    sphParticlesPosition(ZDIM, i) = z;
                    sphParticlesVelocity(XDIM, i) = 0;

                    sphParticlesVelocity(YDIM, i) = vy;

                    sphParticlesVelocity(ZDIM, i) = 0;

                    sphParticlesRung(i) = 0;
                    sphParticlesThermodynamicState(i).internalEnergy = eint;

                    i++;
                }
            }
        }
    }

    constexpr float kT = 10.0;
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The Rayleigh-Taylor scenario produced no gas particles; "
                                 "increase initial.particlesPerDimension");
    }
    return ScenarioPlan{{{kT, 256}}};
}

}    // namespace scenarios
