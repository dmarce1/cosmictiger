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
#include <cosmictiger/scenarios/kelvin_helmholtz.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace scenarios {

ScenarioPlan initializeKelvinHelmholtz() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double rho1 = 0.5;
    double rho0 = 1.0;
    double vy1 = 1.0;
    double vy0 = -1.0;
    double p1 = 19.2;
    double p0 = 19.2;
    constexpr double kEta = 1e-1;
    constexpr double kFreq = 2.0;
    ParticleInteger leftDim =
        pow(0.25 * totalNumberOfParticles * rho1 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
    ParticleInteger rightDim =
        pow(0.25 * totalNumberOfParticles * rho0 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
    if (leftDim < 1 || rightDim < 1) {
        throw std::runtime_error(
            "The Kelvin-Helmholtz scenario needs a resolved lattice in both layers; "
            "increase initial.particlesPerDimension");
    }
    ParticleInteger numberOfLeftParticles = 2 * leftDim * sqr(2 * leftDim);
    ParticleInteger numberOfRightParticles = 2 * rightDim * sqr(2 * rightDim);
    totalNumberOfParticles = numberOfLeftParticles + numberOfRightParticles;
    double dx = 0.5 / leftDim;
    ParticleInteger i = 0;
    auto opts = getOptions();
    opts.initial.gasMass = rho1 * 0.5 / numberOfLeftParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    rho1 = numberOfLeftParticles * m / 0.5f;
    rho0 = numberOfRightParticles * m / 0.5f;
    for (int ix = 0; ix < leftDim; ix++) {
        for (int iy = 0; iy < 2 * leftDim; iy++) {
            for (int iz = 0; iz < 2 * leftDim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double eint = p1 / rho1 / (getOptions().hydro.gamma - 1);
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy1;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
                x = (ix) *dx;
                y = (iy) *dx;
                z = (iz) *dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy1;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
            }
        }
    }
    dx = 0.5 / rightDim;
    for (int ix0 = rightDim; ix0 < 2 * rightDim; ix0++) {
        int ix = ix0;
        for (int iy = 0; iy < 2 * rightDim; iy++) {
            for (int iz = 0; iz < 2 * rightDim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double eint = p0 / rho0 / (getOptions().hydro.gamma - 1);
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy0;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
                x = (ix) *dx;
                y = (iy) *dx;
                z = (iz) *dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy0;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The Kelvin-Helmholtz scenario produced no gas particles; "
                                 "increase initial.particlesPerDimension");
    }
    return ScenarioPlan{{{10.0, 1024}}};
}

}    // namespace scenarios
