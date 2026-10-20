/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/blast.hpp>
#include <cosmictiger/sph/sedov.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace scenarios {

namespace detail {

    constexpr double kDuration = 5e-3;

    static void finishBlast(double endTime) {
        if (sphParticlesSize() == 0) {
            return;
        }
        FILE* fp = fopen(outputPath("blast.txt").c_str(), "wt");
        double l1 = 0.0, l2 = 0.0, lmax = 0.0;
        double norm1 = 0.0, norm2 = 0.0;
        constexpr int kN = 100000;
        for (int l = 0; l < kN; l++) {
            int const i = rand() % sphParticlesSize();
            int const j = sphParticlesParticleIndex(i);
            float x = particlesPos(XDIM, j).toDouble() - 0.5;
            float y = particlesPos(YDIM, j).toDouble() - 0.5;
            float z = particlesPos(ZDIM, j).toDouble() - 0.5;
            float const rho = sphParticlesDensity(i);
            double time = endTime;
            double r = sqrt(sqr(x, y, z));
            double rmax = sqrt(1.5);
            double d, v, p;
            int ndim = 3;
            sedov::solution(time, r, rmax, d, v, p, ndim);
            double dif = fabs(rho - d);
            norm1 += d;
            norm2 += sqr(d);
            l1 += dif;
            l2 += dif * dif;
            lmax = std::max(lmax, dif);
            fprintf(fp, "%e %e %e\n", r, rho, d);
        }
        l1 /= norm1;
        l2 /= norm2;
        l2 = sqrt(l2);
        PRINT("L1 = %e L2 = %e Lmax = %e\n", l1, l2, lmax);
        fclose(fp);
    }

}    // namespace detail

ScenarioPlan initializeBlast() {
    double rho0 = 1.0;
    auto opts = getOptions();
    ParticleInteger ndim = getOptions().initial.particlesPerDimension;
    ParticleInteger numberOfParticles = std::pow(ndim, NDIM);
    ndim = pow(numberOfParticles / 2, 1. / 3.);
    if (ndim < 1) {
        throw std::runtime_error("The blast scenario needs a resolved lattice; "
                                 "increase initial.particlesPerDimension");
    }
    numberOfParticles = 2 * ndim * ndim * ndim;
    opts.initial.gasMass = rho0 / numberOfParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    ParticleInteger i = 0;
    double const dx = 1.0 / ndim;
    double sigma = .015;
    for (int ix = 0; ix < ndim; ix++) {
        for (int iy = 0; iy < ndim; iy++) {
            for (int iz = 0; iz < ndim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
                double eint = std::max(1.0 / sigma / sigma / sigma /
                        sqrt(8.0 * M_PI * M_PI * M_PI) * exp(-0.5 * sqr(r / sigma)),
                    1e-20);
                double vx = 0.0;
                double vy = 0.0;
                double vz = 0.0;
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = vx;
                sphParticlesVelocity(YDIM, i) = vy;
                sphParticlesVelocity(ZDIM, i) = vz;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
                x = (ix + 0.0) * dx;
                y = (iy + 0.0) * dx;
                z = (iz + 0.0) * dx;
                r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
                eint = std::max(1.0 / sigma / sigma / sigma / sqrt(8.0 * M_PI * M_PI * M_PI) *
                        exp(-0.5 * sqr(r / sigma)),
                    1e-20);
                vx = 0.0;
                vy = 0.0;
                vz = 0.0;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = vx;
                sphParticlesVelocity(YDIM, i) = vy;
                sphParticlesVelocity(ZDIM, i) = vz;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The blast scenario produced no gas particles; "
                                 "increase initial.particlesPerDimension");
    }
    return ScenarioPlan{{{detail::kDuration, 16}}, detail::finishBlast};
}

}    // namespace scenarios
