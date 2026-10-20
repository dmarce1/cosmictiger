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
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/sod.hpp>
#include <cosmictiger/sph/exact_sod.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace scenarios {

namespace detail {

    constexpr float kDuration = 0.09f;
    static SodInitT initialState{};

    static void finishSod(double time) {
        if (sphParticlesSize() == 0) {
            return;
        }
        constexpr int kN = 1000;
        FILE* fp = fopen(outputPath("sod.txt").c_str(), "wt");
        double l1 = 0.0, l2 = 0.0, lmax = 0.0;
        double norm1 = 0.0, norm2 = 0.0;
        for (int l = 0; l < kN; l++) {
            int const i = rand() % sphParticlesSize();
            int const j = sphParticlesParticleIndex(i);
            float x = particlesPos(XDIM, j).toDouble();
            float const rho = sphParticlesDensity(i);
            float const eint = sphParticlesInternalEnergy(i);
            float const p = (getOptions().hydro.gamma - 1.0) * rho * eint;
            SodStateT state;
            float x0 = x;
            x0 -= 0.5;
            if (x0 > 0.25) {
                x0 = 0.5 - x0;
            } else if (x0 < -0.25) {
                x0 = -0.5 - x0;
            }
            exactSod(&state, &initialState, x0, time, 0);
            double dif = fabs(rho - state.rho);
            norm1 += state.rho;
            norm2 += sqr(state.rho);
            l1 += dif;
            l2 += dif * dif;
            lmax = std::max(lmax, dif);
            fprintf(fp, "%e %e %e %e %e\n", x, rho, state.rho, p, state.p);
        }
        l1 /= norm1;
        l2 /= norm2;
        l2 = sqrt(l2);
        PRINT("L1 = %e L2 = %e Lmax = %e\n", l1, l2, lmax);
        fclose(fp);
    }

}    // namespace detail

ScenarioPlan initializeSod() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double rho0 = 0.125;
    double rho1 = 1.0;

    double vx0 = 0.0;
    double vy0 = -0.0;
    double vz0 = -0.0e-1;
    double p0 = .1;
    double p1 = 1.0;
    auto& sod = detail::initialState;
    ParticleInteger leftDim =
        pow(0.25 * totalNumberOfParticles * rho1 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
    ParticleInteger rightDim =
        pow(0.25 * totalNumberOfParticles * rho0 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
    if (leftDim < 1 || rightDim < 1) {
        throw std::runtime_error("The Sod scenario needs a resolved lattice in both layers; "
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
    rho1 = numberOfLeftParticles * m / 0.5;
    rho0 = numberOfRightParticles * m / 0.5;
    sod.rhol = rho1;
    sod.rhor = rho0;
    sod.pl = p1;
    sod.pr = p0;
    sod.gamma = getOptions().hydro.gamma;
    constexpr float kEta = 0.0;
    double const ne = getOptions().hydro.neighborNumber;
    terminal::detailPrint("Sod dimensions are %i and %i rho0 %e rho1 %e\n", 2 * rightDim,
        2 * leftDim, rho0 / (m * ne) * (4.0 * M_PI / 3.0), rho1 / (m * ne) * (4.0 * M_PI / 3.0));
    for (int ix = 0; ix < leftDim; ix++) {
        for (int iy = 0; iy < 2 * leftDim; iy++) {
            for (int iz = 0; iz < 2 * leftDim; iz++) {
                double x = (ix) *dx;
                double y = (iy) *dx;
                double z = (iz) *dx;

                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(YDIM, i) = y + rand1() * dx * kEta;
                sphParticlesPosition(ZDIM, i) = z + rand1() * dx * kEta;
                sphParticlesVelocity(XDIM, i) = vx0;
                sphParticlesVelocity(YDIM, i) = vy0;
                sphParticlesVelocity(ZDIM, i) = vz0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy =
                    p1 / (getOptions().hydro.gamma - 1.0) / rho1;
                i++;
                x = (ix + 0.5) * dx;
                y = (iy + 0.5) * dx;
                z = (iz + 0.5) * dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(YDIM, i) = y + rand1() * dx * kEta;
                sphParticlesPosition(ZDIM, i) = z + rand1() * dx * kEta;
                sphParticlesVelocity(XDIM, i) = vx0;
                sphParticlesVelocity(YDIM, i) = vy0;
                sphParticlesVelocity(ZDIM, i) = vz0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy =
                    p1 / (getOptions().hydro.gamma - 1.0) / rho1;
                i++;
            }
        }
    }
    dx = 0.5 / rightDim;
    for (int ix0 = rightDim; ix0 < 2 * rightDim; ix0++) {
        int ix = ix0;
        for (int iy = 0; iy < 2 * rightDim; iy++) {
            for (int iz = 0; iz < 2 * rightDim; iz++) {
                double x = (ix) *dx;
                double y = (iy) *dx;
                double z = (iz) *dx;

                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(YDIM, i) = y + rand1() * dx * kEta;
                sphParticlesPosition(ZDIM, i) = z + rand1() * dx * kEta;
                sphParticlesVelocity(XDIM, i) = 0;
                sphParticlesVelocity(YDIM, i) = 0;
                sphParticlesVelocity(ZDIM, i) = 0;

                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy =
                    p0 / (getOptions().hydro.gamma - 1.0) / rho0;

                i++;
                x = (ix + 0.5) * dx;
                y = (iy + 0.5) * dx;
                z = (iz + 0.5) * dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(YDIM, i) = y + rand1() * dx * kEta;
                sphParticlesPosition(ZDIM, i) = z + rand1() * dx * kEta;
                sphParticlesVelocity(XDIM, i) = 0;
                sphParticlesVelocity(YDIM, i) = 0;
                sphParticlesVelocity(ZDIM, i) = 0;

                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy =
                    p0 / (getOptions().hydro.gamma - 1.0) / rho0;

                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The Sod scenario produced no gas particles; "
                                 "increase initial.particlesPerDimension");
    }
    return ScenarioPlan{{{detail::kDuration, 20}}, detail::finishSod};
}

}    // namespace scenarios
