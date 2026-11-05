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
#include <format>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/diagnostics.hpp>
#include <cosmictiger/scenarios/exact_sod.hpp>
#include <cosmictiger/scenarios/sod.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace scenarios {

namespace detail {

    constexpr float kDuration = 0.09f;
    static SodInitT initialState{};

    static void finishSod(double time) {
        constexpr int kN = 1000;
        FILE* fp = fopen(outputPath("sod.txt").c_str(), "wt");
        double l1 = 0.0, l2 = 0.0, lmax = 0.0;
        double norm1 = 0.0, norm2 = 0.0;
        if (fp == nullptr) {
            throw std::runtime_error("Unable to open sod.txt for writing");
        }
        for (auto const& sample : diagnosticSamples(kN)) {
            float const position = sample.position[kXDimension];
            float const rho = sample.density;
            float const p = sample.pressure;
            SodStateT state;
            float x0 = position;
            x0 -= 0.5;
            if (x0 > 0.25) {
                x0 = 0.5 - x0;
            } else if (x0 < -0.25) {
                x0 = -0.5 - x0;
            }
            exactSod(&state, &initialState, x0, time, 0);
            double dif = fabs(rho - state.rho);
            norm1 += sample.weight * state.rho;
            norm2 += sample.weight * sqr(state.rho);
            l1 += sample.weight * dif;
            l2 += sample.weight * dif * dif;
            lmax = std::max(lmax, dif);
            fprintf(fp, "%e %e %e %e %e\n", position, rho, state.rho, p, state.p);
        }
        l1 /= norm1;
        l2 /= norm2;
        l2 = sqrt(l2);
        std::cout << std::format("L1 = {:e} L2 = {:e} Lmax = {:e}\n", l1, l2, lmax) << std::flush;
        fclose(fp);
    }

}    // namespace detail

ScenarioPlan initializeSod() {
    ParticleInteger totalNumberOfParticles = getOptions().run.gasParticleCount;
    double rho0 = 0.125;
    double rho1 = 1.0;

    double vx0 = 0.0;
    double vy0 = -0.0;
    double vz0 = -0.0e-1;
    double p0 = .1;
    double p1 = 1.0;
    auto& sod = detail::initialState;
    ParticleInteger const rightDim = particle_count::roundLattice(
        totalNumberOfParticles, particle_count::kSodLatticeMultiplicity)
                                         .side;
    ParticleInteger const leftDim =
        (kSpatialDimensions == 1 ? 8 : (kSpatialDimensions == 2 ? 4 : 2)) * rightDim;
    // In 2D a rectangular lattice realizes exactly rho_left/rho_right = 8:
    // 2(4n)(4n) left particles versus 2(n)(2n) right particles.
    ParticleInteger const leftTransverseSide = kSpatialDimensions == 2 ? leftDim : 2 * leftDim;
    if (leftDim < 1 || rightDim < 1) {
        throw std::runtime_error("The Sod scenario needs a resolved lattice in both layers; "
                                 "increase run.gasParticleCount");
    }
    ParticleInteger numberOfLeftParticles = kSpatialDimensions == 2 ?
        2 * leftDim * leftTransverseSide :
        particle_count::spatialPower(2 * leftDim);
    ParticleInteger numberOfRightParticles = particle_count::spatialPower(2 * rightDim);
    totalNumberOfParticles = numberOfLeftParticles + numberOfRightParticles;
    double dx = 0.5 / leftDim;
    double dy = kSpatialDimensions == 2 ? 1.0 / leftTransverseSide : dx;
    ParticleInteger i = 0;
    auto opts = getOptions();
    opts.run.gasParticleCount = totalNumberOfParticles;
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
    terminal::detailPrint(
        "Sod dimensions are %i and %i rho0 %e rho1 %e\n", 2 * rightDim, 2 * leftDim, rho0, rho1);
    for (int ix = 0; ix < leftDim; ix++) {
        for (int iy = 0; iy < (kSpatialDimensions >= 2 ? leftTransverseSide : 1); iy++) {
            for (int iz = 0; iz < (kSpatialDimensions >= 3 ? 2 * leftDim : 1); iz++) {
                double x = (ix) *dx;
                double y = iy * dy;
                double z = (iz) *dx;

                double h = sph_geometry::supportRadius(m, rho1, getOptions().hydro.neighborNumber);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(kYDimension, i) =
                    kSpatialDimensions >= 2 ? y + rand1() * dx * kEta : 0.5;
                sphParticlesPosition(kZDimension, i) =
                    kSpatialDimensions >= 3 ? z + rand1() * dx * kEta : 0.5;
                sphParticlesVelocity(kXDimension, i) = vx0;
                sphParticlesVelocity(kYDimension, i) = vy0;
                sphParticlesVelocity(kZDimension, i) = vz0;
                sphParticlesRung(i) = 0;
                sphParticlesSetInternalEnergy(i, p1 / (getOptions().hydro.gamma - 1.0) / rho1);
                i++;
                x = (ix + 0.5) * dx;
                y = (iy + 0.5) * dy;
                z = (iz + 0.5) * dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(kYDimension, i) =
                    kSpatialDimensions >= 2 ? y + rand1() * dx * kEta : 0.5;
                sphParticlesPosition(kZDimension, i) =
                    kSpatialDimensions >= 3 ? z + rand1() * dx * kEta : 0.5;
                sphParticlesVelocity(kXDimension, i) = vx0;
                sphParticlesVelocity(kYDimension, i) = vy0;
                sphParticlesVelocity(kZDimension, i) = vz0;
                sphParticlesRung(i) = 0;
                sphParticlesSetInternalEnergy(i, p1 / (getOptions().hydro.gamma - 1.0) / rho1);
                i++;
            }
        }
    }
    dx = 0.5 / rightDim;
    dy = dx;
    for (int ix0 = rightDim; ix0 < 2 * rightDim; ix0++) {
        int ix = ix0;
        for (int iy = 0; iy < (kSpatialDimensions >= 2 ? 2 * rightDim : 1); iy++) {
            for (int iz = 0; iz < (kSpatialDimensions >= 3 ? 2 * rightDim : 1); iz++) {
                double x = (ix) *dx;
                double y = iy * dy;
                double z = (iz) *dx;

                double h = sph_geometry::supportRadius(m, rho0, getOptions().hydro.neighborNumber);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(kYDimension, i) =
                    kSpatialDimensions >= 2 ? y + rand1() * dx * kEta : 0.5;
                sphParticlesPosition(kZDimension, i) =
                    kSpatialDimensions >= 3 ? z + rand1() * dx * kEta : 0.5;
                sphParticlesVelocity(kXDimension, i) = 0;
                sphParticlesVelocity(kYDimension, i) = 0;
                sphParticlesVelocity(kZDimension, i) = 0;

                sphParticlesRung(i) = 0;
                sphParticlesSetInternalEnergy(i, p0 / (getOptions().hydro.gamma - 1.0) / rho0);

                i++;
                x = (ix + 0.5) * dx;
                y = (iy + 0.5) * dy;
                z = (iz + 0.5) * dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x + rand1() * dx * kEta;
                sphParticlesPosition(kYDimension, i) =
                    kSpatialDimensions >= 2 ? y + rand1() * dx * kEta : 0.5;
                sphParticlesPosition(kZDimension, i) =
                    kSpatialDimensions >= 3 ? z + rand1() * dx * kEta : 0.5;
                sphParticlesVelocity(kXDimension, i) = 0;
                sphParticlesVelocity(kYDimension, i) = 0;
                sphParticlesVelocity(kZDimension, i) = 0;

                sphParticlesRung(i) = 0;
                sphParticlesSetInternalEnergy(i, p0 / (getOptions().hydro.gamma - 1.0) / rho0);

                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The Sod scenario produced no gas particles; "
                                 "increase run.gasParticleCount");
    }
    return ScenarioPlan{{{detail::kDuration, 20}}, detail::finishSod};
}

}    // namespace scenarios
