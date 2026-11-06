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
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/blast.hpp>
#include <cosmictiger/scenarios/diagnostics.hpp>
#include <cosmictiger/scenarios/sedov.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace scenarios {

namespace detail {

    constexpr double kDuration = 5e-3;

    static void finishBlast(double endTime) {
        FILE* fp = fopen(outputPath("blast.txt").c_str(), "wt");
        double l1 = 0.0, l2 = 0.0, lmax = 0.0;
        double norm1 = 0.0, norm2 = 0.0;
        constexpr int kN = 100000;
        if (fp == nullptr) {
            throw std::runtime_error("Unable to open blast.txt for writing");
        }
        for (auto const& sample : diagnosticSamples(kN)) {
            float const x = sample.position[kXDimension] - 0.5;
            float const y = sample.position[kYDimension] - 0.5;
            float const z = sample.position[kZDimension] - 0.5;
            float const rho = sample.density;
            double time = endTime;
            double r = sqrt(sqr(x, y, z));
            double rmax = sqrt(1.5);
            double d, velocity, p;
            int ndim = kSpatialDimensions;
            sedov::solution(time, r, rmax, d, velocity, p, ndim, getOptions().hydro.gamma);
            double dif = fabs(rho - d);
            norm1 += sample.weight * d;
            norm2 += sample.weight * sqr(d);
            l1 += sample.weight * dif;
            l2 += sample.weight * dif * dif;
            lmax = std::max(lmax, dif);
            fprintf(fp, "%e %e %e\n", r, rho, d);
        }
        l1 /= norm1;
        l2 /= norm2;
        l2 = sqrt(l2);
        std::cout << std::format("L1 = {:e} L2 = {:e} Lmax = {:e}\n", l1, l2, lmax) << std::flush;
        fclose(fp);
    }

}    // namespace detail

ScenarioPlan initializeBlast() {
    double rho0 = 1.0;
    auto opts = getOptions();
    auto const layout = particle_count::roundLattice(opts.run.gasParticleCount, 2);
    ParticleInteger const ndim = layout.side;
    ParticleInteger const numberOfParticles = layout.count;
    opts.run.gasParticleCount = layout.count;
    opts.initial.gasMass = rho0 / numberOfParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    ParticleInteger i = 0;
    double const dx = 1.0 / ndim;
    double sigma = .015;
    for (int ix = 0; ix < ndim; ix++) {
        for (int iy = 0; iy < (kSpatialDimensions >= 2 ? ndim : 1); iy++) {
            for (int iz = 0; iz < (kSpatialDimensions >= 3 ? ndim : 1); iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double r = sqrt(sqr(x - 0.5) + (kSpatialDimensions >= 2 ? sqr(y - 0.5) : 0.0) +
                    (kSpatialDimensions >= 3 ? sqr(z - 0.5) : 0.0));
                double eint = std::max(
                    std::pow(2.0 * std::numbers::pi * sigma * sigma, -0.5 * kSpatialDimensions) *
                        exp(-0.5 * sqr(r / sigma)),
                    1e-20);
                double vx = 0.0;
                double vy = 0.0;
                double vz = 0.0;
                double h = sph_geometry::supportRadius(m, rho0, getOptions().hydro.neighborNumber);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x;
                if constexpr (kSpatialDimensions > 1) {
                    sphParticlesPosition(kYDimension, i) = kSpatialDimensions >= 2 ? y : 0.5;
                }
                if constexpr (kSpatialDimensions > 2) {
                    sphParticlesPosition(kZDimension, i) = kSpatialDimensions >= 3 ? z : 0.5;
                }
                sphParticlesVelocity(kXDimension, i) = vx;
                if constexpr (kSpatialDimensions > 1) {
                    sphParticlesVelocity(kYDimension, i) = vy;
                }
                if constexpr (kSpatialDimensions > 2) {
                    sphParticlesVelocity(kZDimension, i) = vz;
                }
                sphParticlesRung(i) = 0;
                sphParticlesSetInternalEnergy(i, eint);

                i++;
                x = (ix + 0.0) * dx;
                y = (iy + 0.0) * dx;
                z = (iz + 0.0) * dx;
                r = sqrt(sqr(x - 0.5) + (kSpatialDimensions >= 2 ? sqr(y - 0.5) : 0.0) +
                    (kSpatialDimensions >= 3 ? sqr(z - 0.5) : 0.0));
                eint = std::max(
                    std::pow(2.0 * std::numbers::pi * sigma * sigma, -0.5 * kSpatialDimensions) *
                        exp(-0.5 * sqr(r / sigma)),
                    1e-20);
                vx = 0.0;
                vy = 0.0;
                vz = 0.0;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x;
                if constexpr (kSpatialDimensions > 1) {
                    sphParticlesPosition(kYDimension, i) = kSpatialDimensions >= 2 ? y : 0.5;
                }
                if constexpr (kSpatialDimensions > 2) {
                    sphParticlesPosition(kZDimension, i) = kSpatialDimensions >= 3 ? z : 0.5;
                }
                sphParticlesVelocity(kXDimension, i) = vx;
                if constexpr (kSpatialDimensions > 1) {
                    sphParticlesVelocity(kYDimension, i) = vy;
                }
                if constexpr (kSpatialDimensions > 2) {
                    sphParticlesVelocity(kZDimension, i) = vz;
                }
                sphParticlesRung(i) = 0;
                sphParticlesSetInternalEnergy(i, eint);

                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The blast scenario produced no gas particles; "
                                 "increase run.gasParticleCount");
    }
    double depositedEnergy = 0.0;
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        depositedEnergy += m * sphParticlesInternalEnergy(particle);
    }
    if (!(depositedEnergy > 0.0)) {
        throw std::runtime_error("The blast lattice has no deposited energy");
    }
    for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
        sphParticlesSetInternalEnergy(
            particle, sphParticlesInternalEnergy(particle) / (depositedEnergy));
    }
    return ScenarioPlan{{{detail::kDuration, 16}}, detail::finishBlast};
}

}    // namespace scenarios
