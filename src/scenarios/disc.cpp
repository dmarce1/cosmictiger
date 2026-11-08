/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/disc.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace scenarios {

ScenarioPlan initializeDisc() {
    ParticleInteger totalNumberOfParticles = getOptions().run.gasParticleCount;
    double const innerRadius = 0.2;
    double const outerRadius = 0.3;
    double const rho = totalNumberOfParticles;
    // Keep the disc cold while resolving viscous heating with the relative
    // thermal timestep bound. Specific energy is independent of resolution.
    double const internalEnergy = 1.0e-4;
    auto opts = getOptions();
    if (!(opts.scenario.disc.centralAcceleration > 0.0)) {
        throw std::invalid_argument(
            "The disc scenario requires scenario.disc.centralAcceleration > 0");
    }
    opts.initial.gasMass = 1.0;
    setOptions(opts);
    double const m = opts.initial.gasMass;
    double h = sph_geometry::supportRadius(m, rho, getOptions().hydro.neighborNumber);
    int i = 0;
    int const nz = 3;
    // A complete six-plane annular lattice is the valid resolution unit.
    auto const population = [&](int side) {
        size_t count = 0;
        double const spacing = 1.0 / side;
        for (int plane = side / 2 - nz; plane < side / 2 + nz; ++plane) {
            for (int row = 0; row < side; ++row) {
                for (int column = 0; column < side; ++column) {
                    double const radius = std::sqrt(
                        sqr(spacing * row - 0.5, spacing * column - 0.5, spacing * plane - 0.5));
                    count += radius > innerRadius && radius < outerRadius ? 1 : 0;
                }
            }
        }
        return count;
    };
    int dim = std::max(1,
        int(std::sqrt(double(totalNumberOfParticles) /
            (2 * nz * std::numbers::pi *
                (outerRadius * outerRadius - innerRadius * innerRadius)))));
    while (population(dim) < size_t(totalNumberOfParticles)) {
        ++dim;
    }
    opts.run.gasParticleCount = population(dim);
    setOptions(opts);
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
                    sphParticlesSmoothingLength(i) = h;
                    sphParticlesPosition(kXi, i) = x;
                    sphParticlesPosition(kYi, i) = y;
                    sphParticlesPosition(kZi, i) = z;
                    double const velocity = sqrt(opts.scenario.disc.centralAcceleration / r);
                    double const vx = -(y - 0.5) / r * velocity;
                    double const vy = (x - 0.5) / r * velocity;
                    double const vz = 0.0;
                    sphParticlesVelocity(kXi, i) = vx;
                    sphParticlesVelocity(kYi, i) = vy;
                    sphParticlesVelocity(kZi, i) = vz;
                    sphParticlesRung(i) = 0;
                    sphParticlesSetInternalEnergy(i, internalEnergy);

                    i++;
                }
            }
        }
    }
    double const t = 16.0;
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The disc scenario produced no gas particles; "
                                 "increase run.gasParticleCount");
    }
    return ScenarioPlan{{{t, 1024}}};
}

}    // namespace scenarios
