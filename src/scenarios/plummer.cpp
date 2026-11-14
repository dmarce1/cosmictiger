/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <numbers>
#include <random>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/scenarios/plummer.hpp>

namespace scenarios {

/// Sample a reproducible Plummer realization in opposite position/velocity pairs.
void initializePlummerParticles() {
    auto const& options = getOptions();
    auto const numberOfParticles = options.initial.numberOfParticles;
    particlesResize(numberOfParticles);
    // Runtime and profiling libraries may reseed the process-global C RNG.
    // An owned generator keeps this initial realization reproducible.
    std::mt19937 generator(42);
    auto const sampleUniform = [&generator] {
        return (static_cast<double>(generator()) + 0.5) / 4294967296.0;
    };
    double x0 = 0.5;
    double y0 = 0.5;
    double z0 = 0.5;

    for (ParticleInteger i = 0; i < numberOfParticles; i += 2) {
        // Invert M(<r)=r^3/(1+r^2)^(3/2), conditioned on staying inside
        // the periodic box. This avoids a numerical integral's finite plateau.
        double const scale = options.scenario.plummer.radius;
        double const maximumMass = std::pow(1.0 + 4.0 * scale * scale, -1.5);
        double const massFraction = sampleUniform() * maximumMass;
        double const r = scale / std::sqrt(std::expm1(-2.0 / 3.0 * std::log(massFraction)));
        double nx, ny, nz, n2;
        do {
            nx = 2.0 * sampleUniform() - 1.0;
            ny = 2.0 * sampleUniform() - 1.0;
            nz = 2.0 * sampleUniform() - 1.0;
            n2 = sqr(nx, ny, nz);
        } while (n2 == 0.0 || n2 > 1.0);
        double ninv = 1.0 / sqrt(n2);
        nx *= ninv;
        ny *= ninv;
        nz *= ninv;
        double x = x0 + r * nx;
        double y = y0 + r * ny;
        double z = z0 + r * nz;
        double velocity = sqrt(options.gravity.gm * numberOfParticles /
            sqrt(sqr(options.scenario.plummer.radius) + sqr(r)) / 2);
        do {
            nx = 2.0 * sampleUniform() - 1.0;
            ny = 2.0 * sampleUniform() - 1.0;
            nz = 2.0 * sampleUniform() - 1.0;
            n2 = sqr(nx, ny, nz);
        } while (n2 == 0.0 || n2 > 1.0);
        ninv = 1.0 / sqrt(n2);
        nx *= ninv;
        ny *= ninv;
        nz *= ninv;
        double vx = velocity * nx;
        double vy = velocity * ny;
        double vz = velocity * nz;
        particlesPos(kXi, i) = x;
        particlesPos(kYi, i) = y;
        particlesPos(kZi, i) = z;
        particlesVel(kXi, i) = vx;
        particlesVel(kYi, i) = vy;
        particlesVel(kZi, i) = vz;
        particlesRung(i) = 0;
        if (i + 1 == numberOfParticles) {
            break;
        }
        particlesPos(kXi, i + 1) = 2.0 * x0 - x;
        particlesPos(kYi, i + 1) = 2.0 * y0 - y;
        particlesPos(kZi, i + 1) = 2.0 * z0 - z;
        particlesVel(kXi, i + 1) = -vx;
        particlesVel(kYi, i + 1) = -vy;
        particlesVel(kZi, i + 1) = -vz;
        particlesRung(i + 1) = 0;
        particlesRung(i) = 0;
    }
}
/// Set the fixed Plummer background and duration, then populate its particle realization.
void initializePlummer(DriverParameters& state) {
    state.a = 1.0;
    state.adot = 0.0;
    state.tauMax = 1.0e-4;
    initializePlummerParticles();
}

}    // namespace scenarios
