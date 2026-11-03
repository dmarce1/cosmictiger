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

void initializePlummerParticles() {
    auto const numberOfParticles = getOptions().initial.numberOfParticles;
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
        double const scale = getOptions().initial.plummerRadius;
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
        double velocity = sqrt(getOptions().gravity.gm * numberOfParticles /
            sqrt(sqr(getOptions().initial.plummerRadius) + sqr(r)) / 2);
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
        particlesPos(kXDimension, i) = x;
        particlesPos(kYDimension, i) = y;
        particlesPos(kZDimension, i) = z;
        particlesVel(kXDimension, i) = vx;
        particlesVel(kYDimension, i) = vy;
        particlesVel(kZDimension, i) = vz;
        particlesRung(i) = 0;
        if (i + 1 == numberOfParticles) {
            break;
        }
        particlesPos(kXDimension, i + 1) = 2.0 * x0 - x;
        particlesPos(kYDimension, i + 1) = 2.0 * y0 - y;
        particlesPos(kZDimension, i + 1) = 2.0 * z0 - z;
        particlesVel(kXDimension, i + 1) = -vx;
        particlesVel(kYDimension, i + 1) = -vy;
        particlesVel(kZDimension, i + 1) = -vz;
        particlesRung(i + 1) = 0;
        particlesRung(i) = 0;
    }
}
void initializePlummer(DriverParameters& state) {
    state.a = 1.0;
    state.adot = 0.0;
    state.tauMax = 1.0e-4;
    initializePlummerParticles();
}

}    // namespace scenarios
