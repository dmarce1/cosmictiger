/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <functional>
#include <numbers>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/scenarios/plummer.hpp>

namespace scenarios {

namespace detail {

    double sampleDensityDistribution(std::function<double(double)> const& f) {
        double target = rand1();
        double menc = 0.0;
        double r = 0.0;
        double dr = 0.001;
        double iValue, i0;
        iValue = 0.0;
        while (menc < target) {
            menc += 0.5 * iValue * dr;
            i0 = 0.5 * iValue;
            r += dr;
            iValue = f(r) * 4.0 * std::numbers::pi * sqr(r);
            i0 += 0.5 * iValue;
            menc += 0.5 * iValue * dr;
        }
        double dr2 = (menc - target) / i0;
        r -= dr2;
        return r;
    }
}    // namespace detail

void initializePlummerParticles() {
    auto const numberOfParticles = getOptions().initial.numberOfParticles;
    particlesResize(numberOfParticles);
    double x0 = 0.5;
    double y0 = 0.5;
    double z0 = 0.5;

    for (ParticleInteger i = 0; i < numberOfParticles; i += 2) {
        double r;
        r = detail::sampleDensityDistribution(
            [](double r) { return (3.0 / (4.0 * std::numbers::pi)) * pow(1.0 + sqr(r), -2.5); });
        r *= getOptions().initial.plummerRadius;
        double nx, ny, nz, n2;
        do {
            nx = 2.0 * rand1() - 1.0;
            ny = 2.0 * rand1() - 1.0;
            nz = 2.0 * rand1() - 1.0;
            n2 = sqr(nx, ny, nz);
        } while (n2 == 0.0 || n2 > 1.0);
        double ninv = 1.0 / sqrt(n2);
        nx *= ninv;
        ny *= ninv;
        nz *= ninv;
        double x = x0 + r * nx;
        double y = y0 + r * ny;
        double z = z0 + r * nz;
        double v = sqrt(getOptions().gravity.gm * numberOfParticles /
            sqrt(sqr(getOptions().initial.plummerRadius) + sqr(r)) / 2);
        do {
            nx = 2.0 * rand1() - 1.0;
            ny = 2.0 * rand1() - 1.0;
            nz = 2.0 * rand1() - 1.0;
            n2 = sqr(nx, ny, nz);
        } while (n2 == 0.0 || n2 > 1.0);
        ninv = 1.0 / sqrt(n2);
        nx *= ninv;
        ny *= ninv;
        nz *= ninv;
        double vx = v * nx;
        double vy = v * ny;
        double vz = v * nz;
        particlesPos(kXDimension, i) = x;
        particlesPos(kYDimension, i) = y;
        particlesPos(kZDimension, i) = z;
        particlesVel(kXDimension, i) = vx;
        particlesVel(kYDimension, i) = vy;
        particlesVel(kZDimension, i) = vz;
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
