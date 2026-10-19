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
#include <functional>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/problems/plummer.hpp>

double nfwDensity(double r) {
    double rmax = 10.0;
    return 1.0 / r / sqr(1.0 + r) / (4.0 * M_PI * (-1.0 + 1.0 / (1.0 + rmax) + log(1.0 + rmax)));
}

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
        iValue = f(r) * 4.0 * M_PI * sqr(r);
        i0 += 0.5 * iValue;
        menc += 0.5 * iValue * dr;
    }
    double dr2 = (menc - target) / i0;
    r -= dr2;
    return r;
}
void plummerInit(double r0) {
    auto const numberOfParticles = getOptions().initial.numberOfParticles;
    particlesResize(numberOfParticles);
    auto const pdf = [](double x) {
        if (x > 0.0) {
            return sqr(x) * pow(1.0 + sqr(x), -2.5);
        } else {
            return pow(1.0 + sqr(1.0 / x), -2.5) / sqr(sqr(x));
        }
    };

    /*	distribution_init([](float x, float y, float z) {
     float r = sqrt(sqr(x - 0.51, y - 0.51, z - 0.51));
     r *= 4.0;
     return 1.0 / r / sqr(1+r);
     return 1000*pow(1.0+sqr(r),-2.5);
     });
     return;*/

    double x0 = 0.5;
    double y0 = 0.5;
    double z0 = 0.5;

    for (ParticleInteger i = 0; i < numberOfParticles; i += 2) {
        double r, p;
        //		r = sample_density_distribution(nfw_density);
        r = sampleDensityDistribution(
            [](double r) { return (3.0 / (4.0 * M_PI)) * pow(1.0 + sqr(r), -2.5); });
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
        // double e = abs();
        //	double v = sqrt(-2.0 * log(rand1())) * abs(cos(2.0 * M_PI * rand1()));
        double v = sqrt(getOptions().gravity.gm * numberOfParticles /
            sqrt(sqr(getOptions().initial.plummerRadius) + sqr(r)) / 2);
        double v0 = sqrt(getOptions().gravity.gm * numberOfParticles * 0.25);
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
        particlesPos(XDIM, i) = x;
        particlesPos(YDIM, i) = y;
        particlesPos(ZDIM, i) = z;
        particlesVel(XDIM, i) = vx;
        particlesVel(YDIM, i) = vy;
        particlesVel(ZDIM, i) = vz;
        particlesPos(XDIM, i + 1) = 2.0 * x0 - x;
        particlesPos(YDIM, i + 1) = 2.0 * y0 - y;
        particlesPos(ZDIM, i + 1) = 2.0 * z0 - z;
        particlesVel(XDIM, i + 1) = -vx;
        particlesVel(YDIM, i + 1) = -vy;
        particlesVel(ZDIM, i + 1) = -vz;
        particlesRung(i + 1) = 0;
        particlesRung(i) = 0;
        /*if (i % 4 == 0) {
            particles_pos(XDIM, i) += 0.025;
            particles_vel(XDIM, i) -= 50.0 * v0;
            particles_pos(XDIM, i + 1) += 0.025;
            particles_vel(XDIM, i + 1) -= 50.0 * v0;
        } else {
            particles_pos(XDIM, i) -= 0.025;
            particles_vel(XDIM, i) += 50.0 * v0;
            particles_pos(XDIM, i + 1) -= 0.025;
            particles_vel(XDIM, i + 1) += 50.0 * v0;
        }*/
    }
}
