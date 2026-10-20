/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/polytropic_star.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/sphere.hpp>

namespace scenarios {

namespace detail {

    constexpr double kRadiusScale = 20.0;
    constexpr double kCentralDensity = 1.0;
    constexpr double kPolytropicIndex = 1.5;
    constexpr double kMaximumRadius = 3.65375 / kRadiusScale;

    static inline double powN(double y, double n) {
        return std::pow(y, n);
    }

    static inline double fy(double, double z, double) {
        return z;
    }

    static inline double fz(double y, double z, double r, double n) {
        if (r != 0.0) {
            return -(powN(y, n) + 2.0 * z / r);
        }
        return -3.0;
    }

    static inline double fm(double theta, double, double r, double n) {
        constexpr static double kFourPi = double(4) * M_PI;
        return kFourPi * powN(theta, n) * r * r;
    }

    static double laneEmden(double r0, double dr, double n, double& menc) {
        double dy1, dz1, y, z, r, dy2, dz2, dy3, dz3, dy4, dz4, y0, z0;
        double dm1, m, dm2, dm3, dm4, m0;
        int done = 0;
        y = 1.0;
        z = 0.0;
        m = 0.0;
        r = 0.0;
        do {
            if (r + dr > r0) {
                dr = r0 - r;
                done = 1;
            }
            y0 = y;
            z0 = z;
            m0 = m;
            dy1 = fy(y, z, r) * dr;
            dz1 = fz(y, z, r, n) * dr;
            dm1 = fm(y, z, r, n) * dr;
            y += 0.5 * dy1;
            z += 0.5 * dz1;
            m += 0.5 * dm1;
            if (y <= 0.0) {
                y = 0.0;
                break;
            }
            double rdr2 = r + 0.5 * dr;
            dy2 = fy(y, z, rdr2) * dr;
            dz2 = fz(y, z, rdr2, n) * dr;
            dm2 = fm(y, z, rdr2, n) * dr;
            y = y0 + 0.5 * dy2;
            z = z0 + 0.5 * dz2;
            m = m0 + 0.5 * dm2;
            if (y <= 0.0) {
                y = 0.0;
                break;
            }
            dy3 = fy(y, z, rdr2) * dr;
            dz3 = fz(y, z, rdr2, n) * dr;
            dm3 = fm(y, z, rdr2, n) * dr;
            y = y0 + dy3;
            z = z0 + dz3;
            m = m0 + dm3;
            if (y <= 0.0) {
                y = 0.0;
                break;
            }
            double rdr = r + dr;
            dy4 = fy(y, z, rdr) * dr;
            dz4 = fz(y, z, rdr, n) * dr;
            dm4 = fm(y, z, rdr, n) * dr;
            y = y0 + (dy1 + dy4 + 2.0 * (dy3 + dy2)) / 6.0;
            z = z0 + (dz1 + dz4 + 2.0 * (dz3 + dz2)) / 6.0;
            m = m0 + (dm1 + dm4 + 2.0 * (dm3 + dm2)) / 6.0;
            if (y <= 0.0) {
                y = 0.0;
                break;
            }
            r += dr;
        } while (done == 0);
        menc = m;
        if (y < 0.0) {
            return 0.0;
        }
        return pow(y, n);
    }

    static double polytropicDensity(double radius) {
        if (radius == 0.0) {
            return kCentralDensity;
        }
        double enclosedMass;
        return kCentralDensity * laneEmden(radius, radius / 100.0, kPolytropicIndex, enclosedMass);
    }

    static void finishPolytropicStar(double) {
        if (sphParticlesSize() == 0) {
            return;
        }
        int const nsample = 1000;
        double l2 = 0.0;
        double norm = 0.0;
        for (int i = 0; i < nsample; i++) {
            ParticleInteger const k = rand() % sphParticlesSize();
            float const d0 = sphParticlesDensity(k);
            float const x = sphParticlesPosition(XDIM, k).toDouble();
            float const y = sphParticlesPosition(YDIM, k).toDouble();
            float const z = sphParticlesPosition(ZDIM, k).toDouble();
            float const r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
            float d1 = 0.0;
            if (r < kMaximumRadius) {
                d1 = polytropicDensity(r * kRadiusScale);
            }
            l2 += sqr(d1 - d0);
            norm += d1 * d0;
        }
        l2 /= norm;
        l2 = sqrt(l2);
        PRINT("L2 Error = %e\n", l2);
    }

}    // namespace detail

ScenarioPlan initializePolytropicStar() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double const r0 = detail::kRadiusScale;
    int const nValue = totalNumberOfParticles;
    auto opts = getOptions();
    PRINT("  Initializing: polytropic star and particle shells\n");
    opts.initial.gasMass = 203. / nValue / (r0 * r0 * r0) / 5.99071;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    double const rho0 = detail::kCentralDensity;
    double const npoly = detail::kPolytropicIndex;
    auto const rho = detail::polytropicDensity;
    double kValue =
        4.0 * M_PI * opts.gravity.gm / (npoly + 1.0) * powf(rho0, 1.0 - 1.0 / npoly) / sqr(r0);
    double d;
    double r = 0.0;
    int ntot = 0;
    int nr = 0;
    double const rmax = detail::kMaximumRadius;
    Vector<Array<Vector<float>, NDIM>> xValue;
    Vector<float> radius;
    do {
        d = rho(r * r0);
        double dr1 = pow(m / d, 1.0 / 3.0);
        double dr2;
        r += 0.5 * dr1;
        if (r < rmax) {
            d = rho(r * r0);
            dr2 = pow(m / d, 1.0 / 3.0);
            r += 0.5 * (dr2 - dr1);
            if (r < rmax) {
                dr1 = dr2;
                int const nValue = 4.0 * M_PI * r * r * dr2 * (d / m) + 0.5;
                Array<Vector<float>, NDIM> x;
                for (int dim = 0; dim < NDIM; dim++) {
                    x[dim].resize(nValue);
                }
                xValue.push_back(std::move(x));
                terminal::detailPrint(
                    "Star shell %i: radius=%e particles=%i thickness=%e\n", nr, r, nValue, dr1);
                ntot += nValue;
                radius.push_back(r);
                r += 0.5 * dr1;
                nr++;
            }
        }
    } while (r < rmax);
    terminal::detailPrint("Star particles: %i\n", ntot);
    solveSphereSurfaceProblem(xValue);
    for (int i = 0; i < xValue.size(); i++) {
        double const d = rho(radius[i] * r0);
        double h = pow(m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * d), 1.0 / 3.0);
        int const nValue = xValue[i][XDIM].size();
        for (int l = 0; l < nValue; l++) {
            ParticleInteger const k = sphParticlesSize();
            sphParticlesResize(k + 1);
            sphParticlesPosition(XDIM, k) = xValue[i][XDIM][l] * radius[i] + 0.5;
            sphParticlesPosition(YDIM, k) = xValue[i][YDIM][l] * radius[i] + 0.5;
            sphParticlesPosition(ZDIM, k) = xValue[i][ZDIM][l] * radius[i] + 0.5;
            sphParticlesSupportRadius(k) = h;
            double const pValue = kValue * pow(d, 1.0 + 1.0 / npoly);
            double const eValue = pValue * npoly;
            double const eint = eValue / d;
            sphParticlesThermodynamicState(k).internalEnergy = eint;
            sphParticlesVelocity(XDIM, k) = 0;
            sphParticlesVelocity(YDIM, k) = 0;
            sphParticlesVelocity(ZDIM, k) = 0;
            sphParticlesRung(k) = 0;
        }
    }
    double const tdyn = sqrt(1.0 / (opts.gravity.gm * rho0));
    terminal::detailPrint("Star dynamical time: %e (code units)\n", tdyn);

    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The polytropic star scenario produced no gas particles; "
                                 "increase initial.particlesPerDimension");
    }
    // Damp on the dynamical time before the undamped evolution stage.
    return ScenarioPlan{{{10.0 * tdyn, 25, 1.0 / tdyn, 0.0}, {100.0 * tdyn, 250, 0.0, 0.05}},
        detail::finishPolytropicStar};
}

}    // namespace scenarios
