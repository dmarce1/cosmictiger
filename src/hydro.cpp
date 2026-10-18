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

#include <cosmictiger/domain.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/driver.hpp>
#include <cosmictiger/exact_sod.hpp>
#include <cosmictiger/sedov.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/sphere.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/view.hpp>

static inline double powN(double y, double n) {
    return std::pow(y, n);
}

static inline double fy(double y, double z, double r) {
    return z;
}

static inline double fz(double y, double z, double r, double n) {
    if (r != 0.0) {
        return -(powN(y, n) + 2.0 * z / r);
    }
    return -3.0;
}

static inline double fm(double theta, double dummy, double r, double n) {
    constexpr static double kFourPi = double(4) * M_PI;
    return kFourPi * powN(theta, n) * r * r;
}

double laneEmden(double r0, double dr, double n, double& menc) {
    double dy1, dz1, y, z, r, dy2, dz2, dy3, dz3, dy4, dz4, y0, z0;
    double dm1, m, dm2, dm3, dm4, m0;
    int done = 0;
    y = 1.0;
    z = 0.0;
    m = 0.0;
    int nValue = static_cast<int>(r0 / dr + 0.5);
    if (nValue < 1) {
        nValue = 1;
    }
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

void hydroDriver(double tmax, int numberOfSteps = 64) {
    TimeType itime = 0;
    int minrung = 0;
    double t = 0.0;
    double t0 = tmax / numberOfSteps;
    int step = 0;
    int mainStep = 0;
    double etot0 = 0.0;
    double const m = getOptions().initial.gasMass;
    domainsRebound();
    do {
        int minrung = minRung(itime);
        EnergiesT dummy;
        if (minrung == 0) {
            viewOutputViews(mainStep, 1.0);
            //			output_line(main_step);
            mainStep++;
        }
        KickReturn kr{};
        if (getOptions().gravity.enabled) {
            auto tmp = kickStep(minrung, 1.0, 0.0, t0, 0.5, t == 0.0, minrung == 0);
            kr = tmp.first;
        }
        sphStep1(minrung, true);
        SphRunReturn rc2 = sphStep2(minrung, 1.0, t, t0, 1, 0.0, &dummy, true);
        if (minrung == 0) {
            double const ekin = dummy.kin + kr.darkMatterKineticCorrection;
            double const eint = dummy.therm;
            double rhoMax = 0.0;
            double const epot = kr.pot;
            double const xmom = dummy.xmom + kr.darkMatterXMomentumCorrection;
            double const ymom = dummy.ymom + kr.darkMatterYMomentumCorrection;
            double const zmom = dummy.zmom + kr.darkMatterZMomentumCorrection;
            for (ParticleInteger i = 0; i < sphParticlesSize(); i++) {
                rhoMax = std::max(rhoMax, double(sphParticlesDensity(i)));
            }
            FILE* fp = fopen("energy.dat", "at");
            double const etot = ekin + eint + epot;
            if (t == 0.0) {
                etot0 = etot;
            }
            fprintf(fp, "%e %e %e %e %e %e %e %e %e\n", t, xmom, ymom, zmom, ekin, eint, etot, epot,
                (etot - etot0) / etot);
            fclose(fp);
            fp = fopen("rho.dat", "at");
            fprintf(fp, "%e %e\n", t, rhoMax);
            fclose(fp);
        }
        if (!getOptions().gravity.enabled) {
            kr.maxRung = rc2.maximumRung;
        }
        int maxrung = std::max((int) rc2.maximumRung, (int) kr.maxRung);
        double dt = t0 / (1 << maxrung);
        drift(1.0, dt, t, t + dt, tmax);
        itime = inc(itime, maxrung);
        t += dt;
        PRINT("%i %e %e %i %i\n", step, t, dt, minrung, maxrung);
        step++;
    } while (t < tmax);
    viewOutputViews(mainStep, 1.0, true);
}

// Problem initializer: run.test=plummer exercises gas and dark matter in a bound sphere.
void hydroPlummer() {
    ParticleInteger nValue = pow(getOptions().initial.particlesPerDimension, 3);
    double const a = 0.01;
    auto opts = getOptions();
    opts.initial.gasMass = 1.;
    opts.initial.darkMatterMass = 1.;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    double const g = opts.gravity.gm;
    double const m0Value = nValue * m;
    double const rho0 = 3.0 * m0Value / (4.0 * M_PI) / (a * a * a);
    PRINT("Central density = %e\n", rho0);
    double pot = 0.0;
    double ekin = 0.0;
    double maxr = 5.0 * a;
    while (particlesSize() < nValue) {
        double x = maxr * (2.0 * rand1() - 1.0);
        double y = maxr * (2.0 * rand1() - 1.0);
        double z = maxr * (2.0 * rand1() - 1.0);
        double r = sqrt(sqr(x, y, z));
        double dist = powf(1.0 + sqr(r / a), -2.5);
        if (rand1() < dist) {
            double h = pow(m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0 * dist),
                1.0 / 3.0);
            ParticleInteger k = particlesSize();
            particlesResize(k + 1);
            double phi = -g * m0Value / sqrt(r * r + a * a);
            double v = sqrt(-phi / 2.0);
            particlesPos(XDIM, k) = x - 0.5;
            particlesPos(YDIM, k) = y - 0.5;
            particlesPos(ZDIM, k) = z - 0.5;
            x = rand1();
            y = rand1();
            v *= sqrt(-2.0 * log(x)) * cos(2.0 * M_PI * y);
            x = rand1() - 0.5;
            y = rand1() - 0.5;
            z = rand1() - 0.5;
            double norminv = 1.0 / sqrt(sqr(x, y, z));
            x *= norminv;
            y *= norminv;
            z *= norminv;
            particlesVel(XDIM, k) = x * v;
            particlesVel(YDIM, k) = y * v;
            particlesVel(ZDIM, k) = z * v;
            particlesRung(k) = 0;
        }
    }
    sphParticlesResize(0);
    int k = 0;
    while (k < 0) {
        double x = maxr * (2.0 * rand1() - 1.0);
        double y = maxr * (2.0 * rand1() - 1.0);
        double z = maxr * (2.0 * rand1() - 1.0);
        double r = sqrt(sqr(x, y, z));
        double dist = powf(1.0 + sqr(r / a), -2.5);
        if (rand1() < dist) {
            double h = pow(m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0 * dist),
                1.0 / 3.0);
            //			part_int k = sph_particles_size();
            //			sph_particles_resize(k + 1);
            double phi = -g * m0Value / sqrt(r * r + a * a);
            double v = sqrt(-phi / 2.0);
            sphParticlesPosition(XDIM, k) = x - 0.5;
            sphParticlesPosition(YDIM, k) = y - 0.5;
            sphParticlesPosition(ZDIM, k) = z - 0.5;
            x = rand1();
            y = rand1();
            v *= sqrt(-2.0 * log(x)) * cos(2.0 * M_PI * y);
            x = rand1() - 0.5;
            y = rand1() - 0.5;
            z = rand1() - 0.5;
            double norminv = 1.0 / sqrt(sqr(x, y, z));
            x *= norminv;
            y *= norminv;
            z *= norminv;
            sphParticlesVelocity(XDIM, k) = x * v;
            sphParticlesVelocity(YDIM, k) = y * v;
            sphParticlesVelocity(ZDIM, k) = z * v;
            float const eint = 1e-3 * (v * v * 0.5);
            sphParticlesThermodynamicState(k).internalEnergy = eint;
            sphParticlesRung(k) = 0;
            sphParticlesSupportRadius(k) = h;
            ekin += 0.5 * m * sqr(v);
            pot += 0.5 * m * phi;
            k++;
        }
    }
    PRINT("Virial error is %e\n", (2.0 * ekin + pot) / (2.0 * ekin - pot));
    double const tdyn = sqrt(4.0 * M_PI * a * a * a / (3.0 * g * m0Value));
    hydroDriver(100.0 * tdyn, 1024);
}

// Problem initializer: run.test=star is a self-gravitating polytropic gas sphere.
// It uses only SPH particles; damping below relaxes the initial gas distribution.
void hydroStarTest() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double const r0 = 20.0;
    int const nValue = totalNumberOfParticles;
    auto opts = getOptions();
    PRINT("Making star\n");
    opts.initial.gasMass = 203. / nValue / (r0 * r0 * r0) / 5.99071;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    double rho0 = 1.0;
    double const npoly = 1.5;
    auto const rho = [rho0, npoly](double r) {
        if (r == 0.0) {
            return rho0;
        } else {
            double menc;
            return rho0 * laneEmden(r, r / 100.0, npoly, menc);
        }
    };
    double kValue =
        4.0 * M_PI * opts.gravity.gm / (npoly + 1.0) * powf(rho0, 1.0 - 1.0 / npoly) / sqr(r0);
    double d;
    double r = 0.0;
    int ntot = 0;
    int nr = 0;
    double rmax = 3.65375 / r0;
    Vector<Array<Vector<float>, NDIM>> xValue;
    Vector<float> radius;
    do {
        d = rho(r * r0);
        double dr1 = pow(m / d, 1.0 / 3.0);
        double dr2;
        r += 0.5 * dr1;
        if (r < rmax) {
            double diff;
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
                printf("%i %e %i %e\n", nr, r, nValue, dr1);
                ntot += nValue;
                radius.push_back(r);
                r += 0.5 * dr1;
                nr++;
            }
        }
    } while (r < rmax);
    printf("Ntot = %i\n", ntot);
#ifdef USE_CUDA
    if (!cudaEnabled()) {
        THROW_ERROR("Stellar surface initialization requires compute.cuda=true\n");
    }
    solveSphereSurfaceProblem(xValue);
#else
    THROW_ERROR("Stellar surface initialization requires CUDA support\n");
#endif
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
    PRINT("************************************\n");
    PRINT("tdyn = %e\n", 1.0 / sqrt(opts.gravity.gm * rho0));
    PRINT("************************************\n");
    opts = getOptions();
    opts.hydro.damping = 1.0;
    opts.hydro.viscosityAlpha = 0.0;
    setOptions(opts);
    hydroDriver(10.0 * tdyn, 25);
    opts.hydro.damping = 0.0;
    opts.hydro.viscosityAlpha = 0.05;
    setOptions(opts);
    hydroDriver(100.0 * tdyn, 250);
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
        if (r < rmax) {
            d1 = rho(r * r0);
        }
        l2 += sqr(d1 - d0);
        norm += d1 * d0;
    }
    l2 /= norm;
    l2 = sqrt(l2);
    PRINT("L2 Error = %e\n", l2);
}

// Problem initializer: run.test=rt uses uniform y acceleration for Rayleigh-Taylor instability.
void hydroRtTest() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    ParticleInteger hiParts = 2 * totalNumberOfParticles / 3;
    ParticleInteger loParts = totalNumberOfParticles / 3;
    ParticleInteger dim = pow(loParts * 0.25, (1. / 3.));
    double rho1 = 2.0;
    double rho0 = 1.0;
    loParts = 4 * dim * dim * dim;
    hiParts = 2 * loParts;
    double const yAcceleration = getOptions().hydro.yAcceleration;
    auto opts = getOptions();
    opts.initial.gasMass = rho0 / loParts * 0.5;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    rho1 = hiParts * m / 0.5;
    rho0 = loParts * m / 0.5;

    if (yAcceleration >= 0.0) {
        //	PRINT("Error gy must be negative for RT test\n");
        //	abort();
    }
    constexpr double kEta = 0.01;
    double dx = .5 / dim;
    int i = 0;
    double v0 = 0.1;
    for (int ix = 0; ix < 2 * dim; ix++) {
        for (int iy = dim / 2; iy <= 3 * dim / 2; iy++) {
            for (int iz = 0; iz < 2 * dim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double vy = v0 / 4.0 *
                    (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
                double p = 1.5 + fabs(rho0 * yAcceleration * (y - 0.5));
                double eint = p / (getOptions().hydro.gamma - 1) / rho0;
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = 0;
                sphParticlesVelocity(YDIM, i) = vy;
                ;
                sphParticlesVelocity(ZDIM, i) = 0;
                ;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                ;
                i++;
            }
        }
    }
    for (int ix = 0; ix < 2 * dim; ix++) {
        for (int iy = 0; iy < 2 * dim; iy++) {
            for (int iz = 0; iz < 2 * dim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double p;
                double vy = v0 / 4.0 *
                    (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
                if (y > 0.75 || y < 0.25) {
                    if (y >= 0.75) {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.75)) +
                            fabs(rho0 * yAcceleration * .25);
                    } else {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.25)) +
                            fabs(rho0 * yAcceleration * .25);
                    }
                    double eint = p / (getOptions().hydro.gamma - 1) / rho1;
                    sphParticlesResize(sphParticlesSize() + 1);
                    sphParticlesSupportRadius(i) = h;
                    sphParticlesPosition(XDIM, i) = x;
                    sphParticlesPosition(YDIM, i) = y;
                    sphParticlesPosition(ZDIM, i) = z;
                    sphParticlesVelocity(XDIM, i) = 0;
                    ;
                    sphParticlesVelocity(YDIM, i) = vy;
                    ;
                    sphParticlesVelocity(ZDIM, i) = 0;
                    ;
                    sphParticlesRung(i) = 0;
                    sphParticlesThermodynamicState(i).internalEnergy = eint;

                    ;
                    i++;
                }
                x = (ix + 0.0) * dx;
                y = (iy + 0.0) * dx;
                z = (iz + 0.0) * dx;
                vy = v0 / 4.0 *
                    (1.0 + cos(8.0 * M_PI * (x - 0.5)) * (1.0 + cos(6.0 * M_PI * (x - 0.5))));
                if (y > 0.75 || y < 0.25) {
                    if (y >= 0.75) {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.75)) +
                            fabs(rho0 * yAcceleration * .25);
                    } else {
                        p = 1.5 + fabs(rho1 * yAcceleration * (y - 0.25)) +
                            fabs(rho0 * yAcceleration * .25);
                    }
                    double eint = p / (getOptions().hydro.gamma - 1) / rho1;
                    sphParticlesResize(sphParticlesSize() + 1);
                    sphParticlesSupportRadius(i) = h;
                    sphParticlesPosition(XDIM, i) = x;
                    sphParticlesPosition(YDIM, i) = y;
                    sphParticlesPosition(ZDIM, i) = z;
                    sphParticlesVelocity(XDIM, i) = 0;
                    ;
                    sphParticlesVelocity(YDIM, i) = vy;
                    ;
                    sphParticlesVelocity(ZDIM, i) = 0;
                    ;
                    sphParticlesRung(i) = 0;
                    sphParticlesThermodynamicState(i).internalEnergy = eint;

                    i++;
                }
            }
        }
    }
    /*	for (int ix0 = 0; ix0 < 2 * dim; ix0++) {
     int ix = ix0;
     for (int iy = dim; iy < 2 * dim; iy++) {
     for (int iz = 0; iz < 2 * dim; iz++) {
     double x = (ix + 0.25) * dx;
     double y = (iy + 0.25) * dx;
     double z = (iz + 0.25) * dx;
     double p = -(1.0 - y) * rho1 * gy;
     double eint = p / rho1 / (get_options().gamma - 1);
     double h = pow(m * get_options().sneighbor_number / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
     sph_particles_resize(sph_particles_size() + 1);
     sph_particles_smooth_len (i) = h;
     sph_particles_pos(XDIM, i) = x;
     sph_particles_pos(YDIM, i) = y;
     sph_particles_pos(ZDIM, i) = z;
     sph_particles_vel(XDIM, i) = 0.0;
     sph_particles_vel(YDIM, i) = 0.0;
     sph_particles_vel(ZDIM, i) = 0.0;
     sph_particles_rung(i) = 0;
     sph_particles_eint(i) = eint;
     i++;
     x = (ix + 0.75) * dx;
     y = (iy + 0.75) * dx;
     z = (iz + 0.75) * dx;
     p = -rho1 * (1.0 - y) * gy;
     eint = p / rho1 / (get_options().gamma - 1);
     sph_particles_resize(sph_particles_size() + 1);
     sph_particles_smooth_len(i) = h;
     sph_particles_pos(XDIM, i) = x;
     sph_particles_pos(YDIM, i) = y;
     sph_particles_pos(ZDIM, i) = z;
     sph_particles_vel(XDIM, i) = 0.0;
     sph_particles_vel(YDIM, i) = 0.0;
     sph_particles_vel(ZDIM, i) = 0.0;
     sph_particles_rung(i) = 0;
     sph_particles_eint(i) = eint;
     i++;
     }
     }
     }*/
    constexpr float kT = 10.0;
    constexpr int kN = 1000;
    hydroDriver(kT, 256);
}

// Problem initializer: run.test=disc uses an imposed central potential for an orbiting gas disc.
void hydroDiscTest() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double const rinner = 0.2;
    double const router = 0.3;
    double const rho = totalNumberOfParticles;
    double const p0 = 1.0e-30;
    double const m = 1.0;
    double h = pow(m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);
    int i = 0;
    int const nz = 3;
    int dim = sqrt(totalNumberOfParticles) / nz;
    double dx = 1.0 / dim;
    for (int iz = dim / 2 - nz; iz < dim / 2 + nz; iz++) {
        for (int k = 0; k < dim; k++) {
            for (int l = 0; l < dim; l++) {
                double r, x, y, z;
                x = dx * k;
                y = dx * l;
                z = iz * dx;
                r = sqrt(sqr(x - 0.5, y - 0.5, z - 0.5));
                if (r > rinner && r < router) {
                    double eint = p0 / rho / (getOptions().hydro.gamma - 1);
                    sphParticlesResize(sphParticlesSize() + 1);
                    sphParticlesSupportRadius(i) = h;
                    sphParticlesPosition(XDIM, i) = x;
                    sphParticlesPosition(YDIM, i) = y;
                    sphParticlesPosition(ZDIM, i) = z;
                    double const v = 1.00 * sqrt(1 / r);
                    double const vx = -(y - 0.5) / r * v;
                    double const vy = (x - 0.5) / r * v;
                    double const vz = 0.0;
                    double sgn;
                    if (r > (rinner + router) / 2.0) {
                        sgn = 1.0;
                    } else {
                        sgn = 1.0;
                    }
                    sphParticlesVelocity(XDIM, i) = sgn * vx;
                    sphParticlesVelocity(YDIM, i) = sgn * vy;
                    sphParticlesVelocity(ZDIM, i) = vz;
                    sphParticlesRung(i) = 0;
                    sphParticlesThermodynamicState(i).internalEnergy = eint;

                    ;
                    i++;
                }
            }
        }
    }
    double const t = 16.0;
    hydroDriver(t, 1024);
}

// Problem initializer: run.test=sod checks the ideal-gas shock tube without self-gravity.
void hydroSodTest() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double rho0 = 0.125;
    double rho1 = 1.0;
    double vx1 = 0.0;
    double vy1 = 0.0;
    double vz1 = 0.0;
    double vx0 = 0.0;
    double vy0 = -0.0;
    double vz0 = -0.0e-1;
    double p0 = .1;
    double p1 = 1.0;
    SodInitT sod;
    ParticleInteger leftDim =
        pow(0.25 * totalNumberOfParticles * rho1 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
    ParticleInteger rightDim =
        pow(0.25 * totalNumberOfParticles * rho0 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
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
    PRINT("Sod dimensions are %i and %i rho0 %e rho1 %e\n", 2 * rightDim, 2 * leftDim,
        rho0 / (m * ne) * (4.0 * M_PI / 3.0), rho1 / (m * ne) * (4.0 * M_PI / 3.0));
    for (int ix = 0; ix < leftDim; ix++) {
        for (int iy = 0; iy < 2 * leftDim; iy++) {
            for (int iz = 0; iz < 2 * leftDim; iz++) {
                double x = (ix) *dx;
                double y = (iy) *dx;
                double z = (iz) *dx;
                double ent = p1 / powf(rho1, getOptions().hydro.gamma);
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
                //			PRINT("%i\n", i);
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
                double ent = p0 / powf(rho0, getOptions().hydro.gamma);
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
                ;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy =
                    p0 / (getOptions().hydro.gamma - 1.0) / rho0;
                ;
                //		PRINT("%i\n", i);
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
                ;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy =
                    p0 / (getOptions().hydro.gamma - 1.0) / rho0;
                ;
                i++;
            }
        }
    }
    constexpr float kT = .09;
    constexpr int kN = 1000;
    hydroDriver(kT, 20);
    FILE* fp = fopen("sod.txt", "wt");
    double l1 = 0.0, l2 = 0.0, lmax = 0.0;
    double norm1 = 0.0, norm2 = 0.0;
    for (int l = 0; l < kN; l++) {
        int const i = rand() % sphParticlesSize();
        int const j = sphParticlesParticleIndex(i);
        float x = particlesPos(XDIM, j).toDouble();
        float const h = sphParticlesSupportRadius(i);
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
        exactSod(&state, &sod, x0, kT, 0);
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

// Problem initializer: run.test=helmholtz checks Kelvin-Helmholtz shear flow.
void hydroHelmholtzTest() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double rho1 = 0.5;
    double rho0 = 1.0;
    double vx1 = 0.0;
    double vy1 = 1.0;
    double vz1 = 0.0;
    double vx0 = 0.0;
    double vy0 = -1.0;
    double vz0 = 0.0;
    double p1 = 19.2;
    double p0 = 19.2;
    constexpr double kEta = 1e-1;
    constexpr double kFreq = 2.0;
    ParticleInteger leftDim =
        pow(0.25 * totalNumberOfParticles * rho1 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
    ParticleInteger rightDim =
        pow(0.25 * totalNumberOfParticles * rho0 / (rho1 + rho0) / 2, 1.0 / 3.0) + 0.49999;
    ParticleInteger numberOfLeftParticles = 2 * leftDim * sqr(2 * leftDim);
    ParticleInteger numberOfRightParticles = 2 * rightDim * sqr(2 * rightDim);
    totalNumberOfParticles = numberOfLeftParticles + numberOfRightParticles;
    double dx = 0.5 / leftDim;
    ParticleInteger i = 0;
    auto opts = getOptions();
    opts.initial.gasMass = rho1 * 0.5 / numberOfLeftParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    rho1 = numberOfLeftParticles * m / 0.5f;
    rho0 = numberOfRightParticles * m / 0.5f;
    for (int ix = 0; ix < leftDim; ix++) {
        for (int iy = 0; iy < 2 * leftDim; iy++) {
            for (int iz = 0; iz < 2 * leftDim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double eint = p1 / rho1 / (getOptions().hydro.gamma - 1);
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho1), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy1;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
                x = (ix) *dx;
                y = (iy) *dx;
                z = (iz) *dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy1;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                i++;
                //			PRINT("%i\n", i);
            }
        }
    }
    dx = 0.5 / rightDim;
    for (int ix0 = rightDim; ix0 < 2 * rightDim; ix0++) {
        int ix = ix0;
        for (int iy = 0; iy < 2 * rightDim; iy++) {
            for (int iz = 0; iz < 2 * rightDim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double eint = p0 / rho0 / (getOptions().hydro.gamma - 1);
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho0), 1.0 / 3.0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy0;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;
                ;

                //		PRINT("%i\n", i);
                i++;
                x = (ix) *dx;
                y = (iy) *dx;
                z = (iz) *dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = kEta * sin(kFreq * 2.0 * M_PI * y);
                sphParticlesVelocity(YDIM, i) = vy0;
                sphParticlesVelocity(ZDIM, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;

                ;
                //		PRINT("%i\n", i);
                i++;
            }
        }
    }
    hydroDriver(10.00, 1024);
}

// Problem initializer: run.test=blast checks an ideal-gas blast wave.
void hydroBlastTest() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double rho0 = 1.0;
    double eblast = 1.0;
    double p0 = 1.0e-20f;
    auto opts = getOptions();
    ParticleInteger ndim = getOptions().initial.particlesPerDimension;
    ParticleInteger numberOfParticles = std::pow(ndim, NDIM);
    ndim = pow(numberOfParticles / 2, 1. / 3.);
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
                ;
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
                ;
                i++;
            }
        }
    }
    double const t = 5e-3;
    hydroDriver(t, 16);
    FILE* fp = fopen("blast.txt", "wt");
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
        double time = t;
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

void hydroWaveTest() {
    ParticleInteger totalNumberOfParticles = pow(getOptions().initial.particlesPerDimension, 3);
    double rho = 1.0;
    double p0 = 0.845154255;
    auto opts = getOptions();
    ParticleInteger const ndim = getOptions().initial.particlesPerDimension;
    ParticleInteger const numberOfParticles = std::pow(ndim, NDIM);
    opts.initial.gasMass = rho / numberOfParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    ParticleInteger i = 0;
    double const dx = 1.0 / ndim;
    for (int ix = 0; ix < ndim; ix++) {
        for (int iy = 0; iy < ndim; iy++) {
            for (int iz = 0; iz < ndim; iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double eint = p0 / rho / (getOptions().hydro.gamma - 1);
                double h = pow(
                    m * getOptions().hydro.neighborNumber / (4.0 * M_PI / 3.0 * rho), 1.0 / 3.0);
                //	PRINT( "%e\n", h);
                if (ix == ndim / 2) {
                    x += 0.99 * dx;
                } else if (ix == ndim / 2 - 1) {
                    x -= 0.99 * dx;
                }
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSupportRadius(i) = h;
                sphParticlesPosition(XDIM, i) = x;
                sphParticlesPosition(YDIM, i) = y;
                sphParticlesPosition(ZDIM, i) = z;
                sphParticlesVelocity(XDIM, i) = 0.f;
                sphParticlesVelocity(YDIM, i) = 0.f;
                sphParticlesVelocity(ZDIM, i) = 0.f;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy = eint;
                i++;
                //			PRINT("%i\n", i);
            }
        }
    }
    hydroDriver(0.5);
}
