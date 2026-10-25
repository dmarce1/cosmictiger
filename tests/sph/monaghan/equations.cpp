/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/kernels.hpp>
#include <cosmictiger/sph/monaghan/equations.hpp>
#include <cosmictiger/sph/monaghan/viscosity.hpp>

#include <tests/sph/test_helpers.hpp>

namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename Real>
void testViscosity() {
    double const tolerance = 64.0 * std::numeric_limits<Real>::epsilon();
    Real const radiusSquared = Real(0.36);
    Real const velocityDotRadius = Real(-0.24);
    Real const supportI = Real(0.6);
    Real const supportJ = Real(1.0);
    Real const epsilon = Real(0.01);
    Real const mu =
        monaghan::viscosityMu(radiusSquared, velocityDotRadius, supportI, supportJ, epsilon);
    double const expectedMu = -0.096 / 0.3616;
    requireClose("paper smoothing scale in viscosity mu", mu, expectedMu, tolerance);
    Real const supportRadiusPerEffectiveRadius = Real(1.8257418583505538);
    Real const meanEffectiveRadius =
        Real(0.5) * (supportI + supportJ) / supportRadiusPerEffectiveRadius;
    Real const effectiveMu = monaghan::viscosityMu(radiusSquared, velocityDotRadius, supportI,
        supportJ, epsilon, supportRadiusPerEffectiveRadius);
    requireClose("effective smoothing width in viscosity mu", effectiveMu,
        meanEffectiveRadius * velocityDotRadius /
            (radiusSquared + epsilon * meanEffectiveRadius * meanEffectiveRadius),
        tolerance);
    requireClose("calibrated cubic viscosity preserves the paper scale",
        monaghan::viscosityMu(radiusSquared, velocityDotRadius, supportI, supportJ, epsilon,
            supportRadiusPerEffectiveRadius, Real(0.9128709291752769)),
        expectedMu, tolerance);
    requireClose("symmetric viscosity mu", mu,
        monaghan::viscosityMu(radiusSquared, velocityDotRadius, supportJ, supportI, epsilon),
        tolerance);
    requireClose("viscosity independent of length unit", mu,
        monaghan::viscosityMu(Real(9) * radiusSquared, Real(3) * velocityDotRadius,
            Real(3) * supportI, Real(3) * supportJ, epsilon),
        tolerance);
    for (Real const dot : {Real(0), Real(0.24)}) {
        require(monaghan::viscosityMu(radiusSquared, dot, supportI, supportJ, epsilon) == Real(0),
            "no viscosity for static or separating particles");
    }
    Real const rhoI = Real(2.0);
    Real const rhoJ = Real(0.5);
    Real const soundI = Real(1.2);
    Real const soundJ = Real(0.8);
    Real const viscosity =
        monaghan::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(1), Real(2));
    double const expectedViscosity = (-expectedMu + 2.0 * expectedMu * expectedMu) / 1.25;
    requireClose("Monaghan alpha beta viscosity", viscosity, expectedViscosity, tolerance);
    require(viscosity > Real(0), "positive compressive viscosity");
    requireClose("symmetric pair viscosity", viscosity,
        monaghan::artificialViscosity(rhoJ, rhoI, soundJ, soundI, mu, Real(1), Real(2)), tolerance);
    requireClose("viscosity scales as velocity squared", Real(4) * viscosity,
        monaghan::artificialViscosity(
            rhoI, rhoJ, Real(2) * soundI, Real(2) * soundJ, Real(2) * mu, Real(1), Real(2)),
        tolerance);
    require(
        monaghan::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(0), Real(0)) == Real(0),
        "viscosity can be disabled");
    Real const linear =
        monaghan::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(1), Real(0));
    Real const quadratic =
        monaghan::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(0), Real(2));
    requireClose(
        "independent linear and quadratic viscosity", linear + quadratic, viscosity, tolerance);
    for (Real const inactiveMu : {Real(0), Real(0.3)}) {
        require(monaghan::artificialViscosity(
                    rhoI, rhoJ, soundI, soundJ, inactiveMu, Real(1), Real(2)) == Real(0),
            "no artificial expansion heating");
    }
    Real const closeMu =
        monaghan::viscosityMu(Real(1.0e-12), Real(-1.0e-6), supportI, supportJ, epsilon);
    require(std::isfinite(closeMu), "regularized viscosity near coincidence");
    require(std::abs(closeMu) < Real(1.0e-3), "regularization limits close-pair viscosity");
}

using Vector = std::array<double, 3>;

double dot(Vector const& lhs, Vector const& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

Vector difference(Vector const& lhs, Vector const& rhs) {
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

Vector scaled(Vector const& value, double factor) {
    return {factor * value[0], factor * value[1], factor * value[2]};
}

Vector cross(Vector const& lhs, Vector const& rhs) {
    return {lhs[1] * rhs[2] - lhs[2] * rhs[1], lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

struct Particle {
    Vector position;
    Vector velocity;
    double mass;
    double rho;
    double pressure;
    double support;
    double sound;
};

struct Rates {
    Vector acceleration{};
    double energy = 0.0;
    double viscousHeating = 0.0;
};

Rates pairRates(Particle const& sink, Particle const& source, double alpha, double beta) {
    Vector const displacement = difference(sink.position, source.position);
    Vector const relativeVelocity = difference(sink.velocity, source.velocity);
    double const radiusSquared = dot(displacement, displacement);
    Rates result;
    if (radiusSquared == 0.0) {
        return result;
    }
    double const radius = std::sqrt(radiusSquared);
    double const gradientScale = 0.5 / radius *
        (sph_kernels::kernelDerivative(radius / sink.support) / std::pow(sink.support, 4) +
            sph_kernels::kernelDerivative(radius / source.support) / std::pow(source.support, 4));
    Vector const gradient = scaled(displacement, gradientScale);
    double const mu = monaghan::viscosityMu(
        radiusSquared, dot(relativeVelocity, displacement), sink.support, source.support, 0.01);
    double const viscosity = monaghan::artificialViscosity(
        sink.rho, source.rho, sink.sound, source.sound, mu, alpha, beta);
    double const pressure =
        monaghan::pressureCoefficient(sink.rho, source.rho, sink.pressure, source.pressure);
    result.acceleration = scaled(gradient, -source.mass * (pressure + viscosity));
    double const work = dot(relativeVelocity, gradient);
    result.energy = source.mass * monaghan::pairEnergyRate(pressure, viscosity, work, work);
    result.viscousHeating = source.mass * monaghan::pairEnergyRate(0.0, viscosity, work, work);
    return result;
}

void checkConservation(Particle const& first, Particle const& second, bool compressing) {
    Rates const rateI = pairRates(first, second, 1.0, 2.0);
    Rates const rateJ = pairRates(second, first, 1.0, 2.0);
    Vector const forceI = scaled(rateI.acceleration, first.mass);
    Vector const forceJ = scaled(rateJ.acceleration, second.mass);
    Vector const torqueI = cross(first.position, forceI);
    Vector const torqueJ = cross(second.position, forceJ);
    for (size_t dim = 0; dim < forceI.size(); dim++) {
        requireClose("pair momentum conservation", forceI[dim] + forceJ[dim], 0.0, 2.0e-13);
        requireClose(
            "pair angular momentum conservation", torqueI[dim] + torqueJ[dim], 0.0, 2.0e-13);
    }
    double const kineticRate = dot(first.velocity, forceI) + dot(second.velocity, forceJ);
    double const thermalRate = first.mass * rateI.energy + second.mass * rateJ.energy;
    requireClose("pair total energy conservation", kineticRate + thermalRate, 0.0, 2.0e-13);
    if (compressing) {
        require(rateI.viscousHeating > 0.0 && rateJ.viscousHeating > 0.0,
            "compression dissipates into thermal energy for both masses");
        Rates const inviscid = pairRates(first, second, 0.0, 0.0);
        requireClose("viscous thermal increment", rateI.energy - inviscid.energy,
            rateI.viscousHeating, 2.0e-13);
    } else {
        require(rateI.viscousHeating == 0.0 && rateJ.viscousHeating == 0.0,
            "expansion does not produce viscous heating");
    }
}

void testPairConservation() {
    Particle first{{0.3, 0.1, -0.2}, {-0.5, 0.2, 0.0}, 0.7, 2.1, 1.7, 0.65, 1.3};
    Particle second{{-0.4, -0.1, -0.1}, {0.4, -0.1, 0.1}, 1.3, 0.8, 0.4, 1.4, 0.7};
    // This pair is outside one support but inside the other. Both directions must interact.
    double const radius = std::sqrt(dot(
        difference(first.position, second.position), difference(first.position, second.position)));
    require(radius > first.support && radius < second.support, "unequal-support test geometry");
    checkConservation(first, second, true);
    Rates const original = pairRates(first, second, 1.0, 2.0);
    Vector const boost{2.0, -3.0, 0.4};
    for (size_t dim = 0; dim < boost.size(); dim++) {
        first.velocity[dim] += boost[dim];
        second.velocity[dim] += boost[dim];
    }
    Rates const boosted = pairRates(first, second, 1.0, 2.0);
    for (size_t dim = 0; dim < boost.size(); dim++) {
        requireClose("Galilean-invariant acceleration", boosted.acceleration[dim],
            original.acceleration[dim], 2.0e-13);
    }
    requireClose("Galilean-invariant heating", boosted.energy, original.energy, 2.0e-13);
    checkConservation(first, second, true);

    first.velocity = {0.5, -0.2, 0.0};
    second.velocity = {-0.4, 0.1, -0.1};
    checkConservation(first, second, false);
    first.support = 1.1;
    checkConservation(first, second, false);

    first.velocity = second.velocity = {0.2, -0.3, 0.4};
    Rates const staticRate = pairRates(first, second, 1.0, 2.0);
    require(staticRate.energy == 0.0 && staticRate.viscousHeating == 0.0,
        "uniform translation produces no thermal work");
    checkConservation(first, second, false);
    first.position = second.position;
    Rates const coincident = pairRates(first, second, 1.0, 2.0);
    require(
        coincident.energy == 0.0 && dot(coincident.acceleration, coincident.acceleration) == 0.0,
        "coincident pair has zero finite gradient");
}

void testPhysicalVelocitySplit() {
    // Homogeneous expansion has no peculiar pressure work and no artificial viscosity.
    double const expandingMu = monaghan::viscosityMu(0.25, 0.2, 1.0, 1.2, 0.01);
    double const expandingViscosity =
        monaghan::artificialViscosity(1.0, 2.0, 1.2, 0.8, expandingMu, 1.0, 2.0);
    require(monaghan::pairEnergyRate(3.0, expandingViscosity, 0.0, -0.4) == 0.0,
        "homogeneous expansion leaves pair thermal source zero");

    // A contracting background can dissipate even with zero peculiar pressure work.
    double const contractingMu = monaghan::viscosityMu(0.25, -0.2, 1.0, 1.2, 0.01);
    double const contractingViscosity =
        monaghan::artificialViscosity(1.0, 2.0, 1.2, 0.8, contractingMu, 1.0, 2.0);
    double const heat = monaghan::pairEnergyRate(3.0, contractingViscosity, 0.0, 0.4);
    require(heat > 0.0, "physical contraction gives positive viscosity heating");
    requireClose("pressure excludes homogeneous expansion term", heat,
        monaghan::pairEnergyRate(30.0, contractingViscosity, 0.0, 0.4), 1.0e-14);
}

void testVector3() {
    Vector3<double> const u(1.0, -2.0, 3.0);
    Vector3<double> const v(-0.5, 4.0, 2.0);
    requireClose("dot product", u.dot(v), -2.5, 1.0e-15);
    requireClose("squared norm", u.squaredNorm(), 14.0, 1.0e-15);
    requireClose("norm", u.norm(), std::sqrt(14.0), 1.0e-15);
    Vector3<double> w = u + v;
    w -= v;
    w = 2.0 * w;
    w = w * 0.5;
    w = w / 1.0;
    w = -(-w);
    w += Vector3<double>(0.0);
    Vector3<double> const difference = u - v;
    double const expectedDifference[3] = {1.5, -6.0, 1.0};
    Vector3<float> const converted(u);
    Vector3<double> const first = simdSelect(true, u, v);
    Vector3<double> const second = simdSelect(false, u, v);
    for (int dim = 0; dim < 3; dim++) {
        requireClose("vector arithmetic", w[dim], u[dim], 1.0e-15);
        requireClose("vector difference", difference[dim], expectedDifference[dim], 1.0e-15);
        requireClose("vector conversion", converted[dim], u[dim], 1.0e-7);
        require(first[dim] == u[dim] && second[dim] == v[dim], "vector selection");
        require(Vector3<double>(1.5)[dim] == 1.5, "broadcast construction");
    }
    require(Vector3<double>::size() == 3, "three components");
}

}    // namespace detail

TEST(MonaghanSph, ViscosityDouble) {
    detail::testViscosity<double>();
}

TEST(MonaghanSph, ViscosityFloat) {
    detail::testViscosity<float>();
}

TEST(MonaghanSph, PairConservation) {
    detail::testPairConservation();
}

TEST(MonaghanSph, PhysicalVelocitySplit) {
    detail::testPhysicalVelocitySplit();
}

TEST(MonaghanSph, Vector3) {
    detail::testVector3();
}
