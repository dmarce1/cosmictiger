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
#include <cosmictiger/sph/monaghan_sph.hpp>

namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void requireClose(char const* label, double actual, double expected, double tolerance) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

template <typename Real>
void testKernel() {
    double const roundoff = 32.0 * std::numeric_limits<Real>::epsilon();
    requireClose(
        "central kernel normalization", monaghan_sph::kernelValue(Real(0)), 8.0 / kPi, roundoff);
    requireClose(
        "kernel at the spline join", monaghan_sph::kernelValue(Real(0.5)), 2.0 / kPi, roundoff);
    requireClose("gradient at the spline join", monaghan_sph::kernelDerivative(Real(0.5)),
        -12.0 / kPi, roundoff);
    require(monaghan_sph::kernelDerivative(Real(0)) == Real(0), "finite central gradient");
    for (Real const q : {Real(1), Real(1.25), Real(2)}) {
        require(monaghan_sph::kernelValue(q) == Real(0), "zero kernel outside support");
        require(monaghan_sph::kernelDerivative(q) == Real(0), "zero gradient outside support");
    }

    // A radial volume integral checks the actual three-dimensional normalization.
    constexpr int kIntervals = 4096;
    double integral = 0.0;
    for (int i = 0; i <= kIntervals; i++) {
        double const q = double(i) / kIntervals;
        double const weight = (i == 0 || i == kIntervals) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
        double const value = monaghan_sph::kernelValue(Real(q));
        require(value >= 0.0 && std::isfinite(value), "nonnegative finite kernel");
        integral += weight * 4.0 * kPi * q * q * value;
    }
    integral /= 3.0 * kIntervals;
    requireClose(
        "unit three-dimensional kernel integral", integral, 1.0, std::max(roundoff, 2.0e-12));

    Real const step = sizeof(Real) == sizeof(float) ? Real(5.0e-4) : Real(1.0e-6);
    double const derivativeTolerance = sizeof(Real) == sizeof(float) ? 8.0e-4 : 2.0e-9;
    for (Real const q :
        {Real(0.07), Real(0.25), Real(0.49), Real(0.5), Real(0.51), Real(0.79), Real(0.99)}) {
        Real const finiteDifference =
            (monaghan_sph::kernelValue(q + step) - monaghan_sph::kernelValue(q - step)) /
            (Real(2) * step);
        requireClose("kernel gradient finite difference", monaghan_sph::kernelDerivative(q),
            finiteDifference, derivativeTolerance);
    }
    Real const joinStep = sizeof(Real) == sizeof(float) ? Real(1.0e-5) : Real(1.0e-7);
    for (Real const join : {Real(0.5), Real(1)}) {
        requireClose("continuous kernel", monaghan_sph::kernelValue(join - joinStep),
            monaghan_sph::kernelValue(join + joinStep), 20.0 * joinStep);
        requireClose("continuous kernel gradient", monaghan_sph::kernelDerivative(join - joinStep),
            monaghan_sph::kernelDerivative(join + joinStep), 80.0 * joinStep);
    }

    // Compare against the paper's h-based polynomial, where support H is 2h.
    double const smoothingLength = 0.37;
    double const support = 2.0 * smoothingLength;
    for (double const qPaper : {0.0, 0.2, 0.75, 1.0, 1.25, 1.75, 2.0}) {
        double const radius = smoothingLength * qPaper;
        double const shape = qPaper <= 1.0 ?
            1.0 - 1.5 * qPaper * qPaper + 0.75 * qPaper * qPaper * qPaper :
            0.25 * std::pow(2.0 - qPaper, 3);
        double const expected = shape / (kPi * std::pow(smoothingLength, 3));
        double const actual =
            monaghan_sph::kernelValue(Real(radius / support)) / std::pow(support, 3);
        requireClose("support radius equals twice paper h", actual, expected, roundoff);
    }
}

template <typename Real>
void testViscosity() {
    double const tolerance = 64.0 * std::numeric_limits<Real>::epsilon();
    Real const radiusSquared = Real(0.36);
    Real const velocityDotRadius = Real(-0.24);
    Real const supportI = Real(0.6);
    Real const supportJ = Real(1.0);
    Real const epsilon = Real(0.01);
    Real const mu =
        monaghan_sph::viscosityMu(radiusSquared, velocityDotRadius, supportI, supportJ, epsilon);
    double const expectedMu = -0.096 / 0.3616;
    requireClose("paper smoothing scale in viscosity mu", mu, expectedMu, tolerance);
    Real const supportRadiusPerEffectiveRadius = Real(1.8257418583505538);
    Real const meanEffectiveRadius =
        Real(0.5) * (supportI + supportJ) / supportRadiusPerEffectiveRadius;
    Real const effectiveMu = monaghan_sph::viscosityMu(radiusSquared, velocityDotRadius, supportI,
        supportJ, epsilon, supportRadiusPerEffectiveRadius);
    requireClose("effective smoothing width in viscosity mu", effectiveMu,
        meanEffectiveRadius * velocityDotRadius /
            (radiusSquared + epsilon * meanEffectiveRadius * meanEffectiveRadius),
        tolerance);
    requireClose("calibrated cubic viscosity preserves the paper scale",
        monaghan_sph::viscosityMu(radiusSquared, velocityDotRadius, supportI, supportJ, epsilon,
            supportRadiusPerEffectiveRadius, Real(0.9128709291752769)),
        expectedMu, tolerance);
    requireClose("symmetric viscosity mu", mu,
        monaghan_sph::viscosityMu(radiusSquared, velocityDotRadius, supportJ, supportI, epsilon),
        tolerance);
    requireClose("viscosity independent of length unit", mu,
        monaghan_sph::viscosityMu(Real(9) * radiusSquared, Real(3) * velocityDotRadius,
            Real(3) * supportI, Real(3) * supportJ, epsilon),
        tolerance);
    for (Real const dot : {Real(0), Real(0.24)}) {
        require(
            monaghan_sph::viscosityMu(radiusSquared, dot, supportI, supportJ, epsilon) == Real(0),
            "no viscosity for static or separating particles");
    }
    Real const rhoI = Real(2.0);
    Real const rhoJ = Real(0.5);
    Real const soundI = Real(1.2);
    Real const soundJ = Real(0.8);
    Real const viscosity =
        monaghan_sph::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(1), Real(2));
    double const expectedViscosity = (-expectedMu + 2.0 * expectedMu * expectedMu) / 1.25;
    requireClose("Monaghan alpha beta viscosity", viscosity, expectedViscosity, tolerance);
    require(viscosity > Real(0), "positive compressive viscosity");
    requireClose("symmetric pair viscosity", viscosity,
        monaghan_sph::artificialViscosity(rhoJ, rhoI, soundJ, soundI, mu, Real(1), Real(2)),
        tolerance);
    requireClose("viscosity scales as velocity squared", Real(4) * viscosity,
        monaghan_sph::artificialViscosity(
            rhoI, rhoJ, Real(2) * soundI, Real(2) * soundJ, Real(2) * mu, Real(1), Real(2)),
        tolerance);
    require(monaghan_sph::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(0), Real(0)) ==
            Real(0),
        "viscosity can be disabled");
    Real const linear =
        monaghan_sph::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(1), Real(0));
    Real const quadratic =
        monaghan_sph::artificialViscosity(rhoI, rhoJ, soundI, soundJ, mu, Real(0), Real(2));
    requireClose(
        "independent linear and quadratic viscosity", linear + quadratic, viscosity, tolerance);
    for (Real const inactiveMu : {Real(0), Real(0.3)}) {
        require(monaghan_sph::artificialViscosity(
                    rhoI, rhoJ, soundI, soundJ, inactiveMu, Real(1), Real(2)) == Real(0),
            "no artificial expansion heating");
    }
    Real const closeMu =
        monaghan_sph::viscosityMu(Real(1.0e-12), Real(-1.0e-6), supportI, supportJ, epsilon);
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
        (monaghan_sph::kernelDerivative(radius / sink.support) / std::pow(sink.support, 4) +
            monaghan_sph::kernelDerivative(radius / source.support) / std::pow(source.support, 4));
    Vector const gradient = scaled(displacement, gradientScale);
    double const mu = monaghan_sph::viscosityMu(
        radiusSquared, dot(relativeVelocity, displacement), sink.support, source.support, 0.01);
    double const viscosity = monaghan_sph::artificialViscosity(
        sink.rho, source.rho, sink.sound, source.sound, mu, alpha, beta);
    double const pressure =
        monaghan_sph::pressureCoefficient(sink.rho, source.rho, sink.pressure, source.pressure);
    result.acceleration = scaled(gradient, -source.mass * (pressure + viscosity));
    double const work = dot(relativeVelocity, gradient);
    result.energy = source.mass * monaghan_sph::pairEnergyRate(pressure, viscosity, work, work);
    result.viscousHeating = source.mass * monaghan_sph::pairEnergyRate(0.0, viscosity, work, work);
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
    double const expandingMu = monaghan_sph::viscosityMu(0.25, 0.2, 1.0, 1.2, 0.01);
    double const expandingViscosity =
        monaghan_sph::artificialViscosity(1.0, 2.0, 1.2, 0.8, expandingMu, 1.0, 2.0);
    require(monaghan_sph::pairEnergyRate(3.0, expandingViscosity, 0.0, -0.4) == 0.0,
        "homogeneous expansion leaves pair thermal source zero");

    // A contracting background can dissipate even with zero peculiar pressure work.
    double const contractingMu = monaghan_sph::viscosityMu(0.25, -0.2, 1.0, 1.2, 0.01);
    double const contractingViscosity =
        monaghan_sph::artificialViscosity(1.0, 2.0, 1.2, 0.8, contractingMu, 1.0, 2.0);
    double const heat = monaghan_sph::pairEnergyRate(3.0, contractingViscosity, 0.0, 0.4);
    require(heat > 0.0, "physical contraction gives positive viscosity heating");
    requireClose("pressure excludes homogeneous expansion term", heat,
        monaghan_sph::pairEnergyRate(30.0, contractingViscosity, 0.0, 0.4), 1.0e-14);
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

TEST(MonaghanSph, KernelDouble) {
    detail::testKernel<double>();
}

TEST(MonaghanSph, KernelFloat) {
    detail::testKernel<float>();
}

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
