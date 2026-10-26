/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <numbers>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/rayleigh_taylor.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace scenarios {

ScenarioPlan initializeRayleighTaylor() {
    auto opts = getOptions();
    double const requestedSide = opts.initial.particlesPerDimension;
    // Resolve each quarter of the box with whole cells. The heavy layers have
    // two equal-mass particles per cell; the light layer has one.
    int const quarterDim = std::cbrt(requestedSide * requestedSide * requestedSide / 96.0);
    if (quarterDim < 1) {
        throw std::runtime_error("The Rayleigh-Taylor scenario needs a resolved lattice; "
                                 "increase initial.particlesPerDimension");
    }
    int const dim = 2 * quarterDim;
    ParticleInteger const lightCount = ParticleInteger(4) * dim * dim * dim;
    ParticleInteger const particleCount = 3 * lightCount;
    constexpr double kLightDensity = 1.0;
    constexpr double kHeavyDensity = 2.0;
    constexpr double kBoundaryPressure = 1.5;
    constexpr double kVelocityAmplitude = 0.1;
    double const acceleration = opts.hydro.zPlaneAcceleration;
    opts.initial.gasMass = 0.5 * kLightDensity / lightCount;
    setOptions(opts);
    double const mass = opts.initial.gasMass;
    double const dx = 0.5 / dim;
    sphParticlesResize(particleCount);
    ParticleInteger index = 0;
    for (int ix = 0; ix < 2 * dim; ix++) {
        for (int iy = 0; iy < 2 * dim; iy++) {
            for (int iz = 0; iz < dim; iz++) {
                bool const heavy = iz < quarterDim;
                double const density = heavy ? kHeavyDensity : kLightDensity;
                int const samples = heavy ? 2 : 1;
                double const supportRadius = std::cbrt(
                    mass * opts.hydro.neighborNumber / (4.0 * std::numbers::pi / 3.0 * density));
                for (int sample = 0; sample < samples; sample++) {
                    double const offset = heavy ? 0.25 + 0.5 * sample : 0.5;
                    Position const x((ix + offset) * dx);
                    Position const y((iy + offset) * dx);
                    Position const z((iz + offset) * dx);
                    double const height = z.toDouble();
                    double const distanceFromMiddle = 0.5 - height;
                    // Integrate dP/dz = rho * a_z from the outer boundary.
                    // This balances the imposed field; self-gravity is an independent option.
                    double const pressure = kBoundaryPressure +
                        acceleration *
                            (kHeavyDensity * (0.5 - std::fmax(distanceFromMiddle, 0.25)) +
                                kLightDensity * std::fmax(0.25 - distanceFromMiddle, 0.0));
                    double const internalEnergy = pressure / ((opts.hydro.gamma - 1.0) * density);
                    double const transverse = x.toDouble() - 0.5;
                    double const velocity = kVelocityAmplitude / 4.0 *
                        (1.0 +
                            std::cos(8.0 * std::numbers::pi * transverse) *
                                (1.0 + std::cos(6.0 * std::numbers::pi * transverse))) *
                        std::sin(2.0 * std::numbers::pi * height);
                    // Mirror in fixed-point coordinates so paired positions sum to one
                    // exactly. Scalars are even and the normal velocity is odd.
                    for (int side = 0; side < 2; side++) {
                        sphParticlesPosition(kXDimension, index) = x;
                        sphParticlesPosition(kYDimension, index) = y;
                        sphParticlesPosition(kZDimension, index) =
                            side == 0 ? z : Position(0.0) - z;
                        sphParticlesVelocity(kXDimension, index) = 0.0f;
                        sphParticlesVelocity(kYDimension, index) = 0.0f;
                        sphParticlesVelocity(kZDimension, index) = side == 0 ? velocity : -velocity;
                        sphParticlesSupportRadius(index) = supportRadius;
                        sphParticlesThermodynamicState(index).internalEnergy = internalEnergy;
                        sphParticlesRung(index) = 0;
                        index++;
                    }
                }
            }
        }
    }
    return ScenarioPlan{{{10.0, 256}}};
}

}    // namespace scenarios
