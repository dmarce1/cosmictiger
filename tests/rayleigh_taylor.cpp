/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/scenarios/rayleigh_taylor.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

// Ordinary host storage lets this test call the production initializer without
// starting HPX, allocating managed memory, or running the evolution driver.
Array<Fixed32*, NDIM> particlesX;
Array<float, NDIM>* particlesV;
char* particlesR;
ParticleInteger* sphParticlesParticleIndices;
SphThermodynamicState* sphParticlesThermodynamicStates;

namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
Options options;
std::array<std::vector<Fixed32>, NDIM> positions;
std::vector<Array<float, NDIM>> velocities;
std::vector<char> rungs;
std::vector<ParticleInteger> indices;
std::vector<SphThermodynamicState> thermodynamicStates;

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void requireClose(char const* label, double actual, double expected, double tolerance = 2.0e-6) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

}    // namespace detail

Options const& getOptions() {
    return detail::options;
}

void setOptions(Options const& options) {
    detail::options = options;
}

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

ParticleInteger sphParticlesSize() {
    return detail::indices.size();
}

ParticleInteger particlesSize() {
    return sphParticlesSize();
}

void sphParticlesResize(ParticleInteger count, bool) {
    ParticleInteger const oldCount = sphParticlesSize();
    for (int dim = 0; dim < NDIM; dim++) {
        detail::positions[dim].resize(count);
        particlesX[dim] = detail::positions[dim].data();
    }
    detail::velocities.resize(count);
    detail::rungs.resize(count);
    detail::indices.resize(count);
    detail::thermodynamicStates.resize(count);
    particlesV = detail::velocities.data();
    particlesR = detail::rungs.data();
    sphParticlesParticleIndices = detail::indices.data();
    sphParticlesThermodynamicStates = detail::thermodynamicStates.data();
    for (ParticleInteger i = oldCount; i < count; i++) {
        sphParticlesParticleIndices[i] = i;
    }
}

namespace detail {

void testInitialization(int resolution, ParticleInteger expectedCount, double acceleration) {
    sphParticlesResize(0);
    Options configured{};
    configured.initial.particlesPerDimension = resolution;
    configured.hydro.enabled = true;
    configured.hydro.gamma = 5.0 / 3.0;
    configured.hydro.neighborNumber = 42.0;
    configured.hydro.zPlaneAcceleration = acceleration;
    setOptions(configured);
    auto const plan = scenarios::initializeRayleighTaylor();
    require(plan.stages.size() == 1, "Rayleigh-Taylor has one evolution stage");
    requireClose("stage duration", plan.stages.front().duration, 10.0);
    require(plan.stages.front().numberOfSteps == 256, "coarse step count");
    require(sphParticlesSize() == expectedCount, "resolved lattice particle count");
    double const mass = getOptions().initial.gasMass;
    ParticleInteger lightCount = 0;
    ParticleInteger heavyCount = 0;
    std::set<std::array<uint32_t, NDIM>> occupied;
    std::map<double, double> pressureByHeight;
    for (ParticleInteger i = 0; i < sphParticlesSize(); i++) {
        double const x = sphParticlesPosition(XDIM, i).toDouble();
        double const z = sphParticlesPosition(ZDIM, i).toDouble();
        require(z > 0.0 && z < 1.0 && z != 0.25 && z != 0.5 && z != 0.75,
            "particles lie inside material layers away from symmetry planes");
        bool const heavy = z < 0.25 || z > 0.75;
        double const density = heavy ? 2.0 : 1.0;
        if (heavy) {
            heavyCount++;
        } else {
            lightCount++;
        }
        std::array<uint32_t, NDIM> key{};
        for (int dim = 0; dim < NDIM; dim++) {
            key[dim] = sphParticlesPosition(dim, i).raw();
        }
        require(occupied.insert(key).second, "particle positions must be unique");
        double const energy = sphParticlesInternalEnergy(i);
        double const pressure = (configured.hydro.gamma - 1.0) * density * energy;
        double const distance = std::abs(z - 0.5);
        // Integrate the weight inward from P=1.5 at z=0 or z=1.
        double const expectedPressure = heavy ?
            1.5 + 2.0 * acceleration * (0.5 - distance) :
            1.5 + 0.5 * acceleration + acceleration * (0.25 - distance);
        require(energy > 0.0 && pressure > 0.0, "positive thermodynamic state");
        requireClose("hydrostatic pressure", pressure, expectedPressure);
        auto const [position, inserted] = pressureByHeight.emplace(z, pressure);
        if (!inserted) {
            requireClose("pressure depends only on z", pressure, position->second);
        }
        double const support = sphParticlesSupportRadius(i);
        requireClose("support radius encodes layer density",
            mass * configured.hydro.neighborNumber /
                (4.0 * kPi / 3.0 * support * support * support),
            density);
        requireClose("zero transverse x velocity", sphParticlesVelocity(XDIM, i), 0.0);
        requireClose("zero transverse y velocity", sphParticlesVelocity(YDIM, i), 0.0);
        double const expectedVelocity = 0.025 *
            (1.0 + std::cos(8.0 * kPi * (x - 0.5)) * (1.0 + std::cos(6.0 * kPi * (x - 0.5)))) *
            std::sin(2.0 * kPi * z);
        requireClose("normal velocity perturbation", sphParticlesVelocity(ZDIM, i),
            expectedVelocity, 2.0e-8);
        require(sphParticlesRung(i) == 0, "initial rung");
    }
    require(heavyCount == 2 * lightCount, "heavy and light particle counts have density ratio 2:1");
    requireClose("light layer mass", mass * lightCount, 0.5);
    requireClose("heavy layer mass", mass * heavyCount, 1.0);
    requireClose("total mass", mass * sphParticlesSize(), 1.5);
    for (ParticleInteger i = 0; i < sphParticlesSize(); i += 2) {
        require(sphParticlesPosition(ZDIM, i) < Fixed32(0.5), "lower particle precedes its mirror");
        require(sphParticlesPosition(XDIM, i).raw() == sphParticlesPosition(XDIM, i + 1).raw() &&
                sphParticlesPosition(YDIM, i).raw() == sphParticlesPosition(YDIM, i + 1).raw(),
            "mirrored pairs share transverse positions");
        require(uint32_t(sphParticlesPosition(ZDIM, i).raw() +
                    sphParticlesPosition(ZDIM, i + 1).raw()) == 0U,
            "mirrored z coordinates are exact in fixed point");
        require(sphParticlesVelocity(ZDIM, i) == -sphParticlesVelocity(ZDIM, i + 1),
            "normal velocities are exactly odd about the midplane");
        require(sphParticlesInternalEnergy(i) == sphParticlesInternalEnergy(i + 1) &&
                sphParticlesSupportRadius(i) == sphParticlesSupportRadius(i + 1),
            "thermodynamic state is exactly even about the midplane");
    }
    for (auto previous = pressureByHeight.begin(), current = std::next(previous);
        current != pressureByHeight.end(); previous = current++) {
        double const first = previous->first;
        double const second = current->first;
        int const firstLayer = int(4.0 * first);
        if (firstLayer != int(4.0 * second)) {
            continue;
        }
        double const density = (firstLayer == 0 || firstLayer == 3) ? 2.0 : 1.0;
        double const force = first < 0.5 ? acceleration : -acceleration;
        requireClose("hydrostatic pressure gradient",
            (current->second - previous->second) / (second - first), density * force, 2.0e-5);
    }
}

void testUnresolvedLattice() {
    for (int resolution : {1, 4}) {
        sphParticlesResize(0);
        options.initial.particlesPerDimension = resolution;
        bool rejected = false;
        try {
            scenarios::initializeRayleighTaylor();
        } catch (std::runtime_error const&) {
            rejected = true;
        }
        require(rejected, "unresolved lattice must be rejected");
        require(sphParticlesSize() == 0, "rejected lattice leaves no particles");
    }
}

}    // namespace detail

TEST(RayleighTaylor, Resolution8) {
    for (double acceleration : {0.0, 0.5, 1000.0}) {
        detail::testInitialization(8, 96, acceleration);
    }
}

TEST(RayleighTaylor, Resolution16) {
    for (double acceleration : {0.0, 0.5, 1000.0}) {
        detail::testInitialization(16, 2592, acceleration);
    }
}

TEST(RayleighTaylor, Resolution32) {
    for (double acceleration : {0.0, 0.5, 1000.0}) {
        detail::testInitialization(32, 20736, acceleration);
    }
}

TEST(RayleighTaylor, UnresolvedLattice) {
    detail::testUnresolvedLattice();
}
