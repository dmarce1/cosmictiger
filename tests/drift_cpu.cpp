/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/math/simd_backend.hpp>
#include <cosmictiger/particles/drift_cpu.hpp>

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

namespace detail {

constexpr uint64_t kPositionPeriod = uint64_t{1} << 32;

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void requireClose(double actual, double expected, double tolerance, char const* label) {
    // Bit checks remain effective under the production -ffast-math flags.
    uint64_t const exponent = std::bit_cast<uint64_t>(actual) & 0x7ff0000000000000ULL;
    require(exponent != 0x7ff0000000000000ULL, label);
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

struct Fixture {
    std::array<std::vector<Fixed32>, NDIM> positions;
    std::vector<Array<float, NDIM>> velocities;
    std::vector<char> rungs;

    explicit Fixture(size_t size)
      : velocities(size)
      , rungs(size) {
        for (int dim = 0; dim < NDIM; dim++) {
            positions[dim].resize(size);
        }
        for (size_t i = 0; i < size; i++) {
            rungs[i] = i % 5;
            for (int dim = 0; dim < NDIM; dim++) {
                uint32_t raw = uint32_t(i * 2654435761ULL + (dim + 1) * 2246822519ULL);
                if (i % 7 == 0) {
                    raw = dim == XDIM ? 0 : (dim == YDIM ? UINT32_MAX : 1);
                }
                positions[dim][i].setInteger(raw);
                velocities[i][dim] = float(int((i * 11 + dim * 19) % 97) - 48) * 0.375f;
            }
        }
    }

    void run(size_t offset, size_t count, DriftParameters const& parameters,
        std::vector<LightconeEntry>* entries = nullptr) {
        cpuDriftKernel(positions[XDIM].data() + offset, positions[YDIM].data() + offset,
            positions[ZDIM].data() + offset, velocities.data() + offset, rungs.data() + offset,
            count, parameters, entries);
    }
};

uint32_t referencePosition(uint32_t raw, float velocity, DriftParameters const& parameters) {
    long double position = static_cast<long double>(raw) / kPositionPeriod;
    position += static_cast<long double>(velocity) * parameters.timestep / parameters.scale;
    position -= std::floor(position);
    return static_cast<uint32_t>(position * kPositionPeriod);
}

void checkDrift(size_t count, size_t offset, DriftParameters const& parameters) {
    Fixture fixture(count + offset + 5);
    Fixture const original = fixture;
    fixture.run(offset, count, parameters);
    for (size_t i = 0; i < fixture.velocities.size(); i++) {
        bool const active = i >= offset && i < offset + count &&
            (parameters.rung < 0 || original.rungs[i] == parameters.rung);
        require(fixture.rungs[i] == original.rungs[i], "drift preserves rungs");
        for (int dim = 0; dim < NDIM; dim++) {
            uint32_t const actualRaw = fixture.positions[dim][i].raw();
            uint32_t const originalRaw = original.positions[dim][i].raw();
            float const originalVelocity = original.velocities[i][dim];
            if (!active || parameters.timestep == 0.0) {
                require(actualRaw == originalRaw, "inactive and zero-step positions are exact");
                require(std::bit_cast<uint32_t>(fixture.velocities[i][dim]) ==
                        std::bit_cast<uint32_t>(originalVelocity),
                    "inactive and zero-step velocities are exact");
                continue;
            }
            uint32_t const expectedRaw =
                referencePosition(originalRaw, originalVelocity, parameters);
            uint64_t const forward = uint32_t(actualRaw - expectedRaw);
            uint64_t const backward = uint32_t(expectedRaw - actualRaw);
            require(
                std::min(forward, backward) <= 2, "drift agrees with scalar position reference");
            if (parameters.drag == 0.0) {
                require(std::bit_cast<uint32_t>(fixture.velocities[i][dim]) ==
                        std::bit_cast<uint32_t>(originalVelocity),
                    "drift without drag preserves velocity bits");
            } else {
                float const expectedVelocity =
                    double(originalVelocity) / (1.0 + parameters.drag * parameters.timestep);
                double const tolerance = 2.0 * std::numeric_limits<float>::epsilon() *
                    std::max(1.0, std::abs(double(expectedVelocity)));
                requireClose(fixture.velocities[i][dim], expectedVelocity, tolerance,
                    "drag damps velocity after position displacement");
            }
        }
    }
}

void testPositionsAndActivity() {
    constexpr size_t kWidth = NativeSimdVector<double>::size();
    std::vector<size_t> counts = {4095, 4096, 4097, 8193};
    // Exercise every tail length around several packs of the selected native width.
    for (size_t count = 0; count <= 3 * kWidth + 1; ++count) {
        counts.push_back(count);
    }
    for (size_t count : counts) {
        for (size_t offset : {size_t{0}, size_t{3}}) {
            DriftParameters parameters;
            parameters.scale = 0.73;
            parameters.timestep = 0.037;
            checkDrift(count, offset, parameters);
            parameters.rung = 2;
            parameters.drag = 1.7;
            checkDrift(count, offset, parameters);
            parameters.rung = 9;
            checkDrift(count, offset, parameters);
            parameters.rung = -1;
            parameters.timestep = 0.0;
            checkDrift(count, offset, parameters);
            parameters.scale = 0.41;
            parameters.timestep = 2.3;
            checkDrift(count, offset, parameters);
            parameters.drag = 0.0;
            parameters.timestep = -1.7;
            checkDrift(count, offset, parameters);
        }
    }
}

void testEmpty() {
    DriftParameters parameters;
    cpuDriftKernel(nullptr, nullptr, nullptr, nullptr, nullptr, 0, parameters);
    parameters.lightcone = true;
    parameters.tau1 = 0.25;
    parameters.tauMax = 1.0;
    std::vector<LightconeEntry> entries;
    cpuDriftKernel(nullptr, nullptr, nullptr, nullptr, nullptr, 0, parameters, &entries);
    require(entries.empty(), "empty drift produces no lightcone entries");
}

Fixture stationaryParticle(double coordinate) {
    Fixture fixture(1);
    for (int dim = 0; dim < NDIM; dim++) {
        fixture.positions[dim][0] = coordinate;
        fixture.velocities[0][dim] = 0.0f;
    }
    return fixture;
}

void testLightconeImages() {
    Fixture fixture = stationaryParticle(0.5);
    DriftParameters parameters;
    parameters.lightcone = true;
    parameters.timestep = 0.25;
    parameters.tau1 = 0.25;
    parameters.tauMax = 1.0;
    std::vector<LightconeEntry> entries;
    fixture.run(0, 1, parameters, &entries);
    require(entries.size() == 8, "stationary central particle crosses all eight periodic images");
    std::array<bool, 8> images{};
    std::array<bool, 12> pixels{};
    for (LightconeEntry const& entry : entries) {
        double const position[] = {entry.x.toDouble(), entry.y.toDouble(), entry.z.toDouble()};
        int image = 0;
        for (int dim = 0; dim < NDIM; dim++) {
            require(
                std::abs(position[dim]) == 0.5, "lightcone preserves stationary image position");
            image |= int(position[dim] > 0.0) << dim;
        }
        require(!images[image], "lightcone images are unique");
        images[image] = true;
        require(entry.pix >= 0 && entry.pix < 12, "lightcone HEALPix index lies in map");
        require(!pixels[entry.pix], "eight central images occupy distinct HEALPix pixels");
        pixels[entry.pix] = true;
        require(entry.vx == 0.0f && entry.vy == 0.0f && entry.vz == 0.0f,
            "stationary lightcone velocities remain zero");
    }

    fixture = stationaryParticle(0.25);
    parameters.timestep = 0.1;
    parameters.tau0 = 0.5;
    parameters.tau1 = 0.6;
    entries.clear();
    fixture.run(0, 1, parameters, &entries);
    require(entries.size() == 1, "only the positive quarter-box image crosses the lightcone");
    require(entries[0].x.toDouble() == 0.25 && entries[0].y.toDouble() == 0.25 &&
            entries[0].z.toDouble() == 0.25,
        "single stationary crossing has its expected position");
    require(entries[0].pix == 0, "positive diagonal maps to the first north HEALPix face");
}

void testLightconeGates() {
    DriftParameters parameters;
    parameters.timestep = 0.25;
    parameters.tau1 = 0.25;
    parameters.tauMax = 1.0;
    Fixture fixture = stationaryParticle(0.5);
    std::vector<LightconeEntry> entries;
    fixture.run(0, 1, parameters, &entries);
    require(entries.empty(), "disabled lightcone does not emit crossings");
    parameters.lightcone = true;
    parameters.tauMax = 1.25;
    fixture = stationaryParticle(0.0);
    fixture.run(0, 1, parameters, &entries);
    require(entries.empty(), "lightcone waits until the radius falls below one box length");
    parameters.tauMax = 1.0;
    parameters.rung = 2;
    fixture = stationaryParticle(0.5);
    fixture.run(0, 1, parameters, &entries);
    require(entries.empty(), "inactive rung does not emit lightcone crossings");
    parameters.rung = -1;
    fixture.run(0, 1, parameters);
    require(fixture.positions[XDIM][0].toDouble() == 0.5,
        "lightcone supports an omitted output buffer");
}

void testMovingLightcone() {
    for (double drag : {0.0, 2.0}) {
        Fixture fixture = stationaryParticle(0.25);
        for (int dim = 0; dim < NDIM; dim++) {
            fixture.velocities[0][dim] = 0.125f;
        }
        DriftParameters parameters;
        parameters.scale = 2.0;
        parameters.timestep = 0.15;
        parameters.tau0 = 0.5;
        parameters.tau1 = 0.65;
        parameters.tauMax = 1.0;
        parameters.lightcone = true;
        parameters.drag = drag;
        std::vector<LightconeEntry> entries;
        fixture.run(0, 1, parameters, &entries);
        require(entries.size() == 1, "radially moving particle has one lightcone crossing");

        float const expectedVelocity = 0.125 / (1.0 + drag * parameters.timestep);
        double const coordinateSpeed = expectedVelocity / parameters.scale;
        double const crossingTime = (parameters.tauMax - parameters.tau0 - std::sqrt(3.0) * 0.25) /
            (1.0 + std::sqrt(3.0) * coordinateSpeed);
        double const expectedCoordinate = 0.25 + coordinateSpeed * crossingTime;
        LightconeEntry const& entry = entries.front();
        requireClose(entry.x.toDouble(), expectedCoordinate, 2.0e-9,
            "radial crossing x agrees with analytic intersection");
        requireClose(entry.y.toDouble(), expectedCoordinate, 2.0e-9,
            "radial crossing y agrees with analytic intersection");
        requireClose(entry.z.toDouble(), expectedCoordinate, 2.0e-9,
            "radial crossing z agrees with analytic intersection");
        for (float velocity : {entry.vx, entry.vy, entry.vz}) {
            requireClose(velocity, expectedVelocity, 2.0e-8,
                "lightcone stores the post-drag physical velocity");
        }
        require(entry.pix == 0, "radial motion preserves HEALPix direction");
        requireClose(fixture.positions[XDIM][0].toDouble(),
            0.25 + 0.125 * parameters.timestep / parameters.scale, 2.0 / kPositionPeriod,
            "particle displacement uses the velocity before drag");
    }
}

}    // namespace detail

int main() {
    try {
        detail::testEmpty();
        detail::testPositionsAndActivity();
        detail::testLightconeImages();
        detail::testLightconeGates();
        detail::testMovingLightcone();
        std::printf("Drift SIMD CPU regressions passed (%zu native double lanes)\n",
            size_t(NativeSimdVector<double>::size()));
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Drift SIMD CPU regression failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
