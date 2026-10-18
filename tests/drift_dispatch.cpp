/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/hpx_init.hpp>

#include <cosmictiger/drift.hpp>
#include <cosmictiger/lightcone_entry.hpp>

Array<Fixed32*, NDIM> particlesX;
Array<float, NDIM>* particlesV = nullptr;
char* particlesR = nullptr;

namespace detail {

Options options;
ParticleInteger particleCount = 0;
Pair<ParticleInteger> currentRange(0, 0);
std::vector<LightconeEntry> capturedEntries;
size_t capturedWorkers = 0;
size_t lightconeCalls = 0;
double capturedScale = 0.0;
double capturedTau = 0.0;
int profilerDepth = 0;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void resetCapture() {
    capturedEntries.clear();
    capturedWorkers = 0;
    lightconeCalls = 0;
}

struct Fixture {
    std::array<std::vector<Fixed32>, NDIM> positions;
    std::vector<Array<float, NDIM>> velocities;
    std::vector<char> rungs;

    explicit Fixture(size_t count)
      : velocities(count)
      , rungs(count) {
        particleCount = count;
        currentRange = Pair<ParticleInteger>(0, count);
        options = Options{};
        options.compute.cuda = false;
        resetCapture();
        for (int dim = 0; dim < NDIM; ++dim) {
            positions[dim].resize(count);
            particlesX[dim] = positions[dim].data();
        }
        particlesV = velocities.data();
        particlesR = rungs.data();
        for (size_t index = 0; index < count; ++index) {
            rungs[index] = index % 3;
            for (int dim = 0; dim < NDIM; ++dim) {
                positions[dim][index].setInteger(
                    uint32_t(index * 2654435761ULL + (dim + 1) * 2246822519ULL));
                velocities[index][dim] = float(int((index * 11 + dim * 19) % 97) - 48) * 0.25f;
            }
        }
    }

    ~Fixture() {
        for (int dim = 0; dim < NDIM; ++dim) {
            particlesX[dim] = nullptr;
        }
        particlesV = nullptr;
        particlesR = nullptr;
        particleCount = 0;
        currentRange = Pair<ParticleInteger>(0, 0);
    }
};

uint32_t referencePosition(uint32_t raw, float velocity) {
    // All operands are binary fractions, so this independent calculation is exact.
    constexpr long double kPeriod = 4294967296.0L;
    long double position = raw / kPeriod + static_cast<long double>(velocity) * 0.25L;
    position -= std::floor(position);
    return static_cast<uint32_t>(position * kPeriod);
}

void checkDispatch(size_t count, int rung, bool glass) {
    Fixture fixture(count);
    auto const initialPositions = fixture.positions;
    auto const initialVelocities = fixture.velocities;
    currentRange = Pair<ParticleInteger>(3, count - 4);
    options.initial.createGlass = glass;
    // A nonzero caller drag must be overridden in both ordinary and glass modes.
    drift(0.5, 0.125, 0.0, 0.125, 0.0, rung, 123.0);
    double const damping = glass ? 1.0 / (1.0 + std::acos(-1.0) * 0.125) : 1.0;
    for (size_t index = 0; index < count; ++index) {
        bool const selected = rung < 0 ||
            (index >= size_t(currentRange.first) && index < size_t(currentRange.second) &&
                fixture.rungs[index] == rung);
        for (int dim = 0; dim < NDIM; ++dim) {
            uint32_t const expectedPosition = selected ?
                referencePosition(
                    initialPositions[dim][index].raw(), initialVelocities[index][dim]) :
                initialPositions[dim][index].raw();
            require(fixture.positions[dim][index].raw() == expectedPosition,
                "dispatch changed the wrong particle or produced the wrong displacement");
            float const expectedVelocity = selected ?
                float(initialVelocities[index][dim] * damping) :
                initialVelocities[index][dim];
            require(std::abs(fixture.velocities[index][dim] - expectedVelocity) <= 1.0e-6f,
                "dispatch damping differs from the CUDA glass policy");
        }
    }
    require(lightconeCalls == 0, "disabled lightcone unexpectedly received particles");
    require(profilerDepth == 0, "dispatch did not balance profiling calls");
}

void checkEmptyRanges() {
    Fixture fixture(0);
    // Null storage catches accidental formation or dereferencing of element-zero pointers.
    for (int dim = 0; dim < NDIM; ++dim) {
        particlesX[dim] = nullptr;
    }
    particlesV = nullptr;
    particlesR = nullptr;
    drift(0.5, 0.125, 0.0, 0.125, 0.0, -1);
    particleCount = 17;
    currentRange = Pair<ParticleInteger>(9, 9);
    drift(0.5, 0.125, 0.0, 0.125, 0.0, 1);
    require(profilerDepth == 0, "empty dispatch did not balance profiling calls");
    require(lightconeCalls == 0, "empty dispatch unexpectedly emitted particles");
}

void checkLightcone() {
    constexpr size_t kCount = 8201;
    Fixture fixture(kCount);
    options.lightcone.enabled = true;
    options.lightcone.mapSize = 2;
    for (size_t index = 0; index < kCount; ++index) {
        for (int dim = 0; dim < NDIM; ++dim) {
            fixture.positions[dim][index] = 0.25;
            fixture.velocities[index][dim] = 0.0f;
        }
    }
    // Only the positive image at radius sqrt(3)/4 crosses this shrinking shell.
    drift(0.5, 0.125, 0.5, 0.625, 1.0, -1);
    require(lightconeCalls == 1, "host lightcone was not called exactly once");
    require(capturedWorkers == 2, "large host drift did not use both worker batches");
    require(capturedEntries.size() == kCount, "host lightcone lost or duplicated crossings");
    require(capturedScale == 0.5 && capturedTau == 0.5,
        "host lightcone received the wrong scale or time");
    for (auto const& entry : capturedEntries) {
        require(
            entry.x.toDouble() == 0.25 && entry.y.toDouble() == 0.25 && entry.z.toDouble() == 0.25,
            "stationary shell crossing has an incorrect location");
        require(entry.vx == 0.0f && entry.vy == 0.0f && entry.vz == 0.0f,
            "stationary shell crossing acquired velocity");
        require(entry.pix >= 0 && entry.pix < 48, "crossing has an invalid HEALPix index");
    }
    resetCapture();
    drift(0.5, 0.125, 0.5, 0.625, 0.0, -1);
    require(lightconeCalls == 0, "nonpositive maximum time enabled the lightcone");
    drift(0.5, 0.125, 0.5, 0.625, 2.0, -1);
    require(lightconeCalls == 0, "distant lightcone shell was not skipped");
}

}    // namespace detail

Options const& getOptions() {
    return detail::options;
}

ParticleInteger particlesSize() {
    return detail::particleCount;
}

Pair<ParticleInteger> particlesCurrentRange() {
    return detail::currentRange;
}

Vector<hpx::id_type> const& hpxChildren() {
    static Vector<hpx::id_type> const children;
    return children;
}

int hpxHardwareConcurrency() {
    return 2;
}

void profilerEnter(char const*) {
    ++detail::profilerDepth;
}

void profilerExit() {
    --detail::profilerDepth;
}

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

size_t lcAddParticles(
    std::vector<std::vector<LightconeEntry>> const& batches, double scale, double tau) {
    ++detail::lightconeCalls;
    detail::capturedWorkers = batches.size();
    detail::capturedScale = scale;
    detail::capturedTau = tau;
    for (auto const& batch : batches) {
        detail::capturedEntries.insert(detail::capturedEntries.end(), batch.begin(), batch.end());
    }
    return detail::capturedEntries.size();
}

void cudaDrift(int, float, float, float, float, float, double) {
    throw std::runtime_error("CPU dispatch called the CUDA drift kernel");
}

void particlesMemadviseCpu() {
    throw std::runtime_error("CPU dispatch requested CUDA managed-memory advice");
}

int hpx_main(int, char*[]) {
    int status = 0;
    try {
        detail::checkDispatch(37, -1, false);
        detail::checkDispatch(9007, -1, false);
        detail::checkDispatch(9007, 1, false);
        detail::checkDispatch(9007, -1, true);
        detail::checkDispatch(9007, 1, true);
        detail::checkEmptyRanges();
        detail::checkLightcone();
        std::puts("CPU drift dispatch tests passed");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "CPU drift dispatch test failed: %s\n", error.what());
        status = 1;
    }
    hpx::finalize();
    return status;
}

int main(int argc, char* argv[]) {
    std::vector<std::string> configuration = {"hpx.os_threads=2", "hpx.stacks.small_size=524288"};
#ifdef HPX_EARLY
    return hpx::init(argc, argv, configuration);
#else
    hpx::init_params parameters;
    parameters.cfg = std::move(configuration);
    return hpx::init(argc, argv, parameters);
#endif
}
