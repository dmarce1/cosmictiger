/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
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

#include <gtest/gtest.h>
#include <hpx/hpx_init.hpp>

#include <cosmictiger/analysis/lightcone_entry.hpp>
#include <cosmictiger/hydro/hopkins_state.hpp>
#include <cosmictiger/particles/drift.hpp>

Array<Position*, kDimensions> particlesX;
Array<float, kDimensions>* particlesV = nullptr;
char* particlesR = nullptr;
char* particlesTy = nullptr;
ParticleInteger* particlesSph = nullptr;
HopkinsParticleState* sphParticlesHopkinsStates = nullptr;

namespace detail {

Options options;
ParticleInteger particleCount = 0;
Pair<ParticleInteger> currentRange(0, 0);
std::vector<LightconeEntry> capturedEntries;
size_t capturedWorkers = 0;
size_t lightconeCalls = 0;
double capturedScale = 0.0;
double capturedTau = 0.0;

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
    std::array<std::vector<Position>, kDimensions> positions;
    std::vector<Array<float, kDimensions>> velocities;
    std::vector<char> rungs;
    std::vector<char> types;
    std::vector<ParticleInteger> gasIndices;
    std::vector<HopkinsParticleState> meshStates;

    explicit Fixture(size_t count)
      : velocities(count)
      , rungs(count)
      , types(count, kDarkMatterType)
      , gasIndices(count, kNoParticleIndex)
      , meshStates(count) {
        particleCount = count;
        currentRange = Pair<ParticleInteger>(0, count);
        options = Options{};
        options.compute.cuda = false;
        resetCapture();
        for (int dim = 0; dim < kDimensions; ++dim) {
            positions[dim].resize(count);
            particlesX[dim] = positions[dim].data();
        }
        particlesV = velocities.data();
        particlesR = rungs.data();
        particlesTy = types.data();
        particlesSph = gasIndices.data();
        sphParticlesHopkinsStates = meshStates.data();
        for (size_t index = 0; index < count; ++index) {
            rungs[index] = index % 3;
            for (int dim = 0; dim < kDimensions; ++dim) {
                positions[dim][index].setInteger(
                    PositionInteger(uint32_t(index * 2654435761ULL + (dim + 1) * 2246822519ULL))
                    << (kPositionBits - 32));
                velocities[index][dim] = float(int((index * 11 + dim * 19) % 97) - 48) * 0.25f;
            }
        }
    }

    ~Fixture() {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesX[dim] = nullptr;
        }
        particlesV = nullptr;
        particlesR = nullptr;
        particlesTy = nullptr;
        particlesSph = nullptr;
        sphParticlesHopkinsStates = nullptr;
        particleCount = 0;
        currentRange = Pair<ParticleInteger>(0, 0);
    }
};

PositionInteger referencePosition(PositionInteger raw, float velocity) {
    // All operands are binary fractions, so this independent calculation is exact.
    constexpr long double kPeriod = kPositionBits == 64 ? 0x1p64L : 0x1p32L;
    long double const displacement = static_cast<long double>(velocity) * 0.25L;
    PositionInteger const increment =
        static_cast<PositionInteger>(std::fmod(std::abs(displacement), 1.0L) * kPeriod);
    return displacement < 0.0L ? raw - increment : raw + increment;
}

void checkMeshVelocity(std::string const& method, size_t count) {
    Fixture fixture(count);
    options.hydro.enabled = true;
    options.hydro.method = method;
    options.run.scenario = "glass";
    for (size_t index = 0; index < count; index += 2) {
        fixture.types[index] = kSphType;
        fixture.gasIndices[index] = count - 1 - index;
        auto& state = fixture.meshStates[fixture.gasIndices[index]];
        state.initialized = index % 4 == 0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            state.meshVelocity[dim] = -0.5 * fixture.velocities[index][dim];
        }
    }
    for (size_t index = 1; index < count; index += 4) {
        // Stars have no gas sidecar even in a meshless run.
        fixture.types[index] = kStarType;
    }
    auto const initialPositions = fixture.positions;
    auto const initialVelocities = fixture.velocities;
    drift(0.5, 0.125, 0.0, 0.125, 0.0);
    double const damping = 1.0 / (1.0 + std::acos(-1.0) * 0.125);
    for (size_t index = 0; index < count; ++index) {
        bool const meshlessGas = fixture.types[index] == kSphType && method != "sph" &&
            fixture.meshStates[fixture.gasIndices[index]].initialized;
        for (int dim = 0; dim < kDimensions; ++dim) {
            float const driftVelocity = meshlessGas ?
                fixture.meshStates[fixture.gasIndices[index]].meshVelocity[dim] :
                initialVelocities[index][dim];
            require(fixture.positions[dim][index].raw() ==
                    referencePosition(initialPositions[dim][index].raw(), driftVelocity),
                "drift uses the mapped mesh velocity only for meshless gas");
            float const expectedVelocity = meshlessGas ?
                initialVelocities[index][dim] :
                float(initialVelocities[index][dim] * damping);
            require(fixture.velocities[index][dim] == expectedVelocity,
                "meshless positional drift preserves the physical fluid velocity");
        }
    }
}

void checkMeshlessFixed64Motion(std::string const& method, int selectedRung) {
    constexpr size_t kCount = 37;
    constexpr int kSteps = 32;
    constexpr double kTimestep = 0.25;
    Fixture fixture(kCount);
    options.hydro.enabled = true;
    options.hydro.method = method;
    std::array<uint64_t, kDimensions> const positions{
        0x4000000000000123ULL, 0x8000000000000321ULL, UINT64_MAX - 7};
    std::array<double, kDimensions> const velocity{0x1p-60, -0x1p-60, 0x1p-60};
    for (size_t index = 0; index < kCount; ++index) {
        fixture.types[index] = kSphType;
        fixture.gasIndices[index] = kCount - 1 - index;
        auto& state = fixture.meshStates[fixture.gasIndices[index]];
        state.initialized = true;
        state.meshVelocity = velocity;
        state.supportRadius = 0.125;
        state.mass = 1.0;
        state.totalEnergy = 2.0;
        state.momentum = {0.1, -0.2, 0.3};
        for (int dim = 0; dim < kDimensions; ++dim) {
            fixture.positions[dim][index].setInteger(positions[dim]);
            fixture.velocities[index][dim] = 0.0f;
        }
    }
    for (int step = 0; step < kSteps; ++step) {
        drift(1.0, kTimestep, step * kTimestep, (step + 1) * kTimestep, 0.0, selectedRung);
    }
    for (size_t index = 0; index < kCount; ++index) {
        auto const& state = fixture.meshStates[fixture.gasIndices[index]];
        bool const moved =
            method != "sph" && (selectedRung < 0 || fixture.rungs[index] == selectedRung);
        for (int dim = 0; dim < kDimensions; ++dim) {
            uint64_t const displacement = moved ? 4 * kSteps : 0;
            uint64_t const expected =
                dim == 1 ? positions[dim] - displacement : positions[dim] + displacement;
            require(fixture.positions[dim][index].raw() == expected,
                "meshless drift discarded Fixed64 low bits or changed an inactive position");
            require(fixture.velocities[index][dim] == 0.0f,
                "meshless drift changed the physical fluid velocity");
        }
        require(state.mass == 1.0 && state.totalEnergy == 2.0 &&
                state.momentum == std::array<double, 3>{0.1, -0.2, 0.3} &&
                state.supportRadius == 0.125,
            "meshless positional drift changed conserved or support state");
    }
}

void checkDispatch(size_t count, int rung, bool glass) {
    Fixture fixture(count);
    auto const initialPositions = fixture.positions;
    auto const initialVelocities = fixture.velocities;
    currentRange = Pair<ParticleInteger>(3, count - 4);
    options.run.scenario = glass ? "glass" : "cosmology";
    // A nonzero caller drag must be overridden in both ordinary and glass modes.
    drift(0.5, 0.125, 0.0, 0.125, 0.0, rung, 123.0);
    double const damping = glass ? 1.0 / (1.0 + std::acos(-1.0) * 0.125) : 1.0;
    for (size_t index = 0; index < count; ++index) {
        bool const selected = rung < 0 ||
            (index >= size_t(currentRange.first) && index < size_t(currentRange.second) &&
                fixture.rungs[index] == rung);
        for (int dim = 0; dim < kDimensions; ++dim) {
            PositionInteger const expectedPosition = selected ?
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
}

void checkEmptyRanges() {
    Fixture fixture(0);
    // Null storage catches accidental formation or dereferencing of element-zero pointers.
    for (int dim = 0; dim < kDimensions; ++dim) {
        particlesX[dim] = nullptr;
    }
    particlesV = nullptr;
    particlesR = nullptr;
    drift(0.5, 0.125, 0.0, 0.125, 0.0, -1);
    particleCount = 17;
    currentRange = Pair<ParticleInteger>(9, 9);
    drift(0.5, 0.125, 0.0, 0.125, 0.0, 1);
    require(lightconeCalls == 0, "empty dispatch unexpectedly emitted particles");
}

void checkLightcone() {
    constexpr size_t kCount = 8201;
    Fixture fixture(kCount);
    options.lightcone.enabled = true;
    options.lightcone.mapSize = 2;
    for (size_t index = 0; index < kCount; ++index) {
        for (int dim = 0; dim < kDimensions; ++dim) {
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

void cudaDrift(int, double, double, double, double, double, double) {
    throw std::runtime_error("CPU dispatch called the CUDA drift kernel");
}

void particlesMemadviseCpu() {
    throw std::runtime_error("CPU dispatch requested CUDA managed-memory advice");
}

TEST(DriftDispatch, SmallRange) {
    detail::checkDispatch(37, -1, false);
}

TEST(DriftDispatch, MeshlessVelocity) {
    for (auto const* method : {"sph", "mfm", "mfv"}) {
        detail::checkMeshVelocity(method, 37);
        detail::checkMeshVelocity(method, 9007);
    }
}

TEST(DriftDispatch, MeshlessFixed64PositionPreservesLowBits) {
    if constexpr (kPositionBits != 64) {
        GTEST_SKIP() << "Hopkins requires Fixed64 positions";
    } else {
        for (auto const* method : {"sph", "mfm", "mfv"}) {
            detail::checkMeshlessFixed64Motion(method, -1);
            detail::checkMeshlessFixed64Motion(method, 1);
        }
    }
}

TEST(DriftDispatch, MeshlessPreservesDoubleTimestepAndScale) {
    if constexpr (kPositionBits != 64) {
        GTEST_SKIP() << "Hopkins requires Fixed64 positions";
    } else {
        for (auto const* method : {"mfm", "mfv"}) {
            detail::Fixture fixture(1);
            detail::options.hydro.enabled = true;
            detail::options.hydro.method = method;
            fixture.types[0] = kSphType;
            fixture.gasIndices[0] = 0;
            auto& state = fixture.meshStates[0];
            state.initialized = true;
            state.meshVelocity = {0.125, 0.125, 0.125};
            for (int dim = 0; dim < kDimensions; ++dim) {
                fixture.positions[dim][0].setInteger(0x4000000000000123ULL);
            }
            drift(0.7, 0.1, 0.0, 0.1, 0.0);
            PositionInteger const expected =
                0x4000000000000123ULL + Position(0.125 * (0.1 / 0.7)).raw();
            for (int dim = 0; dim < kDimensions; ++dim) {
                EXPECT_EQ(fixture.positions[dim][0].raw(), expected);
            }
        }
    }
}

TEST(DriftDispatch, MeshlessLightconeMassAndFluidVelocity) {
    detail::Fixture fixture(3);
    detail::options.hydro.enabled = true;
    detail::options.hydro.method = "mfv";
    detail::options.lightcone.enabled = true;
    detail::options.lightcone.mapSize = 2;
    fixture.types = {kSphType, kStarType, kDarkMatterType};
    particlesTy = fixture.types.data();
    fixture.gasIndices[0] = 2;
    auto& state = fixture.meshStates[2];
    state.initialized = true;
    state.meshVelocity = {0.0, 0.0, 0.0};
    state.mass = 0.375;
    for (size_t index = 0; index < 3; ++index) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            fixture.positions[dim][index] = 0.25;
            fixture.velocities[index][dim] = index == 0 ? 0.01f * (dim + 1) : 0.0f;
        }
    }
    drift(0.5, 0.125, 0.5, 0.625, 1.0);
    ASSERT_EQ(detail::capturedEntries.size(), 3);
    for (auto const& entry : detail::capturedEntries) {
        EXPECT_DOUBLE_EQ(entry.x.toDouble(), 0.25);
        EXPECT_DOUBLE_EQ(entry.y.toDouble(), 0.25);
        EXPECT_DOUBLE_EQ(entry.z.toDouble(), 0.25);
        if (entry.type == kSphType) {
            EXPECT_FLOAT_EQ(entry.mass, 0.375f);
            EXPECT_FLOAT_EQ(entry.vx, 0.01f);
            EXPECT_FLOAT_EQ(entry.vy, 0.02f);
            EXPECT_FLOAT_EQ(entry.vz, 0.03f);
        } else {
            EXPECT_FLOAT_EQ(entry.mass, -1.0f);
            EXPECT_FLOAT_EQ(entry.vx, 0.0f);
            EXPECT_FLOAT_EQ(entry.vy, 0.0f);
            EXPECT_FLOAT_EQ(entry.vz, 0.0f);
        }
    }
}

TEST(DriftDispatch, WorkerChunks) {
    detail::checkDispatch(9007, -1, false);
    detail::checkDispatch(9007, 1, false);
}

TEST(DriftDispatch, Drag) {
    detail::checkDispatch(9007, -1, true);
    detail::checkDispatch(9007, 1, true);
}

TEST(DriftDispatch, EmptyRanges) {
    detail::checkEmptyRanges();
}

TEST(DriftDispatch, Lightcone) {
    detail::checkLightcone();
}

int hpx_main(int, char*[]) {
    int const status = RUN_ALL_TESTS();
    hpx::finalize();
    return status;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    if (GTEST_FLAG_GET(list_tests)) {
        return RUN_ALL_TESTS();
    }
    std::vector<std::string> configuration = {"hpx.os_threads=2", "hpx.stacks.small_size=524288"};
#ifdef HPX_EARLY
    return hpx::init(argc, argv, configuration);
#else
    hpx::init_params parameters;
    parameters.cfg = std::move(configuration);
    return hpx::init(argc, argv, parameters);
#endif
}
