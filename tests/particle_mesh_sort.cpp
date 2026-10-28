/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cosmictiger/particles/particle_mesh_sort.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph_particles.hpp>
#endif

namespace detail {

void requireParticleMeshSort(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Integer>
uint64_t referenceMeshCoordinate(Integer raw, int meshSize) {
    // Independent oracle: compare to exact cell boundaries by integer division,
    // rather than multiplying the coordinate by the mesh size.
    constexpr int kBits = std::numeric_limits<Integer>::digits;
    unsigned __int128 const period = static_cast<unsigned __int128>(1) << kBits;
    uint64_t lower = 0;
    uint64_t upper = static_cast<uint64_t>(meshSize);
    while (lower < upper) {
        uint64_t const middle = lower + (upper - lower + 1) / 2;
        auto const boundary = (period * middle + meshSize - 1) / meshSize;
        if (static_cast<unsigned __int128>(raw) >= boundary) {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    return lower;
}

template <class Integer>
void checkParticleMeshCoordinates() {
    constexpr int kBits = std::numeric_limits<Integer>::digits;
    unsigned __int128 const period = static_cast<unsigned __int128>(1) << kBits;
    for (int meshSize : {1, 2, 3, 7, 8, 31, 127, 1024, 65537}) {
        for (uint64_t cell : {uint64_t(0), uint64_t(1), uint64_t(meshSize / 2),
                 uint64_t(meshSize - 1), uint64_t(meshSize)}) {
            auto const boundary = (period * cell + meshSize - 1) / meshSize;
            for (int offset : {-1, 0, 1}) {
                auto const raw = static_cast<Integer>(boundary + offset);
                requireParticleMeshSort(
                    particleMeshCoordinate(raw, meshSize) == referenceMeshCoordinate(raw, meshSize),
                    "Particle mesh coordinate misclassified a fixed-point boundary");
            }
        }
    }
}

PositionInteger meshSortTestPosition(size_t identity, int dimension, int pattern) {
    uint64_t raw =
        identity * 0x9e3779b97f4a7c15ULL + uint64_t(dimension + 1) * 0xbf58476d1ce4e5b9ULL;
    raw ^= raw >> 29;
    raw *= 0x94d049bb133111ebULL;
    if (pattern == 1) {
        raw = uint64_t(1) << 62;
    } else if (pattern == 2) {
        switch ((identity + dimension) % 4) {
        case 0:
            raw = 0;
            break;
        case 1:
            raw = std::numeric_limits<uint64_t>::max();
            break;
        case 2:
            raw = uint64_t(1) << 63;
            break;
        default:
            raw = (uint64_t(1) << 63) - 1;
            break;
        }
    }
    return static_cast<PositionInteger>(raw >> (64 - kPositionBits));
}

uint64_t meshSortReferenceKey(ParticleInteger index, int meshSize) {
    uint64_t result = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        result =
            result * meshSize + referenceMeshCoordinate(particlesPos(dim, index).raw(), meshSize);
    }
    return result;
}

void initializeMeshSortFixture(ParticleInteger count, int pattern) {
    particlesResize(count);
#if COSMICTIGER_WITH_SPH
    sphParticlesResize((count + 1) / 2, false);
#endif
    for (ParticleInteger i = 0; i < count; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i).setInteger(meshSortTestPosition(i, dim, pattern));
            particlesVel(dim, i) = float(i) + 0.25f * dim;
            // Saved forces may not yet be initialized at the first force pass.
            // Signal-NaN bit patterns catch accidental arithmetic during swaps.
            particlesGravityAccelerations[dim][i] =
                std::bit_cast<float>(uint32_t(0x7f800001U + 4 * i + dim));
        }
        particlesP[i] = std::bit_cast<float>(uint32_t(0x80000000U + i));
        particlesRung(i) = i % 11;
        particlesLastgroup(i) = 100000 + i;
        particlesGrp[i].store(200000 + i, std::memory_order_relaxed);
        particlesTracer(i) = i % 2;
#if COSMICTIGER_WITH_SPH
        particlesType(i) = i % 2 == 0 ? kSphType : kDarkMatterType;
        particlesCatIndex(i) = i % 2 == 0 ? i / 2 : kNoParticleIndex;
        if (i % 2 == 0) {
            ParticleInteger const gasIndex = i / 2;
            sphParticlesParticleIndex(gasIndex) = i;
            sphParticlesInternalEnergy(gasIndex) = float(i) + 0.5f;
            sphParticlesSupportRadius(gasIndex) = float(i) + 0.75f;
            sphParticlesDensities[gasIndex] = float(i) + 1.0f;
            sphParticlesBalsaraFactors[gasIndex] = float(i) + 1.25f;
            sphParticlesInternalEnergyRates[gasIndex] = float(i) + 1.5f;
            sphParticlesPredictedInternalEnergyRates[gasIndex] = float(i) + 1.75f;
            sphParticlesConvergenceFlags[gasIndex] = i % 7;
            sphParticlesSemiactiveFlags[gasIndex] = i % 5;
            for (int dim = 0; dim < kDimensions; ++dim) {
                sphParticlesAcceleration(dim, gasIndex) = float(i) + 2.0f + dim;
                sphParticlesPredictedAcceleration(dim, gasIndex) = float(i) + 3.0f + dim;
                sphParticlesGravityAcceleration(dim, gasIndex) = float(i) + 4.0f + dim;
            }
        }
#endif
    }
}

void checkMeshSortFixture(int meshSize, int pattern) {
    ParticleInteger const count = particlesSize();
    auto const originalPositions = particlesX;
    auto* const originalVelocities = particlesV;
    auto* const originalRungs = particlesR;
    auto const cells = particleMeshSort(meshSize);
    requireParticleMeshSort(
        particlesSize() == count && particlesV == originalVelocities && particlesR == originalRungs,
        "Particle mesh sort resized particle storage");
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireParticleMeshSort(particlesX[dim] == originalPositions[dim],
            "Particle mesh sort replaced a position array");
    }
    std::vector<bool> seen(count, false);
    uint64_t previousKey = 0;
    for (ParticleInteger i = 0; i < count; ++i) {
        auto const identity = static_cast<ParticleInteger>(particlesVel(0, i));
        requireParticleMeshSort(identity >= 0 && identity < count && !seen[identity],
            "Particle mesh sort lost or duplicated a particle");
        seen[identity] = true;
        auto const key = meshSortReferenceKey(i, meshSize);
        requireParticleMeshSort(i == 0 || key >= previousKey, "Particle mesh keys are not sorted");
        previousKey = key;
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireParticleMeshSort(
                particlesPos(dim, i).raw() == meshSortTestPosition(identity, dim, pattern) &&
                    particlesVel(dim, i) == float(identity) + 0.25f * dim &&
                    std::bit_cast<uint32_t>(particlesGravityAccelerations[dim][i]) ==
                        uint32_t(0x7f800001U + 4 * identity + dim),
                "Particle mesh sort detached position, velocity or saved acceleration");
        }
        requireParticleMeshSort(
            std::bit_cast<uint32_t>(particlesP[i]) == uint32_t(0x80000000U + identity) &&
                particlesRung(i) == identity % 11 && particlesLastgroup(i) == 100000 + identity &&
                particlesGrp[i].load(std::memory_order_relaxed) == 200000 + identity &&
                particlesTracer(i) == identity % 2 &&
                particlesType(i) ==
                    (COSMICTIGER_WITH_SPH && identity % 2 == 0 ? kSphType : kDarkMatterType),
            "Particle mesh sort detached a particle field");
#if COSMICTIGER_WITH_SPH
        if (identity % 2 == 0) {
            ParticleInteger const gasIndex = identity / 2;
            requireParticleMeshSort(particlesCatIndex(i) == gasIndex &&
                    sphParticlesParticleIndex(gasIndex) == i &&
                    sphParticlesInternalEnergy(gasIndex) == float(identity) + 0.5f &&
                    sphParticlesSupportRadius(gasIndex) == float(identity) + 0.75f &&
                    sphParticlesDensities[gasIndex] == float(identity) + 1.0f &&
                    sphParticlesBalsaraFactors[gasIndex] == float(identity) + 1.25f &&
                    sphParticlesInternalEnergyRates[gasIndex] == float(identity) + 1.5f &&
                    sphParticlesPredictedInternalEnergyRates[gasIndex] == float(identity) + 1.75f &&
                    sphParticlesConvergenceFlags[gasIndex] == identity % 7 &&
                    sphParticlesSemiactiveFlags[gasIndex] == identity % 5,
                "Particle mesh sort changed gas state or broke its reverse map");
            for (int dim = 0; dim < kDimensions; ++dim) {
                requireParticleMeshSort(
                    sphParticlesAcceleration(dim, gasIndex) == float(identity) + 2.0f + dim &&
                        sphParticlesPredictedAcceleration(dim, gasIndex) ==
                            float(identity) + 3.0f + dim &&
                        sphParticlesGravityAcceleration(dim, gasIndex) ==
                            float(identity) + 4.0f + dim,
                    "Particle mesh sort changed a gas acceleration");
            }
        } else {
            requireParticleMeshSort(particlesCatIndex(i) == ParticleInteger(kNoParticleIndex),
                "Particle mesh sort changed a dark matter category index");
        }
#endif
    }
    ParticleInteger end = 0;
    previousKey = 0;
    for (size_t index = 0; index < cells.size(); ++index) {
        auto const& cell = cells[index];
        requireParticleMeshSort(cell.begin == end && cell.end > cell.begin && cell.end <= count &&
                (index == 0 || cell.key > previousKey),
            "Particle mesh occupied-cell directory has a gap or duplicate");
        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
            requireParticleMeshSort(meshSortReferenceKey(i, meshSize) == cell.key,
                "Particle mesh occupied-cell range includes the wrong cell");
        }
        end = cell.end;
        previousKey = cell.key;
    }
    requireParticleMeshSort(end == count, "Particle mesh directory omits particles");
}

}    // namespace detail

void particleMeshSortTest() {
    if (hpxSize() != 1 || getOptions().compute.cuda || particlesSize() != 0) {
        throw std::runtime_error("Particle mesh sort fixture requires one empty CPU locality");
    }
    detail::checkParticleMeshCoordinates<uint32_t>();
    detail::checkParticleMeshCoordinates<uint64_t>();
    for (int invalidMesh : {-1, 0, std::numeric_limits<int>::max()}) {
        bool rejected = false;
        try {
            particleMeshSort(invalidMesh);
        } catch (std::exception const&) {
            rejected = true;
        }
        detail::requireParticleMeshSort(rejected, "Particle mesh accepted an invalid mesh size");
    }

    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.hydro.enabled = COSMICTIGER_WITH_SPH != 0;
    options.analysis.saveForce = true;
    options.analysis.tracers = true;
    options.analysis.groups = true;
    options.chemistry.enabled = false;
    setOptions(options);
    constexpr ParticleInteger kMaximumCount = 25003;
    particlesResize(kMaximumCount);
    particlesGroupsInit();
    for (int pattern : {0, 1, 2}) {
        for (ParticleInteger count : {0, 1, 2, 31, 4097, kMaximumCount}) {
            detail::initializeMeshSortFixture(count, pattern);
            for (int meshSize : {1, 7, 16, 31}) {
                detail::checkMeshSortFixture(meshSize, pattern);
                // Reusing already grouped particles exercises equal-key runs.
                detail::checkMeshSortFixture(meshSize, pattern);
            }
        }
    }
    particlesGroupsDestroy();
    particlesResize(0);
#if COSMICTIGER_WITH_SPH
    sphParticlesResize(0, false);
#endif
    setOptions(originalOptions);
    PRINT("Particle mesh in-place sort passed (%s)\n",
        COSMICTIGER_WITH_SPH ? "all fields, gas maps and fixed boundaries" :
                               "collisionless fields and fixed boundaries");
}
