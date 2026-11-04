/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <hpx/threading_base/annotated_function.hpp>

#define SPH_PARTICLES_CPP
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/run_data.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen_particle_exchange.hpp>
#include <cosmictiger/sph/viscosity/hu_etal.hpp>
#include <cosmictiger/sph/viscosity/price_etal.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/chemistry/sph.hpp>
#endif
#include <cosmictiger/sph/particles.hpp>

#include "particle_cache.hpp"

bool sphParticlesEntropyInitialized = false;
double sphParticlesThermalScaleFactor = 1.0;

static ParticleInteger capacity = 0;
static ParticleInteger size = 0;

static Vector3<Position> const* sphParticlesCacheReadLine(detail::SphCacheLineId lineId);
static Pair<float> const* sphParticlesReadThermodynamicCacheLine(detail::SphCacheLineId lineId);
static char const* sphParticlesRungCacheReadLine(detail::SphCacheLineId lineId);
static Vector3<float> const* sphParticlesReadVelocityCacheLine(detail::SphCacheLineId lineId);
struct SphDensityCacheEntry {
    float density{};
    float balsaraFactor{};
    float bulkViscosityCoefficient{};
    float gradHCorrection{};
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & density & balsaraFactor;
        if (getOptions().hydro.viscosityEvolution != "none") {
            archive & bulkViscosityCoefficient;
        }
        archive & gradHCorrection;
    }
};

static SphDensityCacheEntry const* sphParticlesReadDensityCacheLine(detail::SphCacheLineId lineId);

static Vector<Vector3<Position>> sphParticlesFetchCacheLine(ParticleInteger index);
static Vector<char> sphParticlesFetchRungCacheLine(ParticleInteger index);
static Vector<Vector3<float>> sphParticlesFetchVelocityCacheLine(ParticleInteger index);
static Vector<SphDensityCacheEntry> sphParticlesFetchDensityCacheLine(ParticleInteger index);
static Vector<Pair<float>> sphParticlesFetchThermodynamicCacheLine(ParticleInteger index);

HPX_PLAIN_ACTION(sphParticlesFetchCacheLine, SphParticlesFetchCacheLineAction);
HPX_PLAIN_ACTION(
    sphParticlesFetchThermodynamicCacheLine, SphParticlesFetchThermodynamicCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchRungCacheLine, SphParticlesFetchRungCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchVelocityCacheLine, SphParticlesFetchVelocityCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchDensityCacheLine, SphParticlesFetchDensityCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesClearHydroCaches, SphParticlesClearHydroCachesAction);
HPX_PLAIN_ACTION(sphParticlesClearCaches, SphParticlesClearCachesAction);

static Array<
    std::unordered_map<detail::SphCacheLineId,
        std::shared_ptr<hpx::shared_future<Vector<Vector3<Position>>>>, detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    particleCache;
static Array<
    std::unordered_map<detail::SphCacheLineId,
        std::shared_ptr<hpx::shared_future<Vector<Pair<float>>>>, detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    thermodynamicParticleCache;
static Array<std::unordered_map<detail::SphCacheLineId,
                 std::shared_ptr<hpx::shared_future<Vector<char>>>, detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    rungParticleCache;
static Array<
    std::unordered_map<detail::SphCacheLineId,
        std::shared_ptr<hpx::shared_future<Vector<Vector3<float>>>>, detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    particleVelocityCache;
static Array<std::unordered_map<detail::SphCacheLineId,
                 std::shared_ptr<hpx::shared_future<Vector<SphDensityCacheEntry>>>,
                 detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    densityParticleCache;
static Array<SpinlockType, kParticleCacheSize> mutexes;
static Array<SpinlockType, kParticleCacheSize> energyMutexes;
static Array<SpinlockType, kParticleCacheSize> rungMutexes;
static Array<SpinlockType, kParticleCacheSize> velocityMutexes;
static Array<SpinlockType, kParticleCacheSize> densityMutexes;
HPX_PLAIN_ACTION(sphParticlesPrepareThermodynamics, SphParticlesPrepareThermodynamicsAction);
void sphParticlesPrepareThermodynamics(double scaleFactor, bool initializeEntropy) {
    ALWAYS_ASSERT(scaleFactor > 0.0);
    Vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphParticlesPrepareThermodynamicsAction>(
            child, scaleFactor, initializeEntropy));
    }
    sphParticlesThermalScaleFactor = scaleFactor;
    if (initializeEntropy && !sphParticlesEntropyInitialized) {
        for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
            ALWAYS_ASSERT(sphParticlesDensity(gas) > 0.0f && sphParticlesEntropies[gas] > 0.0f);
            sphParticlesEntropies[gas] = sph_entropy::fromStoredEnergy(sphParticlesEntropies[gas],
                sphParticlesDensity(gas), getOptions().hydro.gamma, scaleFactor);
        }
        sphParticlesEntropyInitialized = true;
    }
    for (auto& child : children) {
        child.get();
    }
}

namespace detail {

template <typename Visitor>
void visitSphCheckpointArrays(
    Visitor&& visit, bool includeChemistry = true, bool includeViscosity = true) {
    size_t const count = sphParticlesSize();
    // Persistent physical state and timestep/viscosity history only; force rates are event-local.
    visit(sphParticlesEntropies, count);
    visit(sphParticlesSmoothingLengths, count);
    visit(sphParticlesParticleIndices, count);
    visit(sphParticlesDensities, count);
    visit(sphParticlesFlags, count);
    if (includeViscosity && sphParticlesBulkViscosityCoefficients != nullptr) {
        visit(sphParticlesBulkViscosityCoefficients, count);
    }
    if (includeViscosity && sphScaledDivergence != nullptr) {
        visit(sphScaledDivergence, count);
    }
    if (includeViscosity && getOptions().hydro.viscosityGradientSwitch() &&
        sphParticlesBalsaraFactors != nullptr) {
        visit(sphParticlesBalsaraFactors, count);
    }
    if (includeChemistry && sphParticlesElectronFractions != nullptr) {
        visit(sphParticlesElectronFractions, count);
    }
}

}    // namespace detail

void sphParticlesSwap(ParticleInteger i, ParticleInteger j) {
    if (i != j) {
        sphTreeInvalidate();
    }
    if (sphParticlesBalsaraFactors != nullptr && !getOptions().hydro.viscosityGradientSwitch()) {
        std::swap(sphParticlesBalsaraFactors[i], sphParticlesBalsaraFactors[j]);
    }
    std::swap(sphParticlesGradHCorrections[i], sphParticlesGradHCorrections[j]);
    detail::visitSphCheckpointArrays(
        [i, j](auto* values, size_t) { std::swap(values[i], values[j]); });
}

ParticleInteger sphParticlesSort(
    Pair<ParticleInteger> particleRange, Position splitPosition, int splitDimension) {
    ParticleInteger begin = particleRange.first;
    ParticleInteger end = particleRange.second;
    ParticleInteger lower = begin;
    ParticleInteger upper = end;
    while (lower < upper) {
        if (sphParticlesPosition(splitDimension, lower) >= splitPosition) {
            while (lower != upper) {
                upper--;
                if (sphParticlesPosition(splitDimension, upper) < splitPosition) {
                    sphParticlesSwap(lower, upper);
                    break;
                }
            }
        }
        lower++;
    }
    return upper;
}

ParticleInteger sphParticlesSize() {
    return size;
}

template <typename T>
void sphParticlesArrayResize(T*& ptr, ParticleInteger newCapacity) {
    T* newPtr = static_cast<T*>(allocateManagedMemory(sizeof(T) * newCapacity));
    if (capacity > 0 && ptr != nullptr) {
        hpx_copy(annotatedParallelPolicy("sph.particles.resize.copy"), ptr, ptr + size, newPtr)
            .get();
        freeManagedMemory(ptr);
    }
    ptr = newPtr;
}

void sphParticlesResize(ParticleInteger particleCount, bool resizeGravityParticles) {
    if (particleCount != size) {
        sphTreeInvalidate();
    }
    bool const initializeBalsara =
        getOptions().hydro.balsaraSwitch && sphParticlesBalsaraFactors == nullptr;
    bool const gradientSwitch = getOptions().hydro.viscosityGradientSwitch();
    bool const initializeScaledDivergence = gradientSwitch && sphScaledDivergence == nullptr;
    bool const initializeViscosity = getOptions().hydro.viscosityEvolution != "none" &&
        sphParticlesBulkViscosityCoefficients == nullptr;
    // Stale until a density pass rebuilds it; NaN makes any earlier force use visible.
    float const unbuiltGradHCorrection = std::numeric_limits<float>::quiet_NaN();
    if (particleCount > capacity) {
        ParticleInteger newCapacity = std::max(capacity, (ParticleInteger) 100);
        while (newCapacity < particleCount) {
            newCapacity = size_t(101) * newCapacity / size_t(100);
        }
        sphParticlesArrayResize(sphParticlesDensities, newCapacity);
        if (getOptions().hydro.balsaraSwitch) {
            sphParticlesArrayResize(sphParticlesBalsaraFactors, newCapacity);
        }
        sphParticlesArrayResize(sphParticlesGradHCorrections, newCapacity);
        sphParticlesArrayResize(sphParticlesFlags, newCapacity);
        if (getOptions().hydro.viscosityEvolution != "none") {
            sphParticlesArrayResize(sphParticlesBulkViscosityCoefficients, newCapacity);
        }
        if (gradientSwitch) {
            sphParticlesArrayResize(sphScaledDivergence, newCapacity);
        }
        sphParticlesArrayResize(sphParticlesEntropies, newCapacity);
        sphParticlesArrayResize(sphParticlesSmoothingLengths, newCapacity);
        sphParticlesArrayResize(sphParticlesParticleIndices, newCapacity);
        if (getOptions().chemistry.enabled) {
            sphParticlesArrayResize(sphParticlesElectronFractions, newCapacity);
        }
        capacity = newCapacity;
    }
    if (!getOptions().hydro.balsaraSwitch && sphParticlesBalsaraFactors != nullptr) {
        freeManagedMemory(sphParticlesBalsaraFactors);
        sphParticlesBalsaraFactors = nullptr;
    } else if (getOptions().hydro.balsaraSwitch && sphParticlesBalsaraFactors == nullptr &&
        capacity > 0) {
        sphParticlesArrayResize(sphParticlesBalsaraFactors, capacity);
    }
    if (initializeBalsara) {
        for (ParticleInteger index = 0; index < size; ++index) {
            sphParticlesBalsaraFactors[index] = 0;
        }
    }
    if (getOptions().hydro.viscosityEvolution == "none" &&
        sphParticlesBulkViscosityCoefficients != nullptr) {
        freeManagedMemory(sphParticlesBulkViscosityCoefficients);
        sphParticlesBulkViscosityCoefficients = nullptr;
    } else if (getOptions().hydro.viscosityEvolution != "none" &&
        sphParticlesBulkViscosityCoefficients == nullptr && capacity > 0) {
        sphParticlesArrayResize(sphParticlesBulkViscosityCoefficients, capacity);
    }
    if (!gradientSwitch && sphScaledDivergence != nullptr) {
        freeManagedMemory(sphScaledDivergence);
        sphScaledDivergence = nullptr;
    } else if (gradientSwitch && sphScaledDivergence == nullptr && capacity > 0) {
        sphParticlesArrayResize(sphScaledDivergence, capacity);
    }
    if (initializeScaledDivergence) {
        for (ParticleInteger index = 0; index < size; ++index) {
            sphScaledDivergence[index] = {};
        }
    }
    if (initializeViscosity) {
        float const bulkViscosityCoefficient =
            gradientSwitch ? 0.0f : getOptions().hydro.minimumBulkViscosityCoefficient;
        for (ParticleInteger index = 0; index < size; ++index) {
            sphParticlesBulkViscosityCoefficients[index] = bulkViscosityCoefficient;
        }
    }
    ParticleInteger numberOfNewParticles = particleCount - size;
    ParticleInteger offset = particlesSize();
    if (resizeGravityParticles) {
        particlesResize(particlesSize() + numberOfNewParticles);
    }
    int previousSize = size;
    size = particleCount;
    for (int i = 0; i < numberOfNewParticles; i++) {
        ParticleInteger const slot = previousSize + i;
        sphParticlesEntropies[slot] = 0.0f;
        sphParticlesSmoothingLengths[slot] = 0.0f;
        if (getOptions().chemistry.enabled) {
            sphParticlesElectronFractions[slot] = 1.0f;
        }
        sphParticlesParticleIndices[slot] = kNoParticleIndex;
        sphParticlesDensities[slot] = 0.0f;
        if (getOptions().hydro.balsaraSwitch) {
            sphParticlesBalsaraFactors[slot] = 0;
        }
        sphParticlesGradHCorrections[slot] = unbuiltGradHCorrection;
        sphParticlesFlags[slot] = {};
        sphParticlesSetProposedRung(slot, 0);
        if (getOptions().hydro.viscosityEvolution != "none") {
            float const bulkViscosityCoefficient =
                gradientSwitch ? 0.0f : getOptions().hydro.minimumBulkViscosityCoefficient;
            sphParticlesBulkViscosityCoefficients[slot] = bulkViscosityCoefficient;
        }
        if (gradientSwitch) {
            sphScaledDivergence[slot] = {};
        }
        if (resizeGravityParticles) {
            particlesCatIndex(offset + i) = previousSize + i;
            particlesType(offset + i) = kSphType;
            sphParticlesParticleIndex(previousSize + i) = offset + i;
        }
    }
}

void sphParticlesRemove(ParticleInteger index) {
    ALWAYS_ASSERT(index >= 0 && index < sphParticlesSize());
    ParticleInteger const last = sphParticlesSize() - 1;
    if (index != last) {
        sphParticlesSwap(index, last);
        particlesCatIndex(sphParticlesParticleIndex(index)) = index;
    }
    sphParticlesResize(last, false);
}

void sphParticlesFree() {
    sphTreeInvalidate();
    freeManagedMemory(sphParticlesViscosityDiagnostics);
    sphParticlesViscosityDiagnostics = nullptr;
    sphParticlesEntropyInitialized = false;
    sphParticlesThermalScaleFactor = 1.0;
    if (sphParticlesBalsaraFactors != nullptr) {
        freeManagedMemory(sphParticlesBalsaraFactors);
        sphParticlesBalsaraFactors = nullptr;
    }
    if (sphParticlesGradHCorrections != nullptr) {
        freeManagedMemory(sphParticlesGradHCorrections);
        sphParticlesGradHCorrections = nullptr;
    }
    detail::visitSphCheckpointArrays([](auto*& values, size_t) {
        freeManagedMemory(values);
        values = nullptr;
    });
    size = 0;
    capacity = 0;
}

void sphParticlesResolveWithParticles() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futures;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads] {
                ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
                ParticleInteger const e =
                    (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i < e; i++) {
                    int const j = sphParticlesParticleIndex(i);
                    particlesCatIndex(j) = i;
                }
            },
            "sph.particles.resolveIndices")));
    }
    hpx::wait_all(futures.begin(), futures.end());
}

void sphParticlesGlobalReadPositions(
    ParticleGlobalRange range, Position* x, Position* y, Position* z, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const particleCount = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                x[j] = sphParticlesPosition(kXDimension, i);
                y[j] = sphParticlesPosition(kYDimension, i);
                z[j] = sphParticlesPosition(kZDimension, i);
            }
        } else {
            detail::SphCacheLineId lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    x[destIndex] = ptr[srcIndex][kXDimension];
                    y[destIndex] = ptr[srcIndex][kYDimension];
                    z[destIndex] = ptr[srcIndex][kZDimension];
                    destIndex++;
                }
            }
        }
    }
}

static Vector3<Position> const* sphParticlesCacheReadLine(detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchCacheLineAction>(
        lineId, particleCache, mutexes);
}

static Vector<Vector3<Position>> sphParticlesFetchCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Vector3<Position>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            line[i - begin][dim] = sphParticlesPosition(dim, i);
        }
    }
    return line;
}

void sphParticlesGlobalReadInternalEnergyAndSmoothingLength(ParticleGlobalRange range,
    float* internalEnergy, float* smoothingLength, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    int const particleCount = offset + range.range.second - range.range.first;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const particleCount = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                if (internalEnergy) {
                    internalEnergy[j] = sphParticlesInternalEnergy(i);
                }
                if (smoothingLength) {
                    smoothingLength[j] = sphParticlesSmoothingLength(i);
                }
            }
        } else {
            detail::SphCacheLineId lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesReadThermodynamicCacheLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    if (internalEnergy) {
                        internalEnergy[destIndex] = ptr[srcIndex].first;
                    }
                    if (smoothingLength) {
                        smoothingLength[destIndex] = ptr[srcIndex].second;
                    }
                    destIndex++;
                }
            }
        }
    }
}

static Pair<float> const* sphParticlesReadThermodynamicCacheLine(detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchThermodynamicCacheLineAction>(
        lineId, thermodynamicParticleCache, energyMutexes);
}

static Vector<Pair<float>> sphParticlesFetchThermodynamicCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Pair<float>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin].first = sphParticlesInternalEnergy(i);
        line[i - begin].second = sphParticlesSmoothingLength(i);
    }
    return line;
}

void sphParticlesGlobalReadRungs(ParticleGlobalRange range, char* rungs, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const particleCount = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                rungs[j] = sphParticlesRung(i);
            }
        } else {
            detail::SphCacheLineId lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesRungCacheReadLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    rungs[destIndex] = ptr[srcIndex];
                    destIndex++;
                }
            }
        }
    }
}

static char const* sphParticlesRungCacheReadLine(detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchRungCacheLineAction>(
        lineId, rungParticleCache, rungMutexes);
}

static Vector<char> sphParticlesFetchRungCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<char> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = sphParticlesRung(i);
    }
    return line;
}

void sphParticlesGlobalReadVelocities(
    ParticleGlobalRange range, float* vx, float* vy, float* vz, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const particleCount = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                vx[j] = sphParticlesVelocity(kXDimension, i);
                vy[j] = sphParticlesVelocity(kYDimension, i);
                vz[j] = sphParticlesVelocity(kZDimension, i);
            }
        } else {
            detail::SphCacheLineId lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesReadVelocityCacheLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    vx[destIndex] = ptr[srcIndex][kXDimension];
                    vy[destIndex] = ptr[srcIndex][kYDimension];
                    vz[destIndex] = ptr[srcIndex][kZDimension];
                    destIndex++;
                }
            }
        }
    }
}

static Vector3<float> const* sphParticlesReadVelocityCacheLine(detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchVelocityCacheLineAction>(
        lineId, particleVelocityCache, velocityMutexes);
}

static Vector<Vector3<float>> sphParticlesFetchVelocityCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Vector3<float>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            line[i - begin][dimension] = sphParticlesVelocity(dimension, i);
        }
    }
    return line;
}

void sphParticlesGlobalReadDensity(ParticleGlobalRange range, float* rho, ParticleInteger offset,
    float* balsaraFactor, float* bulkViscosityCoefficient, float* gradHCorrection) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const particleCount = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                if (rho) {
                    rho[j] = sphParticlesDensity(i);
                }
                if (balsaraFactor) {
                    balsaraFactor[j] = sphParticlesBalsaraFactors ?
                        balsara::decodeFactor(sphParticlesBalsaraFactors[i]) :
                        1.0f;
                }
                if (bulkViscosityCoefficient) {
                    bulkViscosityCoefficient[j] = sphParticlesBulkViscosityCoefficients[i];
                }
                if (gradHCorrection) {
                    gradHCorrection[j] = sphParticlesGradHCorrections[i];
                }
            }
        } else {
            detail::SphCacheLineId lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesReadDensityCacheLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    if (rho) {
                        rho[destIndex] = ptr[srcIndex].density;
                    }
                    if (balsaraFactor) {
                        balsaraFactor[destIndex] = ptr[srcIndex].balsaraFactor;
                    }
                    if (bulkViscosityCoefficient) {
                        bulkViscosityCoefficient[destIndex] =
                            ptr[srcIndex].bulkViscosityCoefficient;
                    }
                    if (gradHCorrection) {
                        gradHCorrection[destIndex] = ptr[srcIndex].gradHCorrection;
                    }
                    destIndex++;
                }
            }
        }
    }
}

static SphDensityCacheEntry const* sphParticlesReadDensityCacheLine(detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchDensityCacheLineAction>(
        lineId, densityParticleCache, densityMutexes);
}

static Vector<SphDensityCacheEntry> sphParticlesFetchDensityCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<SphDensityCacheEntry> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = {sphParticlesDensity(i),
            sphParticlesBalsaraFactors ? balsara::decodeFactor(sphParticlesBalsaraFactors[i]) :
                                         1.0f,
            getOptions().hydro.viscosityEvolution != "none" ?
                float(sphParticlesBulkViscosityCoefficients[i]) :
                0.0f,
            sphParticlesGradHCorrections[i]};
    }
    return line;
}

void sphParticlesClearHydroCaches() {
    Vector<hpx::future<void>> futures;
    for (auto const& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesClearHydroCachesAction>(c));
    }
    thermodynamicParticleCache = decltype(thermodynamicParticleCache)();
    particleVelocityCache = decltype(particleVelocityCache)();
    densityParticleCache = decltype(densityParticleCache)();
    cullen_and_dehnen::clearDiagnosticCache();
    hpx::wait_all(futures.begin(), futures.end());
}

void sphParticlesClearCaches() {
    Vector<hpx::future<void>> futures;
    for (auto const& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesClearCachesAction>(c));
    }
    particleCache = decltype(particleCache)();
    thermodynamicParticleCache = decltype(thermodynamicParticleCache)();
    rungParticleCache = decltype(rungParticleCache)();
    particleVelocityCache = decltype(particleVelocityCache)();
    densityParticleCache = decltype(densityParticleCache)();
    cullen_and_dehnen::clearDiagnosticCache();
    hpx::wait_all(futures.begin(), futures.end());
}

namespace detail {

uint32_t sphCheckpointVersion() {
    // Identify the scalar-viscosity payload and its optional model state.
    auto const& options = getOptions();
    // Chemistry payloads 80 and 90 carried a per-particle EOS; 100 and 110 store only the
    // electron fraction because chemistry fixes gamma at 5/3. Payloads below 10000 interleaved
    // internal energy with support radius; later ones store the two arrays separately. From
    // 20000 the predicted rates are no longer stored. From 30000 the thermal array and
    // its rate store entropy, with an explicit initialization flag and scale factor.
    // From 40000 viscosity stores only the scalar bulk viscosity coefficient.
    // From 50000 the transient gravity handoff is no longer checkpointed.
    // From 60000 the hydro rung bound is persistent and the hierarchical
    // predictor/position-clock structure is removed for chronological drifts.
    // From 70000 no force rates persist; gradient switches checkpoint sample time/divergence.
    // From 80000 own-midpoint viscosity history stores the previous sample rung.
    // From 90000 divergences use signed integer significands instead of scaled binary16.
    // From 100000 one half stores Δτ ∇_x·(a v_pec), the scaled stored-velocity gradient.
    // From 110000 endpoint samples need only q; live rung changes rescale it.
    // From 120000 gradient models retain their optional endpoint Balsara byte.
    // From 130000 two working flags and the proposed rung share one byte.
    return 130000U +
        (options.starFormation.enabled ? 110U : (options.chemistry.enabled ? 100U : 2U)) +
        (options.hydro.viscosityGradientSwitch() && options.hydro.balsaraSwitch ? 1U : 0U) +
        (options.hydro.viscosityEvolution == "hu_etal" ?
                8U :
                (options.hydro.viscosityEvolution == "cullenAndDehnen" ?
                        4U :
                        (options.hydro.viscosityEvolution == "priceEtAl" ?
                                6U :
                                (options.hydro.viscosityEvolution != "none" ? 2U : 0U))));
}

}    // namespace detail

void sphParticlesLoad(FILE* fp) {
    sphTreeInvalidate();
    uint32_t header[7];
    ParticleInteger savedCount;
    FREAD(header, sizeof(uint32_t), 7, fp);
    FREAD(&savedCount, sizeof(ParticleInteger), 1, fp);
    if (header[0] != 0x43545350U || header[1] != detail::sphCheckpointVersion() ||
        header[2] != sizeof(ParticleInteger) || header[3] != sizeof(float) || header[4] != 0U ||
        header[5] != kDimensions || header[6] != sizeof(float) ||
        savedCount != sphParticlesSize()) {
        THROW_ERROR("Incompatible SPH checkpoint payload (requires matching "
                    "integrator/viscosity/chemistry layout)\n");
    }
    uint32_t entropyInitialized;
    FREAD(&entropyInitialized, sizeof(entropyInitialized), 1, fp);
    FREAD(&sphParticlesThermalScaleFactor, sizeof(sphParticlesThermalScaleFactor), 1, fp);
    if (entropyInitialized > 1 || !(sphParticlesThermalScaleFactor > 0.0) ||
        !std::isfinite(sphParticlesThermalScaleFactor)) {
        THROW_ERROR("Invalid SPH entropy checkpoint context\n");
    }
    sphParticlesEntropyInitialized = entropyInitialized != 0;
    detail::visitSphCheckpointArrays(
        [fp](auto* values, size_t count) {
            if (count != 0) {
                FREAD(values, sizeof(*values), count, fp);
            }
        },
        false, false);
    for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
        if (!sphParticlesFlags[index].valid()) {
            THROW_ERROR("Invalid packed SPH flags in checkpoint\n");
        }
    }
    if (getOptions().hydro.viscosityEvolution != "none") {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            FREAD(&sphParticlesBulkViscosityCoefficients[index].bits, sizeof(uint16_t), 1, fp);
            auto const coefficientBits = sphParticlesBulkViscosityCoefficients[index].bits;
            if ((coefficientBits & 0x7c00U) == 0x7c00U ||
                ((coefficientBits & 0x8000U) != 0U && (coefficientBits & 0x7fffU) != 0U)) {
                THROW_ERROR("Invalid SPH viscosity coefficient in checkpoint\n");
            }
            if (getOptions().hydro.viscosityGradientSwitch()) {
                auto& state = sphScaledDivergence[index];
                state = {};
                FREAD(&state.bits, sizeof(uint16_t), 1, fp);
                // Validate IEEE bits before decode: finite-math compiler options
                // must not erase rejection of infinities or NaNs.
                if ((state.bits & 0x7c00U) == 0x7c00U) {
                    THROW_ERROR("Invalid SPH divergence history in checkpoint\n");
                }
                if (getOptions().hydro.balsaraSwitch) {
                    FREAD(&sphParticlesBalsaraFactors[index], sizeof(uint8_t), 1, fp);
                }
            }
        }
    }
    if (getOptions().chemistry.enabled) {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            float electronFraction;
            FREAD(&electronFraction, sizeof(float), 1, fp);
            if (!std::isfinite(electronFraction) || electronFraction < 0.0f) {
                THROW_ERROR("Invalid equilibrium electron fraction in checkpoint\n");
            }
            sphParticlesElectronFractions[index] = electronFraction;
        }
    }
}

void sphParticlesSave(FILE* fp) {
    uint32_t header[7] = {0x43545350U, detail::sphCheckpointVersion(), sizeof(ParticleInteger),
        sizeof(float), 0U, kDimensions, sizeof(float)};
    ParticleInteger count = sphParticlesSize();
    FWRITE(header, sizeof(uint32_t), 7, fp);
    FWRITE(&count, sizeof(ParticleInteger), 1, fp);
    uint32_t const entropyInitialized = sphParticlesEntropyInitialized;
    FWRITE(&entropyInitialized, sizeof(entropyInitialized), 1, fp);
    FWRITE(&sphParticlesThermalScaleFactor, sizeof(sphParticlesThermalScaleFactor), 1, fp);
    detail::visitSphCheckpointArrays(
        [fp](auto* values, size_t count) {
            if (count != 0) {
                FWRITE(values, sizeof(*values), count, fp);
            }
        },
        false, false);
    if (getOptions().hydro.viscosityEvolution != "none") {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            FWRITE(&sphParticlesBulkViscosityCoefficients[index].bits, sizeof(uint16_t), 1, fp);
            if (getOptions().hydro.viscosityGradientSwitch()) {
                auto const& state = sphScaledDivergence[index];
                FWRITE(&state.bits, sizeof(uint16_t), 1, fp);
                if (getOptions().hydro.balsaraSwitch) {
                    FWRITE(&sphParticlesBalsaraFactors[index], sizeof(uint8_t), 1, fp);
                }
            }
        }
    }
    if (getOptions().chemistry.enabled) {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            float const electronFraction = sphParticlesElectronFractions[index];
            FWRITE(&electronFraction, sizeof(float), 1, fp);
        }
    }
}

void sphCheckpointTest() {
    // Every valid combination roundtrips and each setter preserves the other fields.
    for (int encoded = 0; encoded < 256; ++encoded) {
        SphParticleFlags flags;
        flags.bits = uint8_t(encoded);
        int const rungCode = encoded >> 2;
        ALWAYS_ASSERT(flags.valid() == (rungCode < kMaxRung));
        ALWAYS_ASSERT(flags.converged() == ((encoded & 1) != 0));
        ALWAYS_ASSERT(flags.semiactive() == ((encoded & 2) != 0));
        ALWAYS_ASSERT(flags.proposedRung() == rungCode);
        flags.setConverged(!flags.converged());
        ALWAYS_ASSERT(flags.bits == uint8_t(encoded ^ 1));
        flags.setSemiactive(!flags.semiactive());
        ALWAYS_ASSERT(flags.bits == uint8_t(encoded ^ 3));
        for (int rung = 0; rung < kMaxRung; ++rung) {
            flags.setProposedRung(rung);
            ALWAYS_ASSERT(flags.proposedRung() == rung && flags.valid());
            ALWAYS_ASSERT((flags.bits & 3U) == ((encoded ^ 3) & 3U));
        }
    }
    Vector<SphParticleFlags> originalFlags;
    if (sphParticlesSize() != 0) {
        originalFlags.assign(sphParticlesFlags, sphParticlesFlags + sphParticlesSize());
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            sphParticlesFlags[index] = {};
            sphParticlesSetConverged(index, index % 2 != 0);
            sphParticlesSetSemiactive(index, index % 3 != 0);
            sphParticlesSetProposedRung(index, int(index % kMaxRung));
        }
    }
    // Compact history is a scalar; its two bytes are both serialized state.
    auto const visitFields = [](auto&& visit) {
        detail::visitSphCheckpointArrays([&visit](auto* values, size_t count) {
            for (size_t index = 0; index < count; ++index) {
                visit(reinterpret_cast<unsigned char*>(values + index), sizeof(*values), index);
            }
        });
    };
    ALWAYS_ASSERT(sphParticlesSize() == 0 ||
        (sphParticlesBulkViscosityCoefficients != nullptr) ==
            (getOptions().hydro.viscosityEvolution != "none"));
    ALWAYS_ASSERT(sphParticlesSize() == 0 ||
        (sphScaledDivergence != nullptr) == getOptions().hydro.viscosityGradientSwitch());
    Vector<Float16Storage> originalScaledDivergences;
    if (getOptions().hydro.viscosityGradientSwitch() && sphParticlesSize() != 0) {
        originalScaledDivergences.assign(
            sphScaledDivergence, sphScaledDivergence + sphParticlesSize());
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            auto& history = sphScaledDivergence[index];
            history = {};
            float const divergence = -0.5f + 0.025f * index;
            ALWAYS_ASSERT(sphEncodeDivergence(history, divergence, 1.0));
        }
    }
    ALWAYS_ASSERT(
        cullen_and_dehnen::evolvedBulkViscosityCoefficient(0.0f, 1.5f, 2.0f, 0.0f) == 1.5f);
    ALWAYS_ASSERT(
        std::fabs(cullen_and_dehnen::evolvedBulkViscosityCoefficient(1.5f, 0.5f, 2.0f, 0.25f) -
            (0.5f + std::exp(-0.5f))) < 1.0e-6f);
    // Price et al. (2018), equation (50): (1.5 + 0.5 * 0.5) / 1.5 after one backward Euler step.
    ALWAYS_ASSERT(price_etal::evolvedBulkViscosityCoefficient(0.0f, 1.5f, 2.0f, 0.0f) == 1.5f);
    ALWAYS_ASSERT(std::fabs(price_etal::evolvedBulkViscosityCoefficient(1.5f, 0.5f, 2.0f, 0.25f) -
                      1.75f / 1.5f) < 1.0e-6f);
    Vector<float> originalViscosity;
    if (getOptions().hydro.viscosityEvolution != "none" && sphParticlesSize() != 0) {
        originalViscosity.assign(sphParticlesBulkViscosityCoefficients,
            sphParticlesBulkViscosityCoefficients + sphParticlesSize());
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            float const bulkViscosityCoefficient = 0.25f + 0.01f * index;
            sphParticlesBulkViscosityCoefficients[index] = bulkViscosityCoefficient;
        }
    }
    float originalElectronFraction{1.0f};
    bool const testChemistry = getOptions().chemistry.enabled && sphParticlesSize() != 0;
    if (testChemistry) {
        originalElectronFraction = sphParticlesElectronFractions[0];
        sphParticlesElectronFractions[0] = 1.0e-25f;
    }
    Vector<unsigned char> expected;
    FILE* file = std::tmpfile();
    if (file == nullptr) {
        THROW_ERROR("Unable to create SPH checkpoint test file\n");
    }
    sphParticlesSave(file);
    visitFields([&expected](unsigned char* values, size_t bytes, size_t) {
        expected.insert(expected.end(), values, values + bytes);
        std::memset(values, 0x5a, bytes);
    });
    bool const expectedEntropyInitialized = sphParticlesEntropyInitialized;
    double const expectedThermalScaleFactor = sphParticlesThermalScaleFactor;
    sphParticlesEntropyInitialized = !expectedEntropyInitialized;
    sphParticlesThermalScaleFactor = 2.0 * expectedThermalScaleFactor;
    std::rewind(file);
    sphParticlesLoad(file);
    ALWAYS_ASSERT(sphParticlesEntropyInitialized == expectedEntropyInitialized);
    ALWAYS_ASSERT(sphParticlesThermalScaleFactor == expectedThermalScaleFactor);
    size_t offset = 0;
    visitFields([&expected, &offset](unsigned char* values, size_t bytes, size_t) {
        if (std::memcmp(values, expected.data() + offset, bytes) != 0) {
            THROW_ERROR("SPH checkpoint did not restore every gas field\n");
        }
        offset += bytes;
    });
    if (sphParticlesSize() >= 2) {
        visitFields([](unsigned char* values, size_t bytes, size_t index) {
            if (index < 2) {
                std::memset(values, index == 0 ? 0x11 : 0x22, bytes);
            }
        });
        auto checkOrder = [&visitFields](unsigned char first, unsigned char second) {
            visitFields([first, second](unsigned char* values, size_t bytes, size_t index) {
                if (index < 2) {
                    for (size_t i = 0; i < bytes; ++i) {
                        if (values[i] != (index == 0 ? first : second)) {
                            THROW_ERROR("SPH reorder did not preserve every gas field\n");
                        }
                    }
                }
            });
        };
        sphParticlesSwap(0, 1);
        checkOrder(0x22, 0x11);
        sphParticlesSwap(0, 1);
        checkOrder(0x11, 0x22);
        std::rewind(file);
        sphParticlesLoad(file);
        std::cout << "SPH reorder passed (all fields, gravity and predictor stages)\n"
                  << std::flush;
    }
    if (sphParticlesSize() != 0) {
        auto const validFlags = sphParticlesFlags[0];
        for (int const rungCode : {kMaxRung, 63}) {
            if (rungCode < kMaxRung || rungCode >= 64) {
                continue;
            }
            sphParticlesFlags[0].bits = uint8_t(rungCode << 2);
            FILE* corruptFile = std::tmpfile();
            if (corruptFile == nullptr) {
                THROW_ERROR("Unable to create corrupt SPH flags test file\n");
            }
            sphParticlesSave(corruptFile);
            std::rewind(corruptFile);
            bool rejected = false;
            try {
                sphParticlesLoad(corruptFile);
            } catch (std::runtime_error const& error) {
                rejected =
                    std::string(error.what()).find("Invalid packed SPH flags") != std::string::npos;
            }
            std::fclose(corruptFile);
            if (!rejected) {
                THROW_ERROR("SPH checkpoint accepted an invalid packed hydro rung\n");
            }
            sphParticlesFlags[0] = validFlags;
        }
    }
    if (getOptions().hydro.viscosityGradientSwitch() && sphParticlesSize() != 0) {
        auto const validHistory = sphScaledDivergence[0];
        uint16_t const invalidSamples[] = {0x7c00, 0xfc00, 0x7e00, 0xfe00};
        for (auto const invalid : invalidSamples) {
            auto& history = sphScaledDivergence[0];
            history = validHistory;
            history.bits = invalid;
            FILE* corruptFile = std::tmpfile();
            if (corruptFile == nullptr) {
                THROW_ERROR("Unable to create corrupt SPH checkpoint test file\n");
            }
            sphParticlesSave(corruptFile);
            std::rewind(corruptFile);
            bool rejected = false;
            try {
                sphParticlesLoad(corruptFile);
            } catch (std::runtime_error const& error) {
                rejected = std::string(error.what()).find("Invalid SPH divergence history") !=
                    std::string::npos;
            }
            std::fclose(corruptFile);
            if (!rejected) {
                THROW_ERROR("SPH checkpoint accepted invalid compact viscosity history\n");
            }
        }
        std::rewind(file);
        sphParticlesLoad(file);
        std::cout << "SPH checkpoint rejected invalid compact viscosity histories\n" << std::flush;
    }
    ParticleInteger const savedSize = size;
    size = 0;
    std::rewind(file);
    sphParticlesSave(file);
    std::rewind(file);
    sphParticlesLoad(file);
    size = savedSize;
    std::fclose(file);
    if (testChemistry) {
        sphParticlesElectronFractions[0] = originalElectronFraction;
    }
    for (size_t index = 0; index < originalViscosity.size(); ++index) {
        sphParticlesBulkViscosityCoefficients[index] = originalViscosity[index];
    }
    for (size_t index = 0; index < originalScaledDivergences.size(); ++index) {
        sphScaledDivergence[index] = originalScaledDivergences[index];
    }
    for (size_t index = 0; index < originalFlags.size(); ++index) {
        sphParticlesFlags[index] = originalFlags[index];
    }
    std::cout << "SPH checkpoint roundtrip passed (all fields and empty gas payload)\n"
              << std::flush;
}

HPX_PLAIN_ACTION(sphParticlesResetConverged, SphParticlesResetConvergedAction);

void sphParticlesResetConverged() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futures;
    for (auto& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesResetConvergedAction>(c));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads] {
                ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
                ParticleInteger const e =
                    (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
                // Each density pass starts here. Discarding every grad-H correction makes
                // the force pass use only corrections that this pass rebuilt, never one left
                // by an earlier step, a migrated slot, or a restart.
                float const unbuiltGradHCorrection = std::numeric_limits<float>::quiet_NaN();
                for (ParticleInteger i = b; i < e; i++) {
                    sphParticlesSetConverged(i, false);
                    sphParticlesGradHCorrections[i] = unbuiltGradHCorrection;
                }
            },
            "sph.particles.resetConverged")));
    }
    hpx::wait_all(futures.begin(), futures.end());
}

HPX_PLAIN_ACTION(sphParticlesResetSemiactive, SphParticlesResetSemiactiveAction);

void sphParticlesResetSemiactive() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futures;
    for (auto& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesResetSemiactiveAction>(c));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads] {
                ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
                ParticleInteger const e =
                    (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i < e; i++) {
                    sphParticlesSetSemiactive(i, false);
                }
            },
            "sph.particles.resetSemiactive")));
    }
    hpx::wait_all(futures.begin(), futures.end());
}

HPX_PLAIN_ACTION(
    sphParticlesBeginViscosityDiagnostics, SphParticlesBeginViscosityDiagnosticsAction);
HPX_PLAIN_ACTION(sphParticlesEndViscosityDiagnostics, SphParticlesEndViscosityDiagnosticsAction);

void sphParticlesEndViscosityDiagnostics() {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesEndViscosityDiagnosticsAction>(child));
    }
    freeManagedMemory(sphParticlesViscosityDiagnostics);
    sphParticlesViscosityDiagnostics = nullptr;
    hpx::wait_all(pending.begin(), pending.end());
    for (auto& future : pending) {
        future.get();
    }
}

void sphParticlesBeginViscosityDiagnostics() {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesBeginViscosityDiagnosticsAction>(child));
    }
    std::exception_ptr failure;
    try {
        ALWAYS_ASSERT(sphParticlesViscosityDiagnostics == nullptr);
        if (getOptions().hydro.viscosityGradientSwitch() && sphParticlesSize() != 0) {
            sphParticlesViscosityDiagnostics = static_cast<SphViscosityDiagnostics*>(
                allocateManagedMemory(sizeof(SphViscosityDiagnostics) * sphParticlesSize()));
            std::fill_n(
                sphParticlesViscosityDiagnostics, sphParticlesSize(), SphViscosityDiagnostics{});
        }
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& future : pending) {
        try {
            future.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        sphParticlesEndViscosityDiagnostics();
        std::rethrow_exception(failure);
    }
}

HPX_PLAIN_ACTION(sphParticlesUpdateTargetAlpha, SphParticlesUpdateTargetAlphaAction);
void sphParticlesUpdateTargetAlpha(SphRunParameters const& parameters) {
    if (!getOptions().hydro.viscosityGradientSwitch()) {
        return;
    }
    bool const cullenDehnen = getOptions().hydro.viscosityEvolution == "cullenAndDehnen";
    bool const huEtAl = getOptions().hydro.viscosityEvolution == "hu_etal";
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesUpdateTargetAlphaAction>(child, parameters));
    }
    sph_flops::Counter flops;
    for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
        int const rung = sphParticlesRung(index);
        bool const measured = parameters.isActiveRung(rung);
        bool const initializing = parameters.viscosityInitializeHistory;
        auto& history = sphScaledDivergence[index];
        if (measured && sphParticlesConverged(index)) {
            ALWAYS_ASSERT(sphParticlesViscosityDiagnostics != nullptr);
            auto const& diagnostic = sphParticlesViscosityDiagnostics[index];
            double const elapsed = sphViscositySampleInterval(
                rung, parameters.viscositySampleInterval, parameters.globalTimestep, initializing);
            float const timeStep = static_cast<float>(elapsed);
            double const sampleTimestep = initializing ?
                sphViscosityStoredSampleTimestep(
                    rung, parameters.viscosityBaseTimestep, 0.0, parameters.globalTimestep) :
                elapsed;
            if (!sphEncodeDivergence(history, diagnostic.pendingDivergence, sampleTimestep)) {
                THROW_ERROR("SPH timestep-scaled divergence exceeds finite float16 storage\n");
            }
            if (elapsed == 0.0) {
                continue;
            }
            auto& bulkViscosityCoefficient = sphParticlesBulkViscosityCoefficients[index];
            float const previousCoefficient = bulkViscosityCoefficient;
            bool const increasing = previousCoefficient < diagnostic.targetBulkViscosityCoefficient;
            // CD (2010), equation (A15); Hu (2014), equation (17); Price (2018), equation (50).
            bulkViscosityCoefficient = huEtAl ?
                hu_etal::evolvedBulkViscosityCoefficient(previousCoefficient,
                    diagnostic.targetBulkViscosityCoefficient, diagnostic.inverseDecayTime,
                    diagnostic.limiter, timeStep) :
                cullenDehnen ? cullen_and_dehnen::evolvedBulkViscosityCoefficient(
                                   previousCoefficient, diagnostic.targetBulkViscosityCoefficient,
                                   diagnostic.inverseDecayTime, timeStep) :
                               price_etal::evolvedBulkViscosityCoefficient(previousCoefficient,
                                   diagnostic.targetBulkViscosityCoefficient,
                                   diagnostic.inverseDecayTime, timeStep);
            flops += (increasing ? (cullenDehnen || huEtAl ? 0 : 1) : 5) + (huEtAl ? 1 : 0);
        }
    }
    addCpuFlops(flops.counts());
    // Every locality must finish consuming the temporary diagnostics before
    // an exception can unwind the driver's collective scratch-buffer scope.
    hpx::wait_all(pending.begin(), pending.end());
    for (auto& future : pending) {
        future.get();
    }
}
