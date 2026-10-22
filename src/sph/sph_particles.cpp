/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>

#define SPH_PARTICLES_CPP
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

static ParticleInteger capacity = 0;
static ParticleInteger size = 0;

struct LineIdType {
    int proc;
    ParticleInteger index;
    inline bool operator==(LineIdType other) const {
        return proc == other.proc && index == other.index;
    }
};

struct LineIdHash {
    inline size_t operator()(LineIdType id) const {
        ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
        ParticleInteger const i = id.index / lineSize;
        return i * (hpxSize() - 1) + ((id.proc < hpxRank()) ? id.proc : id.proc - 1);
    }
};

struct LineIdHashLo {
    inline size_t operator()(LineIdType id) const {
        LineIdHash hash;
        return hash(id) % PART_CACHE_SIZE;
    }
};

struct LineIdHashHi {
    inline size_t operator()(LineIdType id) const {
        LineIdHash hash;
        return hash(id) / PART_CACHE_SIZE;
    }
};

static Array<Position, NDIM> const* sphParticlesCacheReadLine(LineIdType lineId);
static Pair<float> const* sphParticlesReadThermodynamicCacheLine(LineIdType lineId);
static char const* sphParticlesRungCacheReadLine(LineIdType lineId);
static Array<float, NDIM> const* sphParticlesReadVelocityCacheLine(LineIdType lineId);
static Pair<float> const* sphParticlesReadDensityCacheLine(LineIdType lineId);
static Pair<float> const* sphParticlesReadChemistryEosCacheLine(LineIdType lineId);
static Vector<Pair<float>> sphParticlesFetchChemistryEosCacheLine(ParticleInteger index);

static Vector<Array<Position, NDIM>> sphParticlesFetchCacheLine(ParticleInteger index);
static Vector<char> sphParticlesFetchRungCacheLine(ParticleInteger index);
static Vector<Array<float, NDIM>> sphParticlesFetchVelocityCacheLine(ParticleInteger index);
static Vector<Pair<float>> sphParticlesFetchDensityCacheLine(ParticleInteger index);
static Vector<Pair<float>> sphParticlesFetchThermodynamicCacheLine(ParticleInteger index);

HPX_PLAIN_ACTION(sphParticlesFetchCacheLine, SphParticlesFetchCacheLineAction);
HPX_PLAIN_ACTION(
    sphParticlesFetchThermodynamicCacheLine, SphParticlesFetchThermodynamicCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchRungCacheLine, SphParticlesFetchRungCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchVelocityCacheLine, SphParticlesFetchVelocityCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchDensityCacheLine, SphParticlesFetchDensityCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesClearHydroCaches, SphParticlesClearHydroCachesAction);
HPX_PLAIN_ACTION(sphParticlesClearCaches, SphParticlesClearCachesAction);
HPX_PLAIN_ACTION(sphParticlesRefreshChemistryEos, SphParticlesRefreshChemistryEosAction);
HPX_PLAIN_ACTION(
    sphParticlesFetchChemistryEosCacheLine, SphParticlesFetchChemistryEosCacheLineAction);

static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Array<Position, NDIM>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    particleCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Pair<float>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    thermodynamicParticleCache;
static Array<
    std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<char>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    rungParticleCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Array<float, NDIM>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    particleVelocityCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Pair<float>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    densityParticleCache;
static Array<std::unordered_map<LineIdType,
                 std::shared_ptr<hpx::shared_future<Vector<Pair<float>>>>, LineIdHashHi>,
    PART_CACHE_SIZE>
    chemistryEosParticleCache;
static Array<SpinlockType, PART_CACHE_SIZE> chemistryEosMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> mutexes;
static Array<SpinlockType, PART_CACHE_SIZE> energyMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> rungMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> velocityMutexes;
static Array<SpinlockType, PART_CACHE_SIZE> densityMutexes;

template <typename Action, typename T>
static T const* readSphCacheLine(LineIdType lineId,
    Array<std::unordered_map<LineIdType, std::shared_ptr<hpx::shared_future<Vector<T>>>,
              LineIdHashHi>,
        PART_CACHE_SIZE>& cache,
    Array<SpinlockType, PART_CACHE_SIZE>& locks) {
    size_t const bin = LineIdHashLo()(lineId);
    std::shared_ptr<hpx::shared_future<Vector<T>>> future;
    {
        std::lock_guard<SpinlockType> lock(locks[bin]);
        auto const entry = cache[bin].find(lineId);
        if (entry != cache[bin].end()) {
            future = entry->second;
        }
    }
    if (!future) {
        auto promise = std::make_shared<hpx::promise<Vector<T>>>();
        auto candidate =
            std::make_shared<hpx::shared_future<Vector<T>>>(promise->get_future().share());
        bool inserted;
        {
            std::lock_guard<SpinlockType> lock(locks[bin]);
            auto const entry = cache[bin].emplace(lineId, candidate);
            inserted = entry.second;
            future = entry.first->second;
        }
        if (inserted) {
            try {
                hpx::post([promise, lineId]() {
                    try {
                        auto remote =
                            hpx::async<Action>(hpxLocalities()[lineId.proc], lineId.index);
                        promise->set_value(remote.get());
                    } catch (...) {
                        promise->set_exception(std::current_exception());
                    }
                });
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    }
    return future->get().data();
}

HPX_PLAIN_ACTION(sphParticlesApplyUpdates, SphParticlesApplyUpdatesAction);
void sphParticlesApplyUpdates(int minimumRung, int phase, float timeStepScale, float time) {
    // At time zero no step has opened yet, so there is no half-kick to close or correct.
    if (!(phase == 2 || time > 0.0)) {
        return;
    }
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futures;
    for (auto& child : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesApplyUpdatesAction>(
            child, minimumRung, phase, timeStepScale, time));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int threadIndex = 0; threadIndex < numberOfThreads; threadIndex++) {
        futures.push_back(
            hpx::async([timeStepScale, numberOfThreads, threadIndex, phase, minimumRung]() {
                ParticleInteger const rangeBegin =
                    (size_t) threadIndex * sphParticlesSize() / numberOfThreads;
                ParticleInteger const rangeEnd =
                    (size_t) (threadIndex + 1) * sphParticlesSize() / numberOfThreads;
                FlopCounter<> flops;
                for (int gasIndex = rangeBegin; gasIndex < rangeEnd; gasIndex++) {
                    ParticleInteger const gravityIndex = sphParticlesParticleIndex(gasIndex);
                    auto const rung = sphParticlesRung(gasIndex);
                    float const halfTimeStep = 0.5f * timeStepScale / (1 << rung);
                    flops += 2;
                    // Each active particle gets two half-kicks per step: one closes the step
                    // that just drifted, one opens the next. Hydro rates depend on the
                    // velocities and energies they update, so the closing half-kick is first
                    // predicted from the previous rates and then corrected with the new ones.
                    if (phase == 0) {
                        // Predictor (start of sphStep2, before the hydro force run): close
                        // the previous step's half-kick with the rates computed last step,
                        // so the hydro run sees velocities and energies at the end of the
                        // drift. Save those rates so phase 1 can take this kick back out.
                        if (rung >= minimumRung) {
                            flops += 2 * NDIM + 2;
                            for (int dimension = 0; dimension < NDIM; dimension++) {
                                sphParticlesPredictedAcceleration(dimension, gasIndex) =
                                    sphParticlesAcceleration(dimension, gasIndex);
                            }
                            for (int dimension = 0; dimension < NDIM; dimension++) {
                                particlesVel(dimension, gravityIndex) +=
                                    sphParticlesAcceleration(dimension, gasIndex) * halfTimeStep;
                            }
                            ALWAYS_ASSERT(sphParticlesInternalEnergy(gasIndex) > 0.0);
                            sphParticlesPredictedInternalEnergyRate(gasIndex) =
                                sphParticlesInternalEnergyRate(gasIndex);
                            sphParticlesInternalEnergy(gasIndex) +=
                                sphParticlesInternalEnergyRate(gasIndex) * halfTimeStep;
                            ALWAYS_ASSERT(sphParticlesInternalEnergy(gasIndex) > 0.0);
                        }
                    } else if (phase == 1) {
                        // Corrector (after the hydro force run): replace the predicted
                        // half-kick with one from the freshly computed rates. Velocity is
                        // corrected by the rate difference; energy removes the predicted
                        // increment and adds the new one in two steps so a non-positive
                        // energy after either step is reported before the assertion fires.
                        // Particles are now synchronized for the energy diagnostics.
                        if (rung >= minimumRung) {
                            flops += 3 * NDIM + 4;
                            for (int dimension = 0; dimension < NDIM; dimension++) {
                                particlesVel(dimension, gravityIndex) +=
                                    (sphParticlesAcceleration(dimension, gasIndex) -
                                        sphParticlesPredictedAcceleration(dimension, gasIndex)) *
                                    halfTimeStep;
                            }
                            sphParticlesInternalEnergy(gasIndex) +=
                                (-sphParticlesPredictedInternalEnergyRate(gasIndex)) * halfTimeStep;
                            if (sphParticlesInternalEnergy(gasIndex) <= 0.0) {
                                PRINT("%e %e\n", sphParticlesInternalEnergy(gasIndex),
                                    (-sphParticlesPredictedInternalEnergyRate(gasIndex)) *
                                        halfTimeStep);
                            }
                            ALWAYS_ASSERT(sphParticlesInternalEnergy(gasIndex) > 0.0);
                            sphParticlesInternalEnergy(gasIndex) +=
                                (sphParticlesInternalEnergyRate(gasIndex)) * halfTimeStep;
                            if (sphParticlesInternalEnergy(gasIndex) <= 0.0) {
                                PRINT("%e %e\n", sphParticlesInternalEnergy(gasIndex),
                                    (sphParticlesInternalEnergyRate(gasIndex)) * halfTimeStep);
                            }
                            ALWAYS_ASSERT(sphParticlesInternalEnergy(gasIndex) > 0.0);
                        }
                    } else if (phase == 2) {
                        // Opening half-kick (after SPH_RUN_RUNGS assigns new rungs): start
                        // the next step with the synchronized rates and the new rung's
                        // half-step, ahead of the drift. The next step's phase 0 closes it.
                        if (rung >= minimumRung) {
                            flops += 2 * NDIM + 2;
                            for (int dimension = 0; dimension < NDIM; dimension++) {
                                particlesVel(dimension, gravityIndex) +=
                                    sphParticlesAcceleration(dimension, gasIndex) * halfTimeStep;
                            }
                            ALWAYS_ASSERT(sphParticlesInternalEnergy(gasIndex) > 0.0);
                            sphParticlesInternalEnergy(gasIndex) +=
                                sphParticlesInternalEnergyRate(gasIndex) * halfTimeStep;
                            ALWAYS_ASSERT(sphParticlesInternalEnergy(gasIndex) > 0.0);
                        }
                    }
                }
                addCpuFlops(flops);
            }));
    }
    for (auto& future : futures) {
        future.get();
    }
    profilerExit();
}

namespace detail {

template <typename Visitor>
void visitSphCheckpointArrays(Visitor&& visit, bool includeChemistry = true) {
    size_t const count = sphParticlesSize();
    // Keep stage fields as well as the physical fields: fine rungs may be staggered at checkpoints.
    visit(sphParticlesThermodynamicStates, count);
    visit(sphParticlesAccelerationStates, count);
    visit(sphParticlesParticleIndices, count);
    visit(sphParticlesPredictedInternalEnergyRates, count);
    visit(sphParticlesInternalEnergyRates, count);
    visit(sphParticlesDensities, count);
    visit(sphParticlesConvergenceFlags, count);
    visit(sphParticlesSemiactiveFlags, count);
    for (int dim = 0; dim < NDIM; dim++) {
        visit(sphParticlesPredictedAccelerations[dim], count);
        visit(sphParticlesGravityAccelerations[dim], count);
    }
    if (includeChemistry && sphParticlesChemistryStates != nullptr) {
        visit(sphParticlesChemistryStates, count);
        visit(sphParticlesChemistryEos, count);
    }
}

}    // namespace detail

void sphParticlesSwap(ParticleInteger i, ParticleInteger j) {
    std::swap(sphParticlesBalsaraFactors[i], sphParticlesBalsaraFactors[j]);
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
        hpx_copy(PAR_EXECUTION_POLICY, ptr, ptr + size, newPtr).get();
        freeManagedMemory(ptr);
    }
    ptr = newPtr;
}

void sphParticlesResize(ParticleInteger particleCount, bool resizeGravityParticles) {
    if (particleCount > capacity) {
        ParticleInteger newCapacity = std::max(capacity, (ParticleInteger) 100);
        while (newCapacity < particleCount) {
            newCapacity = size_t(101) * newCapacity / size_t(100);
        }
        sphParticlesArrayResize(sphParticlesDensities, newCapacity);
        sphParticlesArrayResize(sphParticlesBalsaraFactors, newCapacity);
        sphParticlesArrayResize(sphParticlesThermodynamicStates, newCapacity);
        sphParticlesArrayResize(sphParticlesAccelerationStates, newCapacity);
        sphParticlesArrayResize(sphParticlesConvergenceFlags, newCapacity);
        sphParticlesArrayResize(sphParticlesSemiactiveFlags, newCapacity);
        sphParticlesArrayResize(sphParticlesParticleIndices, newCapacity);
        sphParticlesArrayResize(sphParticlesPredictedInternalEnergyRates, newCapacity);
        sphParticlesArrayResize(sphParticlesInternalEnergyRates, newCapacity);
        for (int dim = 0; dim < NDIM; dim++) {
            sphParticlesArrayResize(sphParticlesPredictedAccelerations[dim], newCapacity);
            sphParticlesArrayResize(sphParticlesGravityAccelerations[dim], newCapacity);
        }
        if (getOptions().chemistry.enabled) {
            sphParticlesArrayResize(sphParticlesChemistryStates, newCapacity);
            sphParticlesArrayResize(sphParticlesChemistryEos, newCapacity);
        }
        capacity = newCapacity;
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
        sphParticlesThermodynamicStates[slot] = {};
        if (getOptions().chemistry.enabled) {
            sphParticlesChemistryStates[slot] = {};
            sphParticlesChemistryEos[slot] = {};
        }
        sphParticlesAccelerationStates[slot] = {};
        sphParticlesParticleIndices[slot] = NO_INDEX;
        sphParticlesPredictedInternalEnergyRates[slot] = 0.0f;
        sphParticlesInternalEnergyRates[slot] = 0.0f;
        sphParticlesDensities[slot] = 0.0f;
        sphParticlesBalsaraFactors[slot] = 0.0f;
        sphParticlesConvergenceFlags[slot] = 0;
        sphParticlesSemiactiveFlags[slot] = 0;
        for (int dim = 0; dim < NDIM; dim++) {
            sphParticlesPredictedAccelerations[dim][slot] = 0.0f;
            sphParticlesGravityAccelerations[dim][slot] = 0.0f;
        }
        if (resizeGravityParticles) {
            particlesCatIndex(offset + i) = previousSize + i;
            particlesType(offset + i) = SPH_TYPE;
            sphParticlesParticleIndex(previousSize + i) = offset + i;
        }
    }
}

void sphParticlesFree() {
    freeManagedMemory(sphParticlesBalsaraFactors);
    sphParticlesBalsaraFactors = nullptr;
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
        futures.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                int const j = sphParticlesParticleIndex(i);
                particlesCatIndex(j) = i;
            }
        }));
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
                x[j] = sphParticlesPosition(XDIM, i);
                y[j] = sphParticlesPosition(YDIM, i);
                z[j] = sphParticlesPosition(ZDIM, i);
            }
        } else {
            LineIdType lineId;
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
                    x[destIndex] = ptr[srcIndex][XDIM];
                    y[destIndex] = ptr[srcIndex][YDIM];
                    z[destIndex] = ptr[srcIndex][ZDIM];
                    destIndex++;
                }
            }
        }
    }
}

static Array<Position, NDIM> const* sphParticlesCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchCacheLineAction>(lineId, particleCache, mutexes);
}

static Vector<Array<Position, NDIM>> sphParticlesFetchCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Array<Position, NDIM>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            line[i - begin][dim] = sphParticlesPosition(dim, i);
        }
    }
    return line;
}

void sphParticlesGlobalReadInternalEnergyAndSupportRadius(ParticleGlobalRange range,
    float* internalEnergy, float* supportRadius, ParticleInteger offset) {
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
                if (supportRadius) {
                    supportRadius[j] = sphParticlesSupportRadius(i);
                }
            }
        } else {
            LineIdType lineId;
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
                    if (supportRadius) {
                        supportRadius[destIndex] = ptr[srcIndex].second;
                    }
                    destIndex++;
                }
            }
        }
    }
}

static Pair<float> const* sphParticlesReadThermodynamicCacheLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchThermodynamicCacheLineAction>(
        lineId, thermodynamicParticleCache, energyMutexes);
}

static Vector<Pair<float>> sphParticlesFetchThermodynamicCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Pair<float>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin].first = sphParticlesInternalEnergy(i);
        line[i - begin].second = sphParticlesSupportRadius(i);
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
            LineIdType lineId;
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

static char const* sphParticlesRungCacheReadLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchRungCacheLineAction>(
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
                int const k = sphParticlesParticleIndex(i);
                vx[j] = particlesVel(XDIM, k);
                vy[j] = particlesVel(YDIM, k);
                vz[j] = particlesVel(ZDIM, k);
            }
        } else {
            LineIdType lineId;
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
                    vx[destIndex] = ptr[srcIndex][XDIM];
                    vy[destIndex] = ptr[srcIndex][YDIM];
                    vz[destIndex] = ptr[srcIndex][ZDIM];
                    destIndex++;
                }
            }
        }
    }
}

static Array<float, NDIM> const* sphParticlesReadVelocityCacheLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchVelocityCacheLineAction>(
        lineId, particleVelocityCache, velocityMutexes);
}

static Vector<Array<float, NDIM>> sphParticlesFetchVelocityCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Array<float, NDIM>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        int const k = sphParticlesParticleIndex(i);
        line[i - begin][XDIM] = particlesVel(XDIM, k);
        line[i - begin][YDIM] = particlesVel(YDIM, k);
        line[i - begin][ZDIM] = particlesVel(ZDIM, k);
    }
    return line;
}

void sphParticlesGlobalReadDensity(
    ParticleGlobalRange range, float* rho, ParticleInteger offset, float* balsaraFactor) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            ParticleInteger const dif = offset - range.range.first;
            ParticleInteger const particleCount = range.range.second - range.range.first;
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                int const j = offset + i - range.range.first;
                rho[j] = sphParticlesDensity(i);
                if (balsaraFactor) {
                    balsaraFactor[j] = sphParticlesBalsaraFactors[i];
                }
            }
        } else {
            LineIdType lineId;
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
                    rho[destIndex] = ptr[srcIndex].first;
                    if (balsaraFactor) {
                        balsaraFactor[destIndex] = ptr[srcIndex].second;
                    }
                    destIndex++;
                }
            }
        }
    }
}

static Pair<float> const* sphParticlesReadDensityCacheLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchDensityCacheLineAction>(
        lineId, densityParticleCache, densityMutexes);
}

static Vector<Pair<float>> sphParticlesFetchDensityCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Pair<float>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = {sphParticlesDensity(i), sphParticlesBalsaraFactors[i]};
    }
    return line;
}

void sphParticlesGlobalReadChemistryEos(ParticleGlobalRange range, float* pressureEnergyRatio,
    float* adiabaticIndex, ParticleInteger offset) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first != range.range.second) {
        if (range.proc == hpxRank()) {
            for (ParticleInteger i = range.range.first; i < range.range.second; i++) {
                ParticleInteger const j = offset + i - range.range.first;
                pressureEnergyRatio[j] = sphParticlesChemistryEos[i].first;
                adiabaticIndex[j] = sphParticlesChemistryEos[i].second;
            }
        } else {
            LineIdType lineId;
            lineId.proc = range.proc;
            ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
            ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
            ParticleInteger destIndex = offset;
            for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
                lineId.index = line;
                auto const* ptr = sphParticlesReadChemistryEosCacheLine(lineId);
                auto const begin = std::max(lineId.index, range.range.first);
                auto const end = std::min(lineId.index + lineSize, range.range.second);
                for (ParticleInteger i = begin; i < end; i++) {
                    ParticleInteger const srcIndex = i - lineId.index;
                    pressureEnergyRatio[destIndex] = ptr[srcIndex].first;
                    adiabaticIndex[destIndex] = ptr[srcIndex].second;
                    destIndex++;
                }
            }
        }
    }
}

static Pair<float> const* sphParticlesReadChemistryEosCacheLine(LineIdType lineId) {
    return readSphCacheLine<SphParticlesFetchChemistryEosCacheLineAction>(
        lineId, chemistryEosParticleCache, chemistryEosMutexes);
}

static Vector<Pair<float>> sphParticlesFetchChemistryEosCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Pair<float>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        line[i - begin] = sphParticlesChemistryEos[i];
    }
    return line;
}

void sphParticlesRefreshChemistryEos(double scaleFactor) {
    if (!getOptions().chemistry.enabled) {
        return;
    }
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesRefreshChemistryEosAction>(child, scaleFactor));
    }
    for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
        auto const eos = sphChemistryEos(
            sphParticlesChemistryStates[index], sphParticlesInternalEnergy(index), scaleFactor);
        sphParticlesChemistryEos[index] = {eos.pressureEnergyRatio, eos.adiabaticIndex};
    }
    chemistryEosParticleCache = decltype(chemistryEosParticleCache)();
    for (auto& future : futures) {
        future.get();
    }
}

void sphParticlesClearHydroCaches() {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futures;
    for (auto const& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesClearHydroCachesAction>(c));
    }
    thermodynamicParticleCache = decltype(thermodynamicParticleCache)();
    chemistryEosParticleCache = decltype(chemistryEosParticleCache)();
    particleVelocityCache = decltype(particleVelocityCache)();
    hpx::wait_all(futures.begin(), futures.end());
    profilerExit();
}

void sphParticlesClearCaches() {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futures;
    for (auto const& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesClearCachesAction>(c));
    }
    particleCache = decltype(particleCache)();
    thermodynamicParticleCache = decltype(thermodynamicParticleCache)();
    chemistryEosParticleCache = decltype(chemistryEosParticleCache)();
    rungParticleCache = decltype(rungParticleCache)();
    particleVelocityCache = decltype(particleVelocityCache)();
    densityParticleCache = decltype(densityParticleCache)();
    hpx::wait_all(futures.begin(), futures.end());
    profilerExit();
}

void sphParticlesLoad(FILE* fp) {
    uint32_t header[7];
    ParticleInteger savedCount;
    FREAD(header, sizeof(uint32_t), 7, fp);
    FREAD(&savedCount, sizeof(ParticleInteger), 1, fp);
    if (header[0] != 0x43545350U || header[1] != (getOptions().chemistry.enabled ? 3U : 2U) ||
        header[2] != sizeof(ParticleInteger) || header[3] != sizeof(SphThermodynamicState) ||
        header[4] != sizeof(SphAcceleration) || header[5] != NDIM || header[6] != sizeof(float) ||
        savedCount != sphParticlesSize()) {
        THROW_ERROR(
            "Incompatible SPH checkpoint payload (requires matching Monaghan/chemistry layout)\n");
    }
    detail::visitSphCheckpointArrays(
        [fp](auto* values, size_t count) {
            if (count != 0) {
                FREAD(values, sizeof(*values), count, fp);
            }
        },
        false);
    if (getOptions().chemistry.enabled) {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            float words[primordial::kAbundanceCount + 3];
            FREAD(words, sizeof(float), primordial::kAbundanceCount + 3, fp);
            auto& state = sphParticlesChemistryStates[index];
            for (int species = 0; species < primordial::kAbundanceCount; ++species) {
                state.abundances[species].setRaw(words[species]);
            }
            state.thermalEnergy = words[primordial::kAbundanceCount];
            sphParticlesChemistryEos[index] = {
                words[primordial::kAbundanceCount + 1], words[primordial::kAbundanceCount + 2]};
        }
    }
}

void sphParticlesSave(FILE* fp) {
    uint32_t header[7] = {0x43545350U, getOptions().chemistry.enabled ? 3U : 2U,
        sizeof(ParticleInteger), sizeof(SphThermodynamicState), sizeof(SphAcceleration), NDIM,
        sizeof(float)};
    ParticleInteger count = sphParticlesSize();
    FWRITE(header, sizeof(uint32_t), 7, fp);
    FWRITE(&count, sizeof(ParticleInteger), 1, fp);
    detail::visitSphCheckpointArrays(
        [fp](auto* values, size_t count) {
            if (count != 0) {
                FWRITE(values, sizeof(*values), count, fp);
            }
        },
        false);
    if (getOptions().chemistry.enabled) {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            float words[primordial::kAbundanceCount + 3];
            auto const& state = sphParticlesChemistryStates[index];
            for (int species = 0; species < primordial::kAbundanceCount; ++species) {
                words[species] = state.abundances[species].raw();
            }
            words[primordial::kAbundanceCount] = state.thermalEnergy;
            words[primordial::kAbundanceCount + 1] = sphParticlesChemistryEos[index].first;
            words[primordial::kAbundanceCount + 2] = sphParticlesChemistryEos[index].second;
            FWRITE(words, sizeof(float), primordial::kAbundanceCount + 3, fp);
        }
    }
}

void sphCheckpointTest() {
    primordial::State originalChemicalState{};
    bool const testChemistry = getOptions().chemistry.enabled && sphParticlesSize() != 0;
    if (testChemistry) {
        originalChemicalState = sphParticlesChemistryStates[0];
        // A fully ionized fraction is encoded as negative zero. Checkpoint I/O
        // must retain the branch bit even in the application's fast-math build.
        sphParticlesChemistryStates[0].abundances[0] =
            primordial::SignedAbundance::fromMinority(0.0f, true);
        sphParticlesChemistryStates[0].abundances[1] =
            primordial::SignedAbundance::fromMinority(1.0e-25f, true);
    }
    Vector<unsigned char> expected;
    FILE* file = std::tmpfile();
    if (file == nullptr) {
        THROW_ERROR("Unable to create SPH checkpoint test file\n");
    }
    sphParticlesSave(file);
    detail::visitSphCheckpointArrays([&expected](auto* values, size_t count) {
        size_t const bytes = sizeof(*values) * count;
        if (bytes != 0) {
            auto* data = reinterpret_cast<unsigned char*>(values);
            expected.insert(expected.end(), data, data + bytes);
            std::memset(data, 0x5a, bytes);
        }
    });
    std::rewind(file);
    sphParticlesLoad(file);
    size_t offset = 0;
    detail::visitSphCheckpointArrays([&expected, &offset](auto* values, size_t count) {
        size_t const bytes = sizeof(*values) * count;
        if (bytes != 0) {
            if (std::memcmp(values, expected.data() + offset, bytes) != 0) {
                THROW_ERROR("SPH checkpoint did not restore every gas field\n");
            }
            offset += bytes;
        }
    });
    if (sphParticlesSize() >= 2) {
        detail::visitSphCheckpointArrays([](auto* values, size_t) {
            std::memset(values, 0x11, sizeof(*values));
            std::memset(values + 1, 0x22, sizeof(*values));
        });
        auto checkOrder = [](unsigned char first, unsigned char second) {
            detail::visitSphCheckpointArrays([first, second](auto* values, size_t) {
                auto const* bytes = reinterpret_cast<unsigned char const*>(values);
                for (size_t i = 0; i < sizeof(*values); i++) {
                    if (bytes[i] != first || bytes[sizeof(*values) + i] != second) {
                        THROW_ERROR("SPH reorder did not preserve every gas field\n");
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
        PRINT("SPH reorder passed (all fields, gravity and predictor stages)\n");
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
        sphParticlesChemistryStates[0] = originalChemicalState;
    }
    PRINT("SPH checkpoint roundtrip passed (all fields and empty gas payload)\n");
}

HPX_PLAIN_ACTION(sphParticlesResetConverged, SphParticlesResetConvergedAction);

void sphParticlesResetConverged() {
    int const numberOfThreads = hpxHardwareConcurrency();
    std::vector<hpx::future<void>> futures;
    for (auto& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesResetConvergedAction>(c));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futures.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                sphParticlesConverged(i) = false;
            }
        }));
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
        futures.push_back(hpx::async([proc, numberOfThreads] {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                sphParticlesSemiactive(i) = false;
            }
        }));
    }
    hpx::wait_all(futures.begin(), futures.end());
}
