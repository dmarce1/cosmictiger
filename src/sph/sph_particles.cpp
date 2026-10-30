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
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <hpx/threading_base/annotated_function.hpp>

#define SPH_PARTICLES_CPP
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/sph/cullen_and_dehnen/particle_exchange.hpp>
#include <cosmictiger/sph/cullen_and_dehnen/viscosity.hpp>
#include <cosmictiger/sph/morris_and_monaghan/viscosity.hpp>
#include <cosmictiger/sph/price_etal/viscosity.hpp>
#include <cosmictiger/sph/sph.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/sph/sph_chemistry.hpp>
#endif
#include <cosmictiger/sph/sph_particles.hpp>

#include "particle_cache.hpp"

static ParticleInteger capacity = 0;
static ParticleInteger size = 0;

static Array<Position, kDimensions> const* sphParticlesCacheReadLine(detail::SphCacheLineId lineId);
static Pair<float> const* sphParticlesReadThermodynamicCacheLine(detail::SphCacheLineId lineId);
static char const* sphParticlesRungCacheReadLine(detail::SphCacheLineId lineId);
static Array<float, kDimensions> const* sphParticlesReadVelocityCacheLine(
    detail::SphCacheLineId lineId);
struct SphDensityCacheEntry {
    float density{};
    float balsaraFactor{};
    float viscosityAlpha{};
    float gradHCorrection{};
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & density & balsaraFactor;
        if (getOptions().hydro.viscosityEvolution != "none") {
            archive & viscosityAlpha;
        }
        archive & gradHCorrection;
    }
};

static SphDensityCacheEntry const* sphParticlesReadDensityCacheLine(detail::SphCacheLineId lineId);
static Pair<float> const* sphParticlesReadChemistryEosCacheLine(detail::SphCacheLineId lineId);
static Vector<Pair<float>> sphParticlesFetchChemistryEosCacheLine(ParticleInteger index);

static Vector<Array<Position, kDimensions>> sphParticlesFetchCacheLine(ParticleInteger index);
static Vector<char> sphParticlesFetchRungCacheLine(ParticleInteger index);
static Vector<Array<float, kDimensions>> sphParticlesFetchVelocityCacheLine(ParticleInteger index);
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
#if COSMICTIGER_ENABLE_CHEMISTRY
HPX_PLAIN_ACTION(sphParticlesRefreshChemistryEos, SphParticlesRefreshChemistryEosAction);
#endif
HPX_PLAIN_ACTION(
    sphParticlesFetchChemistryEosCacheLine, SphParticlesFetchChemistryEosCacheLineAction);

static Array<std::unordered_map<detail::SphCacheLineId,
                 std::shared_ptr<hpx::shared_future<Vector<Array<Position, kDimensions>>>>,
                 detail::SphCacheLineHashHi>,
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
static Array<std::unordered_map<detail::SphCacheLineId,
                 std::shared_ptr<hpx::shared_future<Vector<Array<float, kDimensions>>>>,
                 detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    particleVelocityCache;
static Array<std::unordered_map<detail::SphCacheLineId,
                 std::shared_ptr<hpx::shared_future<Vector<SphDensityCacheEntry>>>,
                 detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    densityParticleCache;
static Array<
    std::unordered_map<detail::SphCacheLineId,
        std::shared_ptr<hpx::shared_future<Vector<Pair<float>>>>, detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    chemistryEosParticleCache;
static Array<SpinlockType, kParticleCacheSize> chemistryEosMutexes;
static Array<SpinlockType, kParticleCacheSize> mutexes;
static Array<SpinlockType, kParticleCacheSize> energyMutexes;
static Array<SpinlockType, kParticleCacheSize> rungMutexes;
static Array<SpinlockType, kParticleCacheSize> velocityMutexes;
static Array<SpinlockType, kParticleCacheSize> densityMutexes;
HPX_PLAIN_ACTION(sphParticlesApplyUpdates, SphParticlesApplyUpdatesAction);
void sphParticlesApplyUpdates(
    int minimumRung, int phase, float timeStepScale, float time, double explicitTimestep) {
    // At time zero no step has opened yet, so there is no half-kick to close or correct.
    if (!(phase == 2 || time > 0.0)) {
        return;
    }
    Vector<hpx::future<void>> futures;
    for (auto& child : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesApplyUpdatesAction>(
            child, minimumRung, phase, timeStepScale, time, explicitTimestep));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int threadIndex = 0; threadIndex < numberOfThreads; threadIndex++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [timeStepScale, numberOfThreads, threadIndex, phase, minimumRung, explicitTimestep]() {
                ParticleInteger const rangeBegin =
                    (size_t) threadIndex * sphParticlesSize() / numberOfThreads;
                ParticleInteger const rangeEnd =
                    (size_t) (threadIndex + 1) * sphParticlesSize() / numberOfThreads;
                FlopCounter<> flops;
                for (int gasIndex = rangeBegin; gasIndex < rangeEnd; gasIndex++) {
                    ParticleInteger const gravityIndex = sphParticlesParticleIndex(gasIndex);
                    auto const rung = sphParticlesRung(gasIndex);
                    double const halfTimeStep = explicitTimestep >= 0.0 ?
                        0.5 * explicitTimestep :
                        std::ldexp(double(timeStepScale), -1 - int(rung));
                    flops += 2;
                    if (getOptions().hydro.viscosityEvolution == "morrisAndMonaghan" &&
                        rung >= minimumRung) {
                        morris_and_monaghan::applyUpdate(sphParticlesViscosityStates[gasIndex],
                            getOptions().hydro.viscosityAlphaMin, halfTimeStep, phase);
                        flops += 2;
                    }
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
                            flops += 2 * kDimensions + 2;
                            for (int dimension = 0; dimension < kDimensions; dimension++) {
                                sphParticlesPredictedAcceleration(dimension, gasIndex) =
                                    sphParticlesAcceleration(dimension, gasIndex);
                            }
                            for (int dimension = 0; dimension < kDimensions; dimension++) {
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
                            flops += 3 * kDimensions + 4;
                            for (int dimension = 0; dimension < kDimensions; dimension++) {
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
                            flops += 2 * kDimensions + 2;
                            for (int dimension = 0; dimension < kDimensions; dimension++) {
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
            },
            phase == 0     ? "sph.particles.predictKick" :
                phase == 1 ? "sph.particles.correctKick" :
                phase == 2 ? "sph.particles.openKick" :
                             "sph.particles.unknownKickPhase")));
    }
    for (auto& future : futures) {
        future.get();
    }
}

namespace detail {

template <typename Visitor>
void visitSphCheckpointArrays(
    Visitor&& visit, bool includeChemistry = true, bool includeViscosity = true) {
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
    for (int dim = 0; dim < kDimensions; dim++) {
        visit(sphParticlesPredictedAccelerations[dim], count);
        visit(sphParticlesGravityAccelerations[dim], count);
    }
    if (includeViscosity && sphParticlesViscosityStates != nullptr) {
        visit(sphParticlesViscosityStates, count);
    }
    if (includeViscosity && sphParticlesCullenDehnenStates != nullptr) {
        visit(sphParticlesCullenDehnenStates, count);
    }
    if (includeChemistry && sphParticlesChemistryStates != nullptr) {
        visit(sphParticlesChemistryStates, count);
        visit(sphParticlesChemistrySteps, count);
    }
    if (includeChemistry && sphParticlesChemistryEos != nullptr) {
        visit(sphParticlesChemistryEos, count);
    }
}

}    // namespace detail

void sphParticlesSwap(ParticleInteger i, ParticleInteger j) {
    if (sphParticlesBalsaraFactors != nullptr) {
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
    bool const initializeBalsara =
        getOptions().hydro.balsaraSwitch && sphParticlesBalsaraFactors == nullptr;
    bool const gradientSwitch = getOptions().hydro.viscosityGradientSwitch();
    bool const initializeCullenDehnen = gradientSwitch && sphParticlesCullenDehnenStates == nullptr;
    bool const initializeViscosity =
        getOptions().hydro.viscosityEvolution != "none" && sphParticlesViscosityStates == nullptr;
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
        if (getOptions().hydro.viscosityEvolution != "none") {
            sphParticlesArrayResize(sphParticlesViscosityStates, newCapacity);
        }
        if (gradientSwitch) {
            sphParticlesArrayResize(sphParticlesCullenDehnenStates, newCapacity);
        }
        sphParticlesArrayResize(sphParticlesThermodynamicStates, newCapacity);
        sphParticlesArrayResize(sphParticlesAccelerationStates, newCapacity);
        sphParticlesArrayResize(sphParticlesConvergenceFlags, newCapacity);
        sphParticlesArrayResize(sphParticlesSemiactiveFlags, newCapacity);
        sphParticlesArrayResize(sphParticlesParticleIndices, newCapacity);
        sphParticlesArrayResize(sphParticlesPredictedInternalEnergyRates, newCapacity);
        sphParticlesArrayResize(sphParticlesInternalEnergyRates, newCapacity);
        for (int dim = 0; dim < kDimensions; dim++) {
            sphParticlesArrayResize(sphParticlesPredictedAccelerations[dim], newCapacity);
            sphParticlesArrayResize(sphParticlesGravityAccelerations[dim], newCapacity);
        }
        if (getOptions().chemistry.enabled) {
            sphParticlesArrayResize(sphParticlesChemistryStates, newCapacity);
            sphParticlesArrayResize(sphParticlesChemistrySteps, newCapacity);
        }
        if (getOptions().chemistry.enabled || getOptions().starFormation.enabled) {
            sphParticlesArrayResize(sphParticlesChemistryEos, newCapacity);
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
            sphParticlesBalsaraFactors[index] = 0.0f;
        }
    }
    if (getOptions().hydro.viscosityEvolution == "none" && sphParticlesViscosityStates != nullptr) {
        freeManagedMemory(sphParticlesViscosityStates);
        sphParticlesViscosityStates = nullptr;
    } else if (getOptions().hydro.viscosityEvolution != "none" &&
        sphParticlesViscosityStates == nullptr && capacity > 0) {
        sphParticlesArrayResize(sphParticlesViscosityStates, capacity);
    }
    if (!gradientSwitch && sphParticlesCullenDehnenStates != nullptr) {
        freeManagedMemory(sphParticlesCullenDehnenStates);
        sphParticlesCullenDehnenStates = nullptr;
    } else if (gradientSwitch && sphParticlesCullenDehnenStates == nullptr && capacity > 0) {
        sphParticlesArrayResize(sphParticlesCullenDehnenStates, capacity);
    }
    if (initializeCullenDehnen) {
        for (ParticleInteger index = 0; index < size; ++index) {
            sphParticlesCullenDehnenStates[index] = {};
        }
    }
    if (initializeViscosity) {
        float const alpha = gradientSwitch ? 0.0f : getOptions().hydro.viscosityAlphaMin;
        for (ParticleInteger index = 0; index < size; ++index) {
            sphParticlesViscosityStates[index] = {alpha, 0.0f, alpha};
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
        sphParticlesThermodynamicStates[slot] = {};
        if (getOptions().chemistry.enabled) {
            sphParticlesChemistryStates[slot] = {};
            sphParticlesChemistrySteps[slot] = {};
        }
        if (getOptions().chemistry.enabled || getOptions().starFormation.enabled) {
            sphParticlesChemistryEos[slot] = {
                float(getOptions().hydro.gamma - 1.0), float(getOptions().hydro.gamma)};
        }
        sphParticlesAccelerationStates[slot] = {};
        sphParticlesParticleIndices[slot] = kNoParticleIndex;
        sphParticlesPredictedInternalEnergyRates[slot] = 0.0f;
        sphParticlesInternalEnergyRates[slot] = 0.0f;
        sphParticlesDensities[slot] = 0.0f;
        if (getOptions().hydro.balsaraSwitch) {
            sphParticlesBalsaraFactors[slot] = 0.0f;
        }
        sphParticlesGradHCorrections[slot] = unbuiltGradHCorrection;
        if (getOptions().hydro.viscosityEvolution != "none") {
            float const alpha = gradientSwitch ? 0.0f : getOptions().hydro.viscosityAlphaMin;
            sphParticlesViscosityStates[slot] = {alpha, 0.0f, alpha};
        }
        if (gradientSwitch) {
            sphParticlesCullenDehnenStates[slot] = {};
        }
        sphParticlesConvergenceFlags[slot] = 0;
        sphParticlesSemiactiveFlags[slot] = 0;
        for (int dim = 0; dim < kDimensions; dim++) {
            sphParticlesPredictedAccelerations[dim][slot] = 0.0f;
            sphParticlesGravityAccelerations[dim][slot] = 0.0f;
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

static Array<Position, kDimensions> const* sphParticlesCacheReadLine(
    detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchCacheLineAction>(
        lineId, particleCache, mutexes);
}

static Vector<Array<Position, kDimensions>> sphParticlesFetchCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Array<Position, kDimensions>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
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
                    if (supportRadius) {
                        supportRadius[destIndex] = ptr[srcIndex].second;
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
                int const k = sphParticlesParticleIndex(i);
                vx[j] = particlesVel(kXDimension, k);
                vy[j] = particlesVel(kYDimension, k);
                vz[j] = particlesVel(kZDimension, k);
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

static Array<float, kDimensions> const* sphParticlesReadVelocityCacheLine(
    detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchVelocityCacheLineAction>(
        lineId, particleVelocityCache, velocityMutexes);
}

static Vector<Array<float, kDimensions>> sphParticlesFetchVelocityCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Array<float, kDimensions>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; i++) {
        int const k = sphParticlesParticleIndex(i);
        line[i - begin][kXDimension] = particlesVel(kXDimension, k);
        line[i - begin][kYDimension] = particlesVel(kYDimension, k);
        line[i - begin][kZDimension] = particlesVel(kZDimension, k);
    }
    return line;
}

void sphParticlesGlobalReadDensity(ParticleGlobalRange range, float* rho, ParticleInteger offset,
    float* balsaraFactor, float* viscosityAlpha, float* gradHCorrection) {
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
                    balsaraFactor[j] =
                        sphParticlesBalsaraFactors ? sphParticlesBalsaraFactors[i] : 1.0f;
                }
                if (viscosityAlpha) {
                    viscosityAlpha[j] = sphParticlesViscosityStates[i].alpha;
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
                    if (viscosityAlpha) {
                        viscosityAlpha[destIndex] = ptr[srcIndex].viscosityAlpha;
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
            sphParticlesBalsaraFactors ? sphParticlesBalsaraFactors[i] : 1.0f,
            getOptions().hydro.viscosityEvolution != "none" ? sphParticlesViscosityStates[i].alpha :
                                                              0.0f,
            sphParticlesGradHCorrections[i]};
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
            detail::SphCacheLineId lineId;
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

static Pair<float> const* sphParticlesReadChemistryEosCacheLine(detail::SphCacheLineId lineId) {
    return detail::readSphCacheLine<SphParticlesFetchChemistryEosCacheLineAction>(
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

#if COSMICTIGER_ENABLE_CHEMISTRY
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
#endif

void sphParticlesClearHydroCaches() {
    Vector<hpx::future<void>> futures;
    for (auto const& c : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesClearHydroCachesAction>(c));
    }
    thermodynamicParticleCache = decltype(thermodynamicParticleCache)();
    chemistryEosParticleCache = decltype(chemistryEosParticleCache)();
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
    chemistryEosParticleCache = decltype(chemistryEosParticleCache)();
    rungParticleCache = decltype(rungParticleCache)();
    particleVelocityCache = decltype(particleVelocityCache)();
    densityParticleCache = decltype(densityParticleCache)();
    cullen_and_dehnen::clearDiagnosticCache();
    hpx::wait_all(futures.begin(), futures.end());
}

namespace detail {

uint32_t sphCheckpointVersion() {
    // Preserve the fixed-viscosity payload version and identify optional state.
    auto const& options = getOptions();
    return (options.starFormation.enabled ? 40U : (options.chemistry.enabled ? 11U : 2U)) +
        (options.hydro.viscosityEvolution == "cullenAndDehnen" ?
                4U :
                (options.hydro.viscosityEvolution == "priceEtAl" ?
                        6U :
                        (options.hydro.viscosityEvolution != "none" ? 2U : 0U)));
}

}    // namespace detail

void sphParticlesLoad(FILE* fp) {
    uint32_t header[7];
    ParticleInteger savedCount;
    FREAD(header, sizeof(uint32_t), 7, fp);
    FREAD(&savedCount, sizeof(ParticleInteger), 1, fp);
    if (header[0] != 0x43545350U || header[1] != detail::sphCheckpointVersion() ||
        header[2] != sizeof(ParticleInteger) || header[3] != sizeof(SphThermodynamicState) ||
        header[4] != sizeof(SphAcceleration) || header[5] != kDimensions ||
        header[6] != sizeof(float) || savedCount != sphParticlesSize()) {
        THROW_ERROR("Incompatible SPH checkpoint payload (requires matching "
                    "viscosity/chemistry layout)\n");
    }
    detail::visitSphCheckpointArrays(
        [fp](auto* values, size_t count) {
            if (count != 0) {
                FREAD(values, sizeof(*values), count, fp);
            }
        },
        false, false);
    if (getOptions().hydro.viscosityEvolution != "none") {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            float words[2];
            FREAD(words, sizeof(float), 2, fp);
            sphParticlesViscosityStates[index] = {words[0], words[1], words[0]};
            if (getOptions().hydro.viscosityGradientSwitch()) {
                float divergence;
                FREAD(&divergence, sizeof(float), 1, fp);
                sphParticlesCullenDehnenStates[index] = {divergence, 0.0f, 0.0f, 0.0f};
            }
        }
    }
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
            double stepWords[2];
            FREAD(stepWords, sizeof(double), 2, fp);
            sphParticlesChemistrySteps[index] = {stepWords[0], stepWords[1]};
        }
    }
}

void sphParticlesSave(FILE* fp) {
    uint32_t header[7] = {0x43545350U, detail::sphCheckpointVersion(), sizeof(ParticleInteger),
        sizeof(SphThermodynamicState), sizeof(SphAcceleration), kDimensions, sizeof(float)};
    ParticleInteger count = sphParticlesSize();
    FWRITE(header, sizeof(uint32_t), 7, fp);
    FWRITE(&count, sizeof(ParticleInteger), 1, fp);
    detail::visitSphCheckpointArrays(
        [fp](auto* values, size_t count) {
            if (count != 0) {
                FWRITE(values, sizeof(*values), count, fp);
            }
        },
        false, false);
    if (getOptions().hydro.viscosityEvolution != "none") {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            auto const& state = sphParticlesViscosityStates[index];
            float words[2] = {state.alpha, state.rate};
            FWRITE(words, sizeof(float), 2, fp);
            if (getOptions().hydro.viscosityGradientSwitch()) {
                float divergence = sphParticlesCullenDehnenStates[index].divergence;
                FWRITE(&divergence, sizeof(float), 1, fp);
            }
        }
    }
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
            auto const& step = sphParticlesChemistrySteps[index];
            double const stepWords[2] = {step.halfInterval, step.pendingHeating};
            FWRITE(stepWords, sizeof(double), 2, fp);
        }
    }
}

void sphCheckpointTest() {
    ALWAYS_ASSERT(sphParticlesSize() == 0 ||
        (sphParticlesViscosityStates != nullptr) ==
            (getOptions().hydro.viscosityEvolution != "none"));
    ALWAYS_ASSERT(sphParticlesSize() == 0 ||
        (sphParticlesCullenDehnenStates != nullptr) ==
            getOptions().hydro.viscosityGradientSwitch());
    Vector<SphCullenDehnenState> originalCullenDehnen;
    if (getOptions().hydro.viscosityGradientSwitch() && sphParticlesSize() != 0) {
        originalCullenDehnen.assign(
            sphParticlesCullenDehnenStates, sphParticlesCullenDehnenStates + sphParticlesSize());
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            sphParticlesCullenDehnenStates[index] = {-0.5f + 0.025f * index, 0.0f, 0.0f, 0.0f};
        }
    }
    ALWAYS_ASSERT(cullen_and_dehnen::evolvedAlpha(0.0f, 1.5f, 2.0f, 0.0f) == 1.5f);
    ALWAYS_ASSERT(std::fabs(cullen_and_dehnen::evolvedAlpha(1.5f, 0.5f, 2.0f, 0.25f) -
                      (0.5f + std::exp(-0.5f))) < 1.0e-6f);
    // Price et al. (2018), equation (50): (1.5 + 0.5 * 0.5) / 1.5 after one backward Euler step.
    ALWAYS_ASSERT(price_etal::evolvedAlpha(0.0f, 1.5f, 2.0f, 0.0f) == 1.5f);
    ALWAYS_ASSERT(
        std::fabs(price_etal::evolvedAlpha(1.5f, 0.5f, 2.0f, 0.25f) - 1.75f / 1.5f) < 1.0e-6f);
    Vector<SphViscosityState> originalViscosity;
    if (getOptions().hydro.viscosityEvolution != "none" && sphParticlesSize() != 0) {
        originalViscosity.assign(
            sphParticlesViscosityStates, sphParticlesViscosityStates + sphParticlesSize());
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            float const alpha = 0.25f + 0.01f * index;
            sphParticlesViscosityStates[index] = {alpha, -0.125f + 0.001f * index, alpha};
        }
    }
    auto decayError = [](int steps) {
        float const halfStep = 0.5f / steps;
        SphViscosityState state{1.1f, -1.0f, 0.0f};
        morris_and_monaghan::applyUpdate(state, 0.1f, halfStep, 2);
        for (int step = 0; step < steps; ++step) {
            morris_and_monaghan::applyUpdate(state, 0.1f, halfStep, 0);
            state.rate = -(state.alpha - 0.1f);
            morris_and_monaghan::applyUpdate(state, 0.1f, halfStep, 1);
            if (step + 1 < steps) {
                morris_and_monaghan::applyUpdate(state, 0.1f, halfStep, 2);
            }
        }
        return std::fabs(double(state.alpha) - (0.1 + std::exp(-1.0)));
    };
    double const coarseError = decayError(8);
    double const fineError = decayError(16);
    ALWAYS_ASSERT(coarseError > 3.0 * fineError && fineError < 0.001);
    SphViscosityState floorState{0.2f, -1.0f, 0.0f};
    morris_and_monaghan::applyUpdate(floorState, 0.1f, 0.25f, 0);
    ALWAYS_ASSERT(floorState.alpha == 0.1f);
    floorState.rate = 1.0f;
    morris_and_monaghan::applyUpdate(floorState, 0.1f, 0.25f, 1);
    ALWAYS_ASSERT(std::fabs(floorState.alpha - 0.45f) < 1.0e-6f);
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
        detail::visitSphCheckpointArrays(
            [](auto* values, size_t) {
                std::memset(values, 0x11, sizeof(*values));
                std::memset(values + 1, 0x22, sizeof(*values));
            },
            true, true);
        auto checkOrder = [](unsigned char first, unsigned char second) {
            detail::visitSphCheckpointArrays(
                [first, second](auto* values, size_t) {
                    auto const* bytes = reinterpret_cast<unsigned char const*>(values);
                    for (size_t i = 0; i < sizeof(*values); i++) {
                        if (bytes[i] != first || bytes[sizeof(*values) + i] != second) {
                            THROW_ERROR("SPH reorder did not preserve every gas field\n");
                        }
                    }
                },
                true, true);
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
    for (size_t index = 0; index < originalViscosity.size(); ++index) {
        sphParticlesViscosityStates[index] = originalViscosity[index];
    }
    for (size_t index = 0; index < originalCullenDehnen.size(); ++index) {
        sphParticlesCullenDehnenStates[index] = originalCullenDehnen[index];
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
                    sphParticlesConverged(i) = false;
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
                    sphParticlesSemiactive(i) = false;
                }
            },
            "sph.particles.resetSemiactive")));
    }
    hpx::wait_all(futures.begin(), futures.end());
}

HPX_PLAIN_ACTION(sphParticlesUpdateTargetAlpha, SphParticlesUpdateTargetAlphaAction);
void sphParticlesUpdateTargetAlpha(
    int minimumRung, float timeStepScale, float time, double explicitTimestep) {
    if (!getOptions().hydro.viscosityGradientSwitch()) {
        return;
    }
    bool const cullenDehnen = getOptions().hydro.viscosityEvolution == "cullenAndDehnen";
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesUpdateTargetAlphaAction>(
            child, minimumRung, timeStepScale, time, explicitTimestep));
    }
    FlopCounter<> flops;
    for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
        int const rung = sphParticlesRung(index);
        bool const active = rung >= minimumRung;
        auto& diagnostic = sphParticlesCullenDehnenStates[index];
        if (active || sphParticlesSemiactive(index)) {
            diagnostic.divergence = diagnostic.pendingDivergence;
        }
        if (active) {
            float const timeStep = explicitTimestep >= 0.0 ?
                explicitTimestep :
                (time > 0.0f ? timeStepScale / (1 << rung) : 0.0f);
            auto& state = sphParticlesViscosityStates[index];
            // Cullen & Dehnen (2010), equation (A15), or Price et al. (2018), equation (50).
            state.alpha = cullenDehnen ?
                cullen_and_dehnen::evolvedAlpha(
                    state.alpha, diagnostic.targetAlpha, diagnostic.inverseDecayTime, timeStep) :
                price_etal::evolvedAlpha(
                    state.alpha, diagnostic.targetAlpha, diagnostic.inverseDecayTime, timeStep);
            flops += 7;
        }
    }
    addCpuFlops(flops);
    for (auto& future : pending) {
        future.get();
    }
}
