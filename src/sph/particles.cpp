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
static bool gravityStorageEnabled = true;

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
namespace detail {

// Between the predictor (phase 0) and the closing kick (phase 1) of a step, the staged
// velocity and entropy of each active gas particle are predicted to the end of
// its closing interval from the stored rates. The particle state itself stays at the
// half step, so the closing kick needs no record of the rates it replaces.
struct SphPrediction {
    bool open = false;
    int minimumRung = 0;
    float timeStepScale = 0.0f;
    double explicitTimestep = -1.0;
};
static SphPrediction prediction;

static double sphHalfStep(float timeStepScale, double explicitTimestep, int rung) {
    return explicitTimestep >= 0.0 ? 0.5 * explicitTimestep :
                                     std::ldexp(double(timeStepScale), -1 - rung);
}

// Zero unless the prediction is open and the particle closes in this step.
static double predictionHalfStep(ParticleInteger gas) {
    if (!prediction.open || sphParticlesRung(gas) < prediction.minimumRung) {
        return 0.0;
    }
    return sphHalfStep(
        prediction.timeStepScale, prediction.explicitTimestep, sphParticlesRung(gas));
}

static float predictedVelocity(int dimension, ParticleInteger gas, double halfStep) {
    if (sphMidpointEnabled()) {
        return particlesVel(dimension, sphParticlesParticleIndex(gas));
    }
    return particlesVel(dimension, sphParticlesParticleIndex(gas)) +
        sphParticlesAcceleration(dimension, gas) * halfStep;
}

static float predictedInternalEnergy(ParticleInteger gas, double halfStep) {
    if (!sphParticlesEntropyInitialized) {
        return sphParticlesInternalEnergy(gas);
    }
    double const entropy = sphMidpointEnabled() ?
        sphParticlesEntropy(gas) :
        sphParticlesEntropy(gas) + sphParticlesEntropyRate(gas) * halfStep;
    ALWAYS_ASSERT(entropy > 0.0);
    float const energy = sph_entropy::storedEnergy(entropy, sphParticlesDensity(gas),
        getOptions().hydro.gamma, sphParticlesThermalScaleFactor);
    ALWAYS_ASSERT(energy > 0.0f);
    return energy;
}

}    // namespace detail

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
            sphParticlesEntropyRates[gas] = 0.0f;
        }
        sphParticlesEntropyInitialized = true;
    }
    for (auto& child : children) {
        child.get();
    }
}

HPX_PLAIN_ACTION(sphParticlesApplyUpdates, SphParticlesApplyUpdatesAction);
EnergiesT sphParticlesApplyUpdates(
    int minimumRung, int phase, float timeStepScale, float time, double explicitTimestep) {
    // Midpoint integration applies one full hydro kick through its own helper.
    // Legacy predictor/corrector calls must not add another kick or prediction.
    bool const midpoint = sphMidpointEnabled();
    if (midpoint && phase < 2) {
        return {};
    }
    // At time zero no step has opened yet, so there is no half-kick to predict or close.
    if (!(phase >= 2 || time > 0.0)) {
        return {};
    }
    Vector<hpx::future<EnergiesT>> futures;
    for (auto& child : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesApplyUpdatesAction>(
            child, minimumRung, phase, timeStepScale, time, explicitTimestep));
    }
    if (phase == 0) {
        detail::prediction = {true, minimumRung, timeStepScale, explicitTimestep};
    } else if (phase == 1) {
        detail::prediction = {};
    }
    bool const savePotential = saveParticlePotential(getOptions().analysis);
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int threadIndex = 0; threadIndex < numberOfThreads; threadIndex++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [timeStepScale, numberOfThreads, threadIndex, phase, minimumRung, explicitTimestep,
                savePotential, midpoint]() {
                ParticleInteger const rangeBegin =
                    (size_t) threadIndex * sphParticlesSize() / numberOfThreads;
                ParticleInteger const rangeEnd =
                    (size_t) (threadIndex + 1) * sphParticlesSize() / numberOfThreads;
                sph_flops::Counter flops;
                EnergiesT energies;
                for (int gasIndex = rangeBegin; gasIndex < rangeEnd; gasIndex++) {
                    ParticleInteger const gravityIndex = sphParticlesParticleIndex(gasIndex);
                    auto const rung = sphParticlesRung(gasIndex);
                    if (phase >= 2) {
                        // Every gas particle at the synchronized state, before any opening
                        // kick, as particlesSumEnergies would count it.
                        float const vx = particlesVel(kXDimension, gravityIndex);
                        float const vy = particlesVel(kYDimension, gravityIndex);
                        float const vz = particlesVel(kZDimension, gravityIndex);
                        float const m = particlesMass(gravityIndex);
                        if (savePotential) {
                            energies.pot += 0.5 * m * particlesPot(gravityIndex);
                        }
                        energies.kin += 0.5 * m * (sqr(vx) + sqr(vy) + sqr(vz));
                        energies.xmom += m * vx;
                        energies.ymom += m * vy;
                        energies.zmom += m * vz;
                        energies.nmom += m * std::sqrt(sqr(vx) + sqr(vy) + sqr(vz));
                        energies.therm += m * sphParticlesInternalEnergy(gasIndex);
                        if (phase == 3 || midpoint) {
                            continue;
                        }
                    }
                    double const halfTimeStep =
                        detail::sphHalfStep(timeStepScale, explicitTimestep, rung);
                    flops.doublePrecision += explicitTimestep >= 0.0 ? 1 : 2;
                    // Each active particle gets two half-kicks per step: one closes the step
                    // that just drifted, one opens the next. The predictor (phase 0) only
                    // opens the stage-time prediction above, so the hydro run sees end-of-
                    // drift velocities and entropies; the closing kick (phase 1) then applies
                    // the fresh rates to the unchanged half-step state.
                    if (phase == 1 || phase == 2) {
                        // Phase 1 follows the hydro force run; phase 2 follows rung commit
                        // and limiting and opens the next step ahead of the drift.
                        if (rung >= minimumRung) {
                            // Each mixed multiply/add promotes two float values and narrows
                            // the stored result, in addition to two double arithmetic ops.
                            flops.doublePrecision += 5 * (kDimensions + 1);
                            for (int dimension = 0; dimension < kDimensions; dimension++) {
                                particlesVel(dimension, gravityIndex) +=
                                    sphParticlesAcceleration(dimension, gasIndex) * halfTimeStep;
                            }
                            ALWAYS_ASSERT(sphParticlesEntropy(gasIndex) > 0.0f);
                            sphParticlesEntropy(gasIndex) +=
                                sphParticlesEntropyRate(gasIndex) * halfTimeStep;
                            ALWAYS_ASSERT(sphParticlesEntropy(gasIndex) > 0.0f);
                        }
                    }
                }
                addCpuFlops(flops.counts());
                return energies;
            },
            phase == 0     ? "sph.particles.predictKick" :
                phase == 1 ? "sph.particles.closeKick" :
                phase == 2 ? "sph.particles.openKick" :
                phase == 3 ? "sph.particles.sumEnergies" :
                             "sph.particles.unknownKickPhase")));
    }
    EnergiesT energies;
    for (auto& future : futures) {
        energies += future.get();
    }
    return energies;
}

HPX_PLAIN_ACTION(sphParticlesCommitProposedRungs, SphParticlesCommitProposedRungsAction);

void sphParticlesCommitProposedRungs(int minimumRung) {
    ALWAYS_ASSERT(!sphMidpointEnabled());
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesCommitProposedRungsAction>(child, minimumRung));
    }
    int const workers = std::max(1, hpxHardwareConcurrency());
    for (int worker = 0; worker < workers; ++worker) {
        pending.push_back(hpx::async(hpx::annotated_function(
            [minimumRung, workers, worker]() {
                ParticleInteger const begin = size_t(worker) * sphParticlesSize() / workers;
                ParticleInteger const end = size_t(worker + 1) * sphParticlesSize() / workers;
                for (ParticleInteger gas = begin; gas < end; ++gas) {
                    char& proposed = sphParticlesProposedRungs[gas];
                    if (sphParticlesRung(gas) >= minimumRung) {
                        ALWAYS_ASSERT(proposed >= 0);
                    }
                    if (proposed >= 0) {
                        sphParticlesRung(gas) = proposed;
                        proposed = -1;
                    }
                }
            },
            "sph.particles.commitProposedRungs")));
    }
    hpx::wait_all(pending.begin(), pending.end());
    for (auto& future : pending) {
        future.get();
    }
}

HPX_PLAIN_ACTION(sphParticlesCompleteForces, SphParticlesCompleteForcesAction);

void sphParticlesCompleteForces(SphRunParameters const& parameters) {
    ALWAYS_ASSERT(parameters.phase == 0 && parameters.deferAccelerationSources);
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesCompleteForcesAction>(child, parameters));
    }
    int const workers = std::max(1, hpxHardwareConcurrency());
    for (int worker = 0; worker < workers; ++worker) {
        pending.push_back(hpx::async(hpx::annotated_function(
            [parameters, workers, worker]() {
                sph_flops::Counter flops;
                auto const& options = getOptions();
                ParticleInteger const begin = size_t(worker) * sphParticlesSize() / workers;
                ParticleInteger const end = size_t(worker + 1) * sphParticlesSize() / workers;
                for (ParticleInteger gas = begin; gas < end; ++gas) {
                    if (sphParticlesRung(gas) < parameters.minimumRung) {
                        continue;
                    }
                    Vector3<Position> position;
                    Vector3<float> velocity;
                    Vector3<float> gravityAcceleration(0.0f);
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        position[dim] = sphParticlesPosition(dim, gas);
                        velocity[dim] = sphParticlesVelocity(dim, gas);
#if COSMICTIGER_WITH_GRAVITY
                        if (options.gravity.enabled) {
                            gravityAcceleration[dim] = sphParticlesGravityAcceleration(dim, gas);
                        }
#endif
                    }
                    auto& acceleration = sphParticlesAccelerationState(gas);
                    acceleration = sphCompleteAcceleration(parameters, position, velocity,
                        acceleration, gravityAcceleration, options.hydro.centralAcceleration,
                        options.hydro.centralSupportRadius, flops);
                }
                addCpuFlops(flops.counts());
            },
            "sph.particles.completeForces")));
    }
    hpx::wait_all(pending.begin(), pending.end());
    for (auto& future : pending) {
        future.get();
    }
}

namespace detail {

template <typename Visitor>
void visitSphCheckpointArrays(
    Visitor&& visit, bool includeChemistry = true, bool includeViscosity = true) {
    size_t const count = sphParticlesSize();
    // Keep stage fields as well as the physical fields: fine rungs may be staggered at checkpoints.
    visit(sphParticlesEntropies, count);
    visit(sphParticlesSupportRadii, count);
    visit(sphParticlesAccelerationStates, count);
    visit(sphParticlesParticleIndices, count);
    visit(sphParticlesEntropyRates, count);
    visit(sphParticlesDensities, count);
    visit(sphParticlesConvergenceFlags, count);
    visit(sphParticlesSemiactiveFlags, count);
    visit(sphParticlesProposedRungs, count);
    if (includeViscosity && sphParticlesBulkViscosityCoefficients != nullptr) {
        visit(sphParticlesBulkViscosityCoefficients, count);
    }
    if (includeViscosity && sphParticlesCullenDehnenStates != nullptr) {
        visit(sphParticlesCullenDehnenStates, count);
    }
    if (includeChemistry && sphParticlesElectronFractions != nullptr) {
        visit(sphParticlesElectronFractions, count);
    }
}

}    // namespace detail

void sphParticlesSwap(ParticleInteger i, ParticleInteger j) {
    if (i != j) {
        sphTreeInvalidate();
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (sphParticlesGravityAccelerations[dim] != nullptr) {
                std::swap(sphParticlesGravityAccelerations[dim][i],
                    sphParticlesGravityAccelerations[dim][j]);
            }
        }
    }
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

// Legacy standalone gravity/hydro passes use this handoff. Midpoint events
// release it because gravity applies its own kicks directly.
void sphParticlesEnsureGravityStorage() {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    gravityStorageEnabled = true;
    for (int dim = 0; dim < kDimensions; ++dim) {
        if (sphParticlesGravityAccelerations[dim] == nullptr && capacity != 0) {
            sphParticlesGravityAccelerations[dim] =
                static_cast<float*>(allocateManagedMemory(sizeof(float) * capacity));
            std::fill_n(sphParticlesGravityAccelerations[dim], size, 0.0f);
        }
    }
}

HPX_PLAIN_ACTION(sphParticlesReleaseGravityStorage, SphParticlesReleaseGravityStorageAction);
void sphParticlesReleaseGravityStorage() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<SphParticlesReleaseGravityStorageAction>(child));
    }
    gravityStorageEnabled = false;
    for (int dim = 0; dim < kDimensions; ++dim) {
        freeManagedMemory(sphParticlesGravityAccelerations[dim]);
        sphParticlesGravityAccelerations[dim] = nullptr;
    }
    for (auto& future : futures) {
        future.get();
    }
}

void sphParticlesResize(ParticleInteger particleCount, bool resizeGravityParticles) {
    if (particleCount != size) {
        sphTreeInvalidate();
    }
    bool const initializeBalsara =
        getOptions().hydro.balsaraSwitch && sphParticlesBalsaraFactors == nullptr;
    bool const gradientSwitch = getOptions().hydro.viscosityGradientSwitch();
    bool const initializeCullenDehnen = gradientSwitch && sphParticlesCullenDehnenStates == nullptr;
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
        sphParticlesArrayResize(sphParticlesProposedRungs, newCapacity);
        if (getOptions().hydro.viscosityEvolution != "none") {
            sphParticlesArrayResize(sphParticlesBulkViscosityCoefficients, newCapacity);
        }
        if (gradientSwitch) {
            sphParticlesArrayResize(sphParticlesCullenDehnenStates, newCapacity);
        }
        sphParticlesArrayResize(sphParticlesEntropies, newCapacity);
        sphParticlesArrayResize(sphParticlesSupportRadii, newCapacity);
        sphParticlesArrayResize(sphParticlesAccelerationStates, newCapacity);
        sphParticlesArrayResize(sphParticlesConvergenceFlags, newCapacity);
        sphParticlesArrayResize(sphParticlesSemiactiveFlags, newCapacity);
        sphParticlesArrayResize(sphParticlesParticleIndices, newCapacity);
        sphParticlesArrayResize(sphParticlesEntropyRates, newCapacity);
        for (int dim = 0; dim < kDimensions; dim++) {
            if (gravityStorageEnabled) {
                sphParticlesArrayResize(sphParticlesGravityAccelerations[dim], newCapacity);
            }
        }
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
            sphParticlesBalsaraFactors[index] = 0.0f;
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
        sphParticlesSupportRadii[slot] = 0.0f;
        if (getOptions().chemistry.enabled) {
            sphParticlesElectronFractions[slot] = 1.0f;
        }
        sphParticlesAccelerationStates[slot] = {};
        sphParticlesParticleIndices[slot] = kNoParticleIndex;
        sphParticlesEntropyRates[slot] = 0.0f;
        sphParticlesDensities[slot] = 0.0f;
        if (getOptions().hydro.balsaraSwitch) {
            sphParticlesBalsaraFactors[slot] = 0.0f;
        }
        sphParticlesGradHCorrections[slot] = unbuiltGradHCorrection;
        sphParticlesProposedRungs[slot] = sphMidpointEnabled() ? 0 : -1;
        if (getOptions().hydro.viscosityEvolution != "none") {
            float const bulkViscosityCoefficient =
                gradientSwitch ? 0.0f : getOptions().hydro.minimumBulkViscosityCoefficient;
            sphParticlesBulkViscosityCoefficients[slot] = bulkViscosityCoefficient;
        }
        if (gradientSwitch) {
            sphParticlesCullenDehnenStates[slot] = {};
        }
        sphParticlesConvergenceFlags[slot] = 0;
        sphParticlesSemiactiveFlags[slot] = 0;
        for (int dim = 0; dim < kDimensions; dim++) {
            if (sphParticlesGravityAccelerations[dim] != nullptr) {
                sphParticlesGravityAccelerations[dim][slot] = 0.0f;
            }
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
    for (int dim = 0; dim < kDimensions; ++dim) {
        freeManagedMemory(sphParticlesGravityAccelerations[dim]);
        sphParticlesGravityAccelerations[dim] = nullptr;
    }
    gravityStorageEnabled = true;
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
                    internalEnergy[j] =
                        detail::predictedInternalEnergy(i, detail::predictionHalfStep(i));
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
        line[i - begin].first = detail::predictedInternalEnergy(i, detail::predictionHalfStep(i));
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
                double const halfStep = detail::predictionHalfStep(i);
                vx[j] = detail::predictedVelocity(kXDimension, i, halfStep);
                vy[j] = detail::predictedVelocity(kYDimension, i, halfStep);
                vz[j] = detail::predictedVelocity(kZDimension, i, halfStep);
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
        double const halfStep = detail::predictionHalfStep(i);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            line[i - begin][dimension] = detail::predictedVelocity(dimension, i, halfStep);
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
                    balsaraFactor[j] =
                        sphParticlesBalsaraFactors ? sphParticlesBalsaraFactors[i] : 1.0f;
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
            sphParticlesBalsaraFactors ? sphParticlesBalsaraFactors[i] : 1.0f,
            getOptions().hydro.viscosityEvolution != "none" ?
                sphParticlesBulkViscosityCoefficients[i] :
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
    return 60000U +
        (options.starFormation.enabled ? 110U : (options.chemistry.enabled ? 100U : 2U)) +
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
        header[2] != sizeof(ParticleInteger) || header[3] != sizeof(float) ||
        header[4] != sizeof(SphAcceleration) || header[5] != kDimensions ||
        header[6] != sizeof(float) || savedCount != sphParticlesSize()) {
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
    if (getOptions().hydro.viscosityEvolution != "none") {
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            FREAD(&sphParticlesBulkViscosityCoefficients[index], sizeof(float), 1, fp);
            if (getOptions().hydro.viscosityGradientSwitch()) {
                float divergence;
                FREAD(&divergence, sizeof(float), 1, fp);
                sphParticlesCullenDehnenStates[index] = {divergence, 0.0f, 0.0f, 0.0f};
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
        sizeof(float), sizeof(SphAcceleration), kDimensions, sizeof(float)};
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
            FWRITE(&sphParticlesBulkViscosityCoefficients[index], sizeof(float), 1, fp);
            if (getOptions().hydro.viscosityGradientSwitch()) {
                float divergence = sphParticlesCullenDehnenStates[index].divergence;
                FWRITE(&divergence, sizeof(float), 1, fp);
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
    ALWAYS_ASSERT(sphParticlesSize() == 0 ||
        (sphParticlesBulkViscosityCoefficients != nullptr) ==
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
    detail::visitSphCheckpointArrays([&expected](auto* values, size_t count) {
        size_t const bytes = sizeof(*values) * count;
        if (bytes != 0) {
            auto* data = reinterpret_cast<unsigned char*>(values);
            expected.insert(expected.end(), data, data + bytes);
            std::memset(data, 0x5a, bytes);
        }
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
        std::cout << "SPH reorder passed (all fields, gravity and predictor stages)\n"
                  << std::flush;
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
    for (size_t index = 0; index < originalCullenDehnen.size(); ++index) {
        sphParticlesCullenDehnenStates[index] = originalCullenDehnen[index];
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
    bool const huEtAl = getOptions().hydro.viscosityEvolution == "hu_etal";
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesUpdateTargetAlphaAction>(
            child, minimumRung, timeStepScale, time, explicitTimestep));
    }
    sph_flops::Counter flops;
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
            flops.doublePrecision += explicitTimestep >= 0.0 ? 1 : 2;
            flops += explicitTimestep < 0.0 && time > 0.0f ? 1 : 0;
            auto& bulkViscosityCoefficient = sphParticlesBulkViscosityCoefficients[index];
            bool const increasing =
                bulkViscosityCoefficient < diagnostic.targetBulkViscosityCoefficient;
            // CD (2010), equation (A15); Hu (2014), equation (17); Price (2018), equation (50).
            bulkViscosityCoefficient = huEtAl ?
                hu_etal::evolvedBulkViscosityCoefficient(bulkViscosityCoefficient,
                    diagnostic.targetBulkViscosityCoefficient, diagnostic.inverseDecayTime,
                    diagnostic.limiter, timeStep) :
                cullenDehnen ?
                cullen_and_dehnen::evolvedBulkViscosityCoefficient(bulkViscosityCoefficient,
                    diagnostic.targetBulkViscosityCoefficient, diagnostic.inverseDecayTime,
                    timeStep) :
                price_etal::evolvedBulkViscosityCoefficient(bulkViscosityCoefficient,
                    diagnostic.targetBulkViscosityCoefficient, diagnostic.inverseDecayTime,
                    timeStep);
            flops += (increasing ? (cullenDehnen || huEtAl ? 0 : 1) : 5) + (huEtAl ? 1 : 0);
        }
    }
    addCpuFlops(flops.counts());
    for (auto& future : pending) {
        future.get();
    }
}
