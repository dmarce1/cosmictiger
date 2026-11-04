/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/hpx.hpp>
#if COSMICTIGER_ENABLE_FMM
#include <cosmictiger/gravity/kick.hpp>
#endif
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/time_integration.hpp>

#if COSMICTIGER_ENABLE_FMM

namespace detail {

struct SphPositionRestore {
    ParticleInteger particle;
    Vector3<Position> position;
};

static bool eventActive = false;
static bool fullGravity = false;
static double eventTime = 0.0;
static double baseTimestep = 0.0;
static int eventMinimumRung = 0;
static KickParameters gravityParameters;
static Vector<SphPositionRestore> positionsToRestore;

static SphHierarchicalState& hierarchyState(ParticleInteger gas) {
    ALWAYS_ASSERT(sphParticlesHierarchicalStates != nullptr);
    return sphParticlesHierarchicalStates[gas];
}

}    // namespace detail

HPX_PLAIN_ACTION(sphHierarchyBeginEvent, SphHierarchyBeginEventAction);
HPX_PLAIN_ACTION(sphHierarchyEndEvent, SphHierarchyEndEventAction);
HPX_PLAIN_ACTION(sphHierarchyCaptureRungs, SphHierarchyCaptureRungsAction);
HPX_PLAIN_ACTION(sphHierarchyOpenRung, SphHierarchyOpenRungAction);

bool sphHierarchyEventActive() {
    return detail::eventActive;
}

bool sphHierarchyFullGravity() {
    return detail::fullGravity;
}

float sphHierarchyPredictedVelocity(int dimension, ParticleInteger gas) {
    ALWAYS_ASSERT(detail::eventActive);
    return detail::hierarchyState(gas).predictedVelocity[dimension];
}

double sphHierarchyPredictedEntropy(ParticleInteger gas) {
    ALWAYS_ASSERT(detail::eventActive);
    auto const& state = detail::hierarchyState(gas);
    return double(sphParticlesEntropy(gas)) +
        sphParticlesEntropyRate(gas) * (detail::eventTime - state.entropyTime);
}

Vector3<float> sphHierarchyGravityPrediction(ParticleInteger gas) {
    auto const& state = detail::hierarchyState(gas);
    return sphParticlesAccelerationState(gas) - state.hydroAcceleration;
}

void sphHierarchyBeginEvent(double time, int minimumRung, double baseTimestep, bool fullGravity,
    KickParameters const& gravityParameters) {
    ALWAYS_ASSERT(!detail::eventActive && detail::positionsToRestore.empty());
    Vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphHierarchyBeginEventAction>(
            child, time, minimumRung, baseTimestep, fullGravity, gravityParameters));
    }
    detail::eventTime = time;
    detail::baseTimestep = baseTimestep;
    detail::eventMinimumRung = minimumRung;
    detail::fullGravity = fullGravity;
    detail::gravityParameters = gravityParameters;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        auto& state = detail::hierarchyState(gas);
        ParticleInteger const particle = sphParticlesParticleIndex(gas);
        if (!state.initialized) {
            ALWAYS_ASSERT(time == 0.0);
            state.predictedVelocity = particlesVelData()[particle];
            state.positionTime = state.predictionTime = state.entropyTime = time;
            state.initialized = true;
        }
        // Derived predictor, following GADGET-4 predict.cc lines 373-379:
        //   v* (t) = v* (t₀) + (g_full + a_hydro) (t - t₀).
        // g_full + a_hydro -> sphParticlesAccelerationState; t₀ -> predictionTime.
        // Source: https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/predict_8cc_source.html
        double const elapsed = time - state.predictionTime;
        ALWAYS_ASSERT(elapsed >= 0.0);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            state.predictedVelocity[dimension] +=
                sphParticlesAcceleration(dimension, gas) * elapsed;
        }
        state.predictionTime = time;
        if (sphParticlesRung(gas) < minimumRung) {
            ALWAYS_ASSERT(state.positionTime >= time);
            detail::SphPositionRestore restore;
            restore.particle = particle;
            // Locally derived inverse of the actual forward drift:
            //   x(t) = x(t_end) + v_drift (t - t_end) / a_drift.
            // t_end -> positionTime; 1/a_drift -> inverseDriftScale.
            // Use the recorded scale, so projecting a source does not change the
            // cosmological quadrature used to place its stored endpoint.
            double const factor = (time - state.positionTime) * state.inverseDriftScale;
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                auto& position = particlesPos(dimension, particle);
                restore.position[dimension] = position;
                double const displacement = particlesVel(dimension, particle) * factor;
                if constexpr (kPositionBits == 64) {
                    position += Position(displacement);
                } else {
                    position.raw() += static_cast<PositionInteger>(
                        std::llrint(std::remainder(displacement, 1.0) * 0x1p32));
                }
            }
            detail::positionsToRestore.push_back(restore);
        } else {
            double const tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
                std::max(baseTimestep, std::abs(time));
            ALWAYS_ASSERT(std::abs(state.positionTime - time) <= tolerance);
        }
    }
    detail::eventActive = true;
    for (auto& child : children) {
        child.get();
    }
}

void sphHierarchyEndEvent() {
    Vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphHierarchyEndEventAction>(child));
    }
    for (auto const& restore : detail::positionsToRestore) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            particlesPos(dimension, restore.particle) = restore.position[dimension];
        }
    }
    detail::positionsToRestore.clear();
    detail::eventActive = false;
    for (auto& child : children) {
        child.get();
    }
}

void sphHierarchyCloseTarget(
    ParticleInteger gas, Vector3<float> const& oldHydro, Vector3<float> const& newHydro) {
    ALWAYS_ASSERT(detail::eventActive);
    auto& state = detail::hierarchyState(gas);
    double const closingDuration = detail::eventTime - state.entropyTime;
    ALWAYS_ASSERT(closingDuration >= 0.0);
    // GADGET-4 kicks.cc lines 551-567, expressed in this code's conformal time:
    //   v ← v + a_hydro,new Δτ_close;
    //   v* ← v* + (a_hydro,new - a_hydro,old) Δτ_close;
    //   A ← A + Ȧ_new Δτ_close.
    // A -> sphParticlesEntropy; Δτ_close -> closingDuration (always nonnegative).
    // Source: https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/kicks_8cc_source.html
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        sphParticlesVelocity(dimension, gas) += newHydro[dimension] * closingDuration;
        state.predictedVelocity[dimension] +=
            (newHydro[dimension] - oldHydro[dimension]) * closingDuration;
    }
    sphParticlesEntropy(gas) += sphParticlesEntropyRate(gas) * closingDuration;
    ALWAYS_ASSERT(sphParticlesEntropy(gas) > 0.0f);
    state.entropyTime = detail::eventTime;
    state.hydroAcceleration = newHydro;
    state.minimumHydroRung = std::max(0, int(sphParticlesProposedRungs[gas]));
}

void sphHierarchySynchronizeTarget(ParticleInteger gas, Vector3<float> const& velocity) {
    ALWAYS_ASSERT(detail::eventActive);
    if (detail::fullGravity) {
        // Only the complete closing ladder gives the physical synchronized
        // momentum. Fine events retain the independently advanced predictor.
        detail::hierarchyState(gas).predictedVelocity = velocity;
    }
}

int sphHierarchyKickTarget(ParticleInteger gas, Vector3<float> const& oldHydro,
    Vector3<float> const& newHydro, Vector3<float> const& leafGravity) {
    sphHierarchyCloseTarget(gas, oldHydro, newHydro);
    auto const& parameters = detail::gravityParameters;
    auto& state = detail::hierarchyState(gas);
    ParticleInteger const particle = sphParticlesParticleIndex(gas);
    auto& velocity = particlesVelData()[particle];
    float const signedHalfStep = parameters.sign * (parameters.top ? 1.0f : -1.0f) *
        std::ldexp(parameters.t0, -parameters.minRung - 1);
    if (parameters.ascending && !parameters.firstCall) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            velocity[dimension] =
                std::fma(leafGravity[dimension], signedHalfStep, velocity[dimension]);
        }
    }
    sphHierarchySynchronizeTarget(gas, velocity);
    if (parameters.descending) {
        float const gravityMagnitudeSquared = leafGravity.dot(leafGravity) + 1e-35f;
        float const effectiveRadius = parameters.gravitySupportRadius /
            selectedGravityKernel().supportRadiusPerEffectiveRadius;
        float const timestep = std::min(parameters.maxDt,
            std::min(parameters.t0,
                parameters.eta * std::sqrt(parameters.a) *
                    std::sqrt(effectiveRadius / std::sqrt(gravityMagnitudeSquared))));
        int const gravityRung = std::max(parameters.minRung +
                int(int(std::ceil(std::log2(parameters.t0 / timestep))) > parameters.minRung),
            int(particlesRung(particle)) - 1);
        particlesRung(particle) = std::max(gravityRung, state.minimumHydroRung);
        ALWAYS_ASSERT(particlesRung(particle) >= 0 && particlesRung(particle) < kMaxRung);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            velocity[dimension] =
                std::fma(leafGravity[dimension], signedHalfStep, velocity[dimension]);
        }
    }
    return particlesRung(particle);
}

void sphHierarchyCaptureRungs() {
    Vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphHierarchyCaptureRungsAction>(child));
    }
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        if (sphParticlesRung(gas) >= detail::eventMinimumRung) {
            detail::hierarchyState(gas).minimumHydroRung = sphParticlesRung(gas);
        }
    }
    for (auto& child : children) {
        child.get();
    }
}

void sphHierarchyOpenRung(int rung) {
    Vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphHierarchyOpenRungAction>(child, rung));
    }
    double const openingDuration = std::ldexp(detail::baseTimestep, -rung - 1);
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        if (sphParticlesRung(gas) != rung) {
            continue;
        }
        auto& state = detail::hierarchyState(gas);
        ALWAYS_ASSERT(state.entropyTime == detail::eventTime);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            sphParticlesVelocity(dimension, gas) +=
                state.hydroAcceleration[dimension] * openingDuration;
        }
        sphParticlesEntropy(gas) += sphParticlesEntropyRate(gas) * openingDuration;
        ALWAYS_ASSERT(sphParticlesEntropy(gas) > 0.0f);
        state.entropyTime = detail::eventTime + openingDuration;
    }
    for (auto& child : children) {
        child.get();
    }
}

void sphHierarchyRecordDrift(int rung, double endTime, double inverseScale) {
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        if (sphParticlesRung(gas) == rung) {
            auto& state = detail::hierarchyState(gas);
            state.positionTime = endTime;
            state.inverseDriftScale = inverseScale;
        }
    }
}

#else

// SPH-only builds have no hierarchical gravity path. Keep the shared predictor
// accessors available without instantiating gravity templates at expansion order 0.
bool sphHierarchyEventActive() {
    return false;
}

bool sphHierarchyFullGravity() {
    return false;
}

float sphHierarchyPredictedVelocity(int, ParticleInteger) {
    ALWAYS_ASSERT(false);
    return 0.0f;
}

double sphHierarchyPredictedEntropy(ParticleInteger) {
    ALWAYS_ASSERT(false);
    return 0.0;
}

Vector3<float> sphHierarchyGravityPrediction(ParticleInteger) {
    ALWAYS_ASSERT(false);
    return Vector3<float>(0.0f);
}

void sphHierarchyBeginEvent(double, int, double, bool, KickParameters const&) {
    ALWAYS_ASSERT(false);
}

void sphHierarchyEndEvent() {
    ALWAYS_ASSERT(false);
}

void sphHierarchyCaptureRungs() {
    ALWAYS_ASSERT(false);
}

void sphHierarchyOpenRung(int) {
    ALWAYS_ASSERT(false);
}

void sphHierarchyRecordDrift(int, double, double) {
    ALWAYS_ASSERT(false);
}

void sphHierarchyCloseTarget(ParticleInteger, Vector3<float> const&, Vector3<float> const&) {
    ALWAYS_ASSERT(false);
}

void sphHierarchySynchronizeTarget(ParticleInteger, Vector3<float> const&) {
    ALWAYS_ASSERT(false);
}

int sphHierarchyKickTarget(
    ParticleInteger, Vector3<float> const&, Vector3<float> const&, Vector3<float> const&) {
    ALWAYS_ASSERT(false);
    return 0;
}

#endif
