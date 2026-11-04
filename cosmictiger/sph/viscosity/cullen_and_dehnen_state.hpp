/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/float16_storage.hpp>

// Cullen & Dehnen (2010), equations (14)-(17), (A15), and Appendix B3's
// time-difference estimator. Only committed history survives density events,
// checkpoints, migration, and particle sorting.
struct SphCullenDehnenState {
    Float16Storage divergence;    // Scaled physical velocity divergence.
    // Physical divergence for CD/Hu; physical divergence minus 3H for Price.
    Float16Storage sampleDivergence;
    int8_t divergenceExponent = 0;
    int8_t sampleDivergenceExponent = 0;
    // -1: no history; -2: startup boundary sample; otherwise the previous own midpoint rung.
    int8_t sampleRung = -1;

    CUDA_EXPORT float physicalDivergence() const {
        return float16_storage::decodeScaled(divergence, divergenceExponent);
    }

    CUDA_EXPORT float sampledDivergence() const {
        return float16_storage::decodeScaled(sampleDivergence, sampleDivergenceExponent);
    }

    CUDA_EXPORT void setDivergences(float physical, float sampled) {
        auto const physicalStorage = float16_storage::encodeScaled(physical);
        auto const sampledStorage = float16_storage::encodeScaled(sampled);
        divergence = physicalStorage.mantissa;
        divergenceExponent = physicalStorage.exponent;
        sampleDivergence = sampledStorage.mantissa;
        sampleDivergenceExponent = sampledStorage.exponent;
    }
};

/*
 * Derived sampling interval for the Appendix-B3 time-difference estimator:
 *   Δτ = (Δτ_previous + Δτ_current)/2,   Δτ_r = Δτ_base 2⁻ʳ.
 * From an initial boundary sample, Δτ = Δτ_current/2. The global integrator
 * supplies the actual midpoint separation directly. History is committed only
 * at the particle's own midpoints, so no per-particle absolute clock is needed.
 * Symbol directory:
 *   r_previous → state.sampleRung; r_current → currentRung;
 *   Δτ_base (dyadic) or Δτ (global) → interval; Δτ → return value.
 */
CUDA_EXPORT inline double sphViscositySampleInterval(
    SphCullenDehnenState const& state, int currentRung, double interval, bool globalTimestep) {
    if (state.sampleRung == -1 || state.sampleRung < -2 || !(interval > 0.0)) {
        return 0.0;
    }
    if (globalTimestep) {
        return interval;
    }
    if (currentRung < 0) {
        return 0.0;
    }
    double const currentInterval = std::ldexp(interval, -currentRung);
    double const previousInterval =
        state.sampleRung == -2 ? 0.0 : std::ldexp(interval, -int(state.sampleRung));
    return 0.5 * (previousInterval + currentInterval);
}

// Density-pass results retained only until the event's alpha/history update.
// Keep all converged results through density retries, without changing history
// while other local or remote targets still sample its previous divergence.
struct SphViscosityDiagnostics {
    float pendingDivergence = 0.0f;
    float targetBulkViscosityCoefficient = 0.0f;
    float inverseDecayTime = 0.0f;    // Inverse conformal decay time.
    float limiter = 1.0f;             // Hu et al. applies this after the alpha update.
    float pendingSampleDivergence = 0.0f;
};

static_assert(sizeof(SphCullenDehnenState) == 8);
static_assert(sizeof(SphViscosityDiagnostics) == 20);
