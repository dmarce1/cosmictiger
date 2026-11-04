/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/float16_storage.hpp>

namespace sph_viscosity {
namespace detail {

    // Arithmetic is built with finite-math assumptions. Inspect IEEE encodings
    // before any arithmetic so these guards also reject NaN/Inf in that build.
    CUDA_EXPORT inline bool isFinite(float value) {
        return (float16_storage::detail::floatBits(value) & 0x7f800000U) != 0x7f800000U;
    }

    CUDA_EXPORT inline bool isFinite(double value) {
#ifdef __CUDA_ARCH__
        uint64_t const bits = static_cast<uint64_t>(__double_as_longlong(value));
#else
        uint64_t const bits = std::bit_cast<uint64_t>(value);
#endif
        return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    }

}    // namespace detail
}    // namespace sph_viscosity

/*
 * Derived storage normalization for Cullen & Dehnen (2010), Appendix B3:
 *   q = Δτ_r G, G = ∇_x·w = a² ∇_r·v_pec, Δτ_r = Δτ_base 2⁻ʳ.
 * https://arxiv.org/pdf/1006.1524
 * Store one binary16 scalar per gas particle. Endpoint samples use the current
 * live rung's interval; changing the rung rescales q without changing G.
 * Global integration recovers the normalization from its shared previous step.
 * Symbol directory:
 *   q → scaledDivergence; G → divergence; Δτ_r → sampleTimestep.
 */
CUDA_EXPORT inline float sphDivergenceSign(Float16Storage const& scaledDivergence) {
    float const scaled = scaledDivergence;
    return scaled > 0.0f ? 1.0f : (scaled < 0.0f ? -1.0f : 0.0f);
}

CUDA_EXPORT inline double sphDecodeDivergence(
    Float16Storage const& scaledDivergence, double sampleTimestep) {
    return sph_viscosity::detail::isFinite(sampleTimestep) && sampleTimestep > 0.0 ?
        double(float(scaledDivergence)) / sampleTimestep :
        0.0;
}

// Reject overflow rather than silently clamp a shock history. Startup uses the
// shared base interval for global steps, whose first accepted duration is unknown.
CUDA_EXPORT inline bool sphEncodeDivergence(
    Float16Storage& scaledDivergence, float divergence, double sampleTimestep) {
    if (!sph_viscosity::detail::isFinite(divergence) ||
        !sph_viscosity::detail::isFinite(sampleTimestep) || !(sampleTimestep > 0.0)) {
        return false;
    }
    constexpr double kMaximumHalf = 65504.0;
    double const absoluteDivergence = std::abs(double(divergence));
    if (absoluteDivergence > 1.0 && sampleTimestep > kMaximumHalf / absoluteDivergence) {
        return false;
    }
    double const scaled = double(divergence) * sampleTimestep;
    if (std::abs(scaled) > kMaximumHalf) {
        return false;
    }
    scaledDivergence = static_cast<float>(scaled);
    return true;
}

/*
 * Derived change of normalization when a closed interval receives a new rung:
 *   q_new = q_old 2^(r_old − r_new).
 * No new gradient evaluation or sampling time is implied by this operation.
 * Symbol directory: q_old/q_new → scaledDivergence; r_old/r_new → oldRung/newRung.
 */
CUDA_EXPORT inline bool sphRescaleDivergence(
    Float16Storage& scaledDivergence, int oldRung, int newRung) {
    if (oldRung < 0 || oldRung >= kMaxRung || newRung < 0 || newRung >= kMaxRung) {
        return false;
    }
    return sphEncodeDivergence(
        scaledDivergence, float(scaledDivergence), std::ldexp(1.0, oldRung - newRung));
}

CUDA_EXPORT inline double sphViscosityStoredSampleTimestep(
    int liveRung, double baseTimestep, double previousGlobalTimestep, bool globalTimestep) {
    double const timestep = globalTimestep ?
        (previousGlobalTimestep > 0.0 ? previousGlobalTimestep : baseTimestep) :
        (liveRung >= 0 && liveRung < kMaxRung ? std::ldexp(baseTimestep, -liveRung) : 0.0);
    return sph_viscosity::detail::isFinite(timestep) && timestep > 0.0 ? timestep : 0.0;
}

/*
 * The difference of two endpoint samples estimates the gradient derivative at
 * the completed interval's midpoint. Applying its viscosity to the following
 * kick is first order in time. Rung selection happens only after this sample.
 * Δτ → interval (global), or interval 2⁻ʳ (dyadic); r → liveRung.
 */
CUDA_EXPORT inline double sphViscositySampleInterval(
    int liveRung, double interval, bool globalTimestep, bool initialize = false) {
    if (initialize || !sph_viscosity::detail::isFinite(interval) || !(interval > 0.0)) {
        return 0.0;
    }
    if (globalTimestep) {
        return interval;
    }
    return liveRung >= 0 && liveRung < kMaxRung ? std::ldexp(interval, -liveRung) : 0.0;
}

// Density-pass results retained only until the event's alpha/history update.
// Keep all converged results through density retries, without changing history
// while other local or remote targets still sample its previous divergence.
struct SphViscosityDiagnostics {
    float pendingDivergence = 0.0f;
    float targetBulkViscosityCoefficient = 0.0f;
    float inverseDecayTime = 0.0f;    // Inverse conformal decay time.
    float limiter = 1.0f;             // Hu et al. applies this after the alpha update.
};

static_assert(sizeof(Float16Storage) == 2);
static_assert(sizeof(SphViscosityDiagnostics) == 16);
