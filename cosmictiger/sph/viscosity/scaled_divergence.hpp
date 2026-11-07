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
 *   q → scaled divergence; G → divergence; Δτ_r → sample timestep.
 */
CUDA_EXPORT inline float sphDivergenceSign(Float16Storage const& q) {
    float const scaled = q;
    return scaled > 0.0f ? 1.0f : (scaled < 0.0f ? -1.0f : 0.0f);
}

CUDA_EXPORT inline double sphDecodeDivergence(Float16Storage const& q, double Δτ_r) {
    return sph_viscosity::detail::isFinite(Δτ_r) && Δτ_r > 0.0 ? double(float(q)) / Δτ_r : 0.0;
}

// Reject overflow rather than silently clamp a shock history. Startup uses the
// shared base interval for global steps, whose first accepted duration is unknown.
CUDA_EXPORT inline bool sphEncodeDivergence(Float16Storage& q, float G, double Δτ_r) {
    if (!sph_viscosity::detail::isFinite(G) || !sph_viscosity::detail::isFinite(Δτ_r) ||
        !(Δτ_r > 0.0)) {
        return false;
    }
    constexpr double kMaximumHalf = 65504.0;
    double const absoluteDivergence = std::abs(double(G));
    if (absoluteDivergence > 1.0 && Δτ_r > kMaximumHalf / absoluteDivergence) {
        return false;
    }
    double const scaled = double(G) * Δτ_r;
    if (std::abs(scaled) > kMaximumHalf) {
        return false;
    }
    q = static_cast<float>(scaled);
    return true;
}

/*
 * Derived change of normalization when a closed interval receives a new rung:
 *   q_new = q_old 2^(r_old − r_new).
 * No new gradient evaluation or sampling time is implied by this operation.
 * Symbol directory: q_old/q_new → scaled divergence; r_old/r_new → old, new rung.
 * Code names: q holds q_old on entry and q_new on return.
 */
CUDA_EXPORT inline bool sphRescaleDivergence(Float16Storage& q, int r_old, int r_new) {
    if (r_old < 0 || r_old >= kMaxRung || r_new < 0 || r_new >= kMaxRung) {
        return false;
    }
    return sphEncodeDivergence(q, float(q), std::ldexp(1.0, r_old - r_new));
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
 * Δτ → interval (global), or interval 2⁻ʳ (dyadic); r → live rung.
 */
CUDA_EXPORT inline double sphViscositySampleInterval(
    int r, double Δτ, bool globalTimestep, bool initialize = false) {
    if (initialize || !sph_viscosity::detail::isFinite(Δτ) || !(Δτ > 0.0)) {
        return 0.0;
    }
    if (globalTimestep) {
        return Δτ;
    }
    return r >= 0 && r < kMaxRung ? std::ldexp(Δτ, -r) : 0.0;
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
