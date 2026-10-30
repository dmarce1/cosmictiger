/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Compatibility inputs for standalone tests that still specify the historical
// single-solver macros. Application builds define every ENABLE_* capability.
#ifndef COSMICTIGER_WITH_GRAVITY
#define COSMICTIGER_WITH_GRAVITY 1
#endif
#ifndef COSMICTIGER_ENABLE_FMM
#if defined(COSMICTIGER_GRAVITY_PMMM)
#define COSMICTIGER_ENABLE_FMM (COSMICTIGER_WITH_GRAVITY && !COSMICTIGER_GRAVITY_PMMM)
#else
#define COSMICTIGER_ENABLE_FMM COSMICTIGER_WITH_GRAVITY
#endif
#endif
#ifndef COSMICTIGER_ENABLE_PMMM
#if defined(COSMICTIGER_GRAVITY_PMMM)
#define COSMICTIGER_ENABLE_PMMM (COSMICTIGER_WITH_GRAVITY && COSMICTIGER_GRAVITY_PMMM)
#else
#define COSMICTIGER_ENABLE_PMMM 0
#endif
#endif
#ifndef COSMICTIGER_ENABLE_PMMM_NEAR_FMM
#if defined(COSMICTIGER_PMMM_NEAR_FMM)
#define COSMICTIGER_ENABLE_PMMM_NEAR_FMM COSMICTIGER_PMMM_NEAR_FMM
#else
#define COSMICTIGER_ENABLE_PMMM_NEAR_FMM 0
#endif
#endif
#ifndef COSMICTIGER_ENABLE_PMMM_NEAR_PP
#define COSMICTIGER_ENABLE_PMMM_NEAR_PP (!COSMICTIGER_ENABLE_PMMM_NEAR_FMM)
#endif
#ifndef ORDER
#error "ORDER (the default potential expansion degree plus one) must be defined"
#endif

namespace gravity {

inline constexpr bool kBuildHasFmm = COSMICTIGER_ENABLE_FMM != 0;
inline constexpr bool kBuildHasPmmm = COSMICTIGER_ENABLE_PMMM != 0;
inline constexpr bool kBuildHasGravity = kBuildHasFmm || kBuildHasPmmm;
inline constexpr bool kBuildHasPmmmNearPp = kBuildHasPmmm && COSMICTIGER_ENABLE_PMMM_NEAR_PP;
inline constexpr bool kBuildHasPmmmNearFmm = kBuildHasPmmm && COSMICTIGER_ENABLE_PMMM_NEAR_FMM;
static_assert(kBuildHasGravity ? ORDER >= 3 && ORDER <= 9 : ORDER == 0,
    "Gravity builds require potential degree 2 through 8; SPH-only builds use order zero");
static_assert(!kBuildHasPmmm || kBuildHasPmmmNearPp || kBuildHasPmmmNearFmm,
    "PMMM requires an enabled near-interaction algorithm");
inline constexpr int kBuildFmmOrder = ORDER;
inline constexpr char const* kBuildGravityMethod =
    kBuildHasFmm ? "fmm" : (kBuildHasPmmm ? "pmmm" : "none");
inline constexpr char const* kBuildPmmmNearMethod =
    kBuildHasPmmmNearPp ? "pp" : (kBuildHasPmmmNearFmm ? "fmm" : "none");

}    // namespace gravity
