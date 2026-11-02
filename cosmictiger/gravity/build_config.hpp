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
#ifndef COSMICTIGER_ENABLE_PMMM_FMM
#if defined(COSMICTIGER_GRAVITY_PMMM) && defined(COSMICTIGER_PMMM_NEAR_FMM)
#define COSMICTIGER_ENABLE_PMMM_FMM \
    (COSMICTIGER_WITH_GRAVITY && COSMICTIGER_GRAVITY_PMMM && COSMICTIGER_PMMM_NEAR_FMM)
#else
#define COSMICTIGER_ENABLE_PMMM_FMM 0
#endif
#endif
#ifndef COSMICTIGER_ENABLE_PMMM_PP
#if defined(COSMICTIGER_GRAVITY_PMMM)
#define COSMICTIGER_ENABLE_PMMM_PP \
    (COSMICTIGER_WITH_GRAVITY && COSMICTIGER_GRAVITY_PMMM && !COSMICTIGER_ENABLE_PMMM_FMM)
#else
#define COSMICTIGER_ENABLE_PMMM_PP 0
#endif
#endif
#ifndef COSMICTIGER_ENABLE_PM
#define COSMICTIGER_ENABLE_PM 0
#endif
#ifndef COSMICTIGER_ENABLE_P3M_PP
#define COSMICTIGER_ENABLE_P3M_PP 0
#endif
#ifndef COSMICTIGER_ENABLE_P3M_FMM
#define COSMICTIGER_ENABLE_P3M_FMM 0
#endif
// Every PM variant shares the scalar PM/FFT solve and the mandatory outer PM interval.
#define COSMICTIGER_WITH_PARTICLE_MESH \
    (COSMICTIGER_ENABLE_PM || COSMICTIGER_ENABLE_P3M_PP || COSMICTIGER_ENABLE_P3M_FMM)
#define COSMICTIGER_WITH_P3M (COSMICTIGER_ENABLE_P3M_PP || COSMICTIGER_ENABLE_P3M_FMM)
#ifndef ORDER
#error "ORDER (the default potential expansion degree plus one) must be defined"
#endif

namespace gravity {

inline constexpr bool kBuildHasFmm = COSMICTIGER_ENABLE_FMM != 0;
inline constexpr bool kBuildHasPmmm = COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM;
inline constexpr bool kBuildHasPm = COSMICTIGER_ENABLE_PM != 0;
inline constexpr bool kBuildHasP3mPp = COSMICTIGER_ENABLE_P3M_PP != 0;
inline constexpr bool kBuildHasP3mFmm = COSMICTIGER_ENABLE_P3M_FMM != 0;
inline constexpr bool kBuildHasP3m = kBuildHasP3mPp || kBuildHasP3mFmm;
inline constexpr bool kBuildHasParticleMesh = kBuildHasPm || kBuildHasP3m;
inline constexpr bool kBuildHasGravity = kBuildHasFmm || kBuildHasPmmm || kBuildHasParticleMesh;
inline constexpr bool kBuildHasPmmmNearPp = COSMICTIGER_ENABLE_PMMM_PP != 0;
inline constexpr bool kBuildHasPmmmNearFmm = COSMICTIGER_ENABLE_PMMM_FMM != 0;
static_assert(kBuildHasGravity ? ORDER >= 3 && ORDER <= 9 : ORDER == 0,
    "Gravity builds require potential degree 2 through 8; SPH-only builds use order zero");
inline constexpr int kBuildFmmOrder = ORDER;
inline constexpr char const* kBuildGravityMethod = kBuildHasFmm ?
    "fmm" :
    (kBuildHasPmmm ? "pmmm" : (kBuildHasP3m ? "p3m" : (kBuildHasPm ? "pm" : "none")));
inline constexpr char const* kBuildPmmmNearMethod =
    kBuildHasPmmmNearPp ? "pp" : (kBuildHasPmmmNearFmm ? "fmm" : "none");
inline constexpr char const* kBuildP3mNearMethod =
    kBuildHasP3mPp ? "pp" : (kBuildHasP3mFmm ? "fmm" : "none");

}    // namespace gravity
