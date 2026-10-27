/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#ifndef COSMICTIGER_GRAVITY_PMMM
#error "COSMICTIGER_GRAVITY_PMMM must be defined by the build configuration"
#endif
#ifndef COSMICTIGER_PMMM_NEAR_FMM
#error "COSMICTIGER_PMMM_NEAR_FMM must be defined by the build configuration"
#endif
#ifndef ORDER
#error "ORDER (COSMICTIGER_WITH_ORDER) must be defined by the build configuration"
#endif

namespace gravity {

static_assert(COSMICTIGER_GRAVITY_PMMM == 0 || COSMICTIGER_GRAVITY_PMMM == 1);
static_assert(COSMICTIGER_PMMM_NEAR_FMM == 0 || COSMICTIGER_PMMM_NEAR_FMM == 1);
static_assert(ORDER >= 3 && ORDER <= 9, "COSMICTIGER_WITH_ORDER must be from 3 to 9");

inline constexpr bool kBuildUsesPmmm = COSMICTIGER_GRAVITY_PMMM != 0;
inline constexpr bool kBuildPmmmNearUsesFmm = COSMICTIGER_PMMM_NEAR_FMM != 0;
inline constexpr int kBuildFmmOrder = ORDER;
inline constexpr char const* kBuildGravityMethod = kBuildUsesPmmm ? "pmmm" : "fmm";
inline constexpr char const* kBuildPmmmNearMethod = kBuildPmmmNearUsesFmm ? "fmm" : "pp";

}    // namespace gravity
