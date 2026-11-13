/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Application builds define COSMICTIGER_WITH_GRAVITY; standalone tests may not.
#ifndef COSMICTIGER_WITH_GRAVITY
#define COSMICTIGER_WITH_GRAVITY 1
#endif
#ifndef ORDER
#error "ORDER (the build potential expansion degree plus one) must be defined"
#endif

namespace gravity {

// The gravity solver is the global periodic FMM.
// Close leaf interactions use the FMM tree walk's particle-pair kernels.
inline constexpr bool kBuildHasGravity = COSMICTIGER_WITH_GRAVITY != 0;
static_assert(kBuildHasGravity ? ORDER >= 3 && ORDER <= 9 : ORDER == 0,
    "Gravity builds require potential degree 2 through 8; SPH-only builds use order zero");
inline constexpr int kBuildFmmOrder = ORDER;

}    // namespace gravity
