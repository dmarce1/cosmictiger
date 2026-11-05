/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Every application and standalone consumer must receive its actual configured
// capabilities. Missing definitions must not silently change data layouts.
#if !defined(COSMICTIGER_ENABLE_SPH) || !defined(COSMICTIGER_ENABLE_ROCKSTAR) ||    \
    !defined(COSMICTIGER_ENABLE_CHEMISTRY) || !defined(COSMICTIGER_ENABLE_STARS) || \
    !defined(COSMICTIGER_ENABLE_STAR_FORMATION) ||                                  \
    !defined(COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL) ||                          \
    !defined(COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL) ||                         \
    !defined(COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL)
#error "Missing CosmicTiger build capabilities; link cosmictiger_default_config"
#endif
#if COSMICTIGER_ENABLE_STARS != COSMICTIGER_ENABLE_STAR_FORMATION
#error "COSMICTIGER_ENABLE_STARS and COSMICTIGER_ENABLE_STAR_FORMATION must match"
#endif
#if COSMICTIGER_ENABLE_CHEMISTRY && !COSMICTIGER_ENABLE_SPH
#error "The compiled chemistry capability requires SPH"
#endif
#if COSMICTIGER_ENABLE_STARS && !COSMICTIGER_ENABLE_CHEMISTRY
#error "The compiled stars capability requires chemistry"
#endif

#if COSMICTIGER_DIMENSION_COUNT < 3 && (COSMICTIGER_WITH_GRAVITY || !COSMICTIGER_ENABLE_SPH)
#error "Lower-dimensional builds require SPH without gravity"
#endif

namespace integration {

inline constexpr bool kBuildHasTraditional = COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL != 0;
inline constexpr bool kBuildHasHierarchical = COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL != 0;
inline constexpr bool kBuildHasGlobal = COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL != 0;
inline constexpr bool kBuildUsesHierarchical = !kBuildHasTraditional && kBuildHasHierarchical;
inline constexpr char const* kBuildTimeIntegrator =
    kBuildHasTraditional ? "traditional" : (kBuildHasHierarchical ? "hierarchical" : "global");
static_assert(kBuildHasTraditional || kBuildHasHierarchical || kBuildHasGlobal,
    "At least one time integrator must be enabled");

}    // namespace integration
