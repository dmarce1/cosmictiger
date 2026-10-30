/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Legacy standalone test targets may still supply the old exclusive switch.
#ifndef COSMICTIGER_INTEGRATOR_HIERARCHICAL
#define COSMICTIGER_INTEGRATOR_HIERARCHICAL 0
#endif
#ifndef COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL
#define COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL (!COSMICTIGER_INTEGRATOR_HIERARCHICAL)
#endif
#ifndef COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
#define COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL COSMICTIGER_INTEGRATOR_HIERARCHICAL
#endif
#ifndef COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL
#define COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL 0
#endif

namespace integration {

inline constexpr bool kBuildHasTraditional = COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL != 0;
inline constexpr bool kBuildHasHierarchical = COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL != 0;
inline constexpr bool kBuildHasGlobal = COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL != 0;
inline constexpr bool kBuildUsesHierarchical = !kBuildHasTraditional && kBuildHasHierarchical;
inline constexpr char const* kBuildTimeIntegrator = kBuildHasTraditional ? "traditional" :
    (kBuildHasHierarchical ? "hierarchical" : "global");
static_assert(kBuildHasTraditional || kBuildHasHierarchical || kBuildHasGlobal,
    "At least one time integrator must be enabled");

}    // namespace integration
