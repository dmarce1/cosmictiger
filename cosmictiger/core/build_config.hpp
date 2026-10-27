/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#ifndef COSMICTIGER_INTEGRATOR_HIERARCHICAL
#error "COSMICTIGER_INTEGRATOR_HIERARCHICAL must be defined by the build configuration"
#endif

namespace integration {

static_assert(COSMICTIGER_INTEGRATOR_HIERARCHICAL == 0 || COSMICTIGER_INTEGRATOR_HIERARCHICAL == 1);

inline constexpr bool kBuildUsesHierarchical = COSMICTIGER_INTEGRATOR_HIERARCHICAL != 0;
inline constexpr char const* kBuildTimeIntegrator =
    kBuildUsesHierarchical ? "hierarchical" : "leapfrog";

}    // namespace integration
