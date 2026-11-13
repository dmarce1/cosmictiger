/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/particles/particles.hpp>

// Non-owning particle arrays used for local source arithmetic. Explicit masses
// allow independent moment fixtures; production views use particlesMass().
struct FmmSourceView {
    Array<Position*, kDimensions> positions{};
    char* types = nullptr;
    ParticleInteger count = 0;
    float* masses = nullptr;
};

static_assert(std::is_trivially_copyable_v<FmmSourceView>);

FmmSourceView fmmSourceView();

/// Read an explicit source mass or the mass from the local particle store.
inline float fmmSourceMass(FmmSourceView const& sources, ParticleInteger index) {
    return sources.masses != nullptr ? sources.masses[index] : particlesMass(index);
}
