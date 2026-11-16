// Borrowed gravity sources, remote position access, and species-dependent mass lookup.
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/particles/particles.hpp>

struct TreeNode;

using GravityRemoteSourceReader = void (*)(
    TreeNode const&, Position*, Position*, Position*, char*, ParticleInteger);
void gravitySetRemoteSourceReader(GravityRemoteSourceReader reader);
void gravityReadSourcePositions(
    TreeNode const&, Position*, Position*, Position*, char*, ParticleInteger);

// Non-owning position and species arrays used for local source arithmetic.
struct FmmSourceView {
    Array<Position*, kDimensions> positions{};    // Native fixed-point coordinates by axis.
    char* types = nullptr;                        // Species labels; null denotes dark matter only.
    ParticleInteger count = 0;    // Number of readable particles in each source array.
};

static_assert(std::is_trivially_copyable_v<FmmSourceView>);

FmmSourceView fmmSourceView();

/// Derive source mass from this view's species, independent of the local particle store.
inline float fmmSourceMass(FmmSourceView const& sources, ParticleInteger index) {
    return particlesMassForType(sources.types != nullptr ? sources.types[index] : kDarkMatterType);
}
