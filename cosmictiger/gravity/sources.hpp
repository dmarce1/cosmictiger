/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/particles/particles.hpp>

enum class FmmParticleStore { Owned, Ghost };

// The view is passed by value to device work. A node's particle indices refer
// only to its selected store; sink indices always refer to owned particles.
struct FmmSourceView {
    Array<Position*, kDimensions> positions{};
    char* types = nullptr;
    char* rungs = nullptr;
    ParticleInteger count = 0;
};

static_assert(std::is_trivially_copyable_v<FmmSourceView>);

FmmSourceView fmmSourceView(FmmParticleStore store);
// Publish borrowed source storage before building trees or launching readers.
// Ownership and lifetime remain with the solver that supplies the view.
void fmmRegisterSourceView(FmmParticleStore store, FmmSourceView view);
ParticleInteger fmmSourcePartition(FmmParticleStore store, FmmSourceView const& sources,
    Pair<ParticleInteger> range, Position split, int dimension);

inline float fmmSourceMass(FmmSourceView const& sources, ParticleInteger index) {
    if (!getOptions().hydro.enabled) {
        return 1.0f;
    }
    return sources.types[index] == kDarkMatterType ? getOptions().initial.darkMatterMass :
                                                     getOptions().initial.gasMass;
}
