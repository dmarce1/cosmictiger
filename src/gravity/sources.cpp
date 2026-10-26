/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <stdexcept>

#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/particles/particle_partition.hpp>

namespace detail {
static FmmSourceView borrowedSources;
}    // namespace detail

FmmSourceView fmmSourceView(FmmParticleStore store) {
    if (store == FmmParticleStore::Ghost) {
        return detail::borrowedSources;
    }
    return {particlesX, getOptions().hydro.enabled ? particlesTy : nullptr, particlesR,
        particlesSize()};
}

void fmmRegisterSourceView(FmmParticleStore store, FmmSourceView view) {
    if (store != FmmParticleStore::Ghost) {
        throw std::invalid_argument("Owned particle storage cannot be replaced by a borrowed view");
    }
    detail::borrowedSources = view;
}

ParticleInteger fmmSourcePartition(FmmParticleStore store, FmmSourceView const& sources,
    Pair<ParticleInteger> range, Position split, int dimension) {
    if (store == FmmParticleStore::Owned) {
        return particlesPartition(range, split, dimension);
    }
    ParticleInteger lower = range.first;
    ParticleInteger upper = range.second;
    while (lower < upper) {
        if (sources.positions[dimension][lower] >= split) {
            while (lower != upper) {
                --upper;
                if (sources.positions[dimension][upper] < split) {
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        std::swap(sources.positions[dim][lower], sources.positions[dim][upper]);
                    }
                    std::swap(sources.types[lower], sources.types[upper]);
                    std::swap(sources.rungs[lower], sources.rungs[upper]);
                    break;
                }
            }
        }
        ++lower;
    }
    return upper;
}
