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

/// Return the live owned-particle view or the currently registered borrowed ghost view.
FmmSourceView fmmSourceView(FmmParticleStore store) {
    if (store == FmmParticleStore::Ghost) {
        return detail::borrowedSources;
    }
    FmmSourceView view;
    view.positions = particlesX;
    view.types = getOptions().hydro.enabled ? particlesTy : nullptr;
    view.rungs = particlesR;
    view.count = particlesSize();
    return view;
}

/// Replace the ghost source view while rejecting attempts to replace owned particle
/// storage.
void fmmRegisterSourceView(FmmParticleStore store, FmmSourceView view) {
    if (store != FmmParticleStore::Ghost) {
        throw std::invalid_argument("Owned particle storage cannot be replaced by a borrowed view");
    }
    detail::borrowedSources = view;
}

/// Partition the selected source range by one coordinate while keeping its particle fields
/// aligned.
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
                    if (sources.masses != nullptr) {
                        std::swap(sources.masses[lower], sources.masses[upper]);
                    }
                    break;
                }
            }
        }
        ++lower;
    }
    return upper;
}

/// Partition in the expanded frozen-cell frame, preserving every particle field.
ParticleInteger fmmSourcePartitionFrame(FmmParticleStore store, FmmSourceView const& sources,
    Pair<ParticleInteger> range, double split, int dimension) {
    if (!sources.geometry.frozenMembership) {
        return fmmSourcePartition(store, sources, range, Position(split), dimension);
    }
    FlopCounter<> flops;
    auto coordinate = [&](ParticleInteger index) {
        flops += 7;
        Position const point = sources.positions[dimension][index];
        return point.toDouble() +
            detail::fmmCellWinding(
                point, sources.geometry.sinkCell[dimension], sources.geometry.meshSize);
    };
    ParticleInteger lower = range.first;
    ParticleInteger upper = range.second;
    while (lower < upper) {
        if (coordinate(lower) >= split) {
            while (lower != upper) {
                --upper;
                if (coordinate(upper) < split) {
                    if (store == FmmParticleStore::Owned) {
                        particlesSwap(lower, upper);
                    } else {
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            std::swap(sources.positions[dim][lower], sources.positions[dim][upper]);
                        }
                        std::swap(sources.types[lower], sources.types[upper]);
                        std::swap(sources.rungs[lower], sources.rungs[upper]);
                        if (sources.masses != nullptr) {
                            std::swap(sources.masses[lower], sources.masses[upper]);
                        }
                    }
                    break;
                }
            }
        }
        ++lower;
    }
    addCpuFlops(flops, FlopPrecision::Double);
    return upper;
}
