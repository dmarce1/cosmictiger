/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cosmictiger/particles/particles.hpp>

struct ParticleMeshCell {
    uint64_t key;
    ParticleInteger begin;
    ParticleInteger end;
    bool ghost = false;
};

// Local owned-particle sort. The range overload preserves every entry outside
// [first, second), including the inactive prefix and saved rung boundaries.
// The caller must finish migration and all particle readers before sorting,
// then synchronize localities before remote reads. No actions are broadcast.
Vector<ParticleMeshCell> particleMeshSort(int meshSize);
Vector<ParticleMeshCell> particleMeshSort(int meshSize, Pair<ParticleInteger> range);

namespace detail {

template <class Integer>
constexpr uint64_t particleMeshCoordinate(Integer rawPosition, uint32_t meshSize) {
    static_assert(std::is_unsigned_v<Integer>);
    static_assert(
        std::numeric_limits<Integer>::digits == 32 || std::numeric_limits<Integer>::digits == 64);
    // Multiplication before truncation preserves the half-open cells for both
    // fixed-point widths, including odd meshes and positions immediately below 1.
    return static_cast<uint64_t>((static_cast<unsigned __int128>(rawPosition) * meshSize) >>
        std::numeric_limits<Integer>::digits);
}

inline uint64_t particleMeshCellCount(int meshSize) {
    if (meshSize <= 0) {
        throw std::invalid_argument("Particle mesh size must be positive");
    }
    uint64_t const n = static_cast<uint64_t>(meshSize);
    if (n > std::numeric_limits<uint64_t>::max() / n / n) {
        throw std::overflow_error("Particle mesh cell keys overflow uint64_t");
    }
    return n * n * n;
}

}    // namespace detail
