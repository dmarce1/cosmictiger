/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cosmictiger/particles/particle_mesh_sort.hpp>

namespace mesh_domain {

inline double boundaryCoordinate(uint32_t plane, uint32_t meshSize) {
    if (meshSize == 0 || plane > meshSize) {
        throw std::invalid_argument("Invalid mesh-domain boundary");
    }
    return double(plane) / meshSize;
}

// Domain boxes retain their existing double representation. Recover the integer
// plane before comparing particles: double positions cannot distinguish all
// fixed64 particles on opposite sides of an odd-mesh boundary.
inline uint32_t boundaryIndex(double boundary, uint32_t meshSize) {
    if (meshSize == 0 || !std::isfinite(boundary) || boundary < 0 || boundary > 1) {
        throw std::invalid_argument("Invalid mesh-domain boundary");
    }
    return static_cast<uint32_t>(std::llround(boundary * meshSize));
}

template <class Integer>
bool belowBoundary(Integer rawPosition, uint32_t plane, uint32_t meshSize) {
    if (meshSize == 0 || plane > meshSize) {
        throw std::invalid_argument("Invalid mesh-domain boundary");
    }
    return ::detail::particleMeshCoordinate(rawPosition, meshSize) < plane;
}

template <class Integer>
bool contains(Integer rawPosition, uint32_t beginPlane, uint32_t endPlane, uint32_t meshSize) {
    if (meshSize == 0 || beginPlane > endPlane || endPlane > meshSize) {
        throw std::invalid_argument("Invalid mesh-domain interval");
    }
    auto const cell = ::detail::particleMeshCoordinate(rawPosition, meshSize);
    return cell >= beginPlane && cell < endPlane;
}

inline int compareTarget(size_t below, size_t total, uint32_t leftRanks, uint32_t rankCount) {
    if (rankCount == 0 || leftRanks > rankCount || below > total) {
        throw std::invalid_argument("Invalid mesh-domain particle counts");
    }
    unsigned __int128 const actual = static_cast<unsigned __int128>(below) * rankCount;
    unsigned __int128 const target = static_cast<unsigned __int128>(total) * leftRanks;
    return actual < target ? -1 : actual > target ? 1 : 0;
}

namespace detail {

    inline unsigned __int128 absoluteDifference(unsigned __int128 a, unsigned __int128 b) {
        return a < b ? b - a : a - b;
    }

}    // namespace detail

// Minimize the difference between per-locality particle loads in the two child
// groups. Equal scores prefer the rank-proportional geometric division, then
// the lower plane, so empty regions have a reproducible nondegenerate split.
inline bool betterSplit(uint32_t candidatePlane, size_t candidateBelow, uint32_t bestPlane,
    size_t bestBelow, size_t total, uint32_t leftRanks, uint32_t rankCount, uint32_t beginPlane,
    uint32_t endPlane) {
    if (rankCount == 0 || leftRanks > rankCount || candidateBelow > total || bestBelow > total ||
        beginPlane > endPlane || candidatePlane < beginPlane || candidatePlane > endPlane ||
        bestPlane < beginPlane || bestPlane > endPlane) {
        throw std::invalid_argument("Invalid mesh-domain split candidate");
    }
    unsigned __int128 const target = static_cast<unsigned __int128>(total) * leftRanks;
    unsigned __int128 const candidateError = detail::absoluteDifference(
        static_cast<unsigned __int128>(candidateBelow) * rankCount, target);
    unsigned __int128 const bestError =
        detail::absoluteDifference(static_cast<unsigned __int128>(bestBelow) * rankCount, target);
    if (candidateError != bestError) {
        return candidateError < bestError;
    }
    unsigned __int128 const geometricTarget =
        static_cast<unsigned __int128>(endPlane - beginPlane) * leftRanks;
    unsigned __int128 const candidateDistance = detail::absoluteDifference(
        static_cast<unsigned __int128>(candidatePlane - beginPlane) * rankCount, geometricTarget);
    unsigned __int128 const bestDistance = detail::absoluteDifference(
        static_cast<unsigned __int128>(bestPlane - beginPlane) * rankCount, geometricTarget);
    if (candidateDistance != bestDistance) {
        return candidateDistance < bestDistance;
    }
    return candidatePlane < bestPlane;
}

}    // namespace mesh_domain
