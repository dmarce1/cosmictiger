/* CosmicTiger - exact local FMM cell geometry; BSL-1.0. */

// Exact dyadic cell identities, axis schedules, periodic bounds and conservative radii.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/gravity/fmm_box.hpp>
#include <cosmictiger/math/position.hpp>

// Cell bounds must remain exact in FmmBox, and centers must remain exact in Position.
inline constexpr int kGeometricTreeMaxAxisCuts =
    std::min(std::numeric_limits<uint32_t>::digits, kPositionBits - 1);
inline constexpr int kGeometricTreeMaxDepth =
    std::min(kDimensions * kGeometricTreeMaxAxisCuts, kMaxDepth - 1);
inline constexpr std::array<int, 18> kGeometricSplitAxes{
    kXi, kYi, kZi, kZi, kYi, kXi, kZi, kXi, kYi, kYi, kXi, kZi, kYi, kZi, kXi, kXi, kZi, kYi};

struct GeometricTreeCell {
    std::array<uint64_t, kDimensions> coordinate{};    // Cell index along each axis at this depth.
    uint16_t depth = 0;    // Number of binary splits from the unit-box root.

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        for (auto& value : coordinate) {
            archive & value;
        }
        archive & depth;
    }
};

inline constexpr std::array<int, kDimensions> geometricTreeAxisCuts(int depth) {
    std::array<int, kDimensions> cuts{};
    cuts.fill(depth / kDimensions);
    int const base = depth - depth % kDimensions;
    for (int i = base; i < depth; ++i) {
        ++cuts[kGeometricSplitAxes[i % kGeometricSplitAxes.size()]];
    }
    return cuts;
}

inline GeometricTreeCell geometricTreeChild(GeometricTreeCell cell, bool right) {
    if (cell.depth >= kGeometricTreeMaxDepth) {
        throw std::out_of_range("Geometric tree maximum depth exceeded");
    }
    int const axis = kGeometricSplitAxes[cell.depth % kGeometricSplitAxes.size()];
    cell.coordinate[axis] = 2 * cell.coordinate[axis] + unsigned(right);
    ++cell.depth;
    return cell;
}

inline Position geometricTreeCenter(GeometricTreeCell const& cell, int axis) {
    auto const cuts = geometricTreeAxisCuts(cell.depth);
    Position center;
    center.raw() =
        PositionInteger((2 * cell.coordinate[axis] + 1) << (kPositionBits - cuts[axis] - 1));
    return center;
}

// Store exact dyadic bounds as Fixed32 ticks, including the seam at 1.
inline FmmBox geometricTreeBox(GeometricTreeCell const& cell) {
    constexpr int kBoxBits = std::numeric_limits<uint32_t>::digits;
    FmmBox box{};
    auto const cuts = geometricTreeAxisCuts(cell.depth);
    for (int axis = 0; axis < kDimensions; ++axis) {
        int const n = cuts[axis];
        uint64_t const lower = cell.coordinate[axis] << (kBoxBits - n);
        uint64_t const upper = (cell.coordinate[axis] + 1) << (kBoxBits - n);
        box.begin[axis].setInteger(uint32_t(lower));
        box.end[axis].setInteger(uint32_t(upper));
        if (upper - lower == (uint64_t(1) << kBoxBits)) {
            box.fullAxes = uint8_t(box.fullAxes | (1u << axis));
        }
    }
    return box;
}

// Derived geometric half-diagonal: R(d) = ½√(Σᵢ Lᵢ(d)²).
// Symbol directory: R → cell radius limit; Lᵢ → axis side length; d → depth.
// Code names: R → result; Lᵢ²/4 → square; d → depth.
// Each operation is rounded upward under round-to-nearest.
inline float geometricTreeRadiusLimit(int depth) {
    auto const cuts = geometricTreeAxisCuts(depth);
    double sum = 0;
    double const infinity = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < kDimensions; ++axis) {
        double const square = std::ldexp(1.0, -2 * (cuts[axis] + 1));
        sum = std::nextafter(sum + square, infinity);
    }
    double const radius = std::nextafter(std::sqrt(sum), infinity);
    float result = float(radius);
    if (double(result) < radius) {
        result = std::nextafter(result, std::numeric_limits<float>::infinity());
    }
    return result;
}
