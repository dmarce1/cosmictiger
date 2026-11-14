/* CosmicTiger - exact local FMM cell geometry; BSL-1.0. */
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

// Coordinates identify dyadic cells independently of the rounded FmmBox.
// A child at the maximum depth must still have an exactly representable center.
inline constexpr int kGeometricTreeMaxDepth = std::min(3 * (kPositionBits - 1), kMaxDepth - 1);
inline constexpr std::array<int, 18> kGeometricSplitAxes{
    0, 1, 2, 2, 1, 0, 2, 0, 1, 1, 0, 2, 1, 2, 0, 0, 2, 1};

struct GeometricTreeCell {
    std::array<uint64_t, 3> coordinate{};
    uint16_t depth = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        for (auto& value : coordinate) {
            archive & value;
        }
        archive & depth;
    }
};

inline constexpr std::array<int, 3> geometricTreeAxisCuts(int depth) {
    std::array<int, 3> cuts{depth / 3, depth / 3, depth / 3};
    int const base = depth - depth % 3;
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

// Round exact dyadic bounds outward to Fixed32 ticks, including the seam at 1.
inline FmmBox geometricTreeBox(GeometricTreeCell const& cell) {
    FmmBox box{};
    auto const cuts = geometricTreeAxisCuts(cell.depth);
    for (int axis = 0; axis < 3; ++axis) {
        int const n = cuts[axis];
        uint64_t lower;
        uint64_t upper;
        if (n <= 32) {
            lower = cell.coordinate[axis] << (32 - n);
            upper = (cell.coordinate[axis] + 1) << (32 - n);
        } else {
            lower = cell.coordinate[axis] >> (n - 32);
            upper = lower + 1;
        }
        box.begin[axis].setInteger(uint32_t(lower));
        box.end[axis].setInteger(uint32_t(upper));
        if (upper - lower == (uint64_t(1) << 32)) {
            box.fullAxes = uint8_t(box.fullAxes | (1u << axis));
        }
    }
    return box;
}

// Certify the square root without relying on the library sqrt's error bound:
// one downward step bounds each nonnegative rounded square from below.
inline long double geometricTreeSqrtUp(long double value) {
    if (value == 0) {
        return 0;
    }
    long double root = std::sqrt(value);
    long double const infinity = std::numeric_limits<long double>::infinity();
    while (std::nextafter(root * root, -infinity) < value) {
        root = std::nextafter(root, infinity);
    }
    return root;
}

// Derived geometric half-diagonal: R(d) = ½√(Σᵢ Lᵢ(d)²).
// Symbol directory: R → cell radius limit; Lᵢ → axis side length; d → depth.
// Code names: R → result; Lᵢ²/4 → square; d → depth.
// Each operation is rounded upward under round-to-nearest.
inline float geometricTreeRadiusLimit(int depth) {
    auto const cuts = geometricTreeAxisCuts(depth);
    long double sum = 0;
    long double const infinity = std::numeric_limits<long double>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        long double const square = std::ldexp(1.0L, -2 * (cuts[axis] + 1));
        sum = std::nextafter(sum + square, infinity);
    }
    long double const radius = geometricTreeSqrtUp(sum);
    float result = float(radius);
    if (static_cast<long double>(result) < radius) {
        result = std::nextafter(result, std::numeric_limits<float>::infinity());
    }
    return result;
}
