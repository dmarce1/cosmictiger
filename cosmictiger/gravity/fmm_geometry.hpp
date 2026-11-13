/* CosmicTiger - periodic FMM geometry; BSL-1.0. */
#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/fmm_box.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace detail {

/// Accumulate precision-separated geometry work when counting and an output record are
/// enabled.
CUDA_EXPORT inline void addFmmGeometryFlops(
    FlopCounts* counts, FlopCount single, FlopCount doublePrecision) {
    if constexpr (kCountFlops) {
        if (counts != nullptr) {
            counts->singlePrecision += single;
            counts->doublePrecision += doublePrecision;
        }
    }
}

/// Account for the conversions and arithmetic used by periodic fixed-box distance
/// evaluation.
CUDA_EXPORT inline void countPeriodicBoxDistance(FmmBox const& box, FlopCounts* counts) {
    if constexpr (kCountFlops) {
        if (counts != nullptr) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                if (!box.full(dim)) {
                    // Integer displacement scales in double then narrows for the FMA.
                    addFmmGeometryFlops(counts, 2, 2);
                }
            }
        }
    }
}

}    // namespace detail

/// Return the minimum-image sink-minus-source displacement.
CUDA_EXPORT inline float fmmPositionDifference(
    Position sink, Position source, FlopCounts* counts = nullptr) {
    detail::addFmmGeometryFlops(counts, 1, 0);
    return distance(sink, source);
}

/// Measure periodic point-to-box distance and count its arithmetic by precision.
CUDA_EXPORT inline float fmmBoxDistanceSquared(
    FmmBox const& box, Vector3<Position> const& point, FlopCounts* counts = nullptr) {
    detail::countPeriodicBoxDistance(box, counts);
    return boxDistanceSquared(box, point);
}

#ifndef __CUDACC__
/// Evaluate periodic box distances lane by lane when fixed64 precision requires it, with
/// matching work accounting.
inline SimdFloat fmmBoxDistanceSquared(Range<SimdDouble8> const& box,
    Vector3<SimdPosition> const& point, FlopCounts* counts = nullptr) {
    if constexpr (kPositionBits == 64) {
        return SimdFloat([&](auto lane) {
            FmmBox scalarBox;
            Vector3<Position> scalarPoint;
            for (int dim = 0; dim < kDimensions; ++dim) {
                scalarBox.setBounds(dim, box.begin[dim][lane], box.end[dim][lane]);
                scalarPoint[dim].setInteger(static_cast<PositionInteger>(point[dim][lane]));
            }
            detail::addFmmGeometryFlops(counts, 0, 5 * kDimensions);
            detail::countPeriodicBoxDistance(scalarBox, counts);
            return boxDistanceSquared(scalarBox, scalarPoint);
        });
    } else {
        // Per dimension: nine double operations/conversions and one float FMA.
        detail::addFmmGeometryFlops(
            counts, 2 * kDimensions * kSimdFloatSize, 9 * kDimensions * kSimdFloatSize);
        return boxDistanceSquared(box, point);
    }
}

/// Return minimum-image sink-minus-source displacements for a SIMD packet.
inline SimdFloat fmmPositionDifference(
    SimdPosition const& sink, SimdPosition const& source, FlopCounts* counts = nullptr) {
    detail::addFmmGeometryFlops(counts, kSimdFloatSize, 0);
    return simdCast<SimdFloat>(simdPositionDifference(sink, source)) *
        math::broadcast<SimdFloat>(kPositionToFloat);
}
#endif
