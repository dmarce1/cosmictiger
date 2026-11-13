/* CosmicTiger - FMM geometry for periodic mesh-cell forests; BSL-1.0. */
#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/fmm_box.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/vector3.hpp>

// Zero meshSize preserves the ordinary FMM nearest-image geometry. A forest
// instead fixes the source image by mesh cell, consistently through its tree.
struct FmmGeometry {
    int meshSize = 0;
    bool periodicNear = false;
    Array<int, kDimensions> sinkCell{};
    bool frozenMembership = false;
    uint64_t sourceCellKey = 0;

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & meshSize & periodicNear & sinkCell & frozenMembership & sourceCellKey;
    }
};

namespace detail {

/// Decode one coordinate of a cell identity without consulting its moving center.
CUDA_EXPORT inline int fmmCellCoordinate(uint64_t key, int dim, int meshSize) {
    for (int axis = kDimensions - 1; axis > dim; --axis) {
        key /= meshSize;
    }
    return int(key % meshSize);
}

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

/// Unfold a periodic coordinate about the cell that owned it when the interval opened.
CUDA_EXPORT inline int fmmCellWinding(
    Position point, int cell, int meshSize, FlopCounts* counts = nullptr) {
    addFmmGeometryFlops(counts, 0, 5);
    double const offset = point.toDouble() - (double(cell) + .5) / meshSize;
    return -int(nearbyint(offset));
}

/// Choose the source's periodic image relative to the sink cell, retaining the positive
/// half-box tie convention.
CUDA_EXPORT inline int fmmSourceImage(
    Position source, int dim, FmmGeometry const& geometry, uint64_t sourceKey = ~uint64_t(0)) {
    sourceKey = sourceKey == ~uint64_t(0) ? geometry.sourceCellKey : sourceKey;
    int sourceCell;
    if (geometry.frozenMembership) {
        sourceCell = fmmCellCoordinate(sourceKey, dim, geometry.meshSize);
    } else if constexpr (kPositionBits == 64) {
        sourceCell = int(fixedMultiplyHigh(source.raw(), uint64_t(geometry.meshSize)));
    } else {
        sourceCell = int((uint64_t(source.raw()) * uint64_t(geometry.meshSize)) >> 32);
    }
    int const delta = geometry.sinkCell[dim] - sourceCell;
    int const wrapped = (delta % geometry.meshSize + geometry.meshSize) % geometry.meshSize;
    int const nearest = wrapped > geometry.meshSize / 2 ? wrapped - geometry.meshSize : wrapped;
    return (delta - nearest) / geometry.meshSize;
}

// Subtract before conversion so tiny separations across the Q0.64 seam survive.
CUDA_EXPORT inline double fmmShiftedDifference(
    Position sink, Position source, int image, FlopCounts* counts = nullptr) {
    PositionInteger const a = sink.raw();
    PositionInteger const b = source.raw();
    if (image == 1 && a > b) {
        addFmmGeometryFlops(counts, 0, 1);
        return -double(PositionInteger(b - a)) * Position::kToFloatFactor_;
    }
    if (image == -1 && a < b) {
        addFmmGeometryFlops(counts, 0, 1);
        return double(PositionInteger(a - b)) * Position::kToFloatFactor_;
    }
    addFmmGeometryFlops(counts, 0, 2);
    return unwrappedDoubleDistance(sink, source) - image;
}

/// Retain native fixed-point subtraction while unfolding both points in one frozen cell.
CUDA_EXPORT inline double fmmCellDifference(
    Position target, Position source, int cell, int meshSize, FlopCounts* counts = nullptr) {
    int const image = fmmCellWinding(source, cell, meshSize, counts) -
        fmmCellWinding(target, cell, meshSize, counts);
    return fmmShiftedDifference(target, source, image, counts);
}

/// Subtract a particle from an unsigned boundary in the selected periodic image.
CUDA_EXPORT inline double fmmBoundaryDifference(
    Fixed32 boundary, Position point, int image, FlopCounts* counts = nullptr) {
    return fmmShiftedDifference(Position(boundary), point, image, counts);
}

// Restore the box's unwrapped frame around its owning frozen cell. Its
// midpoint, rather than either endpoint, determines the winding at a seam.
CUDA_EXPORT inline int fmmBoxWinding(
    FmmBox const& box, int dim, int cell, int meshSize, FlopCounts* counts = nullptr) {
    double const center = box.begin[dim].toDouble() + double(box.extentTicks(dim)) * 0x1p-33;
    addFmmGeometryFlops(counts, 0, 8);
    return -int(nearbyint(center - (double(cell) + .5) / meshSize));
}

/// Compute point-to-box distance for explicit image offsets and accumulate geometry
/// operation counts.
CUDA_EXPORT inline float fmmBoxDistanceSquared(FmmBox const& box, Vector3<Position> const& point,
    Array<int, kDimensions> const& image, FlopCounts* counts = nullptr) {
    float result = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        double const lower = fmmBoundaryDifference(box.begin[dim], point[dim], image[dim], counts);
        int const endImage = image[dim] - int(box.full(dim) || box.end[dim] < box.begin[dim]);
        double const upper = fmmBoundaryDifference(box.end[dim], point[dim], endImage, counts);
        float const delta = float(lower > 0 ? lower : (upper < 0 ? -upper : 0));
        result = fmaf(delta, delta, result);
        addFmmGeometryFlops(counts, 2, 1);    // Float FMA and double-to-float delta.
    }
    return result;
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

/// Return sink-minus-source displacement. Periodic near FMM uses the ordinary
/// minimum image; the legacy mesh adapter retains its selected-image convention.
CUDA_EXPORT inline float fmmPositionDifference(Position sink, Position source, int dim,
    FmmGeometry const& geometry, FlopCounts* counts = nullptr, uint64_t sourceKey = ~uint64_t(0)) {
    if (geometry.periodicNear || geometry.meshSize == 0) {
        detail::addFmmGeometryFlops(counts, 1, 0);
        return distance(sink, source);
    }
    detail::addFmmGeometryFlops(counts, 0, 1);    // Narrowing to the force precision.
    sourceKey = sourceKey == ~uint64_t(0) ? geometry.sourceCellKey : sourceKey;
    int image = detail::fmmSourceImage(source, dim, geometry, sourceKey);
    if (geometry.frozenMembership) {
        image +=
            detail::fmmCellWinding(source,
                detail::fmmCellCoordinate(sourceKey, dim, geometry.meshSize), geometry.meshSize) -
            detail::fmmCellWinding(sink, geometry.sinkCell[dim], geometry.meshSize);
        detail::addFmmGeometryFlops(counts, 0, 10);
    }
    return float(detail::fmmShiftedDifference(sink, source, image, counts));
}

/// Measure a source's distance to the sink box using the same image convention as the FMM
/// interaction.
CUDA_EXPORT inline float fmmSinkBoxDistanceSquared(FmmBox const& box,
    Vector3<Position> const& source, FmmGeometry const& geometry, FlopCounts* counts = nullptr,
    uint64_t sourceKey = ~uint64_t(0)) {
    if (geometry.periodicNear || geometry.meshSize == 0) {
        detail::countPeriodicBoxDistance(box, counts);
        return boxDistanceSquared(box, source);
    }
    sourceKey = sourceKey == ~uint64_t(0) ? geometry.sourceCellKey : sourceKey;
    Array<int, kDimensions> image;
    for (int dim = 0; dim < kDimensions; ++dim) {
        image[dim] = detail::fmmSourceImage(source[dim], dim, geometry, sourceKey);
        if (geometry.frozenMembership) {
            image[dim] += detail::fmmCellWinding(source[dim],
                detail::fmmCellCoordinate(sourceKey, dim, geometry.meshSize), geometry.meshSize);
            image[dim] -=
                detail::fmmBoxWinding(box, dim, geometry.sinkCell[dim], geometry.meshSize, counts);
            detail::addFmmGeometryFlops(counts, 0, 5);
        }
    }
    return detail::fmmBoxDistanceSquared(box, source, image, counts);
}

/// Measure a sink's distance to the source box using the image selected from the source
/// center.
CUDA_EXPORT inline float fmmSourceBoxDistanceSquared(FmmBox const& box,
    Vector3<Position> const& sink, Vector3<Position> const& source, FmmGeometry const& geometry,
    FlopCounts* counts = nullptr, uint64_t sourceKey = ~uint64_t(0)) {
    if (geometry.periodicNear || geometry.meshSize == 0) {
        detail::countPeriodicBoxDistance(box, counts);
        return boxDistanceSquared(box, sink);
    }
    sourceKey = sourceKey == ~uint64_t(0) ? geometry.sourceCellKey : sourceKey;
    Array<int, kDimensions> image;
    for (int dim = 0; dim < kDimensions; ++dim) {
        image[dim] = -detail::fmmSourceImage(source[dim], dim, geometry, sourceKey);
        if (geometry.frozenMembership) {
            image[dim] +=
                detail::fmmCellWinding(sink[dim], geometry.sinkCell[dim], geometry.meshSize);
            image[dim] -= detail::fmmBoxWinding(box, dim,
                detail::fmmCellCoordinate(sourceKey, dim, geometry.meshSize), geometry.meshSize,
                counts);
            detail::addFmmGeometryFlops(counts, 0, 5);
        }
    }
    return detail::fmmBoxDistanceSquared(box, sink, image, counts);
}

#ifndef __CUDACC__
/// Evaluate periodic box distances lane by lane when fixed64 precision requires it, with
/// matching work accounting.
inline SimdFloat fmmPeriodicBoxDistanceSquared(
    Range<SimdDouble8> const& box, Vector3<SimdPosition> const& point, FlopCounts* counts) {
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

/// Return sink-minus-source displacement. Periodic near FMM uses the ordinary
/// minimum image; the legacy mesh adapter retains its selected-image convention.
inline SimdFloat fmmPositionDifference(SimdPosition const& sink, SimdPosition const& source,
    int dim, FmmGeometry const& geometry, FlopCounts* counts = nullptr,
    Array<uint64_t, kSimdFloatSize> const* sourceKeys = nullptr) {
    if (geometry.periodicNear || geometry.meshSize == 0) {
        detail::addFmmGeometryFlops(counts, kSimdFloatSize, 0);
        return simdCast<SimdFloat>(simdPositionDifference(sink, source)) *
            math::broadcast<SimdFloat>(kPositionToFloat);
    }
    return SimdFloat([&](auto lane) {
        Position a, b;
        a.setInteger(static_cast<PositionInteger>(sink[lane]));
        b.setInteger(static_cast<PositionInteger>(source[lane]));
        return fmmPositionDifference(
            a, b, dim, geometry, counts, sourceKeys == nullptr ? 0 : (*sourceKeys)[lane]);
    });
}

/// Measure a source's distance to the sink box using the same image convention as the FMM
/// interaction.
inline SimdFloat fmmSinkBoxDistanceSquared(Range<SimdDouble8> const& box,
    Vector3<SimdPosition> const& source, FmmGeometry const& geometry, FlopCounts* counts = nullptr,
    Array<uint64_t, kSimdFloatSize> const* sourceKeys = nullptr) {
    if (geometry.periodicNear || geometry.meshSize == 0) {
        return fmmPeriodicBoxDistanceSquared(box, source, counts);
    }
    return SimdFloat([&](auto lane) {
        FmmBox scalarBox;
        Vector3<Position> point;
        for (int dim = 0; dim < kDimensions; ++dim) {
            scalarBox.setBounds(dim, box.begin[dim][lane], box.end[dim][lane]);
            point[dim].setInteger(static_cast<PositionInteger>(source[dim][lane]));
        }
        detail::addFmmGeometryFlops(counts, 0, 5 * kDimensions);
        return fmmSinkBoxDistanceSquared(
            scalarBox, point, geometry, counts, sourceKeys == nullptr ? 0 : (*sourceKeys)[lane]);
    });
}

/// Measure a sink's distance to the source box using the image selected from the source
/// center.
inline SimdFloat fmmSourceBoxDistanceSquared(Range<SimdDouble8> const& box,
    Vector3<SimdPosition> const& sink, Vector3<SimdPosition> const& source,
    FmmGeometry const& geometry, FlopCounts* counts = nullptr,
    Array<uint64_t, kSimdFloatSize> const* sourceKeys = nullptr) {
    if (geometry.periodicNear || geometry.meshSize == 0) {
        return fmmPeriodicBoxDistanceSquared(box, sink, counts);
    }
    return SimdFloat([&](auto lane) {
        FmmBox scalarBox;
        Vector3<Position> a, b;
        for (int dim = 0; dim < kDimensions; ++dim) {
            scalarBox.setBounds(dim, box.begin[dim][lane], box.end[dim][lane]);
            a[dim].setInteger(static_cast<PositionInteger>(sink[dim][lane]));
            b[dim].setInteger(static_cast<PositionInteger>(source[dim][lane]));
        }
        detail::addFmmGeometryFlops(counts, 0, 5 * kDimensions);
        return fmmSourceBoxDistanceSquared(
            scalarBox, a, b, geometry, counts, sourceKeys == nullptr ? 0 : (*sourceKeys)[lane]);
    });
}
#endif
