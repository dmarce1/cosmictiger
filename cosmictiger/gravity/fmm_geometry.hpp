/* CosmicTiger - FMM geometry for periodic mesh-cell forests; BSL-1.0. */
#pragma once

#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/range.hpp>

// Zero meshSize preserves the ordinary FMM nearest-image geometry. A forest
// instead fixes the source image by mesh cell, consistently through its tree.
struct FmmGeometry {
    int meshSize = 0;
    Array<int, kDimensions> sinkCell{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & meshSize & sinkCell;
    }
};

namespace detail {

CUDA_EXPORT inline int fmmSourceImage(Position source, int dim, FmmGeometry const& geometry) {
    int sourceCell;
    if constexpr (kPositionBits == 64) {
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
CUDA_EXPORT inline double fmmShiftedDifference(Position sink, Position source, int image) {
    PositionInteger const a = sink.raw();
    PositionInteger const b = source.raw();
    if (image == 1 && a > b) {
        return -double(PositionInteger(b - a)) * Position::kToFloatFactor_;
    }
    if (image == -1 && a < b) {
        return double(PositionInteger(a - b)) * Position::kToFloatFactor_;
    }
    return unwrappedDoubleDistance(sink, source) - image;
}

CUDA_EXPORT inline double fmmBoundaryDifference(RangeFixed boundary, Position point, int image) {
    double const bound = boundary.toDouble();
    int const whole = int(floor(bound));
    Position const fraction(bound - whole);
    return fmmShiftedDifference(fraction, point, image - whole);
}

CUDA_EXPORT inline float fmmBoxDistanceSquared(Fixed32Range const& box,
    Array<Position, kDimensions> const& point, Array<int, kDimensions> const& image) {
    float result = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        double const lower = fmmBoundaryDifference(box.begin[dim], point[dim], image[dim]);
        double const upper = fmmBoundaryDifference(box.end[dim], point[dim], image[dim]);
        float const delta = float(lower > 0 ? lower : (upper < 0 ? -upper : 0));
        result = fmaf(delta, delta, result);
    }
    return result;
}

}    // namespace detail

CUDA_EXPORT inline float fmmPositionDifference(
    Position sink, Position source, int dim, FmmGeometry const& geometry) {
    if (geometry.meshSize == 0) {
        return distance(sink, source);
    }
    return float(
        detail::fmmShiftedDifference(sink, source, detail::fmmSourceImage(source, dim, geometry)));
}

CUDA_EXPORT inline float fmmSinkBoxDistanceSquared(Fixed32Range const& box,
    Array<Position, kDimensions> const& source, FmmGeometry const& geometry) {
    if (geometry.meshSize == 0) {
        return boxDistanceSquared(box, source);
    }
    Array<int, kDimensions> image;
    for (int dim = 0; dim < kDimensions; ++dim) {
        image[dim] = detail::fmmSourceImage(source[dim], dim, geometry);
    }
    return detail::fmmBoxDistanceSquared(box, source, image);
}

CUDA_EXPORT inline float fmmSourceBoxDistanceSquared(Fixed32Range const& box,
    Array<Position, kDimensions> const& sink, Array<Position, kDimensions> const& source,
    FmmGeometry const& geometry) {
    if (geometry.meshSize == 0) {
        return boxDistanceSquared(box, sink);
    }
    Array<int, kDimensions> image;
    for (int dim = 0; dim < kDimensions; ++dim) {
        image[dim] = -detail::fmmSourceImage(source[dim], dim, geometry);
    }
    return detail::fmmBoxDistanceSquared(box, sink, image);
}

#ifndef __CUDACC__
inline SimdFloat fmmPositionDifference(
    SimdPosition const& sink, SimdPosition const& source, int dim, FmmGeometry const& geometry) {
    if (geometry.meshSize == 0) {
        return simdCast<SimdFloat>(simdPositionDifference(sink, source)) *
            simdBroadcast<SimdFloat>(kPositionToFloat);
    }
    return SimdFloat([&](auto lane) {
        Position a, b;
        a.setInteger(static_cast<PositionInteger>(sink[lane]));
        b.setInteger(static_cast<PositionInteger>(source[lane]));
        return fmmPositionDifference(a, b, dim, geometry);
    });
}

inline SimdFloat fmmSinkBoxDistanceSquared(Range<SimdDouble8> const& box,
    Array<SimdPosition, kDimensions> const& source, FmmGeometry const& geometry) {
    if (geometry.meshSize == 0) {
        return boxDistanceSquared(box, source);
    }
    return SimdFloat([&](auto lane) {
        Fixed32Range scalarBox;
        Array<Position, kDimensions> point;
        for (int dim = 0; dim < kDimensions; ++dim) {
            scalarBox.begin[dim] = box.begin[dim][lane];
            scalarBox.end[dim] = box.end[dim][lane];
            point[dim].setInteger(static_cast<PositionInteger>(source[dim][lane]));
        }
        return fmmSinkBoxDistanceSquared(scalarBox, point, geometry);
    });
}

inline SimdFloat fmmSourceBoxDistanceSquared(Range<SimdDouble8> const& box,
    Array<SimdPosition, kDimensions> const& sink, Array<SimdPosition, kDimensions> const& source,
    FmmGeometry const& geometry) {
    if (geometry.meshSize == 0) {
        return boxDistanceSquared(box, sink);
    }
    return SimdFloat([&](auto lane) {
        Fixed32Range scalarBox;
        Array<Position, kDimensions> a, b;
        for (int dim = 0; dim < kDimensions; ++dim) {
            scalarBox.begin[dim] = box.begin[dim][lane];
            scalarBox.end[dim] = box.end[dim][lane];
            a[dim].setInteger(static_cast<PositionInteger>(sink[dim][lane]));
            b[dim].setInteger(static_cast<PositionInteger>(source[dim][lane]));
        }
        return fmmSourceBoxDistanceSquared(scalarBox, a, b, geometry);
    });
}
#endif
