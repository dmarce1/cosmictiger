/* CosmicTiger - unsigned periodic FMM bounds; BSL-1.0. */
#pragma once

#include <cmath>
#include <cstdint>

#include <cosmictiger/math/range.hpp>

// Each axis runs forward from begin to end, modulo one. Equal endpoints
// describe a point unless that axis is marked full. Do not inherit Range's
// ordered-endpoint operations: a wrapped interval has begin > end.
struct FmmBox {
    Array<Fixed32, kDimensions> begin;
    Array<Fixed32, kDimensions> end;
    uint8_t fullAxes = 0;

    CUDA_EXPORT bool full(int dim) const {
        return (fullAxes & (1u << dim)) != 0;
    }

    CUDA_EXPORT uint64_t extentTicks(int dim) const {
        return full(dim) ? (uint64_t(1) << 32) : uint32_t(end[dim].raw() - begin[dim].raw());
    }

    // Convert finite, ordered, unwrapped tree bounds (in box units) outward.
    // Integer ticks preserve a seam endpoint at one and distinguish a full
    // axis from a point, including when rounding fills the last missing tick.
    CUDA_EXPORT void setBounds(int dim, double lower, double upper) {
        int64_t const lowerTicks = int64_t(floor(lower * 0x1p32));
        int64_t const upperTicks = int64_t(ceil(upper * 0x1p32));
        begin[dim].setInteger(uint32_t(lowerTicks));
        end[dim].setInteger(uint32_t(upperTicks));
        uint8_t const bit = uint8_t(1u << dim);
        fullAxes = uint8_t(fullAxes & ~bit);
        if (upperTicks - lowerTicks >= (int64_t(1) << 32)) {
            fullAxes = uint8_t(fullAxes | bit);
        }
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & begin & end & fullAxes;
    }
};

// Compare in the particle's native integer width. Fixed64 particles therefore
// retain separations smaller than a double ULP at a fixed32 box endpoint.
CUDA_EXPORT inline float boxDistanceSquared(FmmBox const& box, Vector3<Position> const& point) {
    float result = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        if (box.full(dim)) {
            continue;
        }
        PositionInteger const lower = Position(box.begin[dim]).raw();
        PositionInteger const length = Position(box.end[dim]).raw() - lower;
        PositionInteger const offset = point[dim].raw() - lower;
        PositionInteger ticks = 0;
        if (offset > length) {
            PositionInteger const fromUpper = offset - length;
            PositionInteger const fromLower = PositionInteger{0} - offset;
            ticks = fromUpper < fromLower ? fromUpper : fromLower;
        }
        float const delta = float(double(ticks) * Position::kToFloatFactor_);
        result = fmaf(delta, delta, result);
    }
    return result;
}

CUDA_EXPORT inline bool boxIntersectsSphere(
    FmmBox const& box, Vector3<Position> const& point, float radius) {
    return boxDistanceSquared(box, point) < sqr(radius);
}
