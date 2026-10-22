/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cstdint>

#include <cosmictiger/math/fixed.hpp>

#ifndef COSMICTIGER_POSITION_BITS
#define COSMICTIGER_POSITION_BITS 32
#endif

static_assert(COSMICTIGER_POSITION_BITS == 32 || COSMICTIGER_POSITION_BITS == 64);
inline constexpr int kPositionBits = COSMICTIGER_POSITION_BITS;
#if COSMICTIGER_POSITION_BITS == 64
using Position = Fixed64;
using PositionInteger = uint64_t;
using SignedPositionInteger = int64_t;
inline constexpr float kPositionToFloat = 0x1p-64f;
#else
using Position = Fixed32;
using PositionInteger = uint32_t;
using SignedPositionInteger = int32_t;
inline constexpr float kPositionToFloat = 0x1p-32f;
#endif

// Domain boundaries are doubles and use half-open intervals. Round downward
// when converting a coordinate so a point just below a boundary stays below it.
CUDA_EXPORT inline double positionDomainCoordinate(Position position) {
    double coordinate = position.toDouble();
    if constexpr (kPositionBits == 64) {
        if (coordinate >= 1.0 || Position(coordinate).raw() > position.raw()) {
            coordinate = ::nextafter(coordinate, 0.0);
        }
    }
    return coordinate;
}

#ifndef __CUDACC__
using SimdPosition = SimdVector<SignedPositionInteger, SIMD_FLOAT_SIZE>;
#endif
