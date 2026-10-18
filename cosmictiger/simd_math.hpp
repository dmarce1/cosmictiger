/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cosmictiger/simd_backend.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_SIMD_HD __host__ __device__
#else
#define COSMICTIGER_SIMD_HD
#endif

using std::size_t;

template <class Real, class Value>
COSMICTIGER_SIMD_HD inline Real simdBroadcast(Value const& value) {
    if constexpr (std::is_same_v<Real, Value>) {
        return value;
    } else if constexpr (std::is_arithmetic_v<Real>) {
        return static_cast<Real>(value);
    } else {
        return Real(static_cast<typename Real::value_type>(value));
    }
}

template <class Mask, class Real>
COSMICTIGER_SIMD_HD inline Real simdSelect(Mask const& mask, Real const& yes, Real const& no) {
    if constexpr (std::is_same_v<Mask, bool>) {
        return mask ? yes : no;
    } else {
#if COSMICTIGER_SIMD_STANDARD
        return std::simd::select(mask, yes, no);
#elif !defined(__CUDACC__)
        Real result = no;
        std::experimental::where(mask, result) = yes;
        return result;
#endif
    }
}

template <class Real>
COSMICTIGER_SIMD_HD inline Real simdMin(Real const& left, Real const& right) {
    return simdSelect(left < right, left, right);
}

template <class Real>
COSMICTIGER_SIMD_HD inline Real simdMax(Real const& left, Real const& right) {
    return simdSelect(left > right, left, right);
}

template <class Real>
COSMICTIGER_SIMD_HD inline Real simdSqrt(Real const& value) {
    if constexpr (std::is_arithmetic_v<Real>) {
        return std::sqrt(value);
    } else {
#if COSMICTIGER_SIMD_STANDARD
        // Early <simd> implementations lack the optional SIMD math overloads.
        return Real([&](auto lane) { return std::sqrt(value[lane]); });
#elif !defined(__CUDACC__)
        return std::experimental::sqrt(value);
#endif
    }
}

template <class Real>
#if defined(__GNUC__) && !defined(__CUDACC__)
[[gnu::always_inline]]
#endif
COSMICTIGER_SIMD_HD inline Real simdFma(Real const& first, Real const& second, Real const& third) {
    if constexpr (std::is_arithmetic_v<Real>) {
        return std::fma(first, second, third);
    } else {
#if COSMICTIGER_SIMD_STANDARD
        return Real([&](auto lane) { return std::fma(first[lane], second[lane], third[lane]); });
#elif !defined(__CUDACC__)
        // GCC 13 leaves the experimental SIMD math helper out of line in large
        // kernels. This loop exposes the fused operations to the vectorizer
        // without that call boundary and preserves std::fma's fused rounding.
        Real result{};
        for (size_t lane = 0; lane < Real::size(); ++lane) {
            result[lane] = std::fma(first[lane], second[lane], third[lane]);
        }
        return result;
#endif
    }
}

template <class Mask>
COSMICTIGER_SIMD_HD inline bool simdAnyOf(Mask const& mask) {
    if constexpr (std::is_same_v<Mask, bool>) {
        return mask;
    } else {
#if COSMICTIGER_SIMD_STANDARD
        return std::simd::any_of(mask);
#elif !defined(__CUDACC__)
        return std::experimental::any_of(mask);
#endif
    }
}

template <class Mask>
COSMICTIGER_SIMD_HD inline bool simdAllOf(Mask const& mask) {
    if constexpr (std::is_same_v<Mask, bool>) {
        return mask;
    } else {
#if COSMICTIGER_SIMD_STANDARD
        return std::simd::all_of(mask);
#elif !defined(__CUDACC__)
        return std::experimental::all_of(mask);
#endif
    }
}

#ifndef __CUDACC__

template <class Destination, class Source>
inline Destination simdCast(Source const& source) {
#if COSMICTIGER_SIMD_STANDARD
    return Destination(
        [&](auto lane) { return static_cast<typename Destination::value_type>(source[lane]); });
#else
    return std::experimental::static_simd_cast<Destination>(source);
#endif
}

template <class Pack, class Value>
inline void simdSetLane(Pack& pack, size_t lane, Value value) {
    using Element = typename Pack::value_type;
#if COSMICTIGER_SIMD_STANDARD
    // Standard SIMD exposes value-only indexing; preserve the other initialized lanes.
    pack = Pack([&](auto index) {
        return size_t(index) == lane ? static_cast<Element>(value) : pack[index];
    });
#else
    pack[lane] = static_cast<Element>(value);
#endif
}

template <class Pack>
inline auto simdSum(Pack const& value) {
#if COSMICTIGER_SIMD_STANDARD
    return std::simd::reduce(value);
#else
    return std::experimental::reduce(value);
#endif
}

template <class Pack>
inline auto simdMaximum(Pack const& value) {
#if COSMICTIGER_SIMD_STANDARD
    return std::simd::reduce_max(value);
#else
    return std::experimental::hmax(value);
#endif
}

template <class Pack>
inline Pack simdFloor(Pack const& value) {
#if COSMICTIGER_SIMD_STANDARD
    return Pack([&](auto lane) { return std::floor(value[lane]); });
#else
    return std::experimental::floor(value);
#endif
}

template <class Destination, class Source>
inline Destination simdFloorToInt(Source const& source) {
    return simdCast<Destination>(simdFloor(source));
}

template <class Pack>
inline Pack simdRound(Pack const& value) {
    // Unlike std::round, the original kernels use nearest with ties to even.
    return Pack([&](auto lane) {
        using Value = typename Pack::value_type;
#if __has_builtin(__builtin_roundeven)
        if constexpr (std::is_same_v<Value, float>) {
            return __builtin_roundevenf(value[lane]);
        } else {
            return __builtin_roundeven(value[lane]);
        }
#else
        Value const element = value[lane];
        Value const lower = std::floor(element);
        Value const fraction = element - lower;
        Value const rounded = lower +
            Value(fraction > Value(0.5) ||
                (fraction == Value(0.5) && std::fmod(lower, Value(2)) != Value(0)));
        return rounded == Value(0) ? std::copysign(Value(0), element) : rounded;
#endif
    });
}

template <class Pack>
inline Pack simdPositionDifference(Pack const& left, Pack const& right) {
    static_assert(std::is_same_v<typename Pack::value_type, int> && sizeof(int) == 4);
    using UnsignedPack = SimdVector<uint32_t, Pack::size()>;
    // Fixed-point periodic positions subtract modulo 2^32 before signed interpretation.
    return simdCast<Pack>(simdCast<UnsignedPack>(left) - simdCast<UnsignedPack>(right));
}

template <class Pack>
inline Pack simdRsqrt(Pack const& value) {
    return simdBroadcast<Pack>(1) / simdSqrt(value);
}

template <class Pack>
inline Pack simdAbs(Pack const& value) {
    return simdMax(value, -value);
}

template <class Destination, class Source>
inline Destination simdBitCast(Source const& source) {
    static_assert(Destination::size() == Source::size());
    static_assert(sizeof(typename Destination::value_type) == sizeof(typename Source::value_type));
    return Destination(
        [&](auto lane) { return std::bit_cast<typename Destination::value_type>(source[lane]); });
}

#endif

#undef COSMICTIGER_SIMD_HD
