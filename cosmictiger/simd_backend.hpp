/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

// SIMD packs are host-only. CUDA instantiates the shared kernels with scalars.
#ifndef __CUDACC__

#ifndef COSMICTIGER_SIMD_USE_EXPERIMENTAL
#define COSMICTIGER_SIMD_USE_EXPERIMENTAL 0
#endif

#if !COSMICTIGER_SIMD_USE_EXPERIMENTAL && __cplusplus > 202302L && __has_include(<simd>)
#include <simd>
#endif

// Some libstdc++ releases provide <simd> before advertising the complete feature.
#if !COSMICTIGER_SIMD_USE_EXPERIMENTAL && (defined(__cpp_lib_simd) || defined(__glibcxx_simd))
#define COSMICTIGER_SIMD_STANDARD 1
#else
#define COSMICTIGER_SIMD_STANDARD 0
#if __has_include(<experimental/simd>)
#include <experimental/simd>
#else
#error "CosmicTiger requires <simd> in C++26 mode or <experimental/simd>"
#endif
#endif

namespace detail {
#if COSMICTIGER_SIMD_STANDARD
template <class Value, int laneCount>
using SimdStorage = std::simd::vec<Value, laneCount>;
#else
template <class Value, int laneCount>
using SimdStorage = std::experimental::fixed_size_simd<Value, laneCount>;
#endif
}    // namespace detail

template <class Value, int laneCount>
using SimdVector = detail::SimdStorage<Value, laneCount>;

// Logical widths are independent of the target's hardware register width.
#define SIMD_FLOAT4_SIZE 4
#define SIMD_FLOAT8_SIZE 8
#define SIMD_FLOAT16_SIZE 16
#define SIMD_FLOAT_SIZE SIMD_FLOAT8_SIZE

using SimdFloat4 = SimdVector<float, 4>;
using SimdFloat8 = SimdVector<float, 8>;
using SimdFloat16 = SimdVector<float, 16>;
using SimdInt4 = SimdVector<int, 4>;
using SimdInt8 = SimdVector<int, 8>;
using SimdInt16 = SimdVector<int, 16>;
using SimdFloat = SimdFloat8;
using SimdInt = SimdInt8;
using SimdDouble8 = SimdVector<double, 8>;
// The tree combines its LEFT and RIGHT child in these two lanes.
using SimdDouble = SimdVector<double, 2>;

#endif
