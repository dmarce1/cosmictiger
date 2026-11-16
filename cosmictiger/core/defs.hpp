// Shared compile-time capacities and runtime spatial dimension selection.
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_

#include <cstddef>
#include <stdexcept>

#include <cosmictiger/cuda/cuda_export.hpp>

// #define LONG_LONG_PART_INT

#define COUNT_FLOPS

// Density/internal-energy SPH follows Monaghan (1992).
inline constexpr int kSphTreeNodeAllocationSize = 4;

#ifndef NDEBUG

#define CHECK_BOUNDS
#define DOMAINS_CHECK
#endif

// Fixed XYZ storage capacity; active Cartesian axes are selected once at startup.
inline constexpr int kDimensions = 3;
namespace detail {
inline int spatialDimensionCount = kDimensions;
#ifdef __CUDACC__
extern __device__ __constant__ int deviceSpatialDimensionCount;
#endif
}    // namespace detail

CUDA_EXPORT inline int spatialDimensions() {
#ifdef __CUDA_ARCH__
    return detail::deviceSpatialDimensionCount;
#else
    return detail::spatialDimensionCount;
#endif
}

// Zero selects runtime dispatch for shared callers; source-local kernels instantiate
// only dimensions 1, 2, and 3 so their inner loops and scaling are compile-time fixed.
template <int dimension>
CUDA_EXPORT constexpr int spatialDimensions() {
    static_assert(dimension >= 0 && dimension <= kDimensions);
    if constexpr (dimension == 0) {
        return spatialDimensions();
    } else {
        return dimension;
    }
}

// Set on every locality before particle allocation and upload with the kernel tables.
inline void setSpatialDimensions(int dimensions) {
    if (dimensions < 1 || dimensions > kDimensions) {
        throw std::invalid_argument("hydro.dimensions must be 1, 2, or 3");
    }
    detail::spatialDimensionCount = dimensions;
}
inline constexpr int kXi = 0;
inline constexpr int kYi = 1;
inline constexpr int kZi = 2;

inline constexpr int kChildCount = 2;
inline constexpr int kLeftChild = 0;
inline constexpr int kRightChild = 1;

inline constexpr int kBhBucketSize = 64;
inline constexpr int kDomainReboundIterations = 20;

#define USE_CONFORMAL_TIME

inline constexpr int kGroupWaves = 8;
inline constexpr int kGroupBucketSize = 256;

inline constexpr double kMaxLoadImbalance = 0.005;
inline constexpr int kCudaKickPartsMax = (16 * 1024);
inline constexpr unsigned long long kHeapSize = (8ULL * 1024ULL * 1024ULL * 1024ULL);
inline constexpr int kUnorderedSetSize = 1024;
inline constexpr int kWarpSize = 32;
inline constexpr double kEwaldRealCutoff = (2.6);
inline constexpr double kEwaldRealCutoffSquared = (kEwaldRealCutoff * kEwaldRealCutoff);
inline constexpr int kKickOversubscription = 1;
inline constexpr int kMaxDepth = 128;
// Migration vectors are capped by their serialized payload, not record count.
// Leave 64 KiB inside this cap for action/vector metadata and transport framing.
inline constexpr size_t kMaxMigrationParcelBytes = 32ULL * 1024 * 1024;
inline constexpr size_t kMigrationParcelOverheadBytes = 64ULL * 1024;
inline constexpr int kMaxRung = 16;
inline constexpr int kMinKickThreadParts = (1024);
inline constexpr int kMinSortThreadParts = (64 * 1024);

inline constexpr int kCudaMaxDepth = 64;

inline constexpr int kMinTrees = (2 * 1024 * 1024);
inline constexpr int kMinSphTrees = (2 * 1024 * 1024);
inline constexpr int kParticleCacheSize = 1024;
inline constexpr int kSortOversubscription = 2;
inline constexpr int kTreeCacheSize = 1024;
inline constexpr int kTreeNodeAllocationSize = 8;

#endif /* DEFS_HPP_ */
