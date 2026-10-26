
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_

// #define LONG_LONG_PART_INT

#define COUNT_FLOPS

// Density/internal-energy SPH follows Monaghan (1992).
inline constexpr int kSphTreeNodeAllocationSize = 4;

#ifndef NDEBUG

#define CHECK_BOUNDS
#define DOMAINS_CHECK
#endif

inline constexpr int kDimensions = 3;
inline constexpr int kXDimension = 0;
inline constexpr int kYDimension = 1;
inline constexpr int kZDimension = 2;

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
inline constexpr int kMaxParticlesPerParcel = (8 * 1024 * 1024);
inline constexpr int kMaxRung = 16;
inline constexpr int kMinKickThreadParts = (1024);
inline constexpr int kMinSortThreadParts = (64 * 1024);

inline constexpr int kCudaMaxDepth = 64;

inline constexpr int kMinTrees = (2 * 1024 * 1024);
inline constexpr int kMinSphTrees = (2 * 1024 * 1024);
inline constexpr int kParticleCacheSize = 1024;
inline constexpr int kSortOversubscription = 2;
inline constexpr int kTreeCacheSize = 1024;
inline constexpr int kSphTreeCacheSize = 1024;
inline constexpr int kTreeNodeAllocationSize = 8;

// Compile-time cap on compute.bucketSize (COSMICTIGER_WITH_MAX_BUCKET); see kick.hpp.
#ifndef MAX_BUCKET_SIZE
#define MAX_BUCKET_SIZE 128
#endif

#endif /* DEFS_HPP_ */
