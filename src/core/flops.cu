/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/core/flops.hpp>

static __managed__ FlopCount gpuSingleFlops = 0;
static __managed__ FlopCount gpuDoubleFlops = 0;
static __managed__ FlopCount gpuUnclassifiedFlops = 0;

void resetGpuFlops() {
    if (cudaEnabled()) {
        gpuSingleFlops = 0;
        gpuDoubleFlops = 0;
        gpuUnclassifiedFlops = 0;
    }
}

FlopCounts getGpuFlopCounts() {
    return cudaEnabled() ? FlopCounts{gpuSingleFlops, gpuDoubleFlops, gpuUnclassifiedFlops} :
                           FlopCounts{};
}

FlopCount getGpuFlops() {
    return getGpuFlopCounts().total();
}

namespace detail {

__device__ FlopCount reduceFlopCount(FlopCount count, unsigned mask) {
    if (mask == 0xffffffffu) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            count += __shfl_down_sync(mask, count, offset);
        }
    } else {
        // A partial or divergent warp must never shuffle from an inactive
        // lane. All active lanes execute the same sequence of broadcasts.
        FlopCount total = 0;
        unsigned remaining = mask;
        while (remaining != 0) {
            int const source = __ffs(remaining) - 1;
            total += __shfl_sync(mask, count, source);
            remaining &= remaining - 1;
        }
        count = total;
    }
    return count;
}

__device__ FlopCount* gpuFlopCounter(FlopPrecision precision) {
    if (precision == FlopPrecision::Single) {
        return &gpuSingleFlops;
    }
    if (precision == FlopPrecision::Double) {
        return &gpuDoubleFlops;
    }
    return &gpuUnclassifiedFlops;
}

}    // namespace detail

__device__ void addGpuFlops(FlopCount count) {
    addGpuFlops(count, FlopPrecision::Unclassified);
}

__device__ void addGpuFlops(FlopCount count, FlopPrecision precision) {
#ifdef COUNT_FLOPS
    unsigned const mask = __activemask();
    unsigned const thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned const lane = thread % warpSize;
    unsigned const leader = __ffs(mask) - 1;
    if (precision != FlopPrecision::Single && precision != FlopPrecision::Double) {
        precision = FlopPrecision::Unclassified;
    }
    auto const leaderPrecision = FlopPrecision(__shfl_sync(mask, unsigned(precision), leader));
    if (__all_sync(mask, precision == leaderPrecision)) {
        count = detail::reduceFlopCount(count, mask);
        if (lane == leader && count != 0) {
            atomicAdd(detail::gpuFlopCounter(precision), count);
        }
    } else {
        // A caller can select a different precision per lane. Reduce each
        // bucket over the same active mask rather than assigning the entire
        // warp's contribution to its leader's precision.
        for (unsigned bucket = 0; bucket < 3; ++bucket) {
            auto const bucketPrecision = FlopPrecision(bucket);
            FlopCount const total =
                detail::reduceFlopCount(precision == bucketPrecision ? count : 0, mask);
            if (lane == leader && total != 0) {
                atomicAdd(detail::gpuFlopCounter(bucketPrecision), total);
            }
        }
    }
#endif
}
