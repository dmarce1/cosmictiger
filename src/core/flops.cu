/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/core/flops.hpp>

static __managed__ FlopCount gpuFlops = 0;

void resetGpuFlops() {
    if (cudaEnabled()) {
        gpuFlops = 0;
    }
}

FlopCount getGpuFlops() {
    return cudaEnabled() ? gpuFlops : 0;
}

__device__ void addGpuFlops(FlopCount count) {
#ifdef COUNT_FLOPS
    unsigned const mask = __activemask();
    unsigned const thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned const lane = thread % warpSize;
    unsigned const leader = __ffs(mask) - 1;
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
    if (lane == leader && count != 0) {
        atomicAdd(&gpuFlops, count);
    }
#endif
}
