/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>

static __managed__ double gpuFlops = 0.0;

void resetGpuFlops() {
    if (!cudaEnabled()) {
        return;
    }
    gpuFlops = 0.0;
}

double getGpuFlops() {
    return cudaEnabled() ? gpuFlops : 0.0;
}

__device__ void addGpuFlops(int count) {
#ifdef COUNT_FLOPS
    int const& tid = threadIdx.x;
    int total = count;
    sharedReduceAdd(count);
    if (tid == 0) {
        atomicAdd(&gpuFlops, (double) count);
    }
#endif
}
