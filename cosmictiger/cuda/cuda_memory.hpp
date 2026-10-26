#pragma once

/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cstddef>
#include <cstdint>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/math.hpp>

CUDA_EXPORT
inline void syncthreads() {
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
}

#ifndef CUDA_MEM_STACK_SIZE
#define CUDA_MEM_STACK_SIZE (1024 * 1024)
#endif
inline constexpr int kCudaMemoryBinCount = 32;

using CudaMemoryInteger = uint64_t;

class CudaMemory {
    struct Slot {
        CudaMemoryInteger sequence;
        char* pointer;
    };

    static_assert(
        CUDA_MEM_STACK_SIZE >= 2 && (CUDA_MEM_STACK_SIZE & (CUDA_MEM_STACK_SIZE - 1)) == 0);
    Array<Array<Slot, CUDA_MEM_STACK_SIZE>, kCudaMemoryBinCount> q_;
    Array<CudaMemoryInteger, kCudaMemoryBinCount> qin_;
    Array<CudaMemoryInteger, kCudaMemoryBinCount> qout_;
    char* heapBegin_;
    CudaMemoryInteger next_;
    size_t heapSize_;
    CUDA_EXPORT
    void push(int bin, char* ptr);
    CUDA_EXPORT
    char* pop(int bin);
    CUDA_EXPORT
    char* createNewAllocation(int bin);

public:
    CUDA_EXPORT
    void* allocate(size_t sz);
    CUDA_EXPORT
    void free(void* ptr);
    CudaMemory(size_t heapSize);
    ~CudaMemory();
    // All users of the pool must be quiescent before resetting or destroying it.
    void reset();
};

void cudaMemoryInitialize(size_t heapSize);
CUDA_EXPORT void* cudaAllocate(size_t sz);
CUDA_EXPORT void cudaRelease(void* ptr);
CUDA_EXPORT void copyCudaMemory(void* d, void* s, int sz);
