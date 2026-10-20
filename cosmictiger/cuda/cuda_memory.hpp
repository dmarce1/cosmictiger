#pragma once

/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/math.hpp>

CUDA_EXPORT
inline void syncthreads() {
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
}

#define CUDA_MEM_STACK_SIZE (1024 * 1024)
#define CUDA_MEM_NBIN 32

using CudaMemoryInteger = uint64_t;

class CudaMemory {
    Array<Array<char*, CUDA_MEM_STACK_SIZE>, CUDA_MEM_NBIN> q_;
    Array<long long, CUDA_MEM_NBIN> qin_;
    Array<long long, CUDA_MEM_NBIN> qout_;
    char* heapBegin_;
    char* next_;
    char* heapEnd_;
    CUDA_EXPORT
    void push(int bin, char* ptr);
    CUDA_EXPORT
    char* pop(int bin);
    CUDA_EXPORT
    bool createNewAllocation(int bin);

public:
    CUDA_EXPORT
    void* allocate(size_t sz);
    CUDA_EXPORT
    void free(void* ptr);
    CudaMemory(size_t heapSize);
    ~CudaMemory();
    void reset();
};

void cudaMemoryInitialize(size_t heapSize);
CUDA_EXPORT void* cudaAllocate(size_t sz);
CUDA_EXPORT void cudaRelease(void* ptr);
CUDA_EXPORT void copyCudaMemory(void* d, void* s, int sz);
