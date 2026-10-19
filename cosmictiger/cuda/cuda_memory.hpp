#pragma once

/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
