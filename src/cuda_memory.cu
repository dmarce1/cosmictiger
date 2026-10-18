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
#define CUDA_MEM_CU
#include <cooperative_groups.h>
#include <cuda/barrier>

#include <cosmictiger/cuda_memory.hpp>

using BarrierType = cuda::barrier<cuda::thread_scope::thread_scope_block>;

template <class T>
CUDA_EXPORT T systemAtomicCompareExchange(T* self, T expected, T value) {
#ifdef __CUDA_ARCH__
    return atomicCAS_system(self, expected, value);
#else
    return __sync_val_compare_and_swap(self, expected, value);
#endif
}

template <class T>
CUDA_EXPORT T systemAtomicAdd(T* self, T value) {
#ifdef __CUDA_ARCH__
    return atomicAdd_system(self, value);
#else
    T expected, sum, rc;
    do {
        expected = *self;
        sum = expected + value;
        rc = systemAtomicCompareExchange(self, expected, sum);
    } while (rc != expected);
    return expected;
#endif
}

CUDA_EXPORT void fence() {
#ifdef __CUDA_ARCH__
    __threadfence_system();
#else
    __sync_synchronize();
#endif
}

template <class T>
CUDA_EXPORT T systemAtomicMax(T* self, T value) {
#ifdef __CUDA_ARCH__
    return atomicMax_system(self, value);
#else
    T expected, maxvalue, rc;
    do {
        expected = *self;
        maxvalue = std::max(expected, value);
        rc = systemAtomicCompareExchange(self, expected, maxvalue);
    } while (rc != expected);
    return maxvalue;
#endif
}

template <class T>
CUDA_EXPORT T systemAtomicExchange(T* self, T value) {
#ifdef __CUDA_ARCH__
    return atomicExch_system(self, value);
#else
    T expected, rc;
    do {
        expected = *self;
        rc = systemAtomicCompareExchange(self, expected, value);
    } while (rc != expected);
    return expected;
#endif
}

CUDA_EXPORT
void CudaMemory::push(int bin, char* ptr) {
    using Itype = unsigned long long int;
    auto& thisQ = q_[bin];
    auto in = qin_[bin];
    auto const& out = qout_[bin];
    if (in - out >= CUDA_MEM_STACK_SIZE) {
        PRINT("Q full! %li %li\n", out, in);
        ALWAYS_ASSERT(false);
    }
    while (systemAtomicCompareExchange(
               (Itype*) &thisQ[in % CUDA_MEM_STACK_SIZE], (Itype) 0, (Itype) ptr) != 0) {
        in++;
        if (in - out >= CUDA_MEM_STACK_SIZE) {
            PRINT("cuda mem Q full! %li %li\n", out, in);
            ALWAYS_ASSERT(false);
        }
    }
    in++;
    systemAtomicMax((Itype*) &qin_[bin], (Itype) in);
}

CUDA_EXPORT
char* CudaMemory::pop(int bin) {
    using Itype = unsigned long long int;
    auto& thisQ = q_[bin];
    auto const& in = qin_[bin];
    auto out = qout_[bin];
    if (out >= in) {
        return nullptr;
    }
    char* ptr;
    while ((ptr = (char*) systemAtomicExchange(
                (Itype*) &thisQ[out % CUDA_MEM_STACK_SIZE], (Itype) 0)) == nullptr) {
        if (out >= in) {
            return nullptr;
        }
        out++;
    }
    out++;
    systemAtomicMax((Itype*) &qout_[bin], (Itype) out);
    return ptr;
}

CUDA_EXPORT
bool CudaMemory::createNewAllocation(int bin) {
    size_t size = (1 << bin) + sizeof(size_t);
    auto* ptr = (char*) systemAtomicAdd((unsigned long long*) &next_, (unsigned long long) size);
    if (next_ >= heapEnd_) {
        return false;
    } else {
        push(bin, ptr + sizeof(size_t));
        *((size_t*) ptr) = bin;
        fence();
        return true;
    }
}

CUDA_EXPORT
void* CudaMemory::allocate(size_t sz) {
    int allocSize = 8;
    int bin = 3;
    while (allocSize < sz) {
        allocSize *= 2;
        bin++;
    }
    if (bin >= CUDA_MEM_NBIN) {
        printf("Allocation request for %li too large\n", sz);
        ALWAYS_ASSERT(false);
    }
    char* ptr;
    while ((ptr = pop(bin)) == nullptr) {
        if (!createNewAllocation(bin)) {
            return nullptr;
        }
    }
    return ptr;
}

CUDA_EXPORT void CudaMemory::free(void* ptr) {
    size_t* binptr = (size_t*) ((char*) ptr - sizeof(size_t));
    if (*binptr >= CUDA_MEM_NBIN) {
        printf("Corrupt free! %li\n", *binptr);
        ALWAYS_ASSERT(false);
    }
    push(*binptr, (char*) ptr);
}

CudaMemory::CudaMemory(size_t heapSize) {
    CUDA_CHECK(cudaMallocManaged(&heapBegin_, heapSize));
    heapEnd_ = heapBegin_ + heapSize;
    reset();
}

void CudaMemory::reset() {
    next_ = heapBegin_;
    for (int i = 0; i < CUDA_MEM_NBIN; i++) {
        for (int j = 0; j < CUDA_MEM_STACK_SIZE; j++) {
            q_[i][j] = nullptr;
        }
        qin_[i] = 0;
        qout_[i] = 0;
    }
}

CudaMemory::~CudaMemory() {
    CUDA_CHECK(cudaFree(heapBegin_));
}

__managed__ CudaMemory* memory;

CUDA_EXPORT void* cudaAllocate(size_t sz) {
#ifndef __CUDA_ARCH__
    if (!cudaEnabled()) {
        return allocateManagedMemory(sz);
    }
#endif
    return memory->allocate(sz);
}

CUDA_EXPORT void cudaRelease(void* ptr) {
#ifndef __CUDA_ARCH__
    if (!cudaEnabled()) {
        freeManagedMemory(ptr);
        return;
    }
#endif
    memory->free(ptr);
}

void cudaMemoryInitialize(size_t heapSize) {
    CUDA_CHECK(cudaMallocManaged(&memory, sizeof(CudaMemory)));
    new (memory) CudaMemory(heapSize);
}

void copyCudaMemory(void* dest, void* src, int size) {
#ifdef __CUDA_ARCH__
    __shared__ BarrierType barrier;
    auto group = cooperative_groups::this_thread_block();
    if (group.thread_rank() == 0) {
        init(&barrier, group.size());
    }
    syncthreads();
    cuda::memcpy_async(group, dest, src, size, barrier);
    barrier.arrive_and_wait();
#else
    memcpy(dest, src, size);
#endif
}
