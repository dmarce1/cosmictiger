/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include <cuda/atomic>

#include <cosmictiger/cuda/cuda_memory.hpp>

namespace detail {

using CudaMemoryAtomic = cuda::atomic_ref<CudaMemoryInteger, cuda::thread_scope_system>;

void checkCudaMemoryAllocation(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}    // namespace detail

CUDA_EXPORT
void CudaMemory::push(int bin, char* ptr) {
    detail::CudaMemoryAtomic input(qin_[bin]);
    auto position = input.load(cuda::memory_order_relaxed);
    for (;;) {
        auto& slot = q_[bin][position % CUDA_MEM_STACK_SIZE];
        detail::CudaMemoryAtomic sequence(slot.sequence);
        auto const generation = sequence.load(cuda::memory_order_acquire);
        auto const difference = static_cast<int64_t>(generation - position);
        if (difference == 0) {
            if (input.compare_exchange_weak(position, position + 1, cuda::memory_order_relaxed)) {
                // The position CAS reserves this generation of the slot. Its
                // sequence publishes the pointer and the allocation's header.
                slot.pointer = ptr;
                sequence.store(position + 1, cuda::memory_order_release);
                return;
            }
        } else if (difference < 0) {
            PRINT("cuda memory queue full for bin %i\n", bin);
            ALWAYS_ASSERT(false);
            return;
        } else {
            position = input.load(cuda::memory_order_relaxed);
        }
    }
}

CUDA_EXPORT
char* CudaMemory::pop(int bin) {
    detail::CudaMemoryAtomic output(qout_[bin]);
    auto position = output.load(cuda::memory_order_relaxed);
    for (;;) {
        auto& slot = q_[bin][position % CUDA_MEM_STACK_SIZE];
        detail::CudaMemoryAtomic sequence(slot.sequence);
        auto const generation = sequence.load(cuda::memory_order_acquire);
        auto const difference = static_cast<int64_t>(generation - (position + 1));
        if (difference == 0) {
            if (output.compare_exchange_weak(position, position + 1, cuda::memory_order_relaxed)) {
                char* ptr = slot.pointer;
                // A producer cannot reuse this slot until its pointer has been read.
                sequence.store(position + CUDA_MEM_STACK_SIZE, cuda::memory_order_release);
                return ptr;
            }
        } else if (difference < 0) {
            // Empty, or the next producer has not published its reserved slot.
            // Do not spin waiting for a possibly unscheduled GPU thread.
            return nullptr;
        } else {
            position = output.load(cuda::memory_order_relaxed);
        }
    }
}

CUDA_EXPORT
char* CudaMemory::createNewAllocation(int bin) {
    size_t const size = (size_t(1) << bin) + sizeof(size_t);
    detail::CudaMemoryAtomic next(next_);
    auto offset = next.load(cuda::memory_order_relaxed);
    for (;;) {
        if (offset > heapSize_ || size > heapSize_ - offset) {
            return nullptr;
        }
        if (next.compare_exchange_weak(offset, offset + size, cuda::memory_order_relaxed)) {
            char* ptr = heapBegin_ + offset;
            *reinterpret_cast<size_t*>(ptr) = bin;
            // This allocation belongs directly to the caller. It only enters
            // the shared queue once free() publishes its initialized header.
            return ptr + sizeof(size_t);
        }
    }
}

CUDA_EXPORT
void* CudaMemory::allocate(size_t sz) {
    size_t const maximumSize = size_t(1) << (CUDA_MEM_NBIN - 1);
    if (sz > maximumSize) {
        return nullptr;
    }
    size_t allocSize = 8;
    int bin = 3;
    while (allocSize < sz) {
        allocSize *= 2;
        bin++;
    }
    if (char* ptr = pop(bin)) {
        return ptr;
    }
    return createNewAllocation(bin);
}

CUDA_EXPORT void CudaMemory::free(void* ptr) {
    size_t* binptr = (size_t*) ((char*) ptr - sizeof(size_t));
    if (*binptr >= CUDA_MEM_NBIN) {
        printf("Corrupt free! %li\n", *binptr);
        ALWAYS_ASSERT(false);
    }
    push(*binptr, (char*) ptr);
}

CudaMemory::CudaMemory(size_t heapSize)
  : heapBegin_(nullptr)
  , next_(0)
  , heapSize_(heapSize) {
    detail::checkCudaMemoryAllocation(
        cudaMallocManaged(&heapBegin_, heapSize), "Allocate CUDA memory pool heap");
    reset();
}

void CudaMemory::reset() {
    next_ = 0;
    for (int i = 0; i < CUDA_MEM_NBIN; i++) {
        for (int j = 0; j < CUDA_MEM_STACK_SIZE; j++) {
            q_[i][j].sequence = j;
            q_[i][j].pointer = nullptr;
        }
        qin_[i] = 0;
        qout_[i] = 0;
    }
}

CudaMemory::~CudaMemory() {
    cudaError_t const status = cudaFree(heapBegin_);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Free CUDA memory pool heap: %s\n", cudaGetErrorString(status));
    }
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
    CudaMemory* allocation = nullptr;
    detail::checkCudaMemoryAllocation(
        cudaMallocManaged(&allocation, sizeof(CudaMemory)), "Allocate CUDA memory pool object");
    try {
        new (allocation) CudaMemory(heapSize);
    } catch (...) {
        cudaFree(allocation);
        throw;
    }
    memory = allocation;
}

// Collective block copy. A plain strided copy needs no static shared barrier:
// re-initializing one on every call raced with lagging waiters of the previous
// phase (compute-sanitizer synccheck reported missing waits).
void copyCudaMemory(void* dest, void* src, int size) {
#ifdef __CUDA_ARCH__
    int const tid = threadIdx.x;
    int const stride = blockDim.x;
    syncthreads();
    if (size % int(sizeof(int)) == 0 && (reinterpret_cast<size_t>(dest) % sizeof(int)) == 0 &&
        (reinterpret_cast<size_t>(src) % sizeof(int)) == 0) {
        int const count = size / int(sizeof(int));
        auto* destination = static_cast<int*>(dest);
        auto const* source = static_cast<int const*>(src);
        for (int i = tid; i < count; i += stride) {
            destination[i] = source[i];
        }
    } else {
        auto* destination = static_cast<char*>(dest);
        auto const* source = static_cast<char const*>(src);
        for (int i = tid; i < size; i += stride) {
            destination[i] = source[i];
        }
    }
    syncthreads();
#else
    memcpy(dest, src, size);
#endif
}
