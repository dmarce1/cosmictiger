/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

#include <cuda/atomic>

#include <cosmictiger/cuda/cuda_memory.hpp>

namespace detail {

using CudaMemoryAtomic = cuda::atomic_ref<CudaMemoryInteger, cuda::thread_scope_system>;

// Reading a __managed__ symbol can fault before CUDA initializes its storage.
// Host allocation provenance must remain available without touching the runtime.
CudaMemory* hostMemory = nullptr;

/// Translate a failed CUDA allocation operation into a descriptive runtime exception.
void checkCudaMemoryAllocation(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}    // namespace detail

/// Publish a released allocation into its size-bin queue using sequence-number
/// synchronization.
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
#ifdef __CUDA_ARCH__
            printf("cuda memory queue full for bin %i\n", bin);
#else
            std::cout << std::format("cuda memory queue full for bin {:d}\n", bin) << std::flush;
#endif
            ALWAYS_ASSERT(false);
            return;
        } else {
            position = input.load(cuda::memory_order_relaxed);
        }
    }
}

/// Take an available allocation from a size-bin queue, returning null if the next slot is
/// unavailable.
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

/// Reserve aligned space from the pool heap and record its size bin, returning null on
/// exhaustion.
CUDA_EXPORT
char* CudaMemory::createNewAllocation(int bin) {
    size_t const bytes = (size_t(1) << bin) + kCudaPoolHeaderSize;
    size_t const size = (bytes + kCudaPoolAlignment - 1) / kCudaPoolAlignment * kCudaPoolAlignment;
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
            return ptr + kCudaPoolHeaderSize;
        }
    }
}

/// Reuse a size-bin allocation or reserve a new one; return null when the pool cannot
/// satisfy the request.
CUDA_EXPORT
void* CudaMemory::allocate(size_t sz) {
    size_t const maximumSize = size_t(1) << (kCudaMemoryBinCount - 1);
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
        auto& header = *reinterpret_cast<CudaMemoryInteger*>(ptr - kCudaPoolHeaderSize);
        detail::CudaMemoryAtomic state(header);
        ALWAYS_ASSERT(
            state.load(cuda::memory_order_relaxed) == CudaMemoryInteger(bin + kCudaMemoryBinCount));
        state.store(bin, cuda::memory_order_relaxed);
        return ptr;
    }
    return createNewAllocation(bin);
}

/// Mark an allocation as released and publish it back to its size-bin queue.
CUDA_EXPORT void CudaMemory::free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    static_assert(sizeof(size_t) == sizeof(CudaMemoryInteger));
    auto& header =
        *reinterpret_cast<CudaMemoryInteger*>(static_cast<char*>(ptr) - kCudaPoolHeaderSize);
    detail::CudaMemoryAtomic state(header);
    auto bin = state.load(cuda::memory_order_relaxed);
    for (;;) {
        ALWAYS_ASSERT(bin >= 3 && bin < kCudaMemoryBinCount);
        if (state.compare_exchange_weak(
                bin, bin + kCudaMemoryBinCount, cuda::memory_order_relaxed)) {
            break;
        }
    }
    push(int(bin), static_cast<char*>(ptr));
}

/// Test whether a pointer lies inside this pool's heap address range.
CUDA_EXPORT bool CudaMemory::owns(void const* ptr) const {
    auto const address = reinterpret_cast<uintptr_t>(ptr);
    auto const beginning = reinterpret_cast<uintptr_t>(heapBegin_);
    return address >= beginning && address - beginning < heapSize_;
}

/// Allocate the managed pool heap and initialize its free queues.
CudaMemory::CudaMemory(size_t heapSize)
  : heapBegin_(nullptr)
  , next_(0)
  , heapSize_(heapSize) {
    detail::checkCudaMemoryAllocation(
        cudaMallocManaged(&heapBegin_, heapSize), "Allocate CUDA memory pool heap");
    reset();
}

/// Reset the heap cursor and all queue generations; outstanding allocations must no longer
/// be in use.
void CudaMemory::reset() {
    next_ = 0;
    for (int i = 0; i < kCudaMemoryBinCount; i++) {
        for (int j = 0; j < CUDA_MEM_STACK_SIZE; j++) {
            q_[i][j].sequence = j;
            q_[i][j].pointer = nullptr;
        }
        qin_[i] = 0;
        qout_[i] = 0;
    }
}

/// Release the managed heap and report any CUDA deallocation error.
CudaMemory::~CudaMemory() {
    cudaError_t const status = cudaFree(heapBegin_);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Free CUDA memory pool heap: %s\n", cudaGetErrorString(status));
    }
}

__managed__ CudaMemory* memory;

/// Allocate from the shared CUDA pool, using host allocation when CUDA is disabled.
CUDA_EXPORT void* cudaAllocate(size_t sz) {
#ifndef __CUDA_ARCH__
    if (!cudaEnabled()) {
        return allocateManagedMemory(sz);
    }
#endif
#ifdef __CUDA_ARCH__
    CudaMemory* const pool = memory;
    ALWAYS_ASSERT(pool != nullptr);
#else
    CudaMemory* const pool = detail::hostMemory;
    if (pool == nullptr) {
        throw std::logic_error("CUDA memory pool must be initialized before allocation");
    }
#endif
    void* const pointer = pool->allocate(sz);
    if (pointer == nullptr) {
#ifdef __CUDA_ARCH__
        printf("CUDA pool allocation failed for %llu bytes\n", (unsigned long long) sz);
        ALWAYS_ASSERT(false);
#else
        throw std::bad_alloc();
#endif
    }
    return pointer;
}

/// Return pool allocations to their bin, or free non-pool host allocations by provenance.
CUDA_EXPORT void cudaRelease(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
#ifndef __CUDA_ARCH__
    CudaMemory* const pool = detail::hostMemory;
    if (pool == nullptr || !pool->owns(ptr)) {
        freeManagedMemory(ptr);
        return;
    }
#else
    CudaMemory* const pool = memory;
    ALWAYS_ASSERT(pool != nullptr && pool->owns(ptr));
#endif
    pool->free(ptr);
}

/// Construct the shared managed-memory pool and publish its host and device pointers.
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
    detail::hostMemory = allocation;
}

// Collective block copy. A plain strided copy needs no static shared barrier:
// re-initializing one on every call raced with lagging waiters of the previous
// phase (compute-sanitizer synccheck reported missing waits).
void copyCudaMemory(void* dest, void const* src, size_t size) {
#ifdef __CUDA_ARCH__
    ALWAYS_ASSERT(blockDim.y == 1 && blockDim.z == 1);
    size_t const tid = threadIdx.x;
    size_t const stride = blockDim.x;
    syncthreads();
    if (size % sizeof(int) == 0 && (reinterpret_cast<size_t>(dest) % sizeof(int)) == 0 &&
        (reinterpret_cast<size_t>(src) % sizeof(int)) == 0) {
        size_t const count = size / sizeof(int);
        auto* destination = static_cast<int*>(dest);
        auto const* source = static_cast<int const*>(src);
        for (size_t i = tid; i < count; i += stride) {
            destination[i] = source[i];
        }
    } else {
        auto* destination = static_cast<char*>(dest);
        auto const* source = static_cast<char const*>(src);
        for (size_t i = tid; i < size; i += stride) {
            destination[i] = source[i];
        }
    }
    syncthreads();
#else
    memcpy(dest, src, size);
#endif
}
