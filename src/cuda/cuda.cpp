/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>

#ifdef USE_CUDA
#include <hpx/async_cuda/cuda_exception.hpp>
#endif

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particles.hpp>

/// Report whether this build and the resolved options enable a CUDA backend.
bool cudaEnabled() {
#ifdef USE_CUDA
    return usesCuda(getOptions());
#else
    return false;
#endif
}

namespace detail {

#ifdef USE_CUDA
struct CudaHostAllocations {
    std::mutex mutex;
    std::unordered_map<void*, bool> pinned;
};

/// Return the allocation-provenance registry, retaining it through static destruction.
CudaHostAllocations& cudaHostAllocations() {
    // Keep provenance alive until after static host containers are destroyed.
    static auto* allocations = new CudaHostAllocations;
    return *allocations;
}
#endif

/// Allocate pinned or managed storage when CUDA is active, otherwise ordinary host memory.
void* allocateHostAccessibleMemory(size_t bytes, bool pinned) {
    if (bytes == 0) {
        return nullptr;
    }
    void* pointer = nullptr;
#ifdef USE_CUDA
    if (cudaEnabled()) {
        cudaError_t const status = pinned ? cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) :
                                            cudaMallocManaged(&pointer, bytes);
        if (status != cudaSuccess) {
            throw std::bad_alloc();
        }
        auto& allocations = cudaHostAllocations();
        try {
            std::lock_guard<std::mutex> lock(allocations.mutex);
            allocations.pinned.emplace(pointer, pinned);
        } catch (...) {
            if (pinned) {
                cudaFreeHost(pointer);
            } else {
                cudaFree(pointer);
            }
            throw;
        }
        return pointer;
    }
#endif
    pointer = std::malloc(bytes);
    if (pointer == nullptr) {
        throw std::bad_alloc();
    }
    return pointer;
}

/// Free host-accessible storage using its recorded CUDA allocation provenance.
void freeHostAccessibleMemory(void* pointer) noexcept {
    if (pointer == nullptr) {
        return;
    }
#ifdef USE_CUDA
    auto& allocations = cudaHostAllocations();
    {
        std::lock_guard<std::mutex> lock(allocations.mutex);
        auto const allocation = allocations.pinned.find(pointer);
        if (allocation != allocations.pinned.end()) {
            bool const pinned = allocation->second;
            allocations.pinned.erase(allocation);
            if (pinned) {
                cudaFreeHost(pointer);
            } else {
                cudaFree(pointer);
            }
            return;
        }
    }
#endif
    std::free(pointer);
}

}    // namespace detail

/// Allocate storage accessible on the host, using managed memory when CUDA is active.
void* allocateManagedMemory(size_t bytes) {
    return detail::allocateHostAccessibleMemory(bytes, false);
}

/// Release storage obtained through the host-accessible allocation helpers.
void freeManagedMemory(void* pointer) noexcept {
    detail::freeHostAccessibleMemory(pointer);
}

/// Allocate pinned host storage when CUDA is active, otherwise ordinary host storage.
void* allocatePinnedMemory(size_t bytes) {
    return detail::allocateHostAccessibleMemory(bytes, true);
}

/// Release pinned or fallback host storage using its allocation provenance.
void freePinnedMemory(void* pointer) noexcept {
    detail::freeHostAccessibleMemory(pointer);
}

#ifndef USE_CUDA
/// Allocate host-accessible storage for the non-CUDA implementation.
void* cudaAllocate(size_t size) {
    return allocateManagedMemory(size);
}

/// Release the fallback managed-memory allocation.
void cudaRelease(void* pointer) {
    freeManagedMemory(pointer);
}

/// Copy bytes between host-accessible buffers in the non-CUDA implementation.
void copyCudaMemory(void* destination, void const* source, size_t size) {
    std::memcpy(destination, source, size);
}
#endif

#ifdef USE_CUDA

/// Allocate device memory after checking that the request leaves the configured free-memory
/// margin.
void cudaAllocate(void** ptr, size_t size, char const* file, int line) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    double freeMem = cudaFreeMemory();
    double totalMem = cudaTotalMemory();
    if ((freeMem - (double) size) / totalMem < 0.15) {
        std::cout << std::format("Attempt to allocate {:d} bytes on rank {:d} in {:s} on line {:d} "
                                 "leaves less than 15% memory\n",
                         size, hpxRank(), file, line)
                  << std::flush;
        abort();
    }
    CUDA_CHECK(cudaMalloc(ptr, size));
}

/// Select the GPU assigned by node-local rank, failing if no GPU is visible.
void cudaSetDevice() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        throw std::runtime_error("CUDA backend requested without a visible GPU");
    }
    int const deviceNum = hpxLocalRank() % count;
    CUDA_CHECK(cudaSetDevice(deviceNum));
}

/// Return the GPU index assigned by node-local rank modulo visible device count.
int cudaGetDevice() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        throw std::runtime_error("CUDA backend requested without a visible GPU");
    }
    int const deviceNum = hpxLocalRank() % count;
    return deviceNum;
}

/// Select the assigned GPU and return its total memory capacity in bytes.
size_t cudaTotalMemory() {
    size_t total;
    size_t free;
    cudaSetDevice();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total;
}

/// Select the assigned GPU and return its currently free memory in bytes.
size_t cudaFreeMemory() {
    size_t total;
    size_t free;
    cudaSetDevice();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

/// Return the multiprocessor count of the assigned GPU.
int cudaSmpCount() {
    int count;
    cudaSetDevice();
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, cudaGetDevice()));
    return count;
}

/// Initialize the assigned device, restore kernel selections after reset, and initialize
/// runtime support.
void cudaInit() {
    cudaSetDevice();
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        int visibleDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&visibleDevices));
        if (visibleDevices > 1) {
            std::cout << std::format(
                             "CUDA: locality {:d} selects GPU {:d} using node-local rank {:d}. "
                             "Bind each locality to one visible GPU with CUDA_VISIBLE_DEVICES to "
                             "keep the HPX event pool from opening contexts on every GPU.\n",
                             hpxRank(), cudaGetDevice(), hpxLocalRank())
                      << std::flush;
        }
        if (detail::cudaStreamEventsInitialized()) {
            throw std::logic_error("CUDA initialization must precede HPX CUDA stream use");
        }
        hpx::cuda::experimental::check_cuda_error(cudaDeviceReset());
        // Reset clears constant memory populated during option propagation.
        uploadKernelSelections();
        cudaMemoryInitialize(kHeapSize);
    });
    /*	size_t value = 64;
        size_t oval = value;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity , value));
        CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMaxL2FetchGranularity ));
        if (value != oval) {
            throw std::runtime_error(std::format("Unable to set stack size to {}", oval));
        }*/
    detail::initializeCudaStreamRuntime();
}

#endif
