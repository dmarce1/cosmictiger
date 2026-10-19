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

#include <cstdlib>
#include <cstring>
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
#include <cosmictiger/fmm/ewald_indices.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particles.hpp>

bool cudaEnabled() {
#ifdef USE_CUDA
    return getOptions().compute.cuda;
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

CudaHostAllocations& cudaHostAllocations() {
    // Keep provenance alive until after static host containers are destroyed.
    static auto* allocations = new CudaHostAllocations;
    return *allocations;
}
#endif

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

void* allocateManagedMemory(size_t bytes) {
    return detail::allocateHostAccessibleMemory(bytes, false);
}

void freeManagedMemory(void* pointer) noexcept {
    detail::freeHostAccessibleMemory(pointer);
}

void* allocatePinnedMemory(size_t bytes) {
    return detail::allocateHostAccessibleMemory(bytes, true);
}

void freePinnedMemory(void* pointer) noexcept {
    detail::freeHostAccessibleMemory(pointer);
}

#ifndef USE_CUDA
void* cudaAllocate(size_t size) {
    return allocateManagedMemory(size);
}

void cudaRelease(void* pointer) {
    freeManagedMemory(pointer);
}

void copyCudaMemory(void* destination, void* source, int size) {
    std::memcpy(destination, source, size);
}
#endif

#ifdef USE_CUDA

void cudaAllocate(void** ptr, size_t size, char const* file, int line) {
    static MutexType mutex;
    std::lock_guard<MutexType> lock(mutex);
    double freeMem = cudaFreeMemory();
    double totalMem = cudaTotalMemory();
    if ((freeMem - (double) size) / totalMem < 0.15) {
        PRINT("Attempt to allocate %li bytes on rank %i in %s on line %i leaves less than 15%% "
              "memory\n",
            size, hpxRank(), file, line);
        abort();
    }
    CUDA_CHECK(cudaMalloc(ptr, size));
}

void cudaSetDevice() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    int const deviceNum = hpxRank() % count;
    CUDA_CHECK(cudaSetDevice(deviceNum));
}

int cudaGetDevice() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    int const deviceNum = hpxRank() % count;
    return deviceNum;
}

size_t cudaTotalMemory() {
    size_t total;
    size_t free;
    cudaSetDevice();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total;
}

size_t cudaFreeMemory() {
    size_t total;
    size_t free;
    cudaSetDevice();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

int cudaSmpCount() {
    int count;
    cudaSetDevice();
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, cudaGetDevice()));
    return count;
}

void cudaInit() {
    cudaSetDevice();
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        if (detail::cudaStreamEventsInitialized()) {
            throw std::logic_error("CUDA initialization must precede HPX CUDA stream use");
        }
        hpx::cuda::experimental::check_cuda_error(cudaDeviceReset());
        // Reset clears constant memory populated during option propagation.
        uploadKernelSelections();
        cudaMemoryInitialize(HEAP_SIZE);
    });
    /*	size_t value = 64;
        size_t oval = value;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity , value));
        CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMaxL2FetchGranularity ));
        if (value != oval) {
            THROW_ERROR("Unable to set stack size to %li\n", oval);
        }*/
    detail::initializeCudaStreamRuntime();
}

#endif
