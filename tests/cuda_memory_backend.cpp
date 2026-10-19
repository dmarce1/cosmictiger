/* CosmicTiger - standalone CUDA regression support; GPL-2.0-or-later. */
#include <new>

#include <cuda_runtime.h>

#include <cosmictiger/cuda/cuda.hpp>

// Standalone kernel tests initialize the CUDA pool explicitly and have no
// application Options object. They always exercise the CUDA allocation path.
bool cudaEnabled() {
    return true;
}

void* allocateManagedMemory(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* pointer = nullptr;
    if (cudaMallocManaged(&pointer, bytes) != cudaSuccess) {
        throw std::bad_alloc();
    }
    return pointer;
}

void freeManagedMemory(void* pointer) noexcept {
    if (pointer != nullptr) {
        cudaFree(pointer);
    }
}
