#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#define CUDA_EXPORT
#define __managed__
#define ALWAYS_ASSERT(condition) assert(condition)
#define PRINT(...) std::printf(__VA_ARGS__)

enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorInvalidValue };

cudaError_t cudaMallocManaged(void** pointer, size_t bytes);
cudaError_t cudaFree(void* pointer);
char const* cudaGetErrorString(cudaError_t error);

template <class Value>
cudaError_t cudaMallocManaged(Value** pointer, size_t bytes) {
    return cudaMallocManaged(reinterpret_cast<void**>(pointer), bytes);
}

// Match the application's diagnostic-only macro: a test must not accidentally
// supply the exception handling that cuda_memory.cu itself needs to perform.
#define CUDA_CHECK(expression)                                                        \
    do {                                                                              \
        auto const cudaResult = (expression);                                         \
        if (cudaResult != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaResult)); \
        }                                                                             \
    } while (false)

bool cudaEnabled();
void* allocateManagedMemory(size_t bytes);
void freeManagedMemory(void* pointer) noexcept;
