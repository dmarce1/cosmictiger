#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <cosmictiger/cuda/cuda_check.hpp>

#define CUDA_EXPORT
#define __managed__
#define ALWAYS_ASSERT(condition)                                                        \
    do {                                                                                \
        if (!(condition)) {                                                             \
            std::fprintf(stderr, "Condition \"%s\" failed in %s on line %d\n",            \
                #condition, __FILE__, __LINE__);                                         \
            std::abort();                                                               \
        }                                                                               \
    } while (false)
#define PRINT(...) std::printf(__VA_ARGS__)

enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorInvalidValue };

cudaError_t cudaMallocManaged(void** pointer, size_t bytes);
cudaError_t cudaFree(void* pointer);
char const* cudaGetErrorString(cudaError_t error);

template <class Value>
cudaError_t cudaMallocManaged(Value** pointer, size_t bytes) {
    return cudaMallocManaged(reinterpret_cast<void**>(pointer), bytes);
}

// Share the production macros; allocator initialization still owns its explicit
// recoverable error handling instead of relying on these fatal call checks.
#define THROW_ERROR(...) \
    do { \
        std::fprintf(stderr, __VA_ARGS__); \
        ALWAYS_ASSERT(false); \
    } while (false)

bool cudaEnabled();
void* allocateManagedMemory(size_t bytes);
void freeManagedMemory(void* pointer) noexcept;
