/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#ifndef COSMICTIGER_CUDA_HPP_
#define COSMICTIGER_CUDA_HPP_

#include <limits>
#include <memory>
#include <new>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

#include <cosmictiger/assert.hpp>
#ifdef USE_CUDA
#include <cosmictiger/cuda_stream.hpp>
#endif
#include <cosmictiger/defs.hpp>

// Host-accessible storage follows the selected backend, not the build's capabilities.
bool cudaEnabled();
void* allocateManagedMemory(size_t bytes);
void freeManagedMemory(void* pointer) noexcept;
void* allocatePinnedMemory(size_t bytes);
void freePinnedMemory(void* pointer) noexcept;

#ifdef USE_CUDA

#define CUDA_MALLOC(a, b) cudaMalloc2(a, b, __FILE__, __LINE__)
#define CUDA_CHECK(a)     \
    if (a != cudaSuccess) \
    printf("CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

/* error checker from https://forums.developer.nvidia.com/t/cufft-error-handling/29231 */
static char const* cudaGetErrorEnum(cufftResult error) {
    switch (error) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    }
    ALWAYS_ASSERT(false);
    return "->unknown<-";
}

inline void cudaFftCheck(cufftResult err, char const* file, int const line) {
    if (CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n", file,
            line, err, cudaGetErrorEnum(err));
        cudaDeviceReset();
        ASSERT(0);
    }
}

#define CUDA_FFT_CHECK(a) cudaFftCheck(a, __FILE__, __LINE__)

void cudaSetDevice();

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
/**** TAKEN FROM THE CUDA DOCUMENTATION *****/
inline __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#endif

#define CUDA_EXPORT __host__ __device__

// CUDA 12 no longer provides Thrust's experimental pinned allocator.
// Keep the original host-vector contract with CUDA page-locked allocations.
template <typename T>
class PinnedAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    PinnedAllocator() = default;
    template <typename U>
    PinnedAllocator(PinnedAllocator<U> const&) noexcept {}

    T* allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        if (count == 0) {
            return nullptr;
        }
        return static_cast<T*>(allocatePinnedMemory(count * sizeof(T)));
    }

    void deallocate(T* pointer, size_t) noexcept {
        freePinnedMemory(pointer);
    }

    template <typename U>
    bool operator==(PinnedAllocator<U> const&) const noexcept {
        return true;
    }
};

void cudaSetDevice();
size_t cudaFreeMemory();
int cudaSmpCount();
size_t cudaFreeMemory();
size_t cudaTotalMemory();
int cudaSmpCount();
void cudaInit();
int cudaGetDevice();

void cudaAllocate(void** ptr, size_t size, char const* file, int line);

template <class T>
void cudaMalloc2(T** ptr, size_t size, char const* file, int line) {
    cudaAllocate((void**) ptr, size, file, line);
}

/*********************************************************************************/
// From https://localcoder.org/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
#ifdef __CUDACC__
__device__ static float atomicMax(float* address, float val) {
    int* addressAsI = (int*) address;
    int old = *addressAsI, assumed;
    do {
        assumed = old;
        old =
            ::atomicCAS(addressAsI, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ static float atomicMin(float* address, float val) {
    int* addressAsI = (int*) address;
    int old = *addressAsI, assumed;
    do {
        assumed = old;
        old =
            ::atomicCAS(addressAsI, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
#endif
/*********************************************************************************/
#else

template <class T>
using PinnedAllocator = std::allocator<T>;

#define CUDA_EXPORT

#endif

#endif /* COSMICTIGER_CUDA_HPP_ */
