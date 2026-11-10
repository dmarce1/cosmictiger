/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef FIXED_CAPACITY_VECTOR_HPP_
#define FIXED_CAPACITY_VECTOR_HPP_

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cuda/cuda.hpp>

#ifdef __CUDACC__
template <class T, int n>
class FixedCapacityVector {
    Array<T, n> data_;
    int size_;

public:
    /// Initialize an empty vector collectively across the CUDA block.
    __device__ constexpr FixedCapacityVector() {
        int const tid = threadIdx.x;
        if (tid == 0) {
            size_ = 0;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
    /// Reset the logical size to zero and synchronize the participating block.
    __device__ void initialize() {
        int const tid = threadIdx.x;
        if (tid == 0) {
            size_ = 0;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
    FixedCapacityVector(FixedCapacityVector const&) = default;
    FixedCapacityVector& operator=(FixedCapacityVector const&) = default;
    /// Return the logical number of stored elements.
    __device__ inline int size() const {
        return size_;
    }
    /// Set the logical size collectively; reaching the fixed capacity triggers a fatal
    /// bounds error.
    __device__ inline void resize(int newSize) {
        int const tid = threadIdx.x;
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        if (tid == 0) {
            size_ = newSize;
            if (size_ >= n) {
                printf("fixedcapvec exceeded size of %i in %s on line %i\n", n, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
                __trap();
#else
                abort();
#endif
            }
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
    /// Append one shared value using lane zero and synchronize the participating block.
    __device__ inline void pushBack(T const& item) {
        int const tid = threadIdx.x;
        if (tid == 0) {
            data_[size_] = item;
            size_++;
            if (size_ >= n) {
                printf("fixedcapvec exceeded size of %i in %s on line %i\n", n, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
                __trap();
#else
                abort();
#endif
            }
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
    /// Remove the final logical element collectively; the vector must be nonempty.
    __device__ inline void popBack() {
        int const tid = threadIdx.x;
        if (tid == 0) {
            size_--;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
    /// Access the last element; the container must be nonempty.
    __device__ inline T& back() {
        ASSERT(size_ >= 1);
        return data_[size_ - 1];
    }
    /// Access the last element; the container must be nonempty.
    __device__ inline T back() const {
        return data_[size_ - 1];
    }
    /// Return a pointer to the first stored element.
    __device__ T* begin() {
        return data_;
    }
    /// Return the one-past-end pointer for the stored elements.
    __device__ T* end() {
        return data_ + size_;
    }
    /// Access the indexed element; the index must lie within the logical bounds.
    __device__ T& operator[](int i) {
#ifdef CHECK_BOUNDS
        if (i < 0 || i >= size_) {
            printf(
                "index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, size_);
#ifdef __CUDA_ARCH__
            assert(false);
            __trap();
#else
            abort();
#endif
        }
#endif
        return data_[i];
    }
    /// Access the indexed element; the index must lie within the logical bounds.
    __device__ T operator[](int i) const {
#ifdef CHECK_BOUNDS
        if (i < 0 || i >= size_) {
            printf(
                "index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, size_);
#ifdef __CUDA_ARCH__
            __trap();
#else
            abort();
#endif
        }
#endif
        return data_[i];
    }
    /// Expose the contiguous element storage without copying.
    __device__ T* data() {
        return data_;
    }
};
#endif
#endif /* FIXED_CAPACITY_VECTOR_HPP_ */
