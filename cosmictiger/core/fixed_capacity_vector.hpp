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
    __device__ constexpr FixedCapacityVector() {
        int const tid = threadIdx.x;
        if (tid == 0) {
            size_ = 0;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
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
    __device__ inline int size() const {
        return size_;
    }
    __device__ inline void resize(int newSize) {
        int const tid = threadIdx.x;
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        if (tid == 0) {
            size_ = newSize;
            if (size_ >= n) {
                PRINT("fixedcapvec exceeded size of %i in %s on line %i\n", n, __FILE__, __LINE__);
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
    __device__ inline void pushBack(T const& item) {
        int const tid = threadIdx.x;
        if (tid == 0) {
            data_[size_] = item;
            size_++;
            if (size_ >= n) {
                PRINT("fixedcapvec exceeded size of %i in %s on line %i\n", n, __FILE__, __LINE__);
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
    __device__ inline void popBack() {
        int const tid = threadIdx.x;
        if (tid == 0) {
            size_--;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
    __device__ inline T& back() {
        ASSERT(size_ >= 1);
        return data_[size_ - 1];
    }
    __device__ inline T back() const {
        return data_[size_ - 1];
    }
    __device__ T* begin() {
        return data_;
    }
    __device__ T* end() {
        return data_ + size_;
    }
    __device__ T& operator[](int i) {
#ifdef CHECK_BOUNDS
        if (i < 0 || i >= size_) {
            PRINT("index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, size_);
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
    __device__ T operator[](int i) const {
#ifdef CHECK_BOUNDS
        if (i < 0 || i >= size_) {
            PRINT("index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, size_);
#ifdef __CUDA_ARCH__
            __trap();
#else
            abort();
#endif
        }
#endif
        return data_[i];
    }
    __device__ T* data() {
        return data_;
    }
};
#endif
#endif /* FIXED_CAPACITY_VECTOR_HPP_ */
