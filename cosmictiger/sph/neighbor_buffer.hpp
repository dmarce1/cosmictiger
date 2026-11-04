/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstddef>
#include <type_traits>

#include <cosmictiger/cuda/device_vector.hpp>

#ifdef __CUDACC__
namespace detail {

// Budget for neighbor payloads, in addition to block reduction scratch space.
// Keep this bounded: occupancy is also limited by registers and block size.
inline constexpr int kSphNeighborSharedBytes = 16 * 1024;

// Construct collectively inside a __shared__ workspace. Lists that fit stay in
// shared memory; larger lists grow into the allocation pool without truncation.
// One contiguous address range avoids a shared/global branch on every neighbor.
// Clearing the list restores shared storage while retaining overflow capacity
// for subsequent large leaves, avoiding repeated pool allocations.
// resize() and destruction are collective over the whole block; callers synchronize
// after filling a list before cross-thread reads. Elements must be trivial.
template <class Value, int sharedCapacity>
class SharedNeighborBuffer {
    static_assert(sharedCapacity > 0);
    static_assert(std::is_trivially_default_constructible_v<Value>);
    static_assert(std::is_trivially_copyable_v<Value>);
    static_assert(std::is_trivially_destructible_v<Value>);
    Value sharedValues_[sharedCapacity];
    DeviceVector<Value> overflow_;
    Value* data_;
    int size_;

public:
    __device__ SharedNeighborBuffer() {
        if (threadIdx.x == 0) {
            data_ = sharedValues_;
            size_ = 0;
        }
        __syncthreads();
    }

    SharedNeighborBuffer(SharedNeighborBuffer const&) = delete;
    SharedNeighborBuffer& operator=(SharedNeighborBuffer const&) = delete;

    __device__ int size() const {
        return size_;
    }

    __device__ void resize(int newSize) {
        __syncthreads();
        ALWAYS_ASSERT(newSize >= 0);
        if (newSize == 0) {
            overflow_.resize(0);
            if (threadIdx.x == 0) {
                data_ = sharedValues_;
            }
        } else if (newSize > sharedCapacity || data_ != sharedValues_) {
            overflow_.resize(newSize);
            if (data_ == sharedValues_) {
                copyCudaMemory(overflow_.data(), sharedValues_, sizeof(Value) * size_t(size_));
            }
            // All lanes must finish inspecting the old pointer before it changes,
            // including the path that already uses global storage and skips copying.
            __syncthreads();
            if (threadIdx.x == 0) {
                data_ = overflow_.data();
            }
        }
        if (threadIdx.x == 0) {
            size_ = newSize;
        }
        __syncthreads();
    }

    __device__ Value& operator[](int index) {
        return data_[index];
    }

    __device__ Value const& operator[](int index) const {
        return data_[index];
    }
};

}    // namespace detail
#endif
