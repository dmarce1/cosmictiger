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

#pragma once

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cuda/barrier>

#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/cuda/device_vector.hpp>

using BarrierType = cuda::barrier<cuda::thread_scope::thread_scope_block>;

template <class T>
class StackVector {
    DeviceVector<T> dat_;
    DeviceVector<int> bounds_;
    BarrierType barrier_;
    __device__ inline int begin() const {
        ASSERT(bounds_.size() >= 2);
        return bounds_[bounds_.size() - 2];
    }
    __device__ inline int end() const {
        ASSERT(bounds_.size() >= 2);
        return bounds_.back();
    }

public:
    __device__ inline int depth() const {
        return bounds_.size() - 2;
    }
    __device__ inline StackVector() {
        int const& tid = threadIdx.x;
        bounds_.resize(2);
        if (tid == 0) {
            bounds_[0] = 0;
            bounds_[1] = 0;
        }
        auto group = cooperative_groups::this_thread_block();
        if (group.thread_rank() == 0) {
            init(&barrier_, group.size());
        }
        __syncthreads();
    }
    // Collective: reserve storage for `entries` stacked elements and `depth`
    // levels so the traversal grows without pool allocations.
    __device__ inline void reserve(int entries, int depth) {
        dat_.reserve(entries);
        bounds_.reserve(depth + 2);
    }
    __device__ inline void push(T const& a) {
        int const& tid = threadIdx.x;
        ASSERT(bounds_.size() >= 2);
        dat_.pushBack(a);
        if (tid == 0) {
            bounds_.back()++;
        }
        __syncthreads();
    }
    __device__ inline int size() const {
        ASSERT(bounds_.size() >= 2);
        return end() - begin();
    }
    __device__ inline void resize(int sz) {
        int const& tid = threadIdx.x;
        ASSERT(bounds_.size() >= 2);
        dat_.resize(begin() + sz);
        if (tid == 0) {
            bounds_.back() = dat_.size();
        }
        __syncthreads();
    }
    __device__ inline T operator[](int i) const {
        ASSERT(i < size());
        return dat_[begin() + i];
    }
    __device__ inline T& operator[](int i) {
        ASSERT(i < size());
        return dat_[begin() + i];
    }
    __device__ inline T* data() {
        return dat_.data() + begin();
    }
    __device__ inline void pushTop() {
        auto const sz = size();
        bounds_.pushBack(end() + sz);
        if (sz) {
            dat_.resize(dat_.size() + sz);
            auto group = cooperative_groups::this_thread_block();
            cuda::memcpy_async(
                group, &dat_[begin()], &dat_[begin() - sz], sizeof(T) * sz, barrier_);
            barrier_.arrive_and_wait();
        }
    }
    __device__ inline void popTop() {
        ASSERT(bounds_.size() >= 2);
        dat_.resize(begin());
        bounds_.popBack();
    }
};

#endif
