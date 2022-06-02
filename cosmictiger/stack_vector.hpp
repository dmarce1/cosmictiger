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

#pragma once

#include <cosmictiger/cuda_mem.hpp>

#include <cooperative_groups.h>
#include <cuda/barrier>

using barrier_type = cuda::barrier<cuda::thread_scope::thread_scope_block>;

#ifdef __CUDACC__

template<class T>
class stack_vector {
	device_vector<T> data;
	device_vector<int> bounds;
	barrier_type barrier;
	__device__ inline int begin() const {
		ASSERT(bounds.size() >= 2);
		return bounds[bounds.size() - 2];
	}
	__device__ inline int end() const {
		ASSERT(bounds.size() >= 2);
		return bounds.back();
	}
public:
	__device__ inline int depth() const {
		return bounds.size() - 2;
	}
	__device__ inline stack_vector() {
		const int& tid = threadIdx.x;
		bounds.resize(2);
		if (tid == 0) {
			bounds[0] = 0;
			bounds[1] = 0;
		}
		auto group = cooperative_groups::this_thread_block();
		if (group.thread_rank() == 0) {
			init(&barrier, group.size());
		}
		__syncthreads();
	}
	__device__ inline void push(const T &a) {
		const int& tid = threadIdx.x;
		ASSERT(bounds.size() >= 2);
		data.push_back(a);
		if (tid == 0) {
			bounds.back()++;
		}
		__syncthreads();

	}
	__device__ inline int size() const {
		ASSERT(bounds.size() >= 2);
		return end() - begin();
	}
	__device__ inline void resize(int sz) {
		const int& tid = threadIdx.x;
		ASSERT(bounds.size() >= 2);
		data.resize(begin() + sz);
		if (tid == 0) {
			bounds.back() = data.size();
		}
		__syncthreads();
	}
	__device__ inline T operator[](int i) const {
		ASSERT(i < size());
		return data[begin() + i];
	}
	__device__ inline T& operator[](int i) {
		ASSERT(i < size());
		return data[begin() + i];
	}

	__device__ inline void push_top() {
		const auto sz = size();
		bounds.push_back(end() + sz);
		data.resize(data.size() + sz);
		auto group = cooperative_groups::this_thread_block();
		cuda::memcpy_async(group, &data[begin()], &data[begin()-sz], sizeof(T)*sz, barrier);
		barrier.arrive_and_wait();
	}
	__device__ inline void pop_top() {
		ASSERT(bounds.size() >= 2);
		data.resize(begin());
		bounds.pop_back();
	}
};

#endif

