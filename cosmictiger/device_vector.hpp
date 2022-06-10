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

#ifndef DEVICE_VECTOR_HPP_
#define DEVICE_VECTOR_HPP_

#include <cooperative_groups.h>
#include <cuda/barrier>

using barrier_type = cuda::barrier<cuda::thread_scope::thread_scope_block>;

template<class T>
class device_vector {
	barrier_type barrier;
	T* ptr;
	T* new_ptr;
	int sz;
	int cap;
	__device__ inline  void initialize() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			sz = 0;
			cap = 0;
			ptr = nullptr;

		}
		__syncthreads();
		auto group = cooperative_groups::this_thread_block();
		if (group.thread_rank() == 0) {
			init(&barrier, group.size());
		}
	}
public:
	__device__ inline  device_vector() {
		initialize();
	}
	__device__ inline  device_vector(int sz0) {
		initialize();
		resize(sz0);
	}
	__device__ inline  ~device_vector() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			if (ptr) {
				auto* memory = get_cuda_heap();
				memory->free(ptr);
			}
		}
		__syncthreads();
	}
	__device__  inline T* data() {
		return ptr;
	}
	__device__ inline  void shrink_to_fit() {
		const int& tid = threadIdx.x;
		const int& block_size = blockDim.x;
		__syncthreads();
		int new_cap = max(1024 / sizeof(T), (size_t) 1);
		while (new_cap < sz) {
			new_cap *= 2;
		}
		if (tid == 0) {
			if (new_cap < cap) {
				auto* memory = get_cuda_heap();
				new_ptr = (T*) memory->allocate(sizeof(T) * new_cap);
				if (new_ptr == nullptr) {
					PRINT("OOM in device_vector while requesting %i! \n", new_cap);
					__trap();

				}
			}
		}
		__syncthreads();
		if (ptr && new_cap < cap) {
			for (int i = tid; i < sz; i += block_size) {
				new_ptr[i] = ptr[i];
			}
		}
		__syncthreads();
		if (tid == 0 && new_cap < cap) {
			if (ptr) {
				auto* memory = get_cuda_heap();
				memory->free(ptr);
			}
			ptr = new_ptr;
			cap = new_cap;
		}
		__syncthreads();
	}
	__device__ inline
	void resize(int new_sz) {
		const int& tid = threadIdx.x;
		if (new_sz <= cap) {
			__syncthreads();
			if (tid == 0) {
				sz = new_sz;
			}
			__syncthreads();
		} else {
			__syncthreads();
			int new_cap = max(1024 / sizeof(T), (size_t) 1);
			if (tid == 0) {
				while (new_cap < new_sz) {
					new_cap *= 2;
				}
				auto* memory = get_cuda_heap();
				new_ptr = (T*) memory->allocate(sizeof(T) * new_cap);
				if (new_ptr == nullptr) {
					PRINT("OOM in device_vector while requesting %i! \n", new_cap);
					__trap();

				}
			}
			__syncthreads();
			if (ptr) {
				auto group = cooperative_groups::this_thread_block();
				cuda::memcpy_async(group, new_ptr, ptr, sz * sizeof(T), barrier);
				barrier.arrive_and_wait();
			}
			__syncthreads();
			if (tid == 0) {
				if (ptr) {
					auto* memory = get_cuda_heap();
					memory->free(ptr);
				}
				ptr = new_ptr;
				sz = new_sz;
				cap = new_cap;
			}
			__syncthreads();
		}
	}
	__device__ inline  T& back() {
		return ptr[sz - 1];
	}
	__device__ inline   const T& back() const {
		return ptr[sz - 1];
	}
	__device__ inline  void pop_back() {
		if (threadIdx.x == 0) {
			sz--;
		}
		__syncthreads();
	}
	__device__ inline  void push_back(const T& item) {
		resize(size() + 1);
		if (threadIdx.x == 0) {
			back() = item;
		}
		__syncthreads();
	}
	__device__ inline
	int size() const {
		return sz;
	}
	__device__ inline  T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");
			__trap();
		}
#endif
		return ptr[i];
	}
	__device__ inline
	 const T& operator[](int i) const {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");
			__trap();
		}
#endif
		return ptr[i];
	}
};

#include <cosmictiger/cuda_mem.hpp>

#endif /* DEVICE_VECTOR_HPP_ */
