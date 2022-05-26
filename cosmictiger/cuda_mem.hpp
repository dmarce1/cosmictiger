#pragma once

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

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>

#define CUDA_MEM_STACK_SIZE (128*1024)
#define CUDA_MEM_NBIN 32
#define CUDA_MEM_BLOCK_SIZE (4*1024)

using cuda_mem_int = uint64_t;

class cuda_mem {
	array<array<char*, CUDA_MEM_STACK_SIZE>, CUDA_MEM_NBIN> q;
	array<long long, CUDA_MEM_NBIN> qin;
	array<long long, CUDA_MEM_NBIN> qout;
	char* heap;
	int next_block;
	int heap_max;
	__device__
	char* allocate_blocks(int nblocks);
	__device__
	void push(int bin, char* ptr);
	__device__
	char* pop(int bin);
	__device__
	bool create_new_allocations(int bin);
public:

	__device__
	void* allocate(size_t sz);
	__device__
	void free(void* ptr);
	cuda_mem(size_t heap_size);
	~cuda_mem();
	void reset();
};

void cuda_mem_init(size_t heap_size);
__device__ cuda_mem* get_cuda_heap();

#ifdef __CUDACC__

template<class T>
class device_vector {
	int sz;
	int cap;
	T* ptr;
	T* new_ptr;
	__device__ void init() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			sz = 0;
			cap = 0;
			ptr = nullptr;
		}
		__syncthreads();
	}
public:
	__device__ void swap(device_vector<T>& other) {
		__syncthreads();
		if( threadIdx.x == 0 ) {
			const auto sz0 = sz;
			const auto cap0 = cap;
			auto* const ptr0 = ptr;
			sz = other.sz;
			cap = other.cap;
			ptr = other.ptr;
			other.sz = sz0;
			other.cap = cap0;
			other.ptr = ptr0;
		}
		__syncthreads();
	}
	__device__ device_vector() {
		init();
	}
	__device__ device_vector(int sz0) {
		init();
		resize(sz0);
	}
	__device__ ~device_vector() {
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
	__device__ void shrink_to_fit() {
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
	__device__
	void resize(int new_sz) {
		const int& tid = threadIdx.x;
		const int& block_size = blockDim.x;
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
				for (int i = tid; i < sz; i += block_size) {
					new_ptr[i] = ptr[i];
				}
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
	__device__ T& back() {
		return ptr[sz-1];
	}
	__device__ const T& back() const {
		return ptr[sz-1];
	}
	__device__ void pop_back() {
		if( threadIdx.x == 0 ) {
			sz--;
		}
		__syncthreads();
	}
	__device__ void push_back(const T& item) {
		resize(size()+1);
		if( threadIdx.x == 0 ) {
			back() = item;
		}
		__syncthreads();
	}
	__device__
	int size() const {
		return sz;
	}
	__device__ T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");
			__trap();
		}
#endif
		return ptr[i];
	}
	__device__
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

#endif
