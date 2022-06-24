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
#include <cosmictiger/cuda_mem.hpp>

using barrier_type = cuda::barrier<cuda::thread_scope::thread_scope_block>;

template<class T>
class device_vector {
	T* ptr;
	T* new_ptr;
	int sz;
	int cap;

#ifndef __CUDA_ARCH__
	struct threadIdx_t {
		int x;
		threadIdx_t() {
			x = 0;
		}
	} threadIdx;
	inline void __syncthreads() {
	}
	inline void this_memcpy(void* dest, void* src, int size) {
		memcpy(dest, src, size);
	}
#else
	__device__
	inline void this_memcpy( void* dest, void* src, int size ) {
		__shared__ barrier_type barrier;
		auto group = cooperative_groups::this_thread_block();
		if (group.thread_rank() == 0) {
			init(&barrier, group.size());
		}
		cuda::memcpy_async(group, dest, src, size, barrier);
		barrier.arrive_and_wait();
	}
#endif
	CUDA_EXPORT
	inline void initialize() {
		const int tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			sz = 0;
			cap = 0;
			ptr = nullptr;

		}
		__syncthreads();
	}
	CUDA_EXPORT
	inline void construct(int b, int e) {
		for (int i = b; i < e; i++) {
			new (ptr + i) T;
		}
	}
	CUDA_EXPORT
	inline void destruct(int b, int e) {
		for (int i = b; i < e; i++) {
			(ptr + i)->~T();
		}
	}
public:
	CUDA_EXPORT
	inline device_vector() {
		initialize();
	}
	CUDA_EXPORT
	inline device_vector(int sz0) {
		initialize();
		resize(sz0);
	}
	CUDA_EXPORT
	inline device_vector(device_vector<T> && other) {
		initialize();
		swap(other);
	}
	CUDA_EXPORT
	inline device_vector(const device_vector<T> & other) {
		initialize();
		resize(other.size());
		this_memcpy(ptr, other.ptr, sizeof(T) * other.size());
	}
	CUDA_EXPORT
	inline device_vector& operator=(device_vector<T> && other) {
		this->~device_vector();
		initialize();
		swap(other);
		return *this;
	}
	CUDA_EXPORT
	inline device_vector& operator=(const device_vector<T> & other) {
		resize(other.size());
		this_memcpy(ptr, other.ptr, sizeof(T) * other.size());
		return *this;
	}
	CUDA_EXPORT
	void swap(device_vector& other) {
		auto* a = other.ptr;
		auto b = other.sz;
		auto c = other.cap;
		other.ptr = ptr;
		other.sz = sz;
		other.cap = cap;
		ptr = a;
		sz = b;
		cap = c;
	}

	CUDA_EXPORT
	inline ~device_vector() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			if (ptr) {
				cuda_free(ptr);
			}
		}
		__syncthreads();
	}
	CUDA_EXPORT
	inline T* data() {
		return ptr;
	}
	CUDA_EXPORT
	inline const T* data() const {
		return ptr;
	}
	CUDA_EXPORT
	inline
	CUDA_EXPORT void resize(int new_sz) {
		const int& tid = threadIdx.x;
		if (new_sz <= cap) {
			__syncthreads();
			if (tid == 0) {
				sz = new_sz;
			}
			__syncthreads();
		} else {
			__syncthreads();
			int new_cap = 1024 / sizeof(T);
			if (new_cap < 1) {
				new_cap = 1;
			}
			if (tid == 0) {
				while (new_cap < new_sz) {
					new_cap *= 2;
				}
				new_ptr = (T*) cuda_malloc(sizeof(T) * new_cap);
				if (new_ptr == nullptr) {
					PRINT("OOM in device_vector while requesting %i elements and %lli bytes! \n", new_cap, (long long ) new_cap * sizeof(T));ALWAYS_ASSERT(false);
				}
			}
			__syncthreads();
			if (ptr) {
				this_memcpy((void*) new_ptr, (void*) ptr, sz * sizeof(T));
			}
			__syncthreads();
			if (tid == 0) {
				if (ptr) {
					cuda_free(ptr);
				}
				ptr = new_ptr;
				sz = new_sz;
				cap = new_cap;
			}
			__syncthreads();
		}
	}
	CUDA_EXPORT
	inline T& back() {
		return ptr[sz - 1];
	}
	CUDA_EXPORT
	inline const T& back() const {
		return ptr[sz - 1];
	}
	CUDA_EXPORT
	inline void pop_back() {
		if (threadIdx.x == 0) {
			sz--;
		}
		__syncthreads();
	}
	CUDA_EXPORT
	inline void push_back(const T& item) {
		resize(size() + 1);
		if (threadIdx.x == 0) {
			back() = item;
		}
		__syncthreads();
	}
	CUDA_EXPORT
	inline void push_back(T&& item) {
		resize(size() + 1);
		if (threadIdx.x == 0) {
			back() = std::move(item);
		}
		__syncthreads();
	}
	CUDA_EXPORT
	inline int size() const {
		return sz;
	}
	CUDA_EXPORT
	inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");ALWAYS_ASSERT(false);
		}
#endif
		return ptr[i];
	}
	CUDA_EXPORT
	inline const T& operator[](int i) const {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");ALWAYS_ASSERT(false);
		}
#endif
		return ptr[i];
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		int this_size;
		arc & this_size;
		resize(this_size);
		for (int i = 0; i < this_size; i++) {
			arc & (*this)[i];
		}
	}

};

#include <cosmictiger/cuda_mem.hpp>

#endif /* DEVICE_VECTOR_HPP_ */
