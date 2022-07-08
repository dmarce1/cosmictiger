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

#include <cosmictiger/cuda_mem.hpp>

struct threadid {
	CUDA_EXPORT
	inline bool operator==(const int tid) const {
#ifdef __CUDA_ARCH__
		return threadIdx.x == tid;
#else
		return true;
#endif
	}
	CUDA_EXPORT
	inline operator int() const {
#ifdef __CUDA_ARCH__
		return threadIdx.x;
#else
		return 0;
#endif
	}
};

struct blocksz {
	CUDA_EXPORT
	inline operator int() const {
#ifdef __CUDA_ARCH__
		return blockDim.x;
#else
		return 1;
#endif
	}
};

template<class T>
class device_vector {
	T* ptr;
	T* new_ptr;
	int sz;
	int cap;

	CUDA_EXPORT
	inline void construct(int a, int b) {
		threadid tid;
		blocksz bsz;
		syncthreads();
		for (int i = a + tid; i < b; i += (int) bsz) {
			new (ptr + i) T;
		}
		syncthreads();
	}

	CUDA_EXPORT
	inline void destruct(int a, int b) {
		threadid tid;
		blocksz bsz;
		syncthreads();
		for (int i = a + tid; i < b; i += (int) bsz) {
			(ptr + i)->~T();
		}
		syncthreads();
	}

	CUDA_EXPORT
	inline void initialize() {
		threadid tid;
		syncthreads();
		if (tid == 0) {
			sz = 0;
			cap = 0;
			ptr = nullptr;
		}
		syncthreads();
	}
	CUDA_EXPORT
	inline void destroy() {
		threadid tid;
		syncthreads();
		if (tid == 0) {
			if (ptr) {
				cuda_free(ptr);
			}
		}
		syncthreads();
	}
public:
	CUDA_EXPORT
	inline int capacity() const {
		return cap;
	}
	CUDA_EXPORT
	inline void swap(device_vector& other) {
		const auto optr = other.ptr;
		const auto ocap = other.cap;
		const auto osz = other.sz;
		other.ptr = ptr;
		other.cap = cap;
		other.sz = sz;
		ptr = optr;
		cap = ocap;
		sz = osz;
	}
	CUDA_EXPORT
	inline T* begin() {
		return ptr;
	}
	CUDA_EXPORT
	inline T* end() {
		return ptr + sz;
	}
	CUDA_EXPORT
	inline device_vector() {
		initialize();
	}
	CUDA_EXPORT
	inline device_vector(const device_vector& other) {
		initialize();
		*this = other;
	}
	CUDA_EXPORT
	inline device_vector& operator=(const device_vector& other) {
		resize(other.size());
		cuda_memcpy(ptr, other.ptr, size() * sizeof(T));
		return *this;
	}
	CUDA_EXPORT
	inline device_vector& operator=(device_vector&& other) {
		destroy();
		initialize();
		swap(other);
		return *this;
	}
	CUDA_EXPORT
	inline device_vector(device_vector&& other) {
		initialize();
		swap(other);
	}
	CUDA_EXPORT
	inline device_vector(int sz0) {
		initialize();
		resize(sz0);
	}
	CUDA_EXPORT
	inline ~device_vector() {
		destroy();
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
	void resize(int new_sz) {
		threadid tid;
		destruct(new_sz, sz);
		const auto osz = sz;
		if (new_sz <= cap) {
			syncthreads();
			if (tid == 0) {
				sz = new_sz;
			}
			syncthreads();
		} else {
			syncthreads();
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
					PRINT("OOM in device_vector while requesting %i! \n", new_cap);
					ALWAYS_ASSERT(false);

				}
			}
			syncthreads();
			if (ptr) {
				cuda_memcpy(new_ptr, ptr, sizeof(T) * sz);
			}
			syncthreads();
			destroy();
			if (tid == 0) {
				ptr = new_ptr;
				cap = new_cap;
				sz = new_sz;
			}
			syncthreads();
		}
		construct(osz, sz);
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
		threadid tid;
		if (tid == 0) {
			sz--;
		}
		syncthreads();
	}
	CUDA_EXPORT
	inline void push_back(const T& item) {
		threadid tid;
		resize(size() + 1);
		if (tid == 0) {
			back() = item;
		}
		syncthreads();
	}
	CUDA_EXPORT
	inline void push_back(T&& item) {
		threadid tid;
		resize(size() + 1);
		if (tid == 0) {
			back() = std::move(item);
		}
		syncthreads();
	}
	CUDA_EXPORT
	inline
	int size() const {
		return sz;
	}
	CUDA_EXPORT
	inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");
			ALWAYS_ASSERT(false);
		}
#endif
		return ptr[i];
	}
	CUDA_EXPORT
	inline const T& operator[](int i) const {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");
			ALWAYS_ASSERT(false);
		}
#endif
		return ptr[i];
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		int size = sz;
		arc & size;
		resize(size);
		for (int i = 0; i < size; i++) {
			arc & (*this)[i];
		}
	}
};

#include <cosmictiger/cuda_mem.hpp>

#endif /* DEVICE_VECTOR_HPP_ */
