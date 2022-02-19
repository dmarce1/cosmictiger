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


#ifndef CUDA_cuda_vector_HPP_
#define CUDA_cuda_vector_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/cuda.hpp>

#ifdef __CUDACC__

template<class T>
class cuda_vector {
	T *ptr;
	unsigned cap;
	unsigned sz;

public:
	__device__
	T* begin() {
		return ptr;
	}
	__device__
	T* end() {
		return ptr + sz;
	}
	__device__
	inline unsigned capacity() const {
		return cap;
	}
	__device__
	inline cuda_vector() {
		ptr = nullptr;
		cap = 0;
		sz = 0;
	}
	__device__ inline
	void initialize() {
		ptr = nullptr;
		cap = 0;
		sz = 0;
	}
	__device__
	inline cuda_vector(unsigned _sz) {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
		resize(_sz);
	}
	__device__
	inline cuda_vector(unsigned _sz, T ele) {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
		__syncthreads();
		resize(_sz);
		__syncthreads();
		for (unsigned i = tid; i < _sz; i += WARP_SIZE) {
			(*this)[i] = ele;
		}
	}
	__device__
	inline cuda_vector(const cuda_vector &&other) {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
		__syncthreads();
		swap(other);
	}
	__device__
	inline cuda_vector(const cuda_vector &other) {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			sz = 0;
			ptr = nullptr;
			cap = 0;
		};
		__syncthreads();
		reserve(other.cap);
		__syncthreads();
		if (tid == 0) {
			sz = other.sz;
			cap = other.sz;
		}
		__syncthreads();
		for (unsigned i = tid; i < other.sz; i += WARP_SIZE) {
			(*this)[i] = other[i];
		}
	}
	__device__
	inline cuda_vector& operator=(const cuda_vector &other) {
		const int& tid = threadIdx.x;
		reserve(other.cap);
		resize(other.size());
		for (unsigned i = tid; i < other.size(); i += WARP_SIZE) {
			(*this)[i] = other[i];
		}
		return *this;
	}
	__device__
	inline cuda_vector& operator=(cuda_vector &&other) {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			swap(other);
		}
		return *this;
	}
	__device__
	inline cuda_vector(cuda_vector &&other) {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			sz = 0;
			ptr = nullptr;
			cap = 0;
			swap(other);
		}
	}
	__device__
	inline
	void reserve(unsigned new_cap) {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (new_cap > cap) {
			size_t i = 1;
			while (i < new_cap) {
				i *= 2;
			}
			new_cap = i;
			T* new_ptr;
			if (tid == 0) {
				new_ptr = (T*) malloc(new_cap * sizeof(T));
				if( new_ptr == nullptr) {
					printf("CUDA OOM\n");
					__trap();
				}
			}
			size_t new_ptr_int = (size_t) new_ptr;
			new_ptr_int = __shfl_sync(0xFFFFFFFF, new_ptr_int, 0);
			new_ptr = (T*) new_ptr_int;
			for (unsigned i = tid; i < sz; i += WARP_SIZE) {
				new (new_ptr + i) T();
				new_ptr[i] = std::move((*this)[i]);
			}
			__syncthreads();
			if (tid == 0) {
				cap = new_cap;
				if (ptr) {
					free(ptr);
				}
				ptr = new_ptr;
			}
		}
	}
	__device__
	inline
	void resize(unsigned new_size) {
		const int& tid = threadIdx.x;
		__syncthreads();
		reserve(new_size);
		if (tid == 0) {
			sz = new_size;
		}
		__syncthreads();
	}
	__device__
	inline T operator[](unsigned i) const {
		ASSERT(i < sz);
		return ptr[i];
	}
	__device__
	inline T& operator[](unsigned i) {
		ASSERT(i < sz);
		return ptr[i];
	}
	__device__
	inline unsigned size() const {
		return sz;
	}
	__device__
	inline
	void push_back(const T &dat) {
		const int& tid = threadIdx.x;
		resize(size() + 1);
		if (tid == 0) {
			ptr[size() - 1] = dat;
		}
	}
	__device__
	inline
	void push_back(T &&dat) {
		const int& tid = threadIdx.x;
		resize(size() + 1);
		if (tid == 0) {
			ptr[size() - 1] = std::move(dat);
		}
	}
	__device__
	inline T* data() {
		return ptr;
	}
	__device__
	inline const T* data() const {
		return ptr;
	}
	__device__
	inline void destroy() {
		const int& tid = threadIdx.x;
		if (tid == 0 && ptr) {
			free(ptr);
		}
	}
	__device__
	inline ~cuda_vector() {
		destroy();
	}
	__device__
	inline void pop_back() {
		ASSERT(size());
		resize(size() - 1);
	}
	__device__
	inline T back() const {
		return ptr[size() - 1];
	}
	__device__
	inline T& back() {
		return ptr[size() - 1];
	}
	__device__
	inline
	void swap(cuda_vector &other) {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			auto tmp1 = sz;
			auto tmp2 = cap;
			auto tmp3 = ptr;
			sz = other.sz;
			cap = other.cap;
			ptr = other.ptr;
			other.sz = tmp1;
			other.cap = tmp2;
			other.ptr = tmp3;
		}
	}
};

#endif

#endif /* CUDA_cuda_vector_HPP_ */
