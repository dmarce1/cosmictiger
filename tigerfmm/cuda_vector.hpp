/*
 * cuda_cuda_vector.hpp
 *
 *  Created on: Jul 26, 2021
 *      Author: dmarce1
 */

#ifndef CUDA_cuda_vector_HPP_
#define CUDA_cuda_vector_HPP_

#include <tigerfmm/defs.hpp>
#include <tigerfmm/cuda.hpp>

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
		const int& tid = threadIdx.x;
		if (tid == 0) {
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
		__syncwarp();
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
		__syncwarp();
		resize(_sz);
		__syncwarp();
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
		__syncwarp();
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
		__syncwarp();
		reserve(other.cap);
		__syncwarp();
		if (tid == 0) {
			sz = other.sz;
			cap = other.sz;
		}
		__syncwarp();
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
		__syncwarp();
		if (new_cap > cap) {
			size_t i = 1;
			while (i < new_cap) {
				i *= 2;
			}
			new_cap = i;
			T* new_ptr;
			if (tid == 0) {
				new_ptr = (T*) malloc(new_cap * sizeof(T));
			}
			size_t new_ptr_int = (size_t) new_ptr;
			new_ptr_int = __shfl_sync(0xFFFFFFFF, new_ptr_int, 0);
			new_ptr = (T*) new_ptr_int;
			for (unsigned i = tid; i < sz; i += WARP_SIZE) {
				new (new_ptr + i) T();
				new_ptr[i] = std::move((*this)[i]);
			}
			__syncwarp();
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
		__syncwarp();
		reserve(new_size);
		if (tid == 0) {
			sz = new_size;
		}
		__syncwarp();
	}
	__device__
	inline T operator[](unsigned i) const {
		assert(i < sz);
		return ptr[i];
	}
	__device__
	inline T& operator[](unsigned i) {
		assert(i < sz);
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
	inline ~cuda_vector() {
		const int& tid = threadIdx.x;
		if (tid == 0 && ptr) {
			free(ptr);
		}
	}
	__device__
	inline void pop_back() {
		assert(size());
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
