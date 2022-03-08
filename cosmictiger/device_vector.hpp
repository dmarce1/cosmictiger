#include <cosmictiger/cuda_mem.hpp>

#pragma once

#ifdef __CUDACC__

template<class T>
class device_vector {
	int sz;
	int cap;
	T* ptr;
public:
	__device__ device_vector() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			sz = 0;
			cap = 0;
			ptr = nullptr;
		}
		__syncthreads();
	}
	__device__ ~device_vector() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			if (ptr) {
				cuda_mem_free(ptr);
			}
		}
		__syncthreads();
	}
	__device__ void shrink_to_fit() {
		const int& tid = threadIdx.x;
		const int& block_size = blockDim.x;
		__shared__ T* new_ptr;
		__syncthreads();
		int new_cap = max(1024 / sizeof(T), (size_t) 1);
		while (new_cap < sz) {
			new_cap *= 2;
		}
		if (tid == 0) {
			if (new_cap < cap) {
				new_ptr = (T*) cuda_mem_allocate(sizeof(T) * new_cap);
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
				cuda_mem_free(ptr);
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
			__shared__ T* new_ptr;
			__syncthreads();
			int new_cap = max(1024 / sizeof(T), (size_t) 1);
			if (tid == 0) {
				while (new_cap < new_sz) {
					new_cap *= 2;
				}
				new_ptr = (T*) cuda_mem_allocate(sizeof(T) * new_cap);
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
					cuda_mem_free(ptr);
				}
				ptr = new_ptr;
				sz = new_sz;
				cap = new_cap;
			}
			__syncthreads();
		}
	}
	__device__
	int size() const {
		return sz;
	}
	__device__ T& operator[](int i) {
		return ptr[i];
	}
	__device__
	             const T& operator[](int i) const {
		return ptr[i];
	}
	__device__
	void push_back(T&& item) {
		const int& tid = threadIdx.x;
		resize(size() + 1);
		if (tid == 0) {
			ptr[sz - 1] = std::move(item);
			__threadfence();
		}
	}
	__device__
	void push_back(const T& item) {
		const int& tid = threadIdx.x;
		resize(size() + 1);
		if (tid == 0) {
			ptr[sz - 1] = item;
			__threadfence();
		}
	}
	__device__
	void pop_back() {
		const int& tid = threadIdx.x;
		if (tid == 0) {
			sz--;
			__threadfence();
		}
	}
	__device__ T* data() {
		return ptr;
	}
	__device__
	  const T* data() const {
		return ptr;
	}
	__device__ T& front() {
		return ptr[0];
	}
	__device__   const T& front() const {
		return ptr[0];
	}
	__device__ T& back() {
		return ptr[sz - 1];
	}
	__device__
	   const T& back() const {
		return ptr[sz - 1];
	}
	__device__
	void reserve(int new_cap) {
		int old_size = sz;
		resize(new_cap);
		resize(old_size);

	}
};
#endif
