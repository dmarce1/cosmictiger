#pragma once

#include <cosmictiger/fixedcapvec.hpp>

#ifdef __CUDACC__

template<class T, int SIZE, int DEPTH>
class stack_vector {
	fixedcapvec<T,SIZE> data;
	fixedcapvec<int,DEPTH> bounds;
	__device__ inline int begin() const {
		assert(bounds.size() >= 2);
		return bounds[bounds.size() - 2];
	}
	__device__ inline int end() const {
		assert(bounds.size() >= 2);
		return bounds.back();
	}
public:
	__device__ inline int depth() const {
		return bounds.size() - 2;
	}
	__device__ inline void destroy() {
		bounds.destroy();
		data.destroy();
	}
	__device__ inline void initialize() {
		const int& tid = threadIdx.x;
		bounds.resize(2);
		if (tid == 0) {
			bounds[0] = 0;
			bounds[1] = 0;
		}
		__syncwarp();
	}
	__device__ inline stack_vector() {
		const int& tid = threadIdx.x;
		bounds.reserve(CUDA_MAX_DEPTH + 1);
		bounds.resize(2);
		if (tid == 0) {
			bounds[0] = 0;
			bounds[1] = 0;
		}
		__syncwarp();
	}
	__device__ inline void push(const T &a) {
		const int& tid = threadIdx.x;
		assert(bounds.size() >= 2);
		data.push_back(a);
		if (tid == 0) {
			bounds.back()++;
		}
		__syncwarp();

	}
	__device__ inline int size() const {
		assert(bounds.size() >= 2);
		return end() - begin();
	}
	__device__ inline void resize(int sz) {
		const int& tid = threadIdx.x;
		assert(bounds.size() >= 2);
		data.resize(begin() + sz);
		if (tid == 0) {
			bounds.back() = data.size();
		}
		__syncwarp();
	}
	__device__ inline T operator[](int i) const {
		assert(i < size());
		return data[begin() + i];
	}
	__device__ inline T& operator[](int i) {
		assert(i < size());
		return data[begin() + i];
	}

	__device__ inline void push_top() {
		const int& tid = threadIdx.x;

		const auto sz = size();
		bounds.push_back(end() + sz);
		data.resize(data.size() + sz);
		for (int i = begin() + tid; i < end(); i += WARP_SIZE) {
			data[i] = data[i - sz];
		}
		__syncwarp();
	}
	__device__ inline void pop_top() {
		assert(bounds.size() >= 2);
		data.resize(begin());
		bounds.pop_back();
	}
};

#endif
