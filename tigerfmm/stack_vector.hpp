#pragma once

#include <tigerfmm/cuda_vector.hpp>

#ifdef __CUDACC__

template<class T>
class stack_vector {
	cuda_vector<T> data;
	cuda_vector<int> bounds;
	__device__ inline int begin() const {
		assert(bounds.size() >= 2);
		return bounds[bounds.size() - 2];
	}
	__device__ inline int end() const {
		assert(bounds.size() >= 2);
		return bounds.back();
	}
public:
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & data;
		arc & bounds;
	}
	__device__ inline void swap(stack_vector<T>& other) {
		data.swap(other.data);
		bounds.swap(other.bounds);

	}
	__device__ inline int depth() const {
		return bounds.size() - 2;
	}
	__device__ inline stack_vector() {
		const int& tid = threadIdx.x;
		bounds.reserve(CUDA_MAX_DEPTH + 1);
		bounds.resize(2);
		if (tid == 0) {
			bounds[0] = 0;
			bounds[1] = 0;
		}
	}
	__device__ inline void push(const T &a) {
		const int& tid = threadIdx.x;
		assert(bounds.size() >= 2);
		data.push_back(a);
		if (tid == 0) {
			bounds.back()++;}

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
	}
	__device__  inline T operator[](int i) const {
		assert(i < size());
		return data[begin() + i];
	}
	__device__  inline T& operator[](int i) {
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
	}
	__device__ inline void pop_top() {
		assert(bounds.size() >= 2);
		data.resize(begin());
		bounds.pop_back();
	}
};

#endif

