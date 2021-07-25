#pragma once

#include <tigerfmm/fixedcapvec.hpp>

#ifdef __CUDACC__

template<class T, int N, int D>
class stack_vector {
	fixedcapvec<T, N> data;
	fixedcapvec<int, D> bounds;
	__device__
	inline int begin() const {
		return bounds[bounds.size() - 2];
	}
	__device__
	inline int end() const {
		return bounds.back();
	}
public:
	__device__
	inline int depth() const {
		return bounds.size() - 2;
	}
	__device__
	inline stack_vector() {
		bounds.resize(2);
		bounds[0] = 0;
		bounds[1] = 0;
	}
	__device__ __host__
	inline void push(const T &a) {
		data.push_back(a);
			bounds.back()++;
	}
	__device__
	inline int size() const {
		return end() - begin();
	}
	__device__
	inline void resize(int sz) {
		data.resize(begin() + sz);
		bounds.back() = data.size();
	}
	__device__
	inline T operator[](int i) const {
		return data[begin() + i];
	}
	__device__
	inline T& operator[](int i) {
		return data[begin() + i];
	}
	__device__
	inline void push_top() {
		const int& blocksize = blockDim.x;
		const int& tid = threadIdx.x;
		const auto sz = size();
		if( end() + sz >= N ) {
			PRINT( "Maximum stack_vector size exceeded\n");
			__trap();
		}
		if( bounds.size() > D ) {
			PRINT( "Maximum stack_vector depth exceeded\n");
			__trap();
		}
		if( tid == 0 ) {
			bounds.push_back(end() + sz);
			data.resize(data.size() + sz);
		}
		__syncwarp();
		for (int i = begin() + tid; i < end(); i += blocksize) {
			data[i] = data[i - sz];
		}
		__syncwarp();
	}
	__device__
	inline void pop_top() {
		data.resize(begin());
		bounds.pop_back();
	}
};

#endif
