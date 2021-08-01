/*
 * fixedcapvec.hpp
 *
 *  Created on: Jul 9, 2021
 *      Author: dmarce1
 */

#ifndef FIXEDCAPVEC_HPP_
#define FIXEDCAPVEC_HPP_

#include <tigerfmm/defs.hpp>
#include <tigerfmm/containers.hpp>
#include <tigerfmm/cuda.hpp>
#include <tigerfmm/safe_io.hpp>

#ifdef __CUDACC__
template<class T, int N>
class fixedcapvec {
	array<T, N> data_;
	int sz;
public:
	__device__
	constexpr fixedcapvec() {
		const int tid = threadIdx.x;
		if (tid == 0) {
			sz = 0;
		}
		__syncwarp();
	}
	__device__
	void initialize() {
		const int tid = threadIdx.x;
		if (tid == 0) {
			sz = 0;
		}
		__syncwarp();
	}
	fixedcapvec(const fixedcapvec&) = default;
	fixedcapvec& operator=(const fixedcapvec&) = default;
	__device__
	inline int size() const {
		return sz;
	}
	__device__
	inline void resize(int new_sz) {
		const int tid = threadIdx.x;
		__syncwarp();
		if (tid == 0) {
			sz = new_sz;
			if (sz >= N) {
				PRINT("fixedcapvec exceeded size of %i in %s on line %i\n", N, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
				__trap();
#else
				abort();
#endif
			}
		}
		__syncwarp();
	}
	__device__
	inline void push_back(const T& item) {
		const int tid = threadIdx.x;
		if (tid == 0) {
			data_[sz] = item;
			sz++;
			if (sz >= N) {
				PRINT("fixedcapvec exceeded size of %i in %s on line %i\n", N, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
				__trap();
#else
				abort();
#endif
			}
		}
		__syncwarp();
	}
	__device__
	inline void pop_back() {
		const int tid = threadIdx.x;
		if( tid == 0 ) {
			sz--;
		}
		__syncwarp();
	}
	__device__
	inline T& back() {
		assert(sz>=1);
		return data_[sz - 1];
	}
	__device__
	inline T back() const {
		return data_[sz - 1];
	}
	__device__
	T* begin() {
		return data_;
	}
	__device__
	T* end() {
		return data_ + sz;
	}
	__device__
	T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if( i < 0 || i >= sz) {
			PRINT( "index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, sz);
#ifdef __CUDA_ARCH__
			__trap();
#else
			abort();
#endif
		}
#endif
		return data_[i];
	}
	__device__
	T operator[](int i) const {
#ifdef CHECK_BOUNDS
		if( i < 0 || i >= sz) {
			PRINT( "index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, sz);
#ifdef __CUDA_ARCH__
			__trap();
#else
			abort();
#endif
		}
#endif
		return data_[i];
	}
	__device__
	T* data() {
		return data_;
	}

};
#endif
#endif /* FIXEDCAPVEC_HPP_ */
