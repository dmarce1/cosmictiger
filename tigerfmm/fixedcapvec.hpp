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

template<class T, int N>
class fixedcapvec {
	array<T, N> data;
	int sz;
public:
	CUDA_EXPORT constexpr fixedcapvec() : sz(0){
	}
	fixedcapvec(const fixedcapvec&) = default;
	fixedcapvec& operator=(const fixedcapvec&) = default;
	CUDA_EXPORT
	inline int size() const {
		return sz;
	}
	CUDA_EXPORT
	inline void resize(int new_sz) {
		sz = new_sz;
		if( sz >= N ) {
			PRINT( "fixedcapvec exceeded size of %i in %s on line %i\n", N, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
			__trap();
#else
			abort();
#endif
		}
	}
	CUDA_EXPORT
	inline void push_back(const T& item) {
		data[sz] = item;
		sz++;
		if( sz >= N ) {
			PRINT( "fixedcapvec exceeded size of %i in %s on line %i\n", N, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
			__trap();
#else
			abort();
#endif
		}
	}
	CUDA_EXPORT
	inline void pop_back() {
		sz--;
	}
	CUDA_EXPORT
	inline T& back() {
		return data[sz - 1];
	}
	CUDA_EXPORT
	inline T back() const {
		return data[sz - 1];
	}
	CUDA_EXPORT
	T* begin() {
		return data;
	}
	CUDA_EXPORT
	T* end() {
		return data + sz;
	}
	CUDA_EXPORT T& operator[](int i) {
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
		return data[i];
	}
	CUDA_EXPORT T operator[](int i) const {
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
		return data[i];
	}

};

#endif /* FIXEDCAPVEC_HPP_ */
