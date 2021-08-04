/*
 * containers.hpp
 *
 *  Created on: Jun 30, 2021
 *      Author: dmarce1
 */

#ifndef CONTAINERS_HPP_
#define CONTAINERS_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/safe_io.hpp>

#include <array>
#include <vector>

#ifdef __CUDA_ARCH__
#define DO_BOUNDS_CHECK(i) \
		if( i < 0 || i >= this->size() ) { \
			THROW_ERROR( "Bounds error - %i is not between 0 and %i\n",  i, this->size()); \
			__trap(); \
		}
#else
#define DO_BOUNDS_CHECK(i) \
		if( i < 0 || i >= this->size() ) { \
			THROW_ERROR( "Bounds error - %i is not between 0 and %i\n",  i, this->size()); \
			abort(); \
		}
#endif

template<class T, class Alloc = std::allocator<T>>
class vector: public std::vector<T, Alloc> {
	using base_type = std::vector<T, Alloc>;
	using base_type::base_type;

public:
#ifdef CHECK_BOUNDS
	inline T& operator[](int i) {
		DO_BOUNDS_CHECK(i);
		return (*((std::vector<T>*) this))[i];
	}
	inline const T operator[](int i) const {
		DO_BOUNDS_CHECK(i);
		return (*((std::vector<T>*) this))[i];
	}
#endif
};

template<class T, int N>
class array {
	T A[N];
public:
	CUDA_EXPORT
	inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
		DO_BOUNDS_CHECK(i);
#endif
		return A[i];
	}
	CUDA_EXPORT
	inline T operator[](int i) const {
#ifdef CHECK_BOUNDS
		DO_BOUNDS_CHECK(i);
#endif
		return A[i];
	}
	CUDA_EXPORT
	inline const T* data() const {
		return A;
	}
	CUDA_EXPORT
	inline T* data() {
		return A;
	}
	CUDA_EXPORT
	inline int size() const {
		return N;
	}
	CUDA_EXPORT
	inline const T* begin() const {
		return A;
	}
	CUDA_EXPORT
	inline T* begin() {
		return A;
	}
	CUDA_EXPORT
	inline const T* end() const {
		return A + N;
	}
	CUDA_EXPORT
	inline T* end() {
		return A + N;
	}
	template<class Arc>
	void serialize(Arc&& arc, unsigned) {
		for (int i = 0; i < N; i++) {
			arc & A[i];
		}
	}
};

inline array<int, NDIM> operator*(const array<int, NDIM>& a, int b) {
	array<int, NDIM> c;
	for (int dim = 0; dim < NDIM; dim++) {
		c[dim] = a[dim] * b;
	}
	return c;
}

inline bool operator==(const array<int, NDIM>& a, const array<int, NDIM>& b) {
	for (int dim = 0; dim < NDIM; dim++) {
		if (a[dim] != b[dim]) {
			return false;
		}
	}
	return true;
}

inline bool operator!=(const array<int, NDIM>& a, const array<int, NDIM>& b) {
	return !(a == b);
}

inline array<int, NDIM> operator+(const array<int, NDIM>& a, const array<int, NDIM>& b) {
	array<int, NDIM> c;
	for (int dim = 0; dim < NDIM; dim++) {
		c[dim] = a[dim] + b[dim];
	}
	return c;
}

inline array<int, NDIM> operator-(const array<int, NDIM>& a, const array<int, NDIM>& b) {
	array<int, NDIM> c;
	for (int dim = 0; dim < NDIM; dim++) {
		c[dim] = a[dim] - b[dim];
	}
	return c;
}

template<class T, class V = T>
struct pair {
	T first;
	V second;
	pair() = default;
	pair(const pair&) = default;
	pair& operator=(const pair&) = default;
	pair(T a, V b) :
			first(a), second(b) {
	}
	template<class A>
	void serialize(A&& a, unsigned) {
		a & first;
		a & second;
	}
};

#endif /* CONTAINERS_HPP_ */
