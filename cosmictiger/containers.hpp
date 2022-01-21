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

#ifndef CONTAINERS_HPP_
#define CONTAINERS_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/safe_io.hpp>

#include <array>
#include <vector>

#ifdef __CUDA_ARCH__
#define DO_BOUNDS_CHECK(i) \
		if(  i >= this->size() ) { \
			THROW_ERROR( "Bounds error - %i is not between 0 and %i\n",  i, this->size()); \
			__trap(); \
		}
#else
#define DO_BOUNDS_CHECK(i) \
		if( i >= this->size() ) { \
			THROW_ERROR( "Bounds error - %i is not between 0 and %i\n",  i, this->size()); \
		}
#endif

template<class T, class Alloc = std::allocator<T>>
class vector: public std::vector<T, Alloc> {
	using base_type = std::vector<T, Alloc>;
	using base_type::base_type;

public:
#ifdef CHECK_BOUNDS
	inline T& operator[](size_t i) {
		DO_BOUNDS_CHECK(i);
		return (*((std::vector<T>*) this))[i];
	}
	inline const T& operator[](size_t i) const {
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
	inline const T& operator[](int i) const {
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

template<class T>
inline array<T, NDIM> operator*(const array<T, NDIM>& a, T b) {
	array<T, NDIM> c;
	for (int dim = 0; dim < NDIM; dim++) {
		c[dim] = a[dim] * b;
	}
	return c;
}

template<class T>
inline bool operator==(const array<T, NDIM>& a, const array<T, NDIM>& b) {
	for (int dim = 0; dim < NDIM; dim++) {
		if (a[dim] != b[dim]) {
			return false;
		}
	}
	return true;
}

template<class T>
inline bool operator!=(const array<T, NDIM>& a, const array<T, NDIM>& b) {
	return !(a == b);
}

template<class T>
inline array<T, NDIM> operator+(const array<T, NDIM>& a, const array<T, NDIM>& b) {
	array<T, NDIM> c;
	for (int dim = 0; dim < NDIM; dim++) {
		c[dim] = a[dim] + b[dim];
	}
	return c;
}

template<class T>
inline array<T, NDIM> operator-(const array<T, NDIM>& a, const array<T, NDIM>& b) {
	array<T, NDIM> c;
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
