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


#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cosmictiger/assert.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/simd.hpp>

#include <cstdint>
#include <limits>
#include <utility>

#include <limits.h>


template<class >
class fixed;

using fixed32 = fixed<uint32_t>;
using fixed64 = fixed<uint64_t>;

static constexpr float fixed2float = 1.f / float(size_t(1) << size_t(32));

template<class T>
class fixed {
	T i;
	static constexpr float c0 = float(size_t(1) << size_t(32));
	static constexpr float cinv = 1.f / c0;
	static constexpr double dblecinv = 1.f / c0;
	static constexpr T width = (sizeof(float) * CHAR_BIT);
public:
	friend class simd_fixed32;

	CUDA_EXPORT
	inline T raw() const {
		return i;
	}

	CUDA_EXPORT
	inline static fixed<T> max() {
		fixed<T> num;
#ifdef __CUDA_ARCH__
		num.i = 0xFFFFFFFFUL;
#else
		num.i = std::numeric_limits<T>::max();
#endif
		return num;
	}
	CUDA_EXPORT
	inline static fixed<T> min() {
		fixed<T> num;
		num.i = 1;
		return num;
	}
	inline fixed<T>() = default;

	CUDA_EXPORT
	inline
	fixed<T>& operator=(double number) {
		i = (c0 * number);
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T>(float number) :
			i(c0 * number) {
	}
	CUDA_EXPORT
	inline fixed<T>(double number) :
			i(c0 * number) {
	}

	template<class V>

	CUDA_EXPORT
	inline constexpr fixed<T>(fixed<V> other) :
			i(other.i) {
	}

	CUDA_EXPORT
	inline bool operator<(fixed other) const {
		return i < other.i;
	}

	CUDA_EXPORT
	inline bool operator>(fixed other) const {
		return i > other.i;
	}

	CUDA_EXPORT
	inline bool operator<=(fixed other) const {
		return i <= other.i;
	}

	CUDA_EXPORT
	inline bool operator>=(fixed other) const {
		return i >= other.i;
	}

	CUDA_EXPORT
	inline bool operator==(fixed other) const {
		return i == other.i;
	}

	CUDA_EXPORT
	inline bool operator!=(fixed other) const {
		return i != other.i;
	}

	CUDA_EXPORT
	inline float to_float() const {
		return float(i) * cinv;

	}

	CUDA_EXPORT
	inline int to_int() const {
		return i >> width;
	}

	CUDA_EXPORT
	inline double to_double() const {
		return i * dblecinv;

	}

	CUDA_EXPORT
	inline fixed<T> operator*(const fixed<T> &other) const {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = (b * c) >> width;
		fixed<T> res;
		res.i = (T) a;
		return res;
	}

	CUDA_EXPORT
	inline fixed<T> operator*=(const fixed<T> &other) {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = (b * c) >> width;
		i = (T) a;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T> operator*=(int other) {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other;
		a = b * c;
		i = (T) a;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T> operator/(const fixed<T> &other) const {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = b / (c >> width);
		fixed<T> res;
		res.i = (T) a;
		return res;
	}

	CUDA_EXPORT
	inline fixed<T> operator/=(const fixed<T> &other) {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = b / (c >> width);
		i = (T) a;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T> operator+(const fixed<T> &other) const {
		fixed<T> a;
		a.i = i + other.i;
		return a;
	}

	CUDA_EXPORT
	inline fixed<T> operator-(const fixed<T> &other) const {
		fixed<T> a;
		a.i = i - other.i;
		return a;
	}

	CUDA_EXPORT
	inline fixed<T> operator-() const {
		fixed<T> a;
		a.i = -i;
		return a;
	}

	CUDA_EXPORT
	inline fixed<T>& operator+=(const fixed<T> &other) {
		i += other.i;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T>& operator-=(const fixed<T> &other) {
		i -= other.i;
		return *this;
	}

	CUDA_EXPORT
	inline T get_integer() const {
		return i;
	}

	template<class A>
	void serialize(A &arc, unsigned) {
		arc & i;
	}

	template<class >
	friend class fixed;

	template<class V>
	friend void swap(fixed<V> &first, fixed<V> &second);

	friend fixed32 rand_fixed32();


};

template<class T>
CUDA_EXPORT
inline fixed<T> max(const fixed<T>& a, const fixed<T>& b) {
	if( a > b ) {
		return a;
	} else {
		return b;
	}
}

template<class T>
CUDA_EXPORT
inline fixed<T> min(const fixed<T>& a, const fixed<T>& b) {
	if( a < b ) {
		return a;
	} else {
		return b;
	}
}

template<class T>
inline void swap(fixed<T> &first, fixed<T> &second) {
	std::swap(first, second);
}
#endif /* COSMICTIGER_FIXED_HPP_ */
