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

template<class, int>
class fixed;

using fixed32 = fixed<uint32_t, 32>;
using fixed64 = fixed<uint64_t, 32>;

static constexpr float fixed2float = 1.f / float(size_t(1) << size_t(32));

template<class T, int SIGBITS = 32>
class fixed {
	T i;
	static constexpr float c0 = float(size_t(1) << size_t(SIGBITS));
	static constexpr float c0dbl = double(size_t(1) << size_t(SIGBITS));
	static constexpr float cinv = 1.f / c0;
	static constexpr double dblecinv = 1.f / c0dbl;
	static constexpr T width = (SIGBITS);
public:
	friend class simd_fixed32;

	inline void set_integer(T j) {
		i = j;
	}

	template<int M>
	CUDA_EXPORT
	inline fixed operator=(const fixed<T, M>& other) {
		if (M > SIGBITS) {
			i = unsigned(other.i) >> (M - SIGBITS);
		} else {
			i = unsigned(other.i) << (SIGBITS - M);
		}
		return *this;
	}

	CUDA_EXPORT
	inline T raw() const {
		return i;
	}

	CUDA_EXPORT
	inline static fixed<T, SIGBITS> max() {
		fixed<T, SIGBITS> num;
#ifdef __CUDA_ARCH__
		num.i = 0xFFFFFFFFUL;
#else
		num.i = std::numeric_limits < T > ::max();
#endif
		return num;
	}
	CUDA_EXPORT
	inline static fixed<T, SIGBITS> min() {
		fixed<T, SIGBITS> num;
		num.i = 1;
		return num;
	}
	inline fixed<T, SIGBITS>() = default;

	CUDA_EXPORT
	inline fixed<T, SIGBITS>& operator=(double number) {
		i = (c0dbl * number);
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS>(float number) :
			i(c0 * number) {
	}
	CUDA_EXPORT
	inline fixed<T, SIGBITS>(int number) :
			i(c0 * number) {
	}
	CUDA_EXPORT
	inline fixed<T, SIGBITS>(double number) :
			i(c0dbl * number) {
	}

	template<class V, int M>
	CUDA_EXPORT
	inline constexpr fixed<T, SIGBITS>(fixed<V, M> other) {
		if (M > SIGBITS) {
			i = unsigned(other.i) >> abs(M - SIGBITS);
		} else {
			i = unsigned(other.i) << abs(SIGBITS - M);
		}
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
		return (float(i)) * cinv;

	}

	CUDA_EXPORT
	inline int to_int() const {
		return i >> width;
	}

	CUDA_EXPORT
	inline double to_double() const {
		return (i) * dblecinv;

	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator*(const fixed<T, SIGBITS> &other) const {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = (b * c) >> width;
		fixed<T, SIGBITS> res;
		res.i = (T) a;
		return res;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator*=(const fixed<T, SIGBITS> &other) {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = (b * c) >> width;
		i = (T) a;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator*=(int other) {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other;
		a = b * c;
		i = (T) a;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator/(const fixed<T, SIGBITS> &other) const {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = b / (c >> width);
		fixed<T, SIGBITS> res;
		res.i = (T) a;
		return res;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator/=(const fixed<T, SIGBITS> &other) {
		int64_t a;
		const int64_t b = i;
		const int64_t c = other.i;
		a = b / (c >> width);
		i = (T) a;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator+(const fixed<T, SIGBITS> &other) const {
		fixed<T, SIGBITS> a;
		a.i = i + other.i;
		return a;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator-(const fixed<T, SIGBITS> &other) const {
		fixed<T, SIGBITS> a;
		a.i = i - other.i;
		return a;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS> operator-() const {
		fixed<T, SIGBITS> a;
		a.i = -i;
		return a;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS>& operator+=(const fixed<T, SIGBITS> &other) {
		i += other.i;
		return *this;
	}

	CUDA_EXPORT
	inline fixed<T, SIGBITS>& operator-=(const fixed<T, SIGBITS> &other) {
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

	template<class, int>
	friend class fixed;

	template<class V>
	friend void swap(fixed<V> &first, fixed<V> &second);

	friend fixed32 rand_fixed32();

};

template<class T, int M>
CUDA_EXPORT
inline fixed<T, M> max(const fixed<T, M>& a, const fixed<T, M>& b) {
	if (a > b) {
		return a;
	} else {
		return b;
	}
}

template<class T, int M>
CUDA_EXPORT
inline fixed<T, M> min(const fixed<T, M>& a, const fixed<T, M>& b) {
	if (a < b) {
		return a;
	} else {
		return b;
	}
}

template<class T>
inline void swap(fixed<T> &first, fixed<T> &second) {
	std::swap(first, second);
}

CUDA_EXPORT inline float distance(fixed32 a, fixed32 b) {
	return (fixed<int32_t>(a) - fixed<int32_t>(b)).to_float();
}

CUDA_EXPORT inline float distance(double a, double b) {
	double dif = a - b;
	while (dif > 0.5) {
		dif -= 1.0;
	}
	while (dif < -0.5) {
		dif += 1.0;
	}
	return dif;
}

CUDA_EXPORT inline float distance(fixed32 a, double b) {
	double dif = a.to_double() - b;
	while (dif > 0.5) {
		dif -= 1.0;
	}
	while (dif < -0.5) {
		dif += 1.0;
	}
	return dif;
}

CUDA_EXPORT inline float distance(double a, fixed32 b) {
	double dif = a - b.to_double();
	while (dif > 0.5) {
		dif -= 1.0;
	}
	while (dif < -0.5) {
		dif += 1.0;
	}
	return dif;
}

CUDA_EXPORT inline fixed32 sum(fixed32 a, fixed32 b) {
	return (fixed<int32_t>(a) + fixed<int32_t>(b));
}

CUDA_EXPORT inline float double_distance(fixed32 a, fixed32 b) {
	return (fixed<int32_t>(a) - fixed<int32_t>(b)).to_double();
}

#endif /* COSMICTIGER_FIXED_HPP_ */
