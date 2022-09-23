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

#ifndef COMPLEX_HPP_
#define COMPLEX_HPP_

#include <algorithm>
#include <cosmictiger/cuda.hpp>
#include <math.h>

template<class T = float>
class complex {
	T x, y;
public:
	CUDA_EXPORT complex() = default;

	CUDA_EXPORT constexpr complex(T a) :
			x(a), y(0.0) {
	}

	CUDA_EXPORT constexpr complex(T a, T b) :
			x(a), y(b) {
	}

	CUDA_EXPORT complex& operator+=(complex other) {
		x += other.x;
		y += other.y;
		return *this;
	}

	CUDA_EXPORT complex& operator-=(complex other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}

	CUDA_EXPORT constexpr complex operator*(complex other) const {
		complex a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}

	CUDA_EXPORT complex operator/(complex other) const {
		return *this * other.conj() / other.norm();
	}

	CUDA_EXPORT complex operator/=(complex other) {
		*this = *this * other.conj() / other.norm();
		return *this;
	}

	CUDA_EXPORT complex operator/(T other) const {
		complex b;
		b.x = x / other;
		b.y = y / other;
		return b;
	}

	CUDA_EXPORT constexpr complex operator*(T other) const {
		complex b;
		b.x = x * other;
		b.y = y * other;
		return b;
	}

	CUDA_EXPORT complex& operator*=(T other) {
		x *= other;
		y *= other;
		return *this;
	}
	CUDA_EXPORT complex& operator*=(complex other) {
		*this = *this * other;
		return *this;
	}

	CUDA_EXPORT complex operator+(complex other) const {
		complex a;
		a.x = x + other.x;
		a.y = y + other.y;
		return a;
	}

	CUDA_EXPORT  constexpr complex operator-(complex other) const {
		complex a;
		a.x = x - other.x;
		a.y = y - other.y;
		return a;
	}

	CUDA_EXPORT constexpr complex conj() const {
		complex a;
		a.x = x;
		a.y = -y;
		return a;
	}

	CUDA_EXPORT constexpr T real() const {
		return x;
	}

	CUDA_EXPORT  constexpr T imag() const {
		return y;
	}

	CUDA_EXPORT constexpr  T& real() {
		return x;
	}

	CUDA_EXPORT constexpr T& imag() {
		return y;
	}

	CUDA_EXPORT constexpr T norm() const {
		return (x*x+y*y);
	}

	CUDA_EXPORT T abs() const {
		return sqrtf(norm());
	}

	CUDA_EXPORT constexpr complex operator-() const {
		complex a;
		a.x = -x;
		a.y = -y;
		return a;
	}
	template<class A>
	CUDA_EXPORT void serialize(A&& arc, unsigned) {
		arc & x;
		arc & y;
	}
};

template<class T>
CUDA_EXPORT inline constexpr complex<T> operator*(T a, complex<T> b) {
	return complex<T>(a*b.real(),a*b.imag());
}

template<class T>
inline void swap(complex<T>& a, complex<T>& b) {
	std::swap(a.real(), b.real());
	std::swap(a.imag(), b.imag());
}

using cmplx = complex<float>;

template<class T>
inline complex<T> expc(complex<T> z) {
	float x, y, t;
	x = cos(z.imag());
	y = sin(z.imag());
	if (z.real() != 0.0) {
		t = std::exp(z.real());
		x *= t;
		y *= t;
	}
	return complex<T>(x, y);
}
#endif /* COMPLEX_HPP_ */
