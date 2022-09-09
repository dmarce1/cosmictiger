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


#include <math.h>

template<class T = float>
class complex {
	T x, y;
public:
	complex() = default;

	complex(T a) {
		x = a;
		y = T(0.0);
	}

	complex(T a, T b) {
		x = a;
		y = b;
	}

	complex& operator+=(complex other) {
		x += other.x;
		y += other.y;
		return *this;
	}

	complex& operator-=(complex other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}

	complex operator*(complex other) const {
		complex a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}

	complex operator/(complex other) const {
		return *this * other.conj() / other.norm();
	}

	complex operator/=(complex other) {
		*this = *this * other.conj() / other.norm();
		return *this;
	}

	complex operator/(T other) const {
		complex b;
		b.x = x / other;
		b.y = y / other;
		return b;
	}

	complex operator*(T other) const {
		complex b;
		b.x = x * other;
		b.y = y * other;
		return b;
	}

	complex& operator*=(T other) {
		x *= other;
		y *= other;
		return *this;
	}
	complex& operator*=(complex other) {
		*this = *this * other;
		return *this;
	}

	complex operator+(complex other) const {
		complex a;
		a.x = x + other.x;
		a.y = y + other.y;
		return a;
	}

	complex operator-(complex other) const {
		complex a;
		a.x = x - other.x;
		a.y = y - other.y;
		return a;
	}

	complex conj() const {
		complex a;
		a.x = x;
		a.y = -y;
		return a;
	}

	T real() const {
		return x;
	}

	T imag() const {
		return y;
	}

	T& real() {
		return x;
	}

	T& imag() {
		return y;
	}

	T norm() const {
		return ((*this) * conj()).real();
	}

	T abs() const {
		return sqrtf(norm());
	}

	complex operator-() const {
		complex a;
		a.x = -x;
		a.y = -y;
		return a;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & y;
	}
};

template<class T>
inline void swap(complex<T>& a, complex<T>& b) {
	std::swap(a.real(), b.real());
	std::swap(a.imag(), b.imag());
}



using cmplx = complex<float>;


inline cmplx expc(cmplx z) {
	float x, y;
	float t = std::exp(z.real());
	sincosf(z.imag(), &y, &x);
	x *= t;
	y *= t;
	return cmplx(x, y);
}
#endif /* COMPLEX_HPP_ */
