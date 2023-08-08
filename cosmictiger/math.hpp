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

#ifndef MATH_HPP_
#define MATH_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/flops.hpp>

#include <functional>
#include <atomic>

template<class T>
CUDA_EXPORT inline T sqr(const T& a) {
	return a * a;
}

template<class T>
CUDA_EXPORT inline T sqr(const T& a, const T& b, const T& c) {
	return a * a + b * b + c * c;
}

inline float rand1() {
	return ((float) rand() + 0.5) / (float) RAND_MAX;
}

inline float rand_normal() {
	const double x = rand1();
	const double y = rand1();
	return sqrt(-2.0 * log(x)) * cos(2 * M_PI * y);
}

template<class T>
CUDA_EXPORT int constrain_range(T& x) {
	int flops = 2;
	if (x > T(1)) {
		x -= T(1);
		flops++;
	}
	if (x < T(0)) {
		x += T(1);
		flops++;
	}
	if (x > T(1)) {
		PRINT("Print particle out of range %e\n", x);
		ALWAYS_ASSERT(false);
	}
	if (x < T(0)) {
		PRINT("Print particle out of range %e\n", x);
		ALWAYS_ASSERT(false);
	}
	return flops;
}

template<class T>
CUDA_EXPORT
inline T round_up(T a, T b) {
	if (a > 0) {
		return (((a - 1) / b) + 1) * b;
	} else {
		return 0;
	}
}

template<class T>
CUDA_EXPORT
inline T round_down(T a, T b) {
	return (a / b) * b;
}



CUDA_EXPORT
inline double sinc(double x) {
	if (x == 0.0) {
		return 1.0;
	} else {
		return sin(x) / x;
	}
}

inline double double_factorial(int n) {
	if (n < 1) {
		return 1;
	} else {
		return n * double_factorial(n - 2);
	}
}

inline void atomic_add(std::atomic<float>& y, float x) {
	float z = y;
	while (!y.compare_exchange_strong(z, z + x)) {
	}

}

CUDA_EXPORT
inline float tsc(float x) {
	const float absx = fabsf(x);
	if (absx < 0.5) {
		return 0.75 - sqr(x);
	} else if (absx < 1.5) {
		return 0.5 * sqr(1.5 - absx);
	} else {
		return 0.0;
	}
}


inline double difference(const std::function<double(double, double, double)>& f, double x, double y, double z, int n, int m, int l, double dx = 5e-3) {
	if (n > 0) {
		return (difference(f, x + dx, y, z, n - 1, m, l, dx) - difference(f, x - dx, y, z, n - 1, m, l, dx)) / (2.0 * dx);
	} else if (m > 0) {
		return (difference(f, x, y + dx, z, n, m - 1, l, dx) - difference(f, x, y - dx, z, n, m - 1, l, dx)) / (2.0 * dx);
	} else if (l > 0) {
		return (difference(f, x, y, z + dx, n, m, l - 1, dx) - difference(f, x, y, z - dx, n, m, l - 1, dx)) / (2.0 * dx);
	} else {
		return f(x, y, z);
	}
}

#include <cosmictiger/containers.hpp>

pair<double, array<double, NDIM>> find_eigenpair(const array<array<double, NDIM>, NDIM>& A, double mu);



#endif /* MATH_HPP_ */
