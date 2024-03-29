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

#ifdef USE_CUDA
__device__ inline void erfcexp(float x, float* ec, float *ex) {				// 18 + FLOP_DIV + FLOP_EXP
	const float p(0.3275911f);
	const float a1(0.254829592f);
	const float a2(-0.284496736f);
	const float a3(1.421413741f);
	const float a4(-1.453152027f);
	const float a5(1.061405429f);
	const float t1 = 1.f / fma(p, x, 1.f);// FLOP_DIV + 2
	const float t2 = t1 * t1;// 1
	const float t3 = t2 * t1;// 1
	const float t4 = t2 * t2;// 1
	const float t5 = t2 * t3;// 1
	*ex = expf(-x * x);// 2 + FLOP_EXP
	*ec = fmaf(a1, t1, fmaf(a2, t2, fmaf(a3, t3, fmaf(a4, t4, a5 * t5)))) * *ex;// 10
}
#endif

inline void erfcexp(double x, double* ec, double *ex) {				// 18 + FLOP_DIV + FLOP_EXP
	*ex = exp(-x * x);				// 2 + FLOP_EXP
	*ec = erf(x);
}

namespace math {
inline double max(double a, double b) {
	if (a > b) {
		return a;
	} else {
		return b;
	}
}
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

CUDA_EXPORT inline float anytrue(float x) {
	return x;
}

#ifndef __CUDACC__

inline float anytrue(const simd_float& x) {
	return x.sum();
}

#endif

CUDA_EXPORT
inline double sinc(double x) {
	if (x == 0.0) {
		return 1.0;
	} else {
		return sin(x) / x;
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

template<class T>
CUDA_EXPORT inline T erfnearzero(T x) {
	const T c[15] = { 1.1283791670955126,-0.37612638903183752,0.11283791670955126,-0.026866170645131252,0.0052239776254421878,-0.00085483270234508528,0.00012055332981789664,-0.000014925650358406251,
		   1.6462114365889247e-6,-1.6365844691234924e-7,1.4807192815879217e-8,-1.2290555301717927e-9,9.422759064650411e-11,-6.7113668551641104e-12,4.4632242632864773e-13};
	T x2 = x * x;
	T w = c[14];
	for (int i = 13; i >= 0; i--) {
		w = fmaf(w, x2, c[i]);
	}
	w *= x;
	return w;
}

template<class T>
CUDA_EXPORT T expnearzero(T x) {
	T c0 = 1.;
	T c1 = 1.;
	T c2 = 0.5;
	T c3 = 0.166666666666667;
	T c4 = 0.0416666666666667;
	T c5 = 0.00833333333333333;
	T c6 = 0.00138888888888889;
	T c7 = 0.000198412698412698;
	T c8 = 0.0000248015873015873;
	T c9 = 2.75573192239859e-6;
	T c10 = 2.75573192239859e-7;
	T c11 = 2.50521083854417e-8;
	T c12 = 2.08767569878681e-9;
	T c13 = 1.60590438368216e-10;
	T c14 = 1.1470745597729725e-11;
	T c15 = 7.647163731819816e-13;
	T c16 = 4.779477332387385e-14;
	T w = c16;
	w = fmaf(w, x, c15);
	w = fmaf(w, x, c14);
	w = fmaf(w, x, c13);
	w = fmaf(w, x, c12);
	w = fmaf(w, x, c11);
	w = fmaf(w, x, c10);
	w = fmaf(w, x, c9);
	w = fmaf(w, x, c8);
	w = fmaf(w, x, c7);
	w = fmaf(w, x, c6);
	w = fmaf(w, x, c5);
	w = fmaf(w, x, c4);
	w = fmaf(w, x, c3);
	w = fmaf(w, x, c2);
	w = fmaf(w, x, c1);
	w = fmaf(w, x, c0);
	return w;
}

inline double double_factorial(int n) {
	if (n < 1) {
		return 1;
	} else {
		return n * double_factorial(n - 2);
	}
}

#include <cosmictiger/containers.hpp>

pair<double, array<double, NDIM>> find_eigenpair(const array<array<double, NDIM>, NDIM>& A, double mu);
#endif /* MATH_HPP_ */
