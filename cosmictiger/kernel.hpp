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

#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>

#define KERNEL_CUBIC_SPLINE 0
#define KERNEL_QUARTIC_SPLINE 1
#define KERNEL_QUINTIC_SPLINE 2
#define KERNEL_WENDLAND_C2 3
#define KERNEL_WENDLAND_C4 4
#define KERNEL_WENDLAND_C6 5

void kernel_set_type(int type);
void kernel_output();
void kernel_adjust_options(options& opts);

#define Power pow
#define Pi T(M_PI)

template<class T>
CUDA_EXPORT
inline T kernelWfourier(T k) {
	const auto tmp = sin(k / T(4.));
	const auto k2inv = sqr(1.f / k);
	if (k < T(0.02)) {
		return T(1) + T(-3.0 / 80.0) * sqr(k);
	} else {
		return T(-3072.f) * (k * cos(k / T(4.f)) - T(4) * sin(k / T(4.))) * tmp * sqr(tmp) * k2inv * sqr(k2inv);
	}
}

#ifdef __CUDACC__
#ifdef __KERNEL_CU__
__managed__ int kernel_type;
#else
extern __managed__ int kernel_type;
#endif
#endif

template<class T>
CUDA_EXPORT
inline T kernelW(T q) {
	T res, w1, w2, sw1, w3, sw2, sw3;
#ifdef __CUDA_ARCH__
	q = fminf(q,1.f);
#else
#endif
	const T c0 = T(2187.0 / (40.0 * M_PI));
	w1 = T(-10);
	w1 = fmaf(q, w1, T(10));
	w1 *= q;
	w1 = fmaf(q, w1, T(-20.0f / 9.f));
	w1 *= q;
	w1 = fmaf(q, w1, T(22.0f / 81.0f));
	w2 = T(5);
	w2 = fmaf(q, w2, -T(15));
	w2 = fmaf(q, w2, T(50. / 3.));
	w2 = fmaf(q, w2, T(-70. / 9.));
	w2 = fmaf(q, w2, T(25. / 27.));
	w2 = fmaf(q, w2, T(17. / 81.));
	const T tmp = T(1.) - q;
	w3 = sqr(sqr(tmp)) * tmp;
	sw1 = q < T(1. / 3.);
	sw3 = q > T(2. / 3.);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
	T res, w1, w2, sw1, w3, sw2, sw3;
	const T c0 = T(2187.0 / (40.0 * M_PI));
#ifdef __CUDA_ARCH__
	q = fminf(q,1.f);
#else
#endif
	w1 = T(-50);
	w1 = fmaf(q, w1, T(40));
	w1 *= q;
	w1 = fmaf(q, w1, T(-40.0f / 9.f));
	w1 *= q;
	w2 = T(25);
	w2 = fmaf(q, w2, -T(60));
	w2 = fmaf(q, w2, T(50.));
	w2 = fmaf(q, w2, T(-140. / 9.));
	w2 = fmaf(q, w2, T(25. / 27.));
	w3 = -T(5) * sqr(sqr(T(1) - q));
	sw1 = q < T(1. / 3.);
	sw3 = q > T(2. / 3.);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T res, sw1, sw2, w1;
	const auto q0 = q;
	q *= (q < T(1));
	const auto q2 = sqr(q);
	w1 = T(105.0 / 16.0);
	w1 = fmaf(w1, q2, T(-189.0 / 16.0));
	w1 = fmaf(w1, q2, T(135.0 / 16.0));
	w1 = fmaf(w1, q2, T(-35.0 / 16.0));
	res = w1;
	/*w1 = T(32);
	 w1 = fmaf(q, w1, -T(192.f / 5.f));
	 w1 *= q;
	 w1 = fmaf(q, w1, T(32.f / 3.f));
	 w2 = -T(32.f / 3.f);
	 w2 = fmaf(q, w2, T(192 / 5.f));
	 w2 = fmaf(q, w2, -T(48.f));
	 w2 = fmaf(q, w2, T(64.f / 3.f));
	 q3inv = T(1) / (q + T(1.0e-10f));
	 q3inv = sqr(q3inv) * q3inv;
	 w2 -= T(1.f / 15.f) * q3inv;
	 sw1 = q < T(0.5);
	 res = (sw1 * w1 + (T(1) - sw1) * w2);*/
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (sqr(q0) * q0 + T(1e-30))) * (q0 > T(0));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T res, sw1, sw2, w1;
	const auto q0 = q;
	q *= (q < T(1));
	const auto q2 = sqr(q);
	w1 = T(315.0 / 128.0);
	w1 = fmaf(w1, q2, T(-105.0 / 32.0));
	w1 = fmaf(w1, q2, T(189.0 / 64.0));
	w1 = fmaf(w1, q2, T(-45.0 / 32.0));
	w1 = fmaf(w1, q2, T(35.0 / 128.0));
	res = w1;
	/*w1 = -T(32.f / 5.f);
	 w1 = fmaf(q, w1, T(48.f / 5.f));
	 w1 *= q;
	 w1 = fmaf(q, w1, T(-16.f / 3.f));
	 w1 *= q;
	 w1 = fmaf(q, w1, T(14.f / 5.f));
	 w2 = T(32.f / 15.f);
	 w2 = fmaf(q, w2, T(-48.f / 5.f));
	 w2 = fmaf(q, w2, T(16.f));
	 w2 = fmaf(q, w2, T(-32.f / 3.f));
	 w2 *= q;
	 w2 = fmaf(q, w2, T(16.f / 5.f));
	 q1inv = T(1) / (q + T(1.0e-30f));
	 w2 -= T(1.f / 15.f) * q1inv;
	 sw1 = q < T(0.5f);
	 res = (sw1 * w1 + (T(1) - sw1) * w2);*/
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (q0 + T(1e-30f)));
	return res;
}
