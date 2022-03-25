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
	T res, w, mq;
	const T c0 = T(495.f / 32.f / M_PI);
	w = T(35.0f / 3.f);
	mq = T(1) - q;
	w = sqr(mq) * mq;
	w *= w;
	w *= fmaf(fmaf(T(35. / 3.), q, T(6)), q, T(1));
	res = c0 * w;
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
	T res, w, mq;
	const T c0 = T(495.f / 32.f / M_PI);
	mq = q - T(1);
	w = fmaf(T(5), q, T(1));
	w *= q;
	w *= sqr(sqr(mq)) * mq;
	w *= T(56.0 / 3.0);
	res = c0 * w;
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T res, sw1, sw2, w;
	const auto q0 = q;
	const T c0 = T(495.f / 8.f);
	w = T(35.0f / 33.f);
	w = fmaf(q, w, T(-32.f / 5.f));
	w = fmaf(q, w, T(140.f / 9.f));
	w = fmaf(q, w, T(-56.f / 3.f));
	w = fmaf(q, w, T(10));
	w *= q;
	w = fmaf(q, w, T(-28.f / 15.f));
	w *= q;
	w = fmaf(q, w, T(1.0f / 3.0f));
	res = c0 * w;
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (sqr(q0) * q0 + T(1e-30))) * (q0 > T(0));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T res, sw1, sw2, w;
	const auto q0 = q;
	q *= (q < T(1));
	const T c0 = T(495.f / 8.f);
	w = T(-7.0f / 66.f);
	w = fmaf(q, w, T(32.f / 45.f));
	w = fmaf(q, w, T(-35.f / 18.f));
	w = fmaf(q, w, T(8.f / 3.f));
	w = fmaf(q, w, T(-5.f / 3.f));
	w *= q;
	w = fmaf(q, w, T(7.f / 15.f));
	w *= q;
	w = fmaf(q, w, T(-1.0f / 6.0f));
	w *= q;
	w = fmaf(q, w, T(1.0f / 18.0f));
	res = c0 * w;
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (q0 + T(1e-30f))) * (q > T(0));
	return res;
}
