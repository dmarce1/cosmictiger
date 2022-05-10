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

//#define KERNEL_CUBIC_SPLINE
//#define KERNEL_QUARTIC_SPLINE
#define KERNEL_QUINTIC_SPLINE
//#define KERNEL_GAUSSIAN

void kernel_set_type(int type);
void kernel_output();
void kernel_adjust_options(options& opts);

#define Power pow
#define Pi T(M_PI)

template<class T>
CUDA_EXPORT
inline T kernelW(T q) {
#ifdef KERNEL_CUBIC_SPLINE
	T w1, w2, res;
	T sw, mq;
	const T c0 = T(8.0 / M_PI);
	w1 = T(6);
	w1 = fmaf(q, w1, -T(6));
	w1 *= q;
	w1 = fmaf(q, w1, T(1));
	mq = T(1) - q;
	w2 = T(2) * sqr(mq) * mq;
	sw = q < T(0.5);
	res = c0 * (sw * w1 + (T(1) - sw) * w2);
#endif
#ifdef KERNEL_QUARTIC_SPLINE
	T w1, w2, res;
	T w3, sw3, sw1, sw2, mq;
	const T c0 = T(15625.0f / M_PI / 512.0f);
	w1 = T(6);
	w1 *= q;
	w1 = fmaf(q, w1, -T(12.0f / 5.0f));
	w1 *= q;
	w1 = fmaf(q, w1, T(46.0f / 125.0f));
	w2 = -T(4);
	w2 = fmaf(q, w2, T(8));
	w2 = fmaf(q, w2, -T(24.0f / 5.f));
	w2 = fmaf(q, w2, T(8. / 25.));
	w2 = fmaf(q, w2, T(44. / 125.));
	mq = T(1) - q;
	w3 = sqr(sqr(mq));
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);
#endif
#ifdef KERNEL_QUINTIC_SPLINE
	T w1, w2, res;
	T w3, sw3, sw1, sw2;
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
#endif
#ifdef KERNEL_GAUSSIAN
	T res;
	const T q2 = sqr(q);
	res = expf(-16.f * q2);
#endif
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
#ifdef KERNEL_CUBIC_SPLINE
	T w1, w2, res;
	T sw;
	const T c0 = T(8.0 / M_PI);
	w1 = T(18);
	w1 = fmaf(q, w1, -T(12));
	w1 *= q;
	w2 = T(-6);
	w2 *= sqr(T(1) - q);
	sw = q < T(0.5);
	res = c0 * (sw * w1 + (T(1) - sw) * w2);

#endif
#ifdef KERNEL_QUARTIC_SPLINE
	T w1, w2, res;
	T w3, sw3, mq, sw1, sw2;
	const T c0 = T(15625.0f / M_PI / 512.0f);
	w1 = T(24);
	w1 *= q;
	w1 = fmaf(q, w1, -T(24.0f / 5.0f));
	w1 *= q;
	w2 = -T(16);
	w2 = fmaf(q, w2, T(24));
	w2 = fmaf(q, w2, -T(48.0f / 5.f));
	w2 = fmaf(q, w2, T(8. / 25.));
	mq = T(1.) - q;
	w3 = -T(4) * sqr(mq) * mq;
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);

#endif
#ifdef KERNEL_QUINTIC_SPLINE
	T w1, w2, res;
	T w3, sw3, sw2, sw1;
	const T c0 = T(2187.0 / (40.0 * M_PI));
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

#endif
#ifdef KERNEL_GAUSSIAN
	T res;
	const T q2 = sqr(q);
	res = -32.f * q * expf(-16.f * q2);
#endif
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelG(T q) {
	T res;
	const T c0 = T(105.0 / 32.0 / M_PI);
	const T c1 = T(-105.0 / 16.0 / M_PI);
	const T& c2 = c0;
	const T q2 = sqr(q);
	res = c2;
	res = fmaf(res, q2, c1);
	res = fmaf(res, q2, c0);
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelG_dq(T q) {
	T res;
	const T c0 = T(-105.0 / 8.0 / M_PI);
	const T c1 = -c0;
	const T q2 = sqr(q);
	res = c1;
	res = fmaf(res, q2, c0);
	res *= q;
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T sw1, sw2, res;
	const T c0 = T(35.0/8.0);
	const T c1 = T(-21.0/4.0);
	const T c2 = T(15.0/8.0);
	const T q2 = sqr(q);
	res = c2;
	res = fmaf(res, q2, c1);
	res = fmaf(res, q2, c0);
	sw1 = q < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (sqr(q) * q + T(1e-35))) * (q > T(0));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T sw1, sw2, res;
	const T c0 = T(35.0/16.0);
	const T c1 = T(-35.0/16.0);
	const T c2 = T(21.0/16.0);
	const T c3 = T(-5.0/16.0);
	const T q2 = sqr(q);
	res = c3;
	res = fmaf(res, q2, c2);
	res = fmaf(res, q2, c1);
	res = fmaf(res, q2, c0);
	sw1 = q < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (q + T(1e-35f)));
	sw1 = q > T(0);
	res *= sw1;
	return res;
}

CUDA_EXPORT
inline void dsmoothX_dh(float h, float hmin, float hmax, float& x, float& dxdh) {

	if (h < 2.0f * hmin) {
		const float y = (h / hmin - 1.0f);
		x = y;
		dxdh = 1.0f / hmin;
	} else/* if (h < hmax) */{
		x = 1.0f;
		dxdh = 0.0f;
	} /*else {
	 x = (h / hmax);
	 dxdh = 1.f / (hmax);
	 }*/

	return;
}

CUDA_EXPORT inline float dlogsmoothX_dlogh(float h, float hmin, float hmax) {
	float f, dfdh;
	dsmoothX_dh(h, hmin, hmax, f, dfdh);
	return dfdh * h / f;
}

CUDA_EXPORT
inline float smoothX(float h, float hmin, float hmax) {
	float f, dfdh;
	dsmoothX_dh(h, hmin, hmax, f, dfdh);
	return f;
}

