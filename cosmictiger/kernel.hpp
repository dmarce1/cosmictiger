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

#define KERNEL_CUBIC_SPLINE
//#define KERNEL_QUARTIC_SPLINE
//#define KERNEL_QUINTIC_SPLINE

void kernel_set_type(int type);
void kernel_output();
void kernel_adjust_options(options& opts);

#define Power pow
#define Pi T(M_PI)

template<class T>
CUDA_EXPORT
inline T kernelW(T q) {
	T w1, w2, res;
#ifdef KERNEL_CUBIC_SPLINE
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
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
	T w1, w2, res;
#ifdef KERNEL_CUBIC_SPLINE
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
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T w1, w2, res, q3inv, sw1, sw2;
	const auto q0 = q;
#ifdef KERNEL_CUBIC_SPLINE
	T sw;
	w1 = T(32);
	w1 = fmaf(q, w1, -T(192.0 / 5.0));
	w1 *= q;
	w1 = fmaf(q, w1, T(32.0 / 3.0));
	w2 = -T(32.0 / 3.0);
	w2 = fmaf(q, w2, T(192 / 5.0));
	w2 = fmaf(q, w2, -T(48.0));
	w2 = fmaf(q, w2, T(64.0 / 3.0));
	q3inv = T(1) / (q + T(1.0e-10f));
	q3inv = sqr(q3inv) * q3inv;
	w2 -= T(1.0 / 15.0) * q3inv;
	sw = q < T(0.5);
	res = (sw * w1 + (T(1) - sw) * w2);
#endif
#ifdef KERNEL_QUARTIC_SPLINE
	T w3, sw3;
	w1 = T(46875.0f / 448.0f);
	w1 *= q;
	w1 = fmaf(q, w1, -T(1875.f / 32.f));
	w1 *= q;
	w1 = fmaf(q, w1, T(2875.f / 192.f));
	w2 = -T(15625.0f / 224.0f);
	w2 = fmaf(q, w2, T(15625.f / 96.f));
	w2 = fmaf(q, w2, -T(1875.0f / 16.f));
	w2 = fmaf(q, w2, T(625.f / 64.f));
	w2 = fmaf(q, w2, T(1375.f / 96.f));
	q3inv = T(1) / (q + T(1e-10f));
	q3inv = sqr(q3inv) * q3inv;
	w2 += T(1.0f / 6720.0f) * q3inv;
	w3 = T(15625.0f / 896.0f);
	w3 = fmaf(q, w3, -T(15625.0f / 192.0f));
	w3 = fmaf(q, w3, T(9375.0f / 64.0f));
	w3 = fmaf(q, w3, -T(15625.0f / 128.0f));
	w3 = fmaf(q, w3, T(15625.0f / 384.0f));
	w3 -= T(437.0f / 2688.0f) * q3inv;
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = (sw1 * w1 + sw2 * w2 + sw3 * w3);
#endif
#ifdef KERNEL_QUINTIC_SPLINE
	T w3, sw3;
	const T c0 = T(218.7);
	w1 = T(-5. / 4.);
	w1 = fmaf(q, w1, T(10. / 7.));
	w1 *= q;
	w1 = fmaf(q, w1, T(-4.0f / 9.f));
	w1 *= q;
	w1 = fmaf(q, w1, T(22.0f / 243.0f));
	w2 = T(5. / 8.);
	w2 = fmaf(q, w2, -T(15. / 7.));
	w2 = fmaf(q, w2, T(25. / 9.));
	w2 = fmaf(q, w2, T(-14. / 9.));
	w2 = fmaf(q, w2, T(25. / 108.));
	w2 = fmaf(q, w2, T(17. / 243.));
	q3inv = T(1) / (q + T(1e-10));
	q3inv = sqr(q3inv) * q3inv;
	w2 += T(5.0 / 367416.0) * q3inv;
	w3 = T(-1. / 8.);
	w3 = fmaf(q, w3, T(5. / 7.));
	w3 = fmaf(q, w3, T(-5. / 3.));
	w3 = fmaf(q, w3, T(2));
	w3 = fmaf(q, w3, T(-5. / 4.));
	w3 = fmaf(q, w3, T(1. / 3.));
	w3 -= T(169.0 / 122472.0) * q3inv;
	sw1 = q < T(1. / 3.);
	sw3 = q > T(2. / 3.);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);

#endif
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (sqr(q0) * q0 + T(1e-30))) * (q0 > T(0));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T w1, w2, res, q1inv, sw1, sw2;
	const auto q0 = q;
#ifdef KERNEL_CUBIC_SPLINE
	T sw;
	w1 = -T(32.0 / 5.0);
	w1 = fmaf(q, w1, T(48.0 / 5.0));
	w1 *= q;
	w1 = fmaf(q, w1, T(-16.0 / 3.0));
	w1 *= q;
	w1 = fmaf(q, w1, T(14.0 / 5.0));
	w2 = T(32.0 / 15.0);
	w2 = fmaf(q, w2, T(-48.0 / 5.0));
	w2 = fmaf(q, w2, T(16.0));
	w2 = fmaf(q, w2, T(-32.0 / 3.0));
	w2 *= q;
	w2 = fmaf(q, w2, T(16.0 / 5.0));
	q1inv = T(1) / (q + T(1.0e-15));
	w2 -= T(1.0 / 15.0) * q1inv;
	sw = q < T(0.5);
	res = (sw * w1 + (T(1) - sw) * w2);
#endif
#ifdef KERNEL_QUARTIC_SPLINE
	T w3, sw3;
	w1 = T(-15625.0f / 896.0f);
	w1 *= q;
	w1 = fmaf(q, w1, T(1875.f / 128.f));
	w1 *= q;
	w1 = fmaf(q, w1, -T(2875.f / 384.f));
	w1 *= q;
	w1 = fmaf(q, w1, T(1199.f / 384.f));
	w2 = T(15625.0f / 1344.0f);
	w2 = fmaf(q, w2, T(-3125.f / 96.f));
	w2 = fmaf(q, w2, T(1875.0f / 64.f));
	w2 = fmaf(q, w2, -T(625.f / 192.f));
	w2 = fmaf(q, w2, -T(1375.f / 192.f));
	w2 *= q;
	w2 = fmaf(q, w2, T(599.f / 192.f));
	q1inv = T(1) / (q + T(1e-15f));
	w2 += T(1.0f / 6720.0f) * q1inv;
	w3 = -T(15625.0f / 5376.0f);
	w3 = fmaf(q, w3, T(3125.0f / 192.0f));
	w3 = fmaf(q, w3, -T(9375.0f / 256.0f));
	w3 = fmaf(q, w3, T(15625.0f / 384.0f));
	w3 = fmaf(q, w3, -T(15625.0f / 768.0f));
	w3 *= q;
	w3 = fmaf(q, w3, T(3125.0f / 768.0f));
	w3 -= T(437.0f / 2688.0f) * q1inv;
	sw1 = q < T(0.2f);
	sw3 = q > T(0.6f);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = (sw1 * w1 + sw2 * w2 + sw3 * w3);
#endif
#ifdef KERNEL_QUINTIC_SPLINE
	T w3, sw3;
	const T c0 = T(218.7);
	w1 = T(5. / 28.);
	w1 = fmaf(q, w1, T(-5. / 21.));
	w1 *= q;
	w1 = fmaf(q, w1, T(1.0f / 9.f));
	w1 *= q;
	w1 = fmaf(q, w1, T(-11.0f / 243.0f));
	w1 *= q;
	w1 = fmaf(q, w1, T(239. / 15309.));
	w2 = T(-5. / 56.);
	w2 = fmaf(q, w2, T(5. / 14.));
	w2 = fmaf(q, w2, T(-5. / 9.));
	w2 = fmaf(q, w2, T(7. / 18.));
	w2 = fmaf(q, w2, T(-25. / 324.));
	w2 = fmaf(q, w2, T(-17. / 486.));
	w2 *= q;
	w2 = fmaf(q, w2, T(473. / 30618.));
	q1inv = T(1) / (q + T(1e-15));
	w2 += T(5. / 367416.) * q1inv;
	w3 = T(1. / 56.);
	w3 = fmaf(q, w3, T(-5. / 42.));
	w3 = fmaf(q, w3, T(1. / 3.));
	w3 = fmaf(q, w3, T(-.5));
	w3 = fmaf(q, w3, T(5. / 12.));
	w3 = fmaf(q, w3, T(-1. / 6.));
	w3 *= q;
	w3 = fmaf(q, w3, T(1. / 42.));
	w3 -= T(169. / (122472.)) * q1inv;
	sw1 = q < T(1. / 3.);
	sw3 = q > T(2. / 3.);
	sw2 = (T(1) - sw1) * (T(1) - sw3);
	res = c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);
#endif
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (q0 + T(1e-30f)));
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

