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
#define KERNEL_SUPER_GAUSSIAN 6
#define KERNEL_GAUSSIAN 7
#define KERNEL_SILLY 8

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
	T w1, w2, w3, sw, res, sw1, sw2, sw3, w, mq, q2;
#ifdef __CUDA_ARCH__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
		const T c0 = T(8.0 / M_PI);
		w1 = T(6);
		w1 = fmaf(q, w1, -T(6));
		w1 *= q;
		w1 = fmaf(q, w1, T(1));
		mq = T(1) - q;
		w2 = T(2) * sqr(mq) * mq;
		sw = q < T(0.5);
		res = c0 * (sw * w1 + (T(1) - sw) * w2);
	}
		break;
	case KERNEL_QUARTIC_SPLINE: {
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
	}
		break;
	case KERNEL_QUINTIC_SPLINE: {
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
	}
		break;
	case KERNEL_WENDLAND_C2: {
		const T c0 = T(21.f / 2.f / M_PI);
		mq = T(1) - q;
		w = sqr(sqr(mq));
		w *= fmaf(T(4), q, T(1));
		res = c0 * w;
	}
		break;
	case KERNEL_WENDLAND_C4: {
		const T c0 = T(495.f / 32.f / M_PI);
		w = T(35.0f / 3.f);
		mq = T(1) - q;
		w = sqr(mq) * mq;
		w *= w;
		w *= fmaf(fmaf(T(35. / 3.), q, T(6)), q, T(1));
		res = c0 * w;
	}
		break;
	case KERNEL_WENDLAND_C6: {
		const T c0 = T(1365.f / 64.f / M_PI);
		w = T(32);
		w = fmaf(q, w, T(25));
		w = fmaf(q, w, T(8));
		w = fmaf(q, w, T(1));
		const T mq8 = sqr(sqr(sqr(T(1) - q)));
		w *= mq8;
		res = c0 * w;
	}
		break;
	case KERNEL_SUPER_GAUSSIAN: {
		q2 = T(9.f) * sqr(q);
		res = T(27.f / 5.568327997f) * expf(-q2) * (T(2.5f) - q2);
	}
		break;
	case KERNEL_GAUSSIAN: {
		q2 = sqr(q * T(3.54489));
		res = expf(-q2) * 8.f;
	}
		break;
	case KERNEL_SILLY: {
		res = (T(3465) * Power(T(-1) + Power(q, T(2)), T(3)) * (-T(5) + T(13) * Power(q, T(2)))) / (T(1024.) * Pi);
	}
		break;
	};
	res *= q < T(1.f);
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
	T w1, w2, w3, sw, res, sw1, sw2, sw3, w, mq, q2;
#ifdef __CUDA_ARCH__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
		const T c0 = T(8.0 / M_PI);
		w1 = T(18);
		w1 = fmaf(q, w1, -T(12));
		w1 *= q;
		w2 = T(-6);
		w2 *= sqr(T(1) - q);
		sw = q < T(0.5);
		res = c0 * (sw * w1 + (T(1) - sw) * w2);
	}
		break;
	case KERNEL_QUARTIC_SPLINE: {
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
	}
		break;
	case KERNEL_QUINTIC_SPLINE: {
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
	}
		break;

	case KERNEL_WENDLAND_C2: {
		const T c0 = T(21.f / 2.f / M_PI);
		mq = q - T(1);
		w = T(20);
		w *= q * sqr(mq) * mq;
		res = c0 * w;
	}
		break;
	case KERNEL_WENDLAND_C4: {
		const T c0 = T(495.f / 32.f / M_PI);
		mq = q - T(1);
		w = fmaf(T(5), q, T(1));
		w *= q;
		w *= sqr(sqr(mq)) * mq;
		w *= T(56.0 / 3.0);
		res = c0 * w;
	}
		break;
	case KERNEL_WENDLAND_C6: {
		const T c0 = T(-22.0 * 1365.f / 64.f / M_PI);
		w = fmaf(T(16), q, T(7));
		w = fmaf(w, q, T(1));
		w *= q;
		const T mq = T(1) - q;
		const T mq2 = sqr(mq);
		const T mq3 = mq2 * mq;
		const T mq4 = mq2 * mq2;
		const T mq7 = mq3 * mq4;
		w *= mq7;
		res = c0 * w;
	}
		break;
	case KERNEL_SUPER_GAUSSIAN: {
		q2 = T(9.f) * sqr(q);
		res = T(243.f / 5.568327997f) * expf(-q2) * q * (-T(7.f) + T(2.f) * q2);

	}
		break;
	case KERNEL_GAUSSIAN: {
		q2 = sqr(q * T(3.54489));
		res = -T(201.06) * expf(-q2) * q;
	}
		break;
	case KERNEL_SILLY: {
		res = (T(3465) * q * Power(T(-1) + Power(q, T(2)), T(2)) * (T(-7) + T(13) * Power(q, T(2)))) / (T(128.)*Pi);
	}
		break;

	};
	res *= q < T(1.f);
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T w1, w2, sw, res, q3inv, w3, sw1, sw2, sw3, w, q2;
#ifdef __CUDA_ARCH__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
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
	}
		break;
	case KERNEL_QUARTIC_SPLINE:
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
		break;
	case KERNEL_QUINTIC_SPLINE: {
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
	}
		break;

	case KERNEL_WENDLAND_C2: {
		w = T(21);
		w = fmaf(q, w, T(-90));
		w = fmaf(q, w, T(140));
		w = fmaf(q, w, T(-84));
		w *= q;
		w = fmaf(q, w, T(14));
		res = w;
	}
		break;
	case KERNEL_WENDLAND_C4: {
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
	}
		break;
	case KERNEL_WENDLAND_C6: {
		q = min(q, T(1.1));
		const T c0 = T(1365.f / 16.f);
		w = T(16. / 7.);
		w = fmaf(q, w, T(-231. / 13.));
		w = fmaf(q, w, T(176. / 3.));
		w = fmaf(q, w, T(-105));
		w = fmaf(q, w, T(528. / 5.));
		w = fmaf(q, w, T(-154. / 3.));
		w *= q;
		w = fmaf(q, w, T(66. / 7.));
		w *= q;
		w = fmaf(q, w, T(-11. / 5.));
		w *= q;
		w = fmaf(q, w, T(1. / 3.));
		res = c0 * w;
	}
		break;

	case KERNEL_SUPER_GAUSSIAN: {
		q2 = T(9.f) * sqr(q);
		w = T(6.f) * expf(-q2) * q * (T(-1.f) + q2) * T(0.564189584f) + erff(T(3.f) * q);
		res = w / (sqr(q) * q);
	}
		break;
	case KERNEL_GAUSSIAN: {
		q2 = sqr(q * T(3.54489));
		res = (-T(4.00004) * expf(-q2) * q + T(1.00001) * erff(T(3.54489) * q));
		res /= sqr(q) * q + T(1e-30);
	}
		break;
	case KERNEL_SILLY: {
		res = (T(5775) - T(19404) * Power(q, T(2)) + T(26730) * Power(q, T(4)) - 16940 * Power(T(q), T(6)) + T(4095) * Power(T(q), T(8))) / T(256.);
	}
		break;
	};
	sw1 = q < T(1);
	sw2 = T(1) - sw1;
	res = sw1 * res + sw2 / (sqr(q) * q + T(1e-30));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T w1, w2, sw, res, q1inv, w3, sw1, sw2, sw3, w, q2;
#ifdef __CUDA_ARCH__
	switch(kernel_type) {
#else
	static const int kernel_type = get_options().kernel;
	switch (kernel_type) {
#endif
	case KERNEL_CUBIC_SPLINE: {
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
		res = (q > T(0.f)) * (sw * w1 + (T(1) - sw) * w2);
	}
		break;
	case KERNEL_QUARTIC_SPLINE:
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
		res = (q > T(0)) * (sw1 * w1 + sw2 * w2 + sw3 * w3);
		break;

	case KERNEL_QUINTIC_SPLINE: {
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
		res = (q > T(0.f)) * c0 * (sw1 * w1 + sw2 * w2 + sw3 * w3);
	}
		break;

	case KERNEL_WENDLAND_C2: {
		w = T(-3);
		w = fmaf(q, w, T(15));
		w = fmaf(q, w, T(-28));
		w = fmaf(q, w, T(21));
		w *= q;
		w = fmaf(q, w, T(-7));
		w *= q;
		w = fmaf(q, w, T(3));
		res = (q > T(0.f)) * w;
	}
		break;
	case KERNEL_WENDLAND_C4: {
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
		res = (q > T(0)) * c0 * w;
	}
		break;
	case KERNEL_WENDLAND_C6: {
		q = min(q, T(1.1));
		const T c0 = T(1365.f / 16.f);
		w = T(-16. / 91.);
		w = fmaf(q, w, T(77. / 52.));
		w = fmaf(q, w, T(-16. / 3.));
		w = fmaf(q, w, T(21. / 2.));
		w = fmaf(q, w, T(-176. / 15.));
		w = fmaf(q, w, T(77. / 12.));
		w *= q;
		w = fmaf(q, w, T(-11. / 7.));
		w *= q;
		w = fmaf(q, w, T(11. / 20.));
		w *= q;
		w = fmaf(q, w, T(-1. / 6.));
		w *= q;
		w = fmaf(q, w, T(7. / 156.));
		res = (q > T(0)) * c0 * w;
	}
		break;

	case KERNEL_SUPER_GAUSSIAN: {
		q2 = T(9.f) * sqr(q);
		w = -(q - T(1.f)) * erff(T(3.f) * q) / sqr(q);
		w += T(6.f) * expf(-q2) * (-T(1) + q + q2 - q * q2) * T(0.564189584f) / q;
		w += 1.f;
		res = w;
	}
		break;
	case KERNEL_GAUSSIAN: {
		res = T(1.00001) * erff(T(3.54489) * q);
		res /= q + T(1e-30);
		res -= 0.0000144457;
	}
		break;
	case KERNEL_SILLY: {
		res = (T(2079) - T(5775) * Power(q, T(2)) + T(9702) * Power(q, T(4)) - T(8910) * Power(q, T(6)) + T(4235) * Power(q, T(8)) - T(819) * Power(q, T(10)))
				/ T(512.);
	}

		break;
	};
	sw1 = q < T(1);
	sw2 = T(1) - sw1;
	res = sw1 * res + sw2 / (q + T(1e-30f));
	return res;
}
