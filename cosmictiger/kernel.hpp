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
	T w1, w2, w3, sw, res, sw1, sw2, sw3, w, mq;
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
		break;
	}
	};
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T dkernelW_dq(T q) {
	T w1, w2, w3, sw, res, sw1, sw2, sw3, w, mq;
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
	};
	res *= (q < T(1.f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T res, sw1, sw2, w1, w2, q3inv;
	const auto q0 = q;
	q *= (q < T(1));
	const auto q2 = sqr(q);
	w1 = T(105.0/16.0);
	w1 = fmaf(w1, q2, T(-189.0/16.0));
	w1 = fmaf(w1, q2, T(135.0/16.0));
	w1 = fmaf(w1, q2, T(-35.0/16.0));
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
	T res, sw1, sw2, w1, w2, q1inv;
	const auto q0 = q;
	q *= (q < T(1));
	const auto q2 = sqr(q);
	w1 = T(315.0/128.0);
	w1 = fmaf(w1, q2, T(-105.0/32.0));
	w1 = fmaf(w1, q2, T(189.0/64.0));
	w1 = fmaf(w1, q2, T(-45.0/32.0));
	w1 = fmaf(w1, q2, T(35.0/128.0));
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
