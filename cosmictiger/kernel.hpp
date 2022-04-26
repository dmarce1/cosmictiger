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

static constexpr int NPIECE = 128;
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
void kernel_set_type(double type);
void kernel_output();
void kernel_adjust_options(options& opts);

#define Power pow
#define Pi T(M_PI)

#ifdef __KERNEL_CU__
__managed__ pair<float>* WLUT;
__managed__ float kernel_index;
__managed__ float kernel_norm;
#else
extern __managed__ pair<float>* WLUT;
extern __managed__ float kernel_index;
extern __managed__ float kernel_norm;
#endif

CUDA_EXPORT
inline float kernelW(float q) {
	float tmp1 = fmaxf(1.f - q, 0.f);
	float tmp2 = fmaxf((2.f / 3.f) - q, 0.f);
	float tmp3 = fmaxf((1.f / 3.f) - q, 0.f);
	tmp1 *= sqr(sqr(tmp1));
	tmp2 *= -6.f * sqr(sqr(tmp2));
	tmp3 *= 15.f * sqr(sqr(tmp3));
	return 17.403593027f * (tmp1 + tmp2 + tmp3);
}

CUDA_EXPORT
inline float dkernelW_dq(float q, float* w = nullptr, int* flops = nullptr) {
	float tmp1 = fmaxf(1.f - q, 0.f);
	float tmp2 = fmaxf((2.f / 3.f) - q, 0.f);
	float tmp3 = fmaxf((1.f / 3.f) - q, 0.f);
	tmp1 *= -5.f * tmp1 * (sqr(tmp1));
	tmp2 *= 30.f * tmp2 * (sqr(tmp2));
	tmp3 *= -75.f * tmp3 * (sqr(tmp3));
	return 17.403593027f * (tmp1 + tmp2 + tmp3);
}

template<class T>
CUDA_EXPORT
inline T kernelG(T q) {
	T tmp1 = fmaxf(T(1.f) - q, T(0));
	T tmp2 = fmaxf(T(0.5f) - q, T(0));
	tmp1 *= sqr(tmp1);
	tmp2 *= -4.f * sqr(tmp2);
	return 5.092958179f * (tmp1 + tmp2);
}

template<class T>
CUDA_EXPORT
inline T dkernelG_dq(T q) {
	T tmp1 = fmaxf(T(1.f) - q, T(0));
	T tmp2 = fmaxf(T(0.5f) - q, T(0));
	tmp1 *= -3.f * tmp1;
	tmp2 *= 12.f * tmp2;
	return 5.092958179f * (tmp1 + tmp2);
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T w1, w2, sw, res, q3inv, sw1, sw2;
	const auto q0 = q;
	q *= (q < T(1));
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
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (sqr(q0) * q0 + T(1e-30))) * (q0 > T(0));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T w1, w2, sw, res, q1inv, sw1, sw2;
	const auto q0 = q;
	q *= (q < T(1));
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
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (q0 + T(1e-30f)));
	return res;
}
