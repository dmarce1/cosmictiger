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
inline
float W0(float q) {
	const float W0 = kernel_norm;
	const float n = kernel_index;
	q = fminf(q, 1.f);
	float x = float(M_PI) * q;
	x = fminf(x, M_PI * 0.9999999f);
	float w = W0 * powf(sinc(x), n);
	ALWAYS_ASSERT(isfinite(w));
	return w;
}

CUDA_EXPORT
inline float dWdq0(float q) {
//	return (W(q + DX) - W(q)) / DX;
	const float W0 = kernel_norm;
	const float n = kernel_index;
	float x = float(M_PI) * q;
	q = fminf(q, 1.f);
	x = fminf(x, M_PI * 0.9999999f);
	if (q == 0.0f) {
		return 0.f;
	} else {
		float tmp, s, c;
		s = sinf(x);
		if (x < float(0.25f)) {
			tmp = (float(-1. / 3.) + float(1. / 30.) * sqr(x)) * sqr(x) * x;
		} else {
			tmp = (x * sqrtf(1.f - sqr(s)) - s);
		}
		float xinv = 1.f / x;
		float w = W0 * n * tmp * sqr(xinv) * float(M_PI) * powf(s * xinv, n - 1.0f);
		return w;
	}
}

CUDA_EXPORT
inline float kernelW(float q) {
	const float W0 = kernel_norm;
	const float n = kernel_index;
	q = fminf(q, 1.f);
	float x = float(M_PI) * q;
	x = fminf(x, M_PI * 0.9999999f);
	float w = W0 * powf(sinc(x), n);
	return w;
}

CUDA_EXPORT
inline float dkernelW_dq(float q, float* w = nullptr, int* flops = nullptr) {
	const float W0 = kernel_norm;
	const float n = kernel_index;
	float x = float(M_PI) * q;
	q = fminf(q, 1.f);											// 1
	x = fminf(x, M_PI * 0.9999999f);							// 1
	if (q == 0.0f) {												// 1
		if (w) {
			*w = W0;
		}
		if(flops) {
			*flops += 3;
		}
		return 0.f;
	} else {
		float tmp, s, c;
		s = sinf(x);																			// 8
		if (x < float(0.25f)) {																// 1
			const float x2 = sqr(x);														// 2
			tmp = (float(-1. / 3.) + float(1. / 30.) * x2) * x2 * x;				// 4
			if(flops) {
				*flops += 6;
			}
		} else {
			float c = cosf(x);																// 8
			float sgn = -2.0f * (x > float(M_PI / 2.0)) + 1.0;						// 4
			tmp = (sgn * x * sqrtf(1 - sqr(s)) - s);									// 9
			if(flops) {
				*flops += 21;
			}
		}
		float xinv = 1.f / x;																// 4
		float w1 = W0 * powf(s * xinv, n);												// 6
		if (w) {
			*w = w1;
		}
		float dw = n * tmp * xinv * float(M_PI) * w1 / s;							// 8
		if( flops) {
			*flops += 30;
		}
		return dw;
	}
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T res, sw1, sw2, w1;
	const auto q0 = q;
	q *= (q < T(1));
	w1 = T(5.0);
	w1 = fmaf(w1, q, T(-9.0));
	w1 *= q;
	w1 = fmaf(w1, q, T(5.0));
	res = w1;
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
	w1 = T(-1.0);
	w1 = fmaf(w1, q, T(9.0 / 4.0));
	w1 = fmaf(w1, q, T(-2.5));
	w1 *= q;
	w1 = fmaf(w1, q, T(9.0 / 4.0));
	res = w1;

	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (q0 + T(1e-30f)));
	return res;
}

/*

 template<class T>
 CUDA_EXPORT
 inline T kernelFqinv(T q) {
 T res, sw1, sw2, w1;
 q *= (q < T(1));
 w1 = min(q - sinf(T(2 * M_PI) * q) / T(2. * M_PI), T(1)) / (sqr(q) * q + T(1e-30f));
 res = w1;

 return res;
 }

 template<class T>
 CUDA_EXPORT
 inline T kernelPot(T q) {
 T res, sw1, sw2, w1;
 const auto q0 = q;
 q *= (q < T(1));

 const T c0 = -2.437653393057226;
 const T c1 = 3.2898681336964524;
 const T c2 = -3.2469697011334144;
 const T c3 = 2.0346861239688976;
 const T c4 = -0.8367311301649535;
 const T c5 = 0.2402386980306763;
 const T c6 = -0.05066369468793888;
 const T c7 = 0.008163765290898435;
 const auto q2 = sqr(q);
 w1 = c7;
 w1 = fmaf(q2, w1, c6);
 w1 = fmaf(q2, w1, c5);
 w1 = fmaf(q2, w1, c4);
 w1 = fmaf(q2, w1, c3);
 w1 = fmaf(q2, w1, c2);
 w1 = fmaf(q2, w1, c1);
 w1 = fmaf(q2, w1, c0);
 res = w1;
 sw1 = q0 < T(1);
 sw2 = T(1) - sw1;
 res = (sw1 * res + sw2 / (q0 + T(1e-30f))) * (q0 > T(0));
 return res;
 }*/
