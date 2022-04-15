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

static constexpr int NTAYLOR = 16;

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
void kernel_set_type(double type);
void kernel_output();
void kernel_adjust_options(options& opts);

#ifdef __KERNEL_CU__
__managed__ double* pot_series;
__managed__ double* f_series;
__managed__ double kernel_index;
__managed__ double kernel_norm;
#else
extern __managed__ double* pot_series;
extern __managed__ double* f_series;
extern __managed__ double kernel_index;
extern __managed__ double kernel_norm;
#endif

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
inline float dkernelW_dq(float q, float* w = nullptr) {
	const float W0 = kernel_norm;
	const float n = kernel_index;
	float x = float(M_PI) * q;
	q = fminf(q, 1.f);
	x = fminf(x, M_PI * 0.9999999f);
	if (q == 0.0f) {
		if (w) {
			*w = W0;
		}
		return 0.f;
	} else {
		float tmp, s;
		s = sinf(x);
		if (x < float(0.25f)) {
			tmp = (float(-1. / 3.) + float(1. / 30.) * sqr(x)) * sqr(x) * x;
		} else {
			float c = cosf(x);
			tmp = (x * c - s);
		}
		float xinv = 1.f / x;
		float w1 = W0 * powf(s * xinv, n);
		if (w) {
			*w = w1;
		}
		float dw = n * tmp * xinv * float(M_PI) * w1 / s;
		return dw;
	}
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T res, sw1, sw2, w;
	const auto q0 = q;
	q *= (q < T(1));
	const T q2 = sqr(q);
	w = f_series[NTAYLOR - 1];
	for (int n = NTAYLOR - 2; n >= 0; n--) {
		w = fmaf(w, q2, f_series[n]);
	}
	res = w;
	sw1 = q0 < T(1);
	sw2 = T(1) - sw1;
	res = (sw1 * res + sw2 / (sqr(q0) * q0 + T(1e-30)));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T res, sw1, sw2, w;
	const auto q0 = q;
	q *= (q < T(1));
	const T q2 = sqr(q);
	w = pot_series[NTAYLOR - 1];
	for (int n = NTAYLOR - 2; n >= 0; n--) {
		w = fmaf(w, q2, pot_series[n]);
	}
	res = w;
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
