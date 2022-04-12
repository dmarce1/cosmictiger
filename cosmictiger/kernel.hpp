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

static constexpr int NPIECE = 32;
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
void kernel_set_type(double type);
void kernel_output();
void kernel_adjust_options(options& opts);

#define Power pow
#define Cos cos
#define Sin sin
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
		c = cosf(x);
		if (x < float(0.01f)) {
			tmp = float(-1. / 3.) * sqr(x) * x;
		} else {
			tmp = (x * c - s);
		}
		float w = W0 * n * tmp / (x * q) * powf(s / x, n - 1.0);
		return w;
	}
}

CUDA_EXPORT
inline float kernelW(float q) {
//	return W0(q);
	q = fminf(q * NPIECE, NPIECE * 0.9999999f);
	const int q0 = q;
	const int q1 = q0 + 1;
	const float x = q - q0;
	const float y1 = WLUT[q0].first;
	const float k1 = WLUT[q0].second;
	const float y2 = WLUT[q1].first;
	const float k2 = WLUT[q1].second;
	const float dy = y2 - y1;
	const float a = k1 / NPIECE - dy;
	const float b = -k2 / NPIECE + dy;
	const float omx = 1.f - x;
	const float w = omx * y1 + x * y2 + x * omx * (omx * a + x * b);
	return w;
}

CUDA_EXPORT
inline float dkernelW_dq(float q) {
//	return dWdq0(q);
	q = fminf(q * NPIECE, NPIECE * 0.9999999f);
	const int q0 = q;
	const int q1 = q0 + 1;
	const float x = q - q0;
	const float y1 = WLUT[q0].first;
	const float k1 = WLUT[q0].second;
	const float y2 = WLUT[q1].first;
	const float k2 = WLUT[q1].second;
	const float dy = y2 - y1;
	const float a = k1 / NPIECE - dy;
	const float b = -k2 / NPIECE + dy;
	return NPIECE * (b * (2 - 3 * x) * x + a * (-1 + x) * (-1 + 3 * x) - y1 + y2);
}

inline double kernelFourier(double k) {

	constexpr float C[9] = { 1.0, -0.008888390675833106, 0.0001425112054475251, -1.85843699392027e-6, 1.6091255207148952e-8, -9.657273059864925e-11,
			4.2575707364037334e-13, -1.4424066807367349e-15, 3.883913221966831e-18 };
	float w = C[8];
	const float k2 = sqr(k);
	for (int n = 7; n >= 0; n--) {
		w = fmaf(w, k2, C[n]);
	}
	return w;
}

template<class T>
CUDA_EXPORT
inline T kernelFqinv(T q) {
	T res, sw1, sw2, w1;

	const T c0 = 3.2898681336964524 * 2.0;
	const T c1 = -3.2469697011334144 * 4.0;
	const T c2 = 2.0346861239688976 * 6.0;
	const T c3 = -0.8367311301649535 * 8.0;
	const T c4 = 0.2402386980306763 * 10.0;
	const T c5 = -0.05066369468793888 * 12.0;
	const T c6 = 0.008163765290898435 * 14.0;
	const auto q0 = q;
	q *= q < T(1);
	const auto q2 = sqr(q);
	w1 = c6;
	w1 = fmaf(q2, w1, c5);
	w1 = fmaf(q2, w1, c4);
	w1 = fmaf(q2, w1, c3);
	w1 = fmaf(q2, w1, c2);
	w1 = fmaf(q2, w1, c1);
	w1 = fmaf(q2, w1, c0);
	sw1 = q0 < T(1);
	res = w1 * sw1 + (T(1) - sw1) / (sqr(q0) * q0 + T(1e-30f));
	return res;
}

template<class T>
CUDA_EXPORT
inline T kernelPot(T q) {
	T res, sw1, sw2, w1;
	const auto q0 = q;
	q *= (q < T(1));

	const T c0 = 2.437653393057226;
	const T c1 = -3.2898681336964524;
	const T c2 = 3.2469697011334144;
	const T c3 = -2.0346861239688976;
	const T c4 = 0.8367311301649535;
	const T c5 = -0.2402386980306763;
	const T c6 = 0.05066369468793888;
	const T c7 = -0.008163765290898435;
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
}
