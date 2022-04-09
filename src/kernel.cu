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

#define __KERNEL_CU__
#include <cosmictiger/options.hpp>
#include <cosmictiger/kernel.hpp>

#include <cosmictiger/safe_io.hpp>

constexpr int NPIECE = 16;
constexpr float dx = 1.0 / NPIECE;

__managed__ pair<float>* WLUT;
__managed__ pair<float>* dWLUT;

float kernel_index;
float kernel_norm;

CUDA_EXPORT
float kernelW(float q) {
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

#define DX (1.0 / NPIECE)

CUDA_EXPORT
float dkernelW_dq(float q) {
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
	return NPIECE*(b*(2 - 3*x)*x + a*(-1 + x)*(-1 + 3*x) - y1 + y2);
//	const float omx = 1.f - x;
//	const float w = omx * y1 + x * y2 + x * omx * (omx * a + x * b);
//	return w;
}

template<class T>
inline T W(T q) {
	const T W0 = kernel_norm;
	const T n = kernel_index;
	q = fminf(q, 1.f);
	T x = float(M_PI) * q;
	x = fminf(x, M_PI * 0.9999999f);
	T w = W0 * powf(sinc(x), n);
	return w;
}

template<class T>
inline T dWdq(T q) {
//	return (W(q + DX) - W(q)) / DX;
	const T W0 = kernel_norm;
	const T n = kernel_index;
	T x = float(M_PI) * q;
	q = fminf(q, 1.f);
	x = fminf(x, M_PI * 0.9999999f);
	if (q == 0.0f) {
		return 0.f;
	} else {
		T tmp, s, c;
		s = sinf(x);
		c = cosf(x);
		if (x < T(0.01f)) {
			tmp = T(-1. / 3.) * sqr(x) * x;
		} else {
			tmp = (x * c - s);
		}
		T w = W0 * n * tmp / (x * q) * powf(s / x, n - 1.0);
		return w;
	}
}

void kernel_set_type(double n) {
	const auto w = [n](double r) {
		return pow(sinc(M_PI*r),n);
	};
	constexpr int N = 100000;
	double sum = 0.0;
	const double dr = 1.0 / N;
	for (int i = N - 1; i >= 1; i--) {
		double r = double(i) / N;
		sum += r * r * w(r) * 4.0 * M_PI * dr;
	}
	kernel_norm = 1.0 / sum;
	PRINT("KERNEL NORM = %e\n", kernel_norm);
	kernel_index = n;

	CUDA_CHECK(cudaMallocManaged(&WLUT, sizeof(pair<float> ) * (NPIECE + 1)));
	CUDA_CHECK(cudaMallocManaged(&dWLUT, sizeof(pair<float> ) * (NPIECE + 1)));
	WLUT[0].first = kernel_norm;
	WLUT[0].second = 0.0;
	WLUT[NPIECE].first = 0.0;
	WLUT[NPIECE].second = 0.0;
	dWLUT[0].first = 0.0;
	dWLUT[0].second = (dWdq(DX) - 0.0) / DX;
	dWLUT[NPIECE].first = 0.0;
	dWLUT[NPIECE].second = 0.0;
	for (int n = 1; n < NPIECE; n++) {
		const float q = (float) n / NPIECE;
		WLUT[n].first = W(q);
		WLUT[n].second = dWdq(q);
		dWLUT[n].first = dWdq(q);
		dWLUT[n].second = (dWdq(q + DX) - dWdq(q)) / DX;
		printf("%e %e\n", WLUT[n].first, WLUT[n].second);
	}
	FILE* fp = fopen("kernel.txt", "wt");
	for (float r = 0.0; r < 1.0; r += 0.01) {
		float w0 = kernelW(r);
		float w1 = W(r);
		float dw0 = dkernelW_dq(r);
		float dw1 = dWdq(r);
		fprintf(fp, "%e %e %e %e %e\n", r, w1, w0, dw1, dw0);
	}
	fclose(fp);
}

void kernel_output() {
}

double kernel_stddev(std::function<double(double)> W) {
	double sum = 0.f;
	int N = 10000;
	double dq = 1.0 / N;
	for (int i = 0; i < N; i++) {
		double q = (i + 0.5) * dq;
		sum += sqr(sqr(q)) * W(q) * dq;
	}
	sum *= 4.0 * M_PI;
	return sqrt(sum);
}

void kernel_adjust_options(options& opts) {
}
