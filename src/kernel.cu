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

constexpr float dx = 1.0 / NPIECE;


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
	WLUT[0].first = kernel_norm;
	WLUT[0].second = 0.0;
	WLUT[NPIECE].first = 0.0;
	WLUT[NPIECE].second = 0.0;
	for (int n = 1; n < NPIECE; n++) {
		const float q = (float) n / NPIECE;
		WLUT[n].first = W(q);
		WLUT[n].second = dWdq(q);
	}
	FILE* fp = fopen("kernel.txt", "wt");
	float err_max = 0.0;
	float norm;
	for (float r = 0.0; r < 1.0; r += 0.01) {
		float w0 = kernelW(r);
		float w1 = W(r);
		float dw0 = dkernelW_dq(r);
		float dw1 = dWdq(r);
		float dif = fabsf(dw0 - dw1);
		err_max = std::max(err_max, dif);
		float avg = 0.5 * (dw0 + dw1);
		norm += avg;
		fprintf(fp, "%e %e %e %e %e\n", r, kernelW(r), dkernelW_dq(r), kernelFqinv(r), kernelPot(r));
	}
	norm /= 100.0;
	PRINT("Kernel Error is %e\n", err_max / norm);
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

	constexpr float bspline_width = 4.676649e-01;
	constexpr float bspline_n = 60;
	float sum = 0.0;
	constexpr int N = 1024;
	for( int i = 1; i < N; i++) {
		float q = (double) i / N;
		sum += sqr(q) * 4.0 * M_PI  / N * sqr(q) * kernelW(q);
	}
	float width = sqrt(sum);
	PRINT( "kernel width = %e\n", width);
	const float n = pow(width / bspline_width,-3)*bspline_n;
	const float cfl = opts.cfl * width / bspline_width;
	PRINT( "Setting neighbor number to %e\n", n);
	PRINT( "Adjusting CFL to %e\n", cfl);

	opts.neighbor_number = n;
	opts.cfl = cfl;

	opts.sph_bucket_size = 8.0 / M_PI * opts.neighbor_number;

}
