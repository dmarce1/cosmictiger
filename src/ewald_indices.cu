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

#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>

#include <cosmictiger/fmm_kernels.hpp>

#include <algorithm>
ewald_constants ec;
__constant__ ewald_constants ec_dev;

void ewald_const::init_gpu() {

	int n2max = 12;
	int nmax = std::sqrt(n2max) + 1;
	array<float, NDIM> this_h;
	int count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				const int i2 = i * i + j * j + k * k;
				const double x = std::max(abs(i) - 0.5, 0.0);
				const double y = std::max(abs(j) - 0.5, 0.0);
				const double z = std::max(abs(k) - 0.5, 0.0);
				const double d = sqrt(sqr(x, y, z));
				if (d < EWALD_REAL_CUTOFF && i2 > 0) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					ec.real_indices[count++] = this_h;
				}
			}
		}
	}
	PRINT("count = %i %i\n", count, NREAL);
	const auto sort_func = [](const array<float,NDIM>& a, const array<float,NDIM>& b) {
		const auto a2 = sqr(a[0],a[1],a[2]);
		const auto b2 = sqr(b[0],b[1],b[2]);
		return a2 > b2;
	};
	std::sort(ec.real_indices.begin(), ec.real_indices.end(), sort_func);
	n2max = 8;
	nmax = std::sqrt(n2max) + 1;
	count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				if (i * i + j * j + k * k <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					const auto hdot = sqr(this_h[0]) + sqr(this_h[1]) + sqr(this_h[2]);
					if (hdot > 0) {
						ec.four_indices[count++] = this_h;
					}
				}
			}
		}
	}
	std::sort(ec.four_indices.begin(), ec.four_indices.end(), sort_func);
	count = 0;
	for (int i = 0; i < NFOUR; i++) {
		array<float, NDIM> h = ec.four_indices[i];
		auto D0 = vector_to_sym_tensor<float, LORDER>(h);
		const float h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);                     // 5 OP
		const float c0 = -1.0 / h2 * exp(-M_PI * M_PI * h2 / sqr(EWALD_ALPHA)) / M_PI;
		array<int, NDIM> n;
		const int signs[4] = { 1, -1, -1, 1 };
		for (n[0] = 0; n[0] < LORDER; n[0]++) {
			for (n[1] = 0; n[1] < LORDER - n[0]; n[1]++) {
				for (n[2] = 0; n[2] < LORDER - n[0] - n[1]; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					D0(n) *= (signs[n0 % 4] * pow(2.0 * M_PI * SCALE_FACTOR_INV1, n0) * c0) * SCALE_FACTOR_INV1;
				}
			}
		}
		ec.four_expanse[count] = D0.detraceD();
		count++;
	}
	tensor_sym<float, LORDER> D;
	for (int n = 0; n < (LORDER + 2) * (LORDER + 1) * LORDER / 6; n++) {
		D[n] = 0.0;
	}
	constexpr double alpha = EWALD_ALPHA;
	for (int n = 0; n < LORDER; n += 2) {
		for (int m = 0; m < LORDER - n; m += 2) {
			for (int l = 0; l < LORDER - n - m; l += 2) {
				D(n, m, l) = pow(SCALE_FACTOR_INV1, n + m + l + 1) * pow(-2.0, (n + m + l) / 2 + 1) / ((n + m + l + 1.0) * sqrt(M_PI)) * pow(alpha, n + m + l + 1)
						* double_factorial(n - 1) * double_factorial(m - 1) * double_factorial(l - 1);
			}
		}
	}
	ec.D0 = D;
	cuda_set_device();
	CUDA_CHECK(cudaMemcpyToSymbol(ec_dev, &ec, sizeof(ewald_constants)));
}

CUDA_EXPORT int ewald_const::nfour() {
	return NFOUR;
}

CUDA_EXPORT int ewald_const::nreal() {
	return NREAL;
}

CUDA_EXPORT const tensor_sym<float, LORDER> ewald_const::D0() {
#ifdef __CUDA_ARCH__
	return ec_dev.D0;
#else
	return ec.D0;
#endif
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::real_index(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev.real_indices[i];
#else
	return ec.real_indices[i];
#endif
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::four_index(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev.four_indices[i];
#else
	return ec.four_indices[i];
#endif
}

CUDA_EXPORT const tensor_trless_sym<float, LORDER>& ewald_const::four_expansion(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev.four_expanse[i];
#else
	return ec.four_expanse[i];
#endif
}
