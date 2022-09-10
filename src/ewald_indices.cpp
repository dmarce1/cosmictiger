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
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <chealpix.h>
#include <cosmictiger/fmm_kernels.hpp>

#define NHEAL 1024

HPX_PLAIN_ACTION(ewald_const::init, ewald_const_init_action);

#ifndef USE_CUDA

ewald_constants ec;

int ewald_const::nfour() {
	return NFOUR;
}

int ewald_const::nreal() {
	return NREAL;
}

const array<float, NDIM>& ewald_const::real_index(int i) {
	return ec.real_indices[i];
}

const array<float, NDIM>& ewald_const::four_index(int i) {
	return ec.four_indices[i];
}

const tensor_trless_sym<float, LORDER>& ewald_const::four_expansion(int i) {
	return ec.four_expanse[i];
}

const tensor_sym<float, LORDER> ewald_const::D0() {
	return ec.D0;
}

#endif

void ewald_const::init() {
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<ewald_const_init_action>(c));
	}

#ifdef USE_CUDA
	ewald_const::init_gpu();
#else

	int n2max = 10;
	int nmax = std::sqrt(n2max) + 1;
	array<float, NDIM> this_h;
	int count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				const int i2 = i * i + j * j + k * k;
				if (i2 <= n2max && i2 > 0) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					ec.real_indices[count++] = this_h;
				}
			}
		}
	}
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
		const float c0 = -1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) / M_PI;
		array<int, NDIM> n;
		const int signs[4] = { 1, -1, -1, 1 };
		for (n[0] = 0; n[0] < LORDER; n[0]++) {
			for (n[1] = 0; n[1] < LORDER - n[0]; n[1]++) {
				for (n[2] = 0; n[2] < LORDER - n[0] - n[1]; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					D0(n) *= signs[n0 % 4] * pow(2.0 * M_PI, n0) * c0;
				}
			}
		}
		ec.four_expanse[count++] = D0.detraceD();
	}
	tensor_sym<float, LORDER> D;
	for (int n = 0; n < EXPANSION_SIZE; n++) {
		D[n] = 0.0;
	}
	constexpr double alpha = 2.0;
	for (int n = 0; n < LORDER; n += 2) {
		for (int m = 0; m < LORDER - n; m += 2) {
			for (int l = 0; l < LORDER - n - m; l += 2) {
				D(n, m, l) = pow(-2.0, (n + m + l) / 2 + 1) / ((n + m + l + 1.0) * sqrt(M_PI)) * pow(alpha, n + m + l + 1) * double_factorial(n - 1)
						* double_factorial(m - 1) * double_factorial(l - 1);
			}
		}
	}
	ec.D0 = D;
#endif
	hpx::wait_all(futs.begin(), futs.end());

}

double high_precision_ewald(const array<double, NDIM>& X) {
	array<int, NDIM> h;
	double phi = 0.0;
	double pi = acos(-1.0);
	for (h[0] = -4; h[0] <= 4; h[0]++) {
		for (h[1] = -4; h[1] <= 4; h[1]++) {
			for (h[2] = -4; h[2] <= 4; h[2]++) {
				double x = X[XDIM] - h[XDIM];
				double y = X[YDIM] - h[YDIM];
				double z = X[ZDIM] - h[ZDIM];
				double r2 = sqr(x, y, z);
				if (r2 < 3.6 && sqr(h[0], h[1], h[2]) != 0) {
					double r = sqrt(r2);
					double erfc0 = erfc(2.0l * r);
					phi -= erfc0 / r;
				}
			}
		}
	}
	{
		double x = X[XDIM];
		double y = X[YDIM];
		double z = X[ZDIM];
		double r2 = sqr(x, y, z);
		double r = sqrt(r2);
		double erf0 = erf(2.0l * r);
		phi += r > 0.0 ? erf0 / r : 4.0 / sqrt(pi);
	}
	for (h[0] = -3; h[0] <= 3; h[0]++) {
		for (h[1] = -3; h[1] <= 3; h[1]++) {
			for (h[2] = -3; h[2] <= 3; h[2]++) {
				const double h2 = sqr(h[XDIM], h[YDIM], h[ZDIM]);
				if (h2 > 0.0l && h2 < 10.0l) {
					const double hdotx = h[XDIM] * X[XDIM] + h[YDIM] * X[YDIM] + h[ZDIM] * X[ZDIM];
					phi -= 1.0 / (pi * h2) * expl(-sqr(pi) * h2 * 0.25) * cosl(2.0 * pi * hdotx);
				}
			}
		}
	}
	phi += pi * 0.25;
	return phi;
}
