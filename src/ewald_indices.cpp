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
#include <cosmictiger/fmm_kernels.hpp>
#include <chealpix.h>

#define NHEAL 256

HPX_PLAIN_ACTION(ewald_const::init, ewald_const_init_action);

#ifndef USE_CUDA

#define NREAL 179
#define NFOUR 92

struct ewald_constants {
	array<array<float, NDIM>, NREAL> real_indices;
	array<array<float, NDIM>, NFOUR> four_indices;
	array<tensor_trless_sym<float, LORDER>, NFOUR> four_expanse;
};

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
				if (i2 <= n2max) {
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
#endif
	expansion<double> D;
	for (int i = 0; i < EXPANSION_SIZE; i++) {
		D[i] = 0.0;
	}
	constexpr double delta = 0.02;
	const int npix = NHEAL * NHEAL * 12;
	vector<array<double, NDIM>> pos(npix);
	for (int i = 0; i < npix; i++) {
		double theta, phi;
		double vec[NDIM];
		pix2ang_ring(NHEAL, i, &theta, &phi);
		ang2vec(theta, phi, vec);
		//PRINT( "%e %e %e\n", vec[0], vec[1], vec[2]);
		for( int dim = 0; dim < NDIM; dim++) {
			pos[i][dim] = vec[dim] * delta;
		}
	}
	for (int i = 0; i < npix; i++) {
		expansion<double> D1;
		ewald_greens_function(D1, pos[i]);
		for (int j = 0; j < EXPANSION_SIZE; j++) {
			D[j] += D1[j] / npix;
		}
	}
	for (int n = 0; n < LORDER; n++) {
		for (int m = 0; m < LORDER; m++) {
			for (int l = 0; l < LORDER; l++) {
				if (n + m + l < LORDER && ((l < 2) || (n == 0 && m == 0 && l == 2))) {
					PRINT("%i %i %i %e\n", n, m, l, D(n, m, l));
				}
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

}

