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

#include <cosmictiger/driver.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/test.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/unordered_set_ts.hpp>
#include <cosmictiger/memused.hpp>
#include <cosmictiger/healpix.hpp>
#include <cosmictiger/fp16.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cmath>

#include <fenv.h>

int dec_index(array<int, NDIM>& n) {
	for (int dim = 0; dim < NDIM; dim++) {
		if (n[dim] > 0) {
			n[dim]--;
			return dim;
		}
	}
	return -1;
}

double rot_tensor(const array<array<double, NDIM>, NDIM>& rot_mat, array<int, NDIM> m, array<int, NDIM> n) {
	auto n1 = n;
	vector<int> indices;
	int i;
	while ((i = dec_index(m)) != -1) {
		indices.push_back(i);
	}
	std::sort(indices.begin(), indices.end());
	double sum = 0.0;
	do {
		double res = 1.0;
		n = n1;
		int j = 0;
		while ((i = dec_index(n)) != -1) {
			res *= rot_mat[i][indices[j]];
			j++;
		}
		sum += res;
	} while (std::next_permutation(indices.begin(), indices.end()));
	return sum;
}

template<class T, int Q>
tensor_trless_sym<T, Q> rotate_tensor(const tensor_trless_sym<T, Q>& A, array<T, NDIM> X, bool inv = false) {
	T norm = sqrt(sqr(X[XDIM], X[YDIM], X[ZDIM]));
	norm = T(1) / norm;
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] *= norm;
	}
	const double& x = X[XDIM];
	const double& y = X[YDIM];
	const double& z = X[ZDIM];
	const double R = sqrt(sqr(x) + sqr(y));

	array<array<double, NDIM>, NDIM> rot_mat;
	if (R == 0.0) {
		return A;
	}
	if (inv) {
		rot_mat[0][0] = x * z / R;
		rot_mat[0][1] = -y / R;
		rot_mat[0][2] = x;
		rot_mat[1][0] = y * z / R;
		rot_mat[1][1] = x / R;
		rot_mat[1][2] = y;
		rot_mat[2][0] = -R;
		rot_mat[2][1] = 0.0;
		rot_mat[2][2] = z;
	} else {
		rot_mat[0][0] = x * z / R;
		rot_mat[0][1] = y * z / R;
		rot_mat[0][2] = -R;
		rot_mat[1][0] = -y / R;
		rot_mat[1][1] = x / R;
		rot_mat[1][2] = 0.0;
		rot_mat[2][0] = x;
		rot_mat[2][1] = y;
		rot_mat[2][2] = z;

	}
	tensor_trless_sym<T, Q> B;
	for (int n = 0; n < Q; n++) {
		for (int m = 0; m < Q - n; m++) {
			for (int l = 0; l < Q - n - m; l++) {
				if (l >= 2 && !(l == 2 && n == 0 && m == 0)) {
					continue;
				}
				array<int, NDIM> n0, m0;
				n0[XDIM] = n;
				n0[YDIM] = m;
				n0[ZDIM] = l;
				B(n, m, l) = 0.0;
				for (m0[XDIM] = 0; m0[XDIM] < Q; m0[XDIM]++) {
					for (m0[YDIM] = 0; m0[YDIM] < Q - m0[XDIM]; m0[YDIM]++) {
						m0[ZDIM] = n0[XDIM] + n0[YDIM] + n0[ZDIM] - m0[XDIM] - m0[YDIM];
						if (m0[ZDIM] < 0 || m0[ZDIM] >= Q - m0[XDIM] - m0[YDIM]) {
							continue;
						}
						B(n, m, l) += rot_tensor(rot_mat, m0, n0) * A(m0[XDIM], m0[YDIM], m0[ZDIM]);
					}
				}
			}
		}
	}

	return B;

}

void test_multipoles() {
	/*pm_multipole<double> M;
	array<double, NDIM> X;
	for (int i = 0; i < PM_MULTIPOLE_SIZE; i++) {
		M[i] = 1.0;
	}
	X[XDIM] = -1.0;
	X[YDIM] = -1.0;
	X[ZDIM] = -1.0;
	auto M0 = rotate_tensor(M, X);
	M0 = rotate_tensor(M0, X, true);
	const auto M1 = M0;
	for (int n = 0; n < PM_ORDER - 1; n++) {
		for (int m = 0; m < PM_ORDER - 1 - n; m++) {
			for (int l = 0; l < PM_ORDER - 1 - n - m; l++) {
				if (l >= 2 && !(l == 2 && n == 0 && m == 0)) {
					continue;
				}
			}
		}
	}
	X[XDIM] = 0.0;
	X[YDIM] = 0.0;
	X[ZDIM] = 1.0;
	tensor_trless_sym<double, 8> D;
	greens_function(D, X, true);
	for (int n = 0; n < 8; n++) {
		for (int m = 0; m < 8 - n; m++) {
			for (int l = 0; l < 8 - n - m; l++) {
				if (l >= 2 && !(l == 2 && n == 0 && m == 0)) {
					continue;
				}
				PRINT("%i %i %i %e\n", n, m, l, D(n, m, l));
			}
		}
	}
*/
}

int hpx_main(int argc, char *argv[]) {
//	test_multipoles();
//	return 0.0;
	{
		double toler = 1.19e-7 / sqrt(2);
		double norm = 2.83;
		double x;
		for (double alpha = 1.0e-1; alpha < 8.0; alpha += 0.1) {
			x = 4.0 / alpha;
			double error;
			do {
				double f = erfc(alpha * x) / norm * (4.0 * M_PI * x) - toler;
				double dfdx = -8.0 * alpha * exp(-sqr(alpha) * x * x) * sqrt(M_PI) * x / norm + 4.0 * M_PI * erfc(alpha * x) / norm;
				x -= f / dfdx;
				error = fabs(f / dfdx);
			} while (error > 1.e-10);
			double real = x;
			x = 1.26 * alpha;
			error = 1e10;
			do {
				double f = 2.0 * exp(-sqr(M_PI * x / alpha)) / (pow(M_PI, 1.5) * x) - toler / 2.0;
				double dfdx = 4.0 * exp(-sqr(M_PI * x / alpha)) * sqrt(M_PI) * (-1.0 / sqr(alpha) - 0.5 / sqr(M_PI * x));
				x -= f / dfdx;
				error = fabs(f / dfdx);

			} while (error > 1.e-10);
			double four = x;
			PRINT("%e %e %e %e\n", alpha, real, four, sqr(real) * real + sqr(four) * four);
		}

	}

	/*	simd_double8 x;
	 for( x[0] = 0.0; x[0] < 0.87; x[0] += 0.01 ) {
	 PRINT( "%e %e\n", -x[0], (erfnearzero(-x)[0] - erf(-x[0]))/erf(-x[0]));
	 }*/

	array<double, NDIM> X;

	std::atomic<int> i;
	PRINT("%i\n", sizeof(rockstar_record));
	for (double q = 0.0; q < 1.0; q += 0.01) {
//		PRINT( "%e %e %e\n",q, sph_Wh3(q,1.0),sph_dWh3dq(q,1.0));
	}
	FILE* fp = fopen("soft.txt", "wt");
	for (double r = 0.0; r < 1.0; r += 0.01) {
		float f, phi;
		gsoft(f, phi, (float) (r * r), 1.0f, 1.0f, 1.0f, true);
		fprintf(fp, "%e %e %e\n", r, phi, f);
	}
	fclose(fp);
	if (!i.is_lock_free()) {
		PRINT("std::atomic<int> is not lock free!\n");
		PRINT("std::atomic<int> must be lock_free for CosmicTiger to run properly.\n");
		PRINT("Exiting...\n");
		return hpx::finalize();
	} else {
		PRINT("std::atomic<int> is lock free!\n");
	}
	PRINT("tree_node size = %i\n", sizeof(tree_node));
	hpx_init();
	healpix_init();
	PRINT("Starting mem use daemon\n");
	start_memuse_daemon();
	if (process_options(argc, argv)) {
#ifndef TREEPM
		ewald_const::init();
		pm_expansion<double> D;

		FILE* fp = fopen("ewald.txt", "wt");
		X[0] = X[1] = X[2] = 0.00;
		for (double x = 0.0; x <= 0.501; x += 0.01) {
			X[0] = x;
			X[1] = 0.0986421;
			X[2] = -0.3123;
			ewald_greens_function(D, X);
			fprintf(fp, "%e ", x);
			for (int i = 0; i < PM_EXPANSION_SIZE; i++) {
				fprintf(fp, "%e ", D[i]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
#endif
		if (get_options().test != "") {
			test(get_options().test);
		} else {
			driver();
		}
	}
	particles_destroy();
	stop_memuse_daemon();
	return hpx::finalize();
}

#ifndef HPX_LITE
int main(int argc, char *argv[]) {
//	feenableexcept (FE_DIVBYZERO);
//	feenableexcept (FE_INVALID);
//	feenableexcept (FE_OVERFLOW);

	PRINT("STARTING MAIN\n");
	std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=524288");
#ifdef HPX_EARLY
	hpx::init(argc, argv, cfg);
#else
	hpx::init_params init_params;
	init_params.cfg = std::move(cfg);
	hpx::init(argc, argv, init_params);
#endif
}
#endif
