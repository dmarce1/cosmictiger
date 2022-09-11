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
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cmath>

#include <fenv.h>

void test_multipoles() {
	pm_multipole<double> M;
	pm_expansion<double> L;
	constexpr int ntrials = 1;
	for (int i = 0; i < PM_MULTIPOLE_SIZE; i++) {
		M[i] = 1.0;
	}
	array<double, NDIM> x;
	array<double, NDIM> y;
	for (int dim = 0; dim < NDIM; dim++) {
		y[dim] = 0.0;
	}
	for (int i = 0; i < ntrials; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = 2.0 * rand1() - 1.0;
			y[dim] -= x[dim];
		}
		M = M2M(M, x);
	}
	M = M2M(M, y);
	for (int n = 0; n < PM_ORDER - 1; n++) {
		for (int m = 0; m < PM_ORDER - 1 - n; m++) {
			for (int l = 0; l < PM_ORDER - 1 - n - m; l++) {
				if (l >= 2 && !(l == 2 && n == 0 && m == 0)) {
					continue;
				}
				PRINT("%i %i %i %e\n", n, m, l, M(n, m, l));
			}
		}
	}
	PRINT("!!!!!!!!!!!\n");
	for (int i = 0; i < PM_EXPANSION_SIZE; i++) {
		L[i] = 1.0;
	}
	for (int dim = 0; dim < NDIM; dim++) {
		y[dim] = 0.0;
	}
	for (int i = 0; i < ntrials; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = 2.0 * rand1() - 1.0;
			y[dim] -= x[dim];
		}
		L = L2L(L, x, true);
	}
	L = L2L(L, y, true);
	for (int i = 0; i < PM_EXPANSION_SIZE; i++) {
		PRINT("%.16e\n", L[i]);
	}
}

int hpx_main(int argc, char *argv[]) {
	test_multipoles();
	return 0;
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
