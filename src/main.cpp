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
#include <cosmictiger/gravity.hpp>
#include <cmath>

int hpx_main(int argc, char *argv[]) {

	/*	simd_double8 x;
	 for( x[0] = 0.0; x[0] < 0.87; x[0] += 0.01 ) {
	 PRINT( "%e %e\n", -x[0], (erfnearzero(-x)[0] - erf(-x[0]))/erf(-x[0]));
	 }*/

	expansion<simd_float> D;
	array<simd_float, NDIM> X;

	std::atomic<int> i;
	PRINT("%i\n", sizeof(rockstar_record));
	for (double q = 0.0; q < 1.0; q += 0.01) {
//		PRINT( "%e %e %e\n",q, sph_Wh3(q,1.0),sph_dWh3dq(q,1.0));
	}
	FILE* fp = fopen("soft.txt", "wt");
	for (double r = 0.0; r < 1.0; r += 0.01) {
		double f, phi;
		gsoft(f, phi,r*r, 1.0, 1.0, 1.0, true);
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
		ewald_const::init();
		FILE* fp = fopen("ewald.txt", "wt");
		X[0] = X[1] = X[2] = 0.00;
		for (double x = 0.00; x < 0.5; x += 0.001) {
			X[0] = x;
			D = 0.0;
			ewald_greens_function(D, X);
			fprintf(fp, "%e ", x);
			for (int n = 0; n < LORDER; n++) {
				for (int m = 0; m < LORDER - n; m++) {
					for (int l = 0; l < LORDER - n - m; l++) {
						if (l > 1 && !(l == 2 && n == 0 && m == 0)) {
							continue;
						}
						fprintf(fp, "%e ", D(n, m, l)[0]);
					}
				}
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
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
