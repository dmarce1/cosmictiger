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

#include <cosmictiger/float40.hpp>

int hpx_main(int argc, char *argv[]) {
	PRINT( "%7s ", "");
	for( double B = -5.0; B <= 5.0; B += 0.5) {
		PRINT( "%7.3e ", B);
	}
	PRINT( "\n");
	for( double A = -5.0; A <= 5.0; A += 0.3497704) {
		PRINT( "%7.3e ", A);
		for( double B = -5.0; B <= 5.0; B += 0.5) {
			float40 a = A;
			float40 b = B;
			PRINT( "%7.3e ", (double) sqrt(a / b) - sqrt(A/B));
		}
		PRINT( "\n");
	}
	sleep(1000);
	PRINT("%.8e\n", (27.0 / (M_PI * (-6. / exp(9.) + sqrt(M_PI) * erf(3.)))));
	std::atomic<int> i;
	for (double q = 0.0; q < 1.0; q += 0.01) {
//		PRINT( "%e %e %e\n",q, sph_Wh3(q,1.0),sph_dWh3dq(q,1.0));
	}
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
	ewald_const::init();
	start_memuse_daemon();
	if (process_options(argc, argv)) {
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
