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

int hpx_main(int argc, char *argv[]) {
	std::atomic<int> i;
	PRINT( "%i\n", sizeof(rockstar_record));
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
	healpix_init();
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
