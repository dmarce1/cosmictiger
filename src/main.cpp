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
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/unordered_set_ts.hpp>

std::set<std::string> get_hostnames();

HPX_PLAIN_ACTION (get_hostnames);

std::set<std::string> get_hostnames() {
	std::set < std::string > hostnames;
	std::vector < hpx::future<std::set<std::string>>>futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<get_hostnames_action>(c));
	}
	char hostname[256];
	gethostname(hostname, 255);
	hostnames.insert(std::string(hostname));
	for (auto& f : futs) {
		const auto tmp = f.get();
		for (const auto& hname : tmp) {
			if (hostnames.find(hname) == hostnames.end()) {
				hostnames.insert(hname);
			}
		}
	}
	return hostnames;
}

static int compute_procs_per_node() {
	return hpx_size() / get_hostnames().size();
}

int hpx_main(int argc, char *argv[]) {
	hpx_init();
#ifdef USE_CUDA
	const int procs_per_node = compute_procs_per_node();
	cuda_init(procs_per_node);
#endif
	ewald_const::init();
	if (process_options(argc, argv)) {
		if (get_options().test != "") {
			test(get_options().test);
		} else {
			driver();
		}
	}
	particles_destroy();
	return hpx::finalize();
}

#ifndef HPX_LITE
int main(int argc, char *argv[]) {
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
