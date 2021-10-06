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

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/kick_workspace.hpp>

static int rank;
static int nranks;
static vector<hpx::id_type> localities;
static vector<hpx::id_type> children;

HPX_PLAIN_ACTION (hpx_init);

void hpx_yield() {
	hpx::this_thread::yield();
}


int hpx_hardware_concurrency() {
	return hpx::thread::hardware_concurrency();
}

void hpx_init() {
	rank = hpx::get_locality_id();
	auto tmp = hpx::find_all_localities();
	localities.insert(localities.end(), tmp.begin(), tmp.end());
	nranks = localities.size();
	int base = (rank + 1) << 1;
	const int index1 = base - 1;
	const int index2 = base;
	if (index1 < nranks) {
		children.push_back(localities[index1]);
	}
	if (index2 < nranks) {
		children.push_back(localities[index2]);
	}
#ifdef USE_CUDA
	cuda_init();
	kick_workspace::initialize();
#endif

	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < hpx_init_action > (c));
	}
	hpx::wait_all(futs.begin(), futs.end());

}

int hpx_rank() {
	return rank;
}

int hpx_size() {
	return nranks;
}

const vector<hpx::id_type>& hpx_localities() {
	return localities;
}

const vector<hpx::id_type>& hpx_children() {
	return children;
}
