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
#include <cosmictiger/containers.hpp>
#include <string>
#include <set>

static int rank;
static int nranks;
static vector<hpx::id_type> localities;
static vector<hpx::id_type> children;
static int ppnode;
static bool hyper;

HPX_PLAIN_ACTION (hpx_init);

static void detect_processes_per_node();
static bool detect_hyperthreading();

void hpx_yield() {
	hpx::this_thread::yield();
}

int hpx_hardware_concurrency() {
	static bool first_call = true;
	int threads = hpx::thread::hardware_concurrency();
	if (hyper) {
		threads /= 2 * ppnode;
	} else {
		threads /= ppnode;
	}
	if (threads == 0) {
		threads = 1;
	}
	if (first_call) {
		PRINT("Detected a hardware concurrency of %i\n", threads);
		first_call = false;
	}
	return threads;
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
#endif

	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<hpx_init_action>(c));
	}
	hpx::wait_all(futs.begin(), futs.end());
	detect_processes_per_node();
	hyper = detect_hyperthreading();
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

static bool detect_hyperthreading() {
	FILE* fp = fopen("/sys/devices/system/cpu/smt/active", "rt");
	if (fp != nullptr) {
		char c;
		FREAD(&c, sizeof(char), 1, fp);
		auto rc = c == '1';
		if (rc) {
			PRINT("Hyperthreading detected\n");
		} else {
			PRINT("Hyperthreading NOT detected\n");
		}
		return rc;
	} else {
		return false;
	}
}

static std::string get_hostname() {
	char buffer[1024];
	gethostname(buffer, 1023);
	std::string name = buffer;
	return name;

}

HPX_PLAIN_ACTION (get_hostname);

static void detect_processes_per_node() {
	vector < hpx::future < std::string >> futs;
	for (int i = 0; i < localities.size(); i++) {
		futs.push_back(hpx::async<get_hostname_action>(localities[i]));
	}

	std::set < std::string > names;
	for (auto& f : futs) {
		const auto name = f.get();
		if (names.find(name) == names.end()) {
			names.insert(name);
		}
	}
	ppnode = localities.size() / names.size();
	PRINT("Detected %i processes per node\n", ppnode);
}
