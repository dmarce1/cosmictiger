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

#include <cosmictiger/defs.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/timer.hpp>

static double cpu_flops = 0.0;
static spinlock_type mutex;
static timer tm;

HPX_PLAIN_ACTION (reset_flops);
HPX_PLAIN_ACTION (flops_per_second);

void reset_flops() {
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<reset_flops_action>(c));
	}
	reset_gpu_flops();
	cpu_flops = 0.0;
	tm.stop();
	tm.reset();
	tm.start();
	hpx::wait_all(futs.begin(), futs.end());
}

double flops_per_second() {
	vector<hpx::future<double>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<flops_per_second_action>(c));
	}
	tm.stop();
	double result = (get_gpu_flops() + cpu_flops);
	tm.start();
	for (auto& f : futs) {
		result += f.get();
	}
	if (hpx_rank() == 0) {
		result /= tm.read();
	}
	return result;
}

void add_cpu_flops(int count) {
#ifdef COUNT_FLOPS
	std::lock_guard<spinlock_type> lock(mutex);
	cpu_flops += count;
#endif
}
