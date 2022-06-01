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

void reset_flops() {
	reset_gpu_flops();
	cpu_flops = 0.0;
	tm.stop();
	tm.reset();
	tm.start();
}

double flops_per_second() {
	tm.stop();
	double result = (get_gpu_flops() + cpu_flops) / tm.read();
	tm.start();
	return result;
}

void add_cpu_flops(int count) {
	std::lock_guard<spinlock_type> lock(mutex);
	cpu_flops += count;
}
