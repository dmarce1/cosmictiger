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
#include <cosmictiger/drift.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/cosmology.hpp>

HPX_PLAIN_ACTION (drift);

#define CHUNK_SIZE (1024*1024)

void drift(double scale, double dt, double tau0, double tau1, double tau_max, int rung) {
	profiler_enter("FUNCTION");

	particles_memadvise_cpu();
	vector<hpx::future<void>> rfuts;
	for (auto c : hpx_children()) {
		rfuts.push_back(hpx::async<drift_action>(c, scale, dt, tau0, tau1, tau_max, rung));
	}
	const int nthreads = 2 * hpx::thread::hardware_concurrency();
	//PRINT("Drifting on %i with %i threads\n", hpx_rank(), nthreads);
	std::atomic<part_int> next(0);
	const auto func = [dt, scale, tau0, rung, tau1, tau_max, &next](int proc, int nthreads) {
		vector<lc_particle> this_part_buffer;
		const double ainv = 1.0 / scale;
		const double a2inv = 1.0 / sqr(scale);
		auto range = particles_current_range();
		part_int begin = (size_t) proc * (range.second - range.first) / nthreads + range.first;
		part_int end = (size_t) (proc+1) * (range.second - range.first) / nthreads + range.first;
		bool do_lc = get_options().do_lc;
		do_lc = do_lc && (tau_max - tau1 <= 1.0);
		const auto adot = get_options().test != "" ? 0.0 : scale * cosmos_dadt(scale);
		for( part_int i = begin; i < end; i++) {
			if( particles_rung(i) == rung ) {
				double x = particles_pos(XDIM,i).to_double();
				double y = particles_pos(YDIM,i).to_double();
				double z = particles_pos(ZDIM,i).to_double();
				float vx = particles_vel(XDIM,i);
				float vy = particles_vel(YDIM,i);
				float vz = particles_vel(ZDIM,i);
				vx *= ainv;
				vy *= ainv;
				vz *= ainv;
				double x0, y0, z0;
				x0 = x;
				y0 = y;
				z0 = z;
				x += double(vx*dt);
				y += double(vy*dt);
				z += double(vz*dt);
				if( do_lc) {
					lc_add_particle(x0, y0, z0, x, y, z, vx, vy, vz, tau0, tau1, this_part_buffer);
				}
				constrain_range(x);
				constrain_range(y);
				constrain_range(z);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
			}
		}
		if (do_lc) {
			lc_add_parts(std::move(this_part_buffer));
		}
	};
	timer tm;
	tm.start();
	for (int proc = 1; proc < nthreads; proc++) {
		rfuts.push_back(hpx::async(func, proc, nthreads));
	}
	func(0, nthreads);
	for (auto& fut : rfuts) {
		fut.get();
	}
	tm.stop();
	profiler_exit();
}
