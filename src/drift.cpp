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

drift_return drift(double scale, double dt, double tau0, double tau1, double tau_max, int rung) {
	profiler_enter("FUNCTION");

	particles_memadvise_cpu();
	vector<hpx::future<drift_return>> rfuts;
	for (auto c : hpx_children()) {
		rfuts.push_back(hpx::async<drift_action>(HPX_PRIORITY_HI, c, scale, dt, tau0, tau1, tau_max, rung));
	}
	const int nthreads = 2 * hpx::thread::hardware_concurrency();
	//PRINT("Drifting on %i with %i threads\n", hpx_rank(), nthreads);
	std::atomic<part_int> next(0);
	const auto func = [dt, scale, tau0, rung, tau1, tau_max, &next](int proc, int nthreads) {
		vector<lc_particle> this_part_buffer;
		const double ainv = 1.0 / scale;
		const double a2inv = 1.0 / sqr(scale);
		drift_return this_dr;
		this_dr.kin = 0.0;
		this_dr.momx = 0.0;
		this_dr.momy = 0.0;
		this_dr.momz = 0.0;
		this_dr.flops = 0.0;
		this_dr.nmapped = 0;
		this_dr.therm = 0.0;
		this_dr.vol = 0.0;
		this_dr.cold_mass = 0.0;
		auto range = particles_current_range();
		part_int begin = (size_t) proc * (range.second - range.first) / nthreads + range.first;
		part_int end = (size_t) (proc+1) * (range.second - range.first) / nthreads + range.first;
		bool do_lc = get_options().do_lc;
		do_lc = do_lc && (tau_max - tau1 <= 1.0);
		const auto adot = get_options().test != "" ? 0.0 : scale * cosmos_dadt(scale);
		for( part_int i = begin; i < end; i++) {
			if( rung != -1 && particles_rung(i) != rung ) {
				continue;
			}
			double x = particles_pos(XDIM,i).to_double();
			double y = particles_pos(YDIM,i).to_double();
			double z = particles_pos(ZDIM,i).to_double();
			float vx = particles_vel(XDIM,i);
			float vy = particles_vel(YDIM,i);
			float vz = particles_vel(ZDIM,i);
			this_dr.kin += 0.5 * sqr(vx,vy,vz) * a2inv;
			this_dr.momx += vx;
			this_dr.momy += vy;
			this_dr.momz += vz;
			vx *= ainv;
			vy *= ainv;
			vz *= ainv;
			double x0, y0, z0;
			x0 = x;
			y0 = y;
			z0 = z;
			x += double(vx*dt);
			if( std::isnan(vy)) {
				PRINT( "vy is nan\n");
				abort();
			}
			y += double(vy*dt);
			z += double(vz*dt);
			if( do_lc) {
				this_dr.nmapped += lc_add_particle(x0, y0, z0, x, y, z, vx, vy, vz, tau0, tau1, this_part_buffer);
			}
			constrain_range(x);
			constrain_range(y);
			constrain_range(z);
			particles_pos(XDIM,i) = x;
			particles_pos(YDIM,i) = y;
			particles_pos(ZDIM,i) = z;
			this_dr.flops += 34;
		}
		if (do_lc) {
			lc_add_parts(std::move(this_part_buffer));
		}
		return this_dr;
	};
	timer tm;
	tm.start();
	for (int proc = 1; proc < nthreads; proc++) {
		rfuts.push_back(hpx::async(func, proc, nthreads));
	}
	auto dr = func(0, nthreads);
	for (auto& fut : rfuts) {
		auto this_dr = fut.get();
		dr.kin += this_dr.kin;
		dr.momx += this_dr.momx;
		dr.momy += this_dr.momy;
		dr.flops += this_dr.flops;
		dr.momz += this_dr.momz;
		dr.nmapped += this_dr.nmapped;
		dr.therm += this_dr.therm;
		dr.vol += this_dr.vol;
		dr.cold_mass += this_dr.cold_mass;
	}
	tm.stop();
	profiler_exit();
	return dr;
}
