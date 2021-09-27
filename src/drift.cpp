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

HPX_PLAIN_ACTION (drift);

#define CHUNK_SIZE (1024*1024)

drift_return drift(double scale, double t, double dt, double t_max) {
	particles_memadvise_cpu();
	vector<hpx::future<drift_return>> rfuts;
	for (auto c : hpx_children()) {
		rfuts.push_back(hpx::async < drift_action > (HPX_PRIORITY_HI, c, scale, t, dt, t_max));
	}
	const int nthreads = 2 * hpx::thread::hardware_concurrency();
	//PRINT("Drifting on %i with %i threads\n", hpx_rank(), nthreads);
	std::atomic<part_int> next(0);
	const auto func = [dt, scale, t, &next, t_max](int proc, int nthreads) {
		vector<lc_particle> this_part_buffer;
		const double factor = 1.0 / scale;
		const double a2inv = 1.0 / sqr(scale);
		drift_return this_dr;
		this_dr.kin = 0.0;
		this_dr.momx = 0.0;
		this_dr.momy = 0.0;
		this_dr.momz = 0.0;
		this_dr.flops = 0.0;
		this_dr.nmapped = 0;
		part_int begin = (size_t) proc * particles_size() / nthreads;
		part_int end = (size_t) (proc+1) * particles_size() / nthreads;
#ifdef USE_CUDA
			cudaStream_t stream;
			CUDA_CHECK(cudaStreamCreate(&stream));
			CUDA_CHECK(cudaMemPrefetchAsync(&particles_vel(XDIM,begin), (end-begin)*sizeof(float), cudaCpuDeviceId));
			CUDA_CHECK(cudaMemPrefetchAsync(&particles_vel(YDIM,begin), (end-begin)*sizeof(float), cudaCpuDeviceId));
			CUDA_CHECK(cudaMemPrefetchAsync(&particles_vel(ZDIM,begin), (end-begin)*sizeof(float), cudaCpuDeviceId));
			while(cudaStreamQuery(stream)!=cudaSuccess) {
				hpx_yield();
			}
			CUDA_CHECK(cudaStreamSynchronize(stream));
			CUDA_CHECK(cudaStreamDestroy(stream));
#endif
			bool do_lc = get_options().do_lc;
			do_lc = do_lc && (t_max - (t+dt) <= 1.0);
			for( part_int i = begin; i < end; i++) {
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
				vx *= factor;
				vy *= factor;
				vz *= factor;
				double x0, y0, z0;
				x0 = x;
				y0 = y;
				z0 = z;
				x += double(vx*dt);
				y += double(vy*dt);
				z += double(vz*dt);
				if( do_lc) {
					this_dr.nmapped += lc_add_particle(x0, y0, z0, x, y, z, vx, vy, vz, t, dt, this_part_buffer);
				}
				constrain_range(x);
				constrain_range(y);
				constrain_range(z);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
				this_dr.flops += 31;
			}
			if( do_lc) {
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
	}
	tm.stop();
//	PRINT("Drift on %i took %e s\n", hpx_rank(), tm.read());
	return dr;
}
