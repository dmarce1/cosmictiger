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
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>

HPX_PLAIN_ACTION (drift);

#define CHUNK_SIZE (1024*1024)

drift_return drift(double scale, double dt, double tau0, double tau1, double tau_max) {
	particles_memadvise_cpu();
	vector<hpx::future<drift_return>> rfuts;
	for (auto c : hpx_children()) {
		rfuts.push_back(hpx::async<drift_action>(HPX_PRIORITY_HI, c, scale, dt, tau0, tau1, tau_max));
	}
	const int nthreads = 2 * hpx::thread::hardware_concurrency();
	//PRINT("Drifting on %i with %i threads\n", hpx_rank(), nthreads);
	std::atomic<part_int> next(0);
	const auto func = [dt, scale, tau0, tau1, tau_max, &next](int proc, int nthreads) {
		const bool sph = get_options().sph;
		const float dm_mass = get_options().dm_mass;
		const float sph_mass = get_options().sph_mass;
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
			cuda_stream_synchronize(stream);
			CUDA_CHECK(cudaStreamDestroy(stream));
#endif
			bool do_lc = get_options().do_lc;
			do_lc = do_lc && (tau_max - tau1 <= 1.0);
			for( part_int i = begin; i < end; i++) {
				double x = particles_pos(XDIM,i).to_double();
				double y = particles_pos(YDIM,i).to_double();
				double z = particles_pos(ZDIM,i).to_double();
				float vx = particles_vel(XDIM,i);
				float vy = particles_vel(YDIM,i);
				float vz = particles_vel(ZDIM,i);
				int type = DARK_MATTER_TYPE;
				float mass = 1.0f;
				if( sph ) {
					type = particles_type(i);
					if( type == DARK_MATTER_TYPE) {
						mass = dm_mass;
					} else {
						mass = sph_mass;
					}
				}
				this_dr.kin += mass * 0.5 * sqr(vx,vy,vz) * a2inv;
				this_dr.momx += mass * vx;
				this_dr.momy += mass * vy;
				this_dr.momz += mass * vz;
				if( type == SPH_TYPE ) {
					int j = particles_cat_index(i);
					float& h = sph_particles_smooth_len(j);
					char rung = particles_rung(i);
					float dt0 = tau_max / 64 / (1 <<rung);
					const float ent = sph_particles_ent(j);
					if( tau0 != 0.0 ) {
						const float divv = sph_particles_divv(j);
						float dloghdt = (1.f/3.f)*divv/scale;
						if( dloghdt > 1.0 / dt0 ) {
							PRINT( "Clipping dhdt %e\n", dloghdt * dt0);
	//						PRINT( "Hmult = %e\n", c0);
//							abort();
							dloghdt = 1.0 / dt0;
						} else if( dloghdt < -1.0 / dt0) {
							PRINT( "Clippling dhdt %e\n", dloghdt * dt0);
//							PRINT( "Hmult = %e\n", c0);
//							abort();
							dloghdt = -1.0 / dt0;
						}
						float c0 = exp(dloghdt*dt);
						h *= c0;
						if( h > 0.5 ) {
							PRINT( "BIG H!\n");
							abort();
						}
					}
					const float h3 = sqr(h)*h;
					const float vol = (4.0*M_PI/3.0) * h3 / get_options().neighbor_number;
					const float rho = sph_den(1./h3);
#ifdef SPH_TOTAL_ENERGY
			const float ekin = 0.5f * mass * sqr(vx,vy,vz);
			const float etherm = sph_particles_ent(j) - ekin;
			const float e = etherm;
#else
			const float p = ent * pow(rho, SPH_GAMMA);
			const float e = p * (1.0f/(SPH_GAMMA-1.0f)) * vol;
#endif
			this_dr.therm += e * a2inv;
			this_dr.vol += vol;
		}
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
}	;
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
	}
	tm.stop();
//	PRINT("Drift on %i took %e s\n", hpx_rank(), tm.read());
	return dr;
}
