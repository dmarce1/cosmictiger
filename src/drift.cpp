#include <cosmictiger/hpx.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/timer.hpp>

HPX_PLAIN_ACTION (drift);

#define CHUNK_SIZE 1024

drift_return drift(double scale, double t, double dt) {
	const bool do_map = get_options().do_map;
	vector<hpx::future<drift_return>> rfuts;
	for (auto c : hpx_children()) {
		rfuts.push_back(hpx::async < drift_action > (HPX_PRIORITY_BOOST, c, scale, t, dt));
	}
	const int nthreads = hpx::thread::hardware_concurrency() - 1;
	PRINT("Drifting on %i with %i threads\n", hpx_rank(), nthreads);
	std::atomic<part_int> next(0);
	const auto func = [dt, scale, do_map, t, &next]() {
		const double factor = 1.0 / scale;
		const double a2inv = 1.0 / sqr(scale);
		drift_return this_dr;
		healpix_map this_map;
		this_dr.kin = 0.0;
		this_dr.momx = 0.0;
		this_dr.momy = 0.0;
		this_dr.momz = 0.0;
		this_dr.flops = 0.0;
		this_dr.nmapped = 0;
		part_int begin = (next+=CHUNK_SIZE) - CHUNK_SIZE;
		while( begin < particles_size()) {
			part_int end = std::min(begin+CHUNK_SIZE,particles_size());
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
				if( do_map) {
					x0 = x;
					y0 = y;
					z0 = z;
				}
				x += double(vx*dt);
				y += double(vy*dt);
				z += double(vz*dt);
				if( do_map) {
					this_dr.nmapped += this_map.map_add_particle(x0, y0, z0, x, y, z, vx, vy, vz, t, dt);
				}
				constrain_range(x);
				constrain_range(y);
				constrain_range(z);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
				this_dr.flops += 31;
			}
			begin = (next+=CHUNK_SIZE) - CHUNK_SIZE;
		}
		if( do_map ) {
			map_add_map(std::move(this_map));
		}
		return this_dr;
	};
	timer tm;
	tm.start();
	for (int proc = 0; proc < nthreads; proc++) {
		rfuts.push_back(hpx::async(func));
	}
	auto dr = func();
	for (auto& fut : rfuts) {
		auto this_dr = fut.get();
		dr.kin += this_dr.kin;
		dr.momx += this_dr.momx;
		dr.momy += this_dr.momy;
		dr.flops += this_dr.flops;
		dr.momz += this_dr.momz;
	}
	tm.stop();
	PRINT("Drift on %i took %e s\n", hpx_rank(), tm.read());
	return dr;
}
