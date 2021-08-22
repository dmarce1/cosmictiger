#include <cosmictiger/hpx.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>

HPX_PLAIN_ACTION(drift);

drift_return drift(double scale, double t, double dt) {
	const bool do_map = get_options().do_map;
	vector<hpx::future<drift_return>> rfuts;
	for (auto c : hpx_children()) {
		rfuts.push_back(hpx::async<drift_action>(c, scale, t, dt));
	}
	drift_return dr;
	dr.kin = 0.0;
	dr.momx = 0.0;
	dr.momy = 0.0;
	dr.momz = 0.0;
	dr.flops = 0.0;
	dr.nmapped = 0;
	mutex_type mutex;
	std::vector<hpx::future<void>> futs;
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [&mutex, &dr, dt, scale, proc, nthreads, do_map, t]() {
			int64_t flops = 0;
			const double factor = 1.0 / scale;
			const double a2inv = 1.0 / sqr(scale);
			double kin = 0.0;
			healpix_map this_map;
			double momx = 0.0;
			double momy = 0.0;
			double momz = 0.0;
			part_int nmapped = 0;
			const part_int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const part_int end = size_t(proc+1) * size_t(particles_size()) / size_t(nthreads);
			for( part_int i = begin; i < end; i++) {
				double x = particles_pos(XDIM,i).to_double();
				double y = particles_pos(YDIM,i).to_double();
				double z = particles_pos(ZDIM,i).to_double();
				float vx = particles_vel(XDIM,i);
				float vy = particles_vel(YDIM,i);
				float vz = particles_vel(ZDIM,i);
				kin += 0.5 * sqr(vx,vy,vz) * a2inv;
				momx += vx;
				momy += vy;
				momz += vz;
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
					nmapped += this_map.map_add_particle(x0, y0, z0, x, y, z, vx, vy, vz, t, dt);
				}
				constrain_range(x);
				constrain_range(y);
				constrain_range(z);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
				flops += 31;
			}
			map_add_map(std::move(this_map));
			std::lock_guard<mutex_type> lock(mutex);
			dr.kin += kin;
			dr.momx += momx;
			dr.momy += momy;
			dr.momz += momz;
			dr.nmapped += nmapped;
			dr.flops += flops;
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());

	for (auto& fut : rfuts) {
		auto this_dr = fut.get();
		dr.kin += this_dr.kin;
		dr.momx += this_dr.momx;
		dr.momy += this_dr.momy;
		dr.flops += this_dr.flops;
		dr.momz += this_dr.momz;
	}
	return dr;
}
