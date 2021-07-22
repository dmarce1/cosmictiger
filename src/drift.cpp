#include <tigerfmm/hpx.hpp>
#include <tigerfmm/drift.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/particles.hpp>

HPX_PLAIN_ACTION(drift);

drift_return drift(double scale, double dt) {

	vector<hpx::future<drift_return>> rfuts;
	for (auto c : hpx_children()) {
		rfuts.push_back(hpx::async<drift_action>(c,scale, dt));
	}
	drift_return dr;
	dr.kin = 0.0;
	dr.momx = 0.0;
	dr.momy = 0.0;
	dr.momz = 0.0;
	mutex_type mutex;
	std::vector<hpx::future<void>> futs;
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [&mutex, &dr, dt, scale, proc, nthreads]() {
			const double factor = dt / scale;
			const double a2inv = 1.0 / sqr(scale);
			double kin = 0.0;
			double momx = 0.0;
			double momy = 0.0;
			double momz = 0.0;
			const int begin = size_t(proc) * size_t(particles_size()) / size_t(nthreads);
			const int end = size_t(proc+1) * size_t(particles_size()) / size_t(nthreads);
			for( int i = begin; i < end; i++) {
				double x = particles_pos(XDIM,i).to_double();
				double y = particles_pos(YDIM,i).to_double();
				double z = particles_pos(ZDIM,i).to_double();
				const float vx = particles_vel(XDIM,i);
				const float vy = particles_vel(YDIM,i);
				const float vz = particles_vel(ZDIM,i);
				kin += 0.5 * sqr(vx,vy,vz) * a2inv;
				momx += vx;
				momy += vy;
				momz += vz;
				x += double(vx) * factor;
				y += double(vy) * factor;
				z += double(vz) * factor;
				constrain_range(x);
				constrain_range(y);
				constrain_range(z);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
			}
			std::lock_guard<mutex_type> lock(mutex);
			dr.kin += kin;
			dr.momx += momx;
			dr.momy += momy;
			dr.momz += momz;
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());

	for( auto& fut : rfuts ) {
		auto this_dr = fut.get();
		dr.kin += this_dr.kin;
		dr.momx += this_dr.momx;
		dr.momy += this_dr.momy;
		dr.momz += this_dr.momz;
	}
	return dr;
}
