#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>

static void compute_density();

HPX_PLAIN_ACTION (compute_density);


vector<float> power_spectrum_compute() {
	const int N = get_options().parts_dim;
	fft3d_init(N);
	compute_density();
	fft3d_execute();
	auto power = fft3d_power_spectrum();
	fft3d_destroy();
	return power;
}

static void compute_density() {
	vector<float> rho;
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async < compute_density_action > (c));
	}
	const int N = get_options().parts_dim;
	const auto dblbox = domains_find_my_box();
	range<int> intbox;
	for (int dim = 0; dim < NDIM; dim++) {
		intbox.begin[dim] = dblbox.begin[dim] * N;
		intbox.end[dim] = dblbox.end[dim] * N + 2;
	}
	rho.resize(intbox.volume(), 0.0);
	array<int, NDIM> I;
	for (I[0] = intbox.begin[XDIM]; I[0] < intbox.end[XDIM] - 2; I[0]++) {
		for (I[1] = intbox.begin[YDIM]; I[1] < intbox.end[YDIM] - 2; I[1]++) {
			for (I[2] = intbox.begin[ZDIM]; I[2] < intbox.end[ZDIM] - 2; I[2]++) {
				rho[intbox.index(I)] -= 1.0;
			}
		}
	}
	static vector<spinlock_type> mutexes(intbox.end[XDIM] - intbox.begin[XDIM]);

	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,N,intbox,&rho]() {
			const int begin = (size_t) proc * particles_size() / nthreads;
			const int end = (size_t) (proc+1) * particles_size() / nthreads;
			const double Ninv = 1.0 / N;
			for( int i = begin; i < end; i++) {
				const double x = particles_pos(XDIM,i).to_double();
				const double y = particles_pos(YDIM,i).to_double();
				const double z = particles_pos(ZDIM,i).to_double();
				const int i0 = x * N;
				const int j0 = y * N;
				const int k0 = z * N;
				const int i1 = i0 + 1;
				const int j1 = j0 + 1;
				const int k1 = k0 + 1;
				const double wx1 = x * N - i0;
				const double wy1 = y * N - j0;
				const double wz1 = z * N - k0;
				const double wx0 = 1.0 - wx1;
				const double wy0 = 1.0 - wy1;
				const double wz0 = 1.0 - wz1;
				const double w000 = wx0 * wy0 * wz0;
				const double w001 = wx0 * wy0 * wz1;
				const double w010 = wx0 * wy1 * wz0;
				const double w011 = wx0 * wy1 * wz1;
				const double w100 = wx1 * wy0 * wz0;
				const double w101 = wx1 * wy0 * wz1;
				const double w110 = wx1 * wy1 * wz0;
				const double w111 = wx1 * wy1 * wz1;
				const int i000 = intbox.index(i0,j0,k0);
				const int i001 = intbox.index(i0,j0,k1);
				const int i010 = intbox.index(i0,j1,k0);
				const int i011 = intbox.index(i0,j1,k1);
				const int i100 = intbox.index(i1,j0,k0);
				const int i101 = intbox.index(i1,j0,k1);
				const int i110 = intbox.index(i1,j1,k0);
				const int i111 = intbox.index(i1,j1,k1);
				{
					std::lock_guard<spinlock_type> lock(mutexes[i0 - intbox.begin[XDIM]]);
					rho[i000] += w000;
					rho[i001] += w001;
					rho[i010] += w010;
					rho[i011] += w011;
				}
				{
					std::lock_guard<spinlock_type> lock(mutexes[i1 - intbox.begin[XDIM]]);
					rho[i100] += w100;
					rho[i101] += w101;
					rho[i110] += w110;
					rho[i111] += w111;
				}
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_accumulate_real(intbox, rho);
	hpx::wait_all(futs1.begin(), futs1.end());
}