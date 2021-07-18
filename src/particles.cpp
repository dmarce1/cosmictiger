#define PARTICLES_CPP

#include <tigerfmm/hpx.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/particles.hpp>

#include <gsl/gsl_rng.h>

HPX_PLAIN_ACTION(particles_random_init);
HPX_PLAIN_ACTION(particles_destroy);

int particles_size() {
	return particles_r.size();
}

int particles_size_pos() {
	return particles_x[XDIM].size();
}

void particles_destroy() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for( const auto& c : children) {
		futs.push_back(hpx::async<particles_destroy_action>(c));
	}
	particles_x = decltype(particles_x)();
	particles_v = decltype(particles_v)();
	particles_r = decltype(particles_r)();
	hpx::wait_all(futs.begin(),futs.end());
}

void particles_resize(int sz) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_x[dim].resize(sz);
		particles_v[dim].resize(sz);
	}
	particles_r.resize(sz);
}

void particles_resize_pos(int sz) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_x[dim].resize(sz);
	}
}

void particles_random_init() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<particles_random_init_action>(c));
	}
	const size_t total_num_parts = std::pow(get_options().parts_dim, NDIM);
	const size_t begin = (size_t) (hpx_rank()) * total_num_parts / hpx_size();
	const size_t end = (size_t) (hpx_rank() + 1) * total_num_parts / hpx_size();
	const size_t my_num_parts = end - begin;
	particles_resize(my_num_parts);
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads]() {
			const int begin = (size_t) proc * particles_size() / nthreads;
			const int end = (size_t) (proc+1) * particles_size() / nthreads;
			const int seed = 4321 * (hpx_size() * proc + hpx_rank() + 42);
			gsl_rng * rndgen = gsl_rng_alloc(gsl_rng_taus);
			for (int i = begin; i < end; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					particles_pos(dim, i) = gsl_rng_uniform(rndgen);
					particles_vel(dim, i) = 0.0f;
				}
				particles_rung(i) = 0;
			}
			gsl_rng_free(rndgen);
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

int particles_sort(pair<int,int> rng, double xm, int xdim) {
	int begin = rng.first;
	int end = rng.second;
	int lo = begin;
	int hi = end;
	fixed32 xmid(xm);
	auto& xptr_dim = particles_x[xdim];
	auto& x = particles_x[XDIM];
	auto& y = particles_x[YDIM];
	auto& z = particles_x[ZDIM];
	auto& ux = particles_v[XDIM];
	auto& uy = particles_v[YDIM];
	auto& uz = particles_v[ZDIM];
	while (lo < hi) {
		if (xptr_dim[lo] >= xmid) {
			while (lo != hi) {
				hi--;
				if (xptr_dim[hi] < xmid) {
					std::swap(x[hi], x[lo]);
					std::swap(y[hi], y[lo]);
					std::swap(z[hi], z[lo]);
					std::swap(ux[hi], ux[lo]);
					std::swap(uy[hi], uy[lo]);
					std::swap(uz[hi], uz[lo]);
					std::swap(particles_r[hi], particles_r[lo]);
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

