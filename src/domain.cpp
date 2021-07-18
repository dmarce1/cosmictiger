constexpr bool verbose = true;

#include <tigerfmm/domain.hpp>
#include <tigerfmm/containers.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/range.hpp>
#include <tigerfmm/safe_io.hpp>

#include <unordered_map>

void domains_transmit_particles(vector<particle>);

HPX_PLAIN_ACTION(domains_begin);
HPX_PLAIN_ACTION(domains_end);
HPX_PLAIN_ACTION(domains_transmit_particles);

static vector<int> free_indices;
static vector<particle> trans_particles;
static mutex_type mutex;

static void domains_find_all(vector<range<double>>& domains, int begin, int end, range<double> box);
static int find_particle_domain(const array<double, NDIM>& x);

void domains_transmit_particles(vector<particle> parts) {
	PRINT("Receiving %li particles on %i\n", parts.size(), hpx_rank());
	std::lock_guard<mutex_type> lock(mutex);
	const int start = trans_particles.size();
	const int stop = start + parts.size();
	trans_particles.resize(stop);
	auto fut = hpx::parallel::copy(PAR_EXECUTION_POLICY, parts.begin(), parts.end(), trans_particles.begin() + start);
	fut.get();
}

void domains_begin() {
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for (auto& c : children) {
		futs.push_back(hpx::async<domains_begin_action>(c));
	}
	vector<range<double>> domains(hpx_size());
	domains_find_all(domains, 0, hpx_size(), unit_box<double>());
	const auto my_domain = domains[hpx_rank()];
	free_indices.resize(0);
	mutex_type mutex;
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&mutex,my_domain]() {
			std::unordered_map<int,vector<particle>> sends;
			vector<hpx::future<void>> futs;
			vector<int> my_free_indices;
			const int begin = (size_t) proc * particles_size() / nthreads;
			const int end = (size_t) (proc+1) * particles_size() / nthreads;
			for( int i = begin; i < end; i++) {
				array<double, NDIM> x;
				for( int dim = 0; dim < NDIM; dim++) {
					x[dim] = particles_pos(dim,i).to_double();
				}
				if( !my_domain.contains(x)) {
					const int rank = find_particle_domain(x);
					auto& send = sends[rank];
					send.push_back(particles_get_particle(i));
					if( send.size() >= MAX_PARTICLES_PER_PARCEL) {
						PRINT( "%i sending %li particles to %i\n", hpx_rank(), send.size(), rank);
						futs.push_back(hpx::async<domains_transmit_particles_action>(hpx_localities()[rank], std::move(send)));
					}
					my_free_indices.push_back(i);
				}
			}
			for( auto i = sends.begin(); i != sends.end(); i++) {
				if( i->second.size()) {
					PRINT( "%i sending %li particles to %i\n", hpx_rank(), i->second.size(), i->first);
					futs.push_back(hpx::async<domains_transmit_particles_action>(hpx_localities()[i->first], std::move(i->second)));
				}
			}
			std::lock_guard<mutex_type> lock(mutex);
			free_indices.insert(free_indices.end(),my_free_indices.begin(),my_free_indices.end());
			hpx::wait_all(futs.begin(), futs.end());
		}));
	}

	hpx::wait_all(futs.begin(), futs.end());
}

void domains_end() {
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for (auto& c : children) {
		futs.push_back(hpx::async<domains_end_action>(c));
	}
	if (trans_particles.size()) {
		const auto particle_compare = [](particle a, particle b) {
			for( int dim = 0; dim < NDIM; dim++) {
				if( a.x[dim] < b.x[dim]) {
					return true;
				} else if( a.x[dim] > b.x[dim]) {
					return false;
				}
			}
			return false;
		};
		PRINT("Processing %li particles on %i\n", trans_particles.size(), hpx_rank());
		auto fut = hpx::parallel::sort(PAR_EXECUTION_POLICY, free_indices.begin(), free_indices.end());
		hpx::parallel::sort(PAR_EXECUTION_POLICY, trans_particles.begin(), trans_particles.end(), particle_compare).get();
		fut.get();
		if (free_indices.size() < trans_particles.size()) {
			const int diff = trans_particles.size() - free_indices.size();
			for (int i = 0; i < diff; i++) {
				free_indices.push_back(particles_size() + i);
			}
			particles_resize(particles_size() + diff);
		} else {
			while (free_indices.size() > trans_particles.size()) {
				const auto p = particles_get_particle(particles_size() - 1);
				particles_set_particle(p, free_indices.back());
				free_indices.pop_back();
				particles_resize(particles_size() - 1);
			}
		}
		const int nthreads = hpx::thread::hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(hpx::async([nthreads, proc]() {
				const int begin = (size_t) proc * free_indices.size() / nthreads;
				const int end = (size_t) (proc+1) * free_indices.size() / nthreads;
				for( int i = begin; i < end; i++) {
					particles_set_particle(trans_particles[i],free_indices[i]);
				}
			}));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	trans_particles = decltype(trans_particles)();
	free_indices = decltype(free_indices)();
}

static void domains_find_all(vector<range<double>>& domains, int begin, int end, range<double> box) {
	if (end - begin == 1) {
		domains[begin] = box;
	} else {
		auto left = box;
		auto right = box;
		int mid = (begin + end) / 2;
		const double wt = double(end - mid) / double(end - begin);
		const int dim = box.longest_dim();
		const double mid_x = box.begin[dim] * wt + box.end[dim] * (1.0 - wt);
		left.end[dim] = right.begin[dim] = mid_x;
		domains_find_all(domains, begin, mid, left);
		domains_find_all(domains, mid, end, right);
	}
}

static int find_particle_domain(const array<double, NDIM>& x) {
	int begin = 0;
	int end = hpx_size();
	auto box = unit_box<double>();
	while (end - begin != 1) {
		int mid = (begin + end) / 2;
		const double wt = double(end - mid) / double(end - begin);
		const int dim = box.longest_dim();
		const double mid_x = box.begin[dim] * wt + box.end[dim] * (1.0 - wt);
		if (x[dim] < mid_x) {
			box.end[dim] = mid_x;
			end = mid;
		} else {
			box.begin[dim] = mid_x;
			begin = mid;
		}

	}
	return begin;
}