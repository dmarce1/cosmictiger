constexpr bool verbose = true;
#define PARTICLES_CPP

#include <tigerfmm/hpx.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/safe_io.hpp>

#include <gsl/gsl_rng.h>

#include <shared_mutex>
#include <unordered_map>

struct line_id_type;

static vector<array<fixed32, NDIM>> particles_fetch_cache_line(int index);
static const array<fixed32, NDIM>* particles_cache_read_line(line_id_type line_id);
void particles_cache_free();

HPX_PLAIN_ACTION (particles_cache_free);
HPX_PLAIN_ACTION (particles_destroy);
HPX_PLAIN_ACTION (particles_fetch_cache_line);
HPX_PLAIN_ACTION (particles_random_init);
HPX_PLAIN_ACTION (particles_sample);

struct line_id_type {
	int proc;
	int index;
	inline bool operator==(line_id_type other) const {
		return proc == other.proc && index == other.index;
	}
};

struct line_id_hash {
	inline size_t operator()(line_id_type id) const {
		const int line_size = get_options().part_cache_line_size;
		const int i = id.index / line_size;
		return i * (hpx_size() - 1) + ((id.proc < hpx_rank()) ? id.proc : id.proc - 1);
	}
};

struct line_id_hash_lo {
	inline size_t operator()(line_id_type id) const {
		line_id_hash hash;
		return hash(id) % PART_CACHE_SIZE;
	}
};

struct line_id_hash_hi {
	inline size_t operator()(line_id_type id) const {
		line_id_hash hash;
		return hash(id) / PART_CACHE_SIZE;
	}
};

static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<fixed32, NDIM>>> , line_id_hash_hi>,PART_CACHE_SIZE> part_cache;
static array<spinlock_type, PART_CACHE_SIZE> mutexes;

void particles_cache_free() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_cache_free_action>(c));
	}
	for (int i = 0; i < PART_CACHE_SIZE; i++) {
		part_cache[i] = std::unordered_map<line_id_type, hpx::shared_future<vector<array<fixed32, NDIM>>> , line_id_hash_hi>();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, int offset) {
	const int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const int dif = offset - range.range.first;
			const int sz = range.range.second - range.range.first;
			std::memcpy(x + offset, &particles_pos(XDIM, range.range.first), sizeof(float) * sz);
			std::memcpy(y + offset, &particles_pos(YDIM, range.range.first), sizeof(float) * sz);
			std::memcpy(z + offset, &particles_pos(ZDIM, range.range.first), sizeof(float) * sz);
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const int start_line = (range.range.first / line_size) * line_size;
			const int stop_line = ((range.range.second - 1) / line_size) * line_size;
			int dest_index = offset;
			for (int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = particles_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (int i = begin; i < end; i++) {
					const int src_index = i - line_id.index;
					x[dest_index] = ptr[src_index][XDIM];
					y[dest_index] = ptr[src_index][YDIM];
					z[dest_index] = ptr[src_index][ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static const array<fixed32, NDIM>* particles_cache_read_line(line_id_type line_id) {
	const int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = part_cache[bin].find(line_id);
	const array<fixed32, NDIM>* ptr;
	if (iter == part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<array<fixed32, NDIM>>> >();
		part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::async([prms,line_id]() {
			auto line_fut = hpx::async<particles_fetch_cache_line_action>(hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<array<fixed32, NDIM>> particles_fetch_cache_line(int index) {
	const int line_size = get_options().part_cache_line_size;
	vector<array<fixed32, NDIM>> line(line_size);
	const int begin = (index / line_size) * line_size;
	const int end = std::min(particles_size(), begin + line_size);
	for (int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin][dim] = particles_pos(dim, i);
		}
	}
	return line;
}

void particles_destroy() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<particles_destroy_action>(c));
	}
	particles_x = decltype(particles_x)();
	particles_g = decltype(particles_g)();
	particles_v = decltype(particles_v)();
	particles_p = decltype(particles_p)();
	particles_r = decltype(particles_r)();
	hpx::wait_all(futs.begin(), futs.end());
}


int particles_size() {
	return particles_r.size();
}

void particles_resize(int sz) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_x[dim].resize(sz);
		particles_v[dim].resize(sz);
	}
	particles_r.resize(sz);
	if (get_options().save_force) {
		particles_p.resize(sz);
		for (int dim = 0; dim < NDIM; dim++) {
			particles_g[dim].resize(sz);
		}
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
			const int seed = 4321*(hpx_rank() * nthreads + proc) + 42;
			gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
			gsl_rng_set(rndgen, seed);
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

int particles_sort(pair<int, int> rng, double xm, int xdim) {
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

vector<particle_sample> particles_sample(int cnt) {
	const bool save_force = get_options().save_force;
	const auto children = hpx_children();
	const size_t nparts = std::pow(get_options().parts_dim, NDIM);
	vector<hpx::future<vector<particle_sample>>>futs;
	for (int i = 0; i < children.size(); i++) {
		const int begin = size_t(cnt) * nparts / (children.size() + 1);
		const int end = size_t(cnt + 1) * nparts / (children.size() + 1);
		const int this_cnt = end - begin;
		futs.push_back(hpx::async<particles_sample_action>(children[i], this_cnt));
	}
	const int begin = size_t(children.size()) * nparts / (children.size() + 1);
	const int this_cnt = nparts - begin;
	vector<particle_sample> parts;
	const int seed = 4321 * hpx_rank() + 42;
	gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rndgen, seed);
	for (int i = 0; i < this_cnt; i++) {
		particle_sample sample;
		const int index = gsl_rng_get(rndgen) % particles_size();
		for (int dim = 0; dim < NDIM; dim++) {
			sample.x[dim] = particles_pos(dim, index);
		}
		if (save_force) {
			for (int dim = 0; dim < NDIM; dim++) {
				sample.g[dim] = particles_gforce(dim, index);
			}
			sample.p = particles_pot(index);
		}
		parts.push_back(sample);
	}
	gsl_rng_free(rndgen);
	for (auto& f : futs) {
		auto v = f.get();
		parts.insert(parts.end(), v.begin(), v.end());
	}
	return std::move(parts);
}

void particles_load(FILE* fp) {
	int size;
	FREAD(&size, sizeof(int), 1, fp);
	particles_resize(size);
	FREAD(&particles_pos(XDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_pos(YDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_pos(ZDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_vel(XDIM, 0), sizeof(float), particles_size(), fp);
	FREAD(&particles_vel(YDIM, 0), sizeof(float), particles_size(), fp);
	FREAD(&particles_vel(ZDIM, 0), sizeof(float), particles_size(), fp);
	FREAD(&particles_rung(0), sizeof(char), particles_size(), fp);

}

void particles_save(FILE* fp) {
	int size = particles_size();
	fwrite(&size, sizeof(int), 1, fp);
	fwrite(&particles_pos(XDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(&particles_pos(YDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(&particles_pos(ZDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(&particles_vel(XDIM, 0), sizeof(float), particles_size(), fp);
	fwrite(&particles_vel(YDIM, 0), sizeof(float), particles_size(), fp);
	fwrite(&particles_vel(ZDIM, 0), sizeof(float), particles_size(), fp);
	fwrite(&particles_rung(0), sizeof(char), particles_size(), fp);

}
