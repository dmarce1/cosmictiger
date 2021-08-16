constexpr bool verbose = true;
#define PARTICLES_CPP

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>

#include <gsl/gsl_rng.h>

#include <unordered_map>

struct line_id_type;

static vector<group_int> particles_group_refresh_cache_line(part_int index);

static vector<array<fixed32, NDIM>> particles_fetch_cache_line(part_int index);
static const array<fixed32, NDIM>* particles_cache_read_line(line_id_type line_id);
void particles_cache_free();

static vector<group_particle> particles_group_fetch_cache_line(part_int index);
static const group_particle* particles_group_cache_read_line(line_id_type line_id);
void particles_group_cache_free();

static void particles_set_global_offset(vector<part_int>);

static part_int size = 0;
static part_int capacity = 0;
static vector<part_int> global_offsets;

HPX_PLAIN_ACTION (particles_cache_free);
HPX_PLAIN_ACTION (particles_inc_group_cache_epoch);
HPX_PLAIN_ACTION (particles_destroy);
HPX_PLAIN_ACTION (particles_fetch_cache_line);
HPX_PLAIN_ACTION (particles_group_refresh_cache_line);
HPX_PLAIN_ACTION (particles_group_fetch_cache_line);
HPX_PLAIN_ACTION (particles_random_init);
HPX_PLAIN_ACTION (particles_sample);
HPX_PLAIN_ACTION (particles_groups_init);
HPX_PLAIN_ACTION (particles_groups_destroy);
HPX_PLAIN_ACTION (particles_set_global_offset);

struct line_id_type {
	int proc;
	part_int index;
	inline bool operator==(line_id_type other) const {
		return proc == other.proc && index == other.index;
	}
};

struct line_id_hash {
	inline size_t operator()(line_id_type id) const {
		const part_int line_size = get_options().part_cache_line_size;
		const part_int i = id.index / line_size;
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
static int group_cache_epoch = 0;

struct group_cache_entry {
	hpx::shared_future<vector<group_particle>> data;
	int epoch;
};
static array<std::unordered_map<line_id_type, group_cache_entry, line_id_hash_hi>, PART_CACHE_SIZE> group_part_cache;
static array<spinlock_type, PART_CACHE_SIZE> group_mutexes;

std::unordered_map<int, part_int> particles_groups_init() {
	vector<hpx::future<std::unordered_map<int, part_int>>>futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < particles_groups_init_action > (c));
	}

	ALWAYS_ASSERT(!particles_grp);
	particles_grp = new std::atomic<group_int>[size];
	for( int i = 0; i < particles_size(); i++) {
		particles_grp[i] = NO_GROUP;
	}
	group_cache_epoch = 0;

	std::unordered_map<int, part_int> map;
	map[hpx_rank()] = particles_size();
	for (auto& f : futs) {
		auto this_map = f.get();
		for (auto i = this_map.begin(); i != this_map.end(); i++) {
			map[i->first] = i->second;
		}
	}

	if( hpx_rank() == 0 ) {
		vector<size_t> offsets(hpx_size());
		offsets[0] = 0;
		for( int i = 0; i < hpx_size() - 1; i++) {
			offsets[i + 1] = map[i] + offsets[i];
		}
		for( int i = 0; i < hpx_size(); i++) {
			map[i] = offsets[i];
		}
		vector<part_int> vmap(hpx_size());
		for( int i = 0; i < hpx_size(); i++) {
			vmap[i] = map[i];
		}
		particles_set_global_offset(std::move(vmap));
	}

	return map;
}

int particles_group_home(group_int grp) {
	int begin = 0;
	int end = hpx_size();
	while (end - begin > 1) {
		int mid = (begin + end) / 2;
		if (grp >= global_offsets[mid]) {
			begin = mid;
		} else {
			end = mid;
		}
	}
	return begin;
}

static void particles_set_global_offset(vector<part_int> map) {
	particles_global_offset = map[hpx_rank()];
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < particles_set_global_offset_action > (c, map));
	}
	particles_global_offset = map[hpx_rank()];
	global_offsets = std::move(map);
	hpx::wait_all(futs.begin(), futs.end());
}

void particles_groups_destroy() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < particles_groups_destroy_action > (c));
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	vector<hpx::future<void>> futs2;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([nthreads,proc]() {
			const part_int begin = (size_t) proc * particles_size() / nthreads;
			const part_int end = (size_t) (proc + 1) * particles_size() / nthreads;
			for( part_int i = begin; i < end; i++) {
				particles_lastgroup(i) = particles_group(i);
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	delete[] (particles_grp);
	particles_grp = nullptr;
	group_part_cache = decltype(group_part_cache)();
	hpx::wait_all(futs.begin(), futs.end());

}

void particles_cache_free() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < particles_cache_free_action > (c));
	}
	for (int i = 0; i < PART_CACHE_SIZE; i++) {
		part_cache[i] = std::unordered_map<line_id_type, hpx::shared_future<vector<array<fixed32, NDIM>>> , line_id_hash_hi>();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void particles_global_read_pos_and_group(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, group_int* g, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			std::memcpy(x + offset, &particles_pos(XDIM, range.range.first), sizeof(float) * sz);
			std::memcpy(y + offset, &particles_pos(YDIM, range.range.first), sizeof(float) * sz);
			std::memcpy(z + offset, &particles_pos(ZDIM, range.range.first), sizeof(float) * sz);
			for (int i = 0; i < sz; i++) {
				g[i + offset] = particles_group(range.range.first + i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = particles_group_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					x[dest_index] = ptr[src_index].x[XDIM];
					y[dest_index] = ptr[src_index].x[YDIM];
					z[dest_index] = ptr[src_index].x[ZDIM];
					g[dest_index] = ptr[src_index].g;
					dest_index++;
				}
			}
		}
	}
}

void particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			std::memcpy(x + offset, &particles_pos(XDIM, range.range.first), sizeof(float) * sz);
			std::memcpy(y + offset, &particles_pos(YDIM, range.range.first), sizeof(float) * sz);
			std::memcpy(z + offset, &particles_pos(ZDIM, range.range.first), sizeof(float) * sz);
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = particles_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					x[dest_index] = ptr[src_index][XDIM];
					y[dest_index] = ptr[src_index][YDIM];
					z[dest_index] = ptr[src_index][ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static const group_particle* particles_group_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = group_part_cache[bin].find(line_id);
	const group_particle* ptr;
	if (iter == group_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<group_particle>> >();
		auto& entry = group_part_cache[bin][line_id];
		entry.data = prms->get_future();
		entry.epoch = group_cache_epoch;
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<particles_group_fetch_cache_line_action>(hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = group_part_cache[bin].find(line_id);
	} else if (iter->second.epoch < group_cache_epoch) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<group_particle>> >();
		auto old_fut = std::move(iter->second.data);
		auto& entry = group_part_cache[bin][line_id];
		entry.data = prms->get_future();
		entry.epoch = group_cache_epoch;
		lock.unlock();
		auto old_data = old_fut.get();
		hpx::apply([prms,line_id](vector<group_particle> data) {
			auto grp_fut = hpx::async<particles_group_refresh_cache_line_action>(hpx_localities()[line_id.proc],line_id.index);
			const auto grps = grp_fut.get();
			for( int i = 0; i < grps.size(); i++) {
				data[i].g = grps[i];
			}
			prms->set_value(std::move(data));
		}, std::move(old_data));
		lock.lock();
		iter = group_part_cache[bin].find(line_id);
	}
	auto fut = iter->second.data;
	lock.unlock();
	return fut.get().data();
}

static const array<fixed32, NDIM>* particles_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = part_cache[bin].find(line_id);
	const array<fixed32, NDIM>* ptr;
	if (iter == part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<array<fixed32, NDIM>>> >();
		part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
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

static vector<array<fixed32, NDIM>> particles_fetch_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<array<fixed32, NDIM>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin][dim] = particles_pos(dim, i);
		}
	}
	return line;
}

static vector<group_particle> particles_group_fetch_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<group_particle> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin].x[dim] = particles_pos(dim, i);
		}
		line[i - begin].g = particles_group(i);
	}
	return line;
}

void particles_inc_group_cache_epoch() {
	const part_int line_size = get_options().part_cache_line_size;
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async < particles_inc_group_cache_epoch_action > (c));
	}
	group_cache_epoch++;
	hpx::wait_all(futs.begin(), futs.end());
}

static vector<group_int> particles_group_refresh_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<group_int> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = (group_int) particles_group(i);
	}
	return line;
}

void particles_destroy() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async < particles_destroy_action > (c));
	}
	particles_x = decltype(particles_x)();
	particles_g = decltype(particles_g)();
	particles_v = decltype(particles_v)();
	particles_p = decltype(particles_p)();
	particles_r = decltype(particles_r)();
	hpx::wait_all(futs.begin(), futs.end());
}

part_int particles_size() {
	return size;
}

template<class T>
void array_resize(T*& ptr, part_int new_capacity, bool reg) {
	T* new_ptr;
	if (capacity > 0) {
#ifdef USE_CUDA
		if( reg ) {
			CUDA_CHECK(cudaHostUnregister(ptr));
		}
#endif
	}
	new_ptr = (T*) malloc(sizeof(T) * new_capacity);
#ifdef USE_CUDA
	if( reg ) {
		CUDA_CHECK(cudaHostRegister(new_ptr, new_capacity * sizeof(T), cudaHostRegisterPortable | cudaHostRegisterMapped));
	}
#endif
	if (capacity > 0) {
		hpx_copy(PAR_EXECUTION_POLICY, ptr, ptr + size, new_ptr).get();
		free(ptr);
	}
	ptr = new_ptr;

}

void particles_resize(part_int sz) {
	if (sz > capacity) {
		part_int new_capacity = std::max(capacity, 20);
		while (new_capacity < sz) {
			new_capacity = size_t(21) * new_capacity / size_t(20);
		}
		PRINT("Resizing particles to %li from %li\n", new_capacity, capacity);
		for (int dim = 0; dim < NDIM; dim++) {
			array_resize(particles_x[dim], new_capacity, false);
			array_resize(particles_v[dim], new_capacity, true);
		}
		array_resize(particles_r, new_capacity, true);
		if (get_options().do_groups) {
			array_resize(particles_lgrp, new_capacity, false);
			for (part_int i = 0; i < new_capacity; i++) {
				particles_lgrp[i] = NO_GROUP;
			}
		}
		if (get_options().save_force) {
			for (int dim = 0; dim < NDIM; dim++) {
				array_resize(particles_g[dim], new_capacity, true);
			}
			array_resize(particles_p, new_capacity, true);
		}
		capacity = new_capacity;
	}
	size = sz;
}

void particles_random_init() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async < particles_random_init_action > (c));
	}
	const size_t total_num_parts = std::pow(get_options().parts_dim, NDIM);
	const size_t begin = (size_t)(hpx_rank()) * total_num_parts / hpx_size();
	const size_t end = (size_t)(hpx_rank() + 1) * total_num_parts / hpx_size();
	const size_t my_num_parts = end - begin;
	particles_resize(my_num_parts);
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads]() {
			const part_int begin = (size_t) proc * particles_size() / nthreads;
			const part_int end = (size_t) (proc+1) * particles_size() / nthreads;
			const int seed = 4321*(hpx_rank() * nthreads + proc) + 42;
			gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
			gsl_rng_set(rndgen, seed);
			for (part_int i = begin; i < end; i++) {
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

part_int particles_sort(pair<part_int> rng, double xm, int xdim) {
	part_int begin = rng.first;
	part_int end = rng.second;
	part_int lo = begin;
	part_int hi = end;
	fixed32 xmid(xm);
	const bool do_groups = get_options().do_groups;
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
					if (do_groups) {
						std::swap(particles_lgrp[hi], particles_lgrp[lo]);
					}
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
		const part_int begin = size_t(cnt) * nparts / (children.size() + 1);
		const part_int end = size_t(cnt + 1) * nparts / (children.size() + 1);
		const part_int this_cnt = end - begin;
		futs.push_back(hpx::async < particles_sample_action > (children[i], this_cnt));
	}
	const part_int begin = size_t(children.size()) * nparts / (children.size() + 1);
	const part_int this_cnt = nparts - begin;
	vector<particle_sample> parts;
	const int seed = 4321 * hpx_rank() + 42;
	gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rndgen, seed);
	for (part_int i = 0; i < this_cnt; i++) {
		particle_sample sample;
		const part_int index = (gsl_rng_get(rndgen) * gsl_rng_get(rndgen)) % particles_size();
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
	part_int size;
	FREAD(&size, sizeof(part_int), 1, fp);
	particles_resize(size);
	FREAD(&particles_pos(XDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_pos(YDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_pos(ZDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_vel(XDIM, 0), sizeof(float), particles_size(), fp);
	FREAD(&particles_vel(YDIM, 0), sizeof(float), particles_size(), fp);
	FREAD(&particles_vel(ZDIM, 0), sizeof(float), particles_size(), fp);
	FREAD(&particles_rung(0), sizeof(char), particles_size(), fp);
	if (get_options().do_groups) {
		FREAD(&particles_lastgroup(0), sizeof(group_int), particles_size(), fp);
	}
}

void particles_save(std::ofstream& fp) {
	part_int size = particles_size();
	fp.write((const char*) &size, sizeof(part_int));
	fp.write((const char*) &particles_pos(XDIM, 0), sizeof(fixed32) * particles_size());
	fp.write((const char*) &particles_pos(YDIM, 0), sizeof(fixed32) * particles_size());
	fp.write((const char*) &particles_pos(ZDIM, 0), sizeof(fixed32) * particles_size());
	fp.write((const char*) &particles_vel(XDIM, 0), sizeof(float) * particles_size());
	fp.write((const char*) &particles_vel(YDIM, 0), sizeof(float) * particles_size());
	fp.write((const char*) &particles_vel(ZDIM, 0), sizeof(float) * particles_size());
	fp.write((const char*) &particles_rung(0), sizeof(char) * particles_size());
	if (get_options().do_groups) {
		fp.write((const char*) &particles_lastgroup(0), sizeof(group_int) * particles_size());
	}

}
