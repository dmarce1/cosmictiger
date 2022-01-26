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

#define SPH_PARTICLES_CPP
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/hpx.hpp>

static part_int capacity = 0;
static part_int size = 0;

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

static const array<fixed32, NDIM>* sph_particles_cache_read_line(line_id_type line_id);
static const sph_particle* sph_particles_sph_cache_read_line(line_id_type line_id);
static const pair<char, float>* sph_particles_rung_cache_read_line(line_id_type line_id);

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index);
static vector<pair<char, float>> sph_particles_fetch_rung_cache_line(part_int index);
static vector<sph_particle> sph_particles_fetch_sph_cache_line(part_int index);

HPX_PLAIN_ACTION (sph_particles_fetch_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_sph_cache_line);
HPX_PLAIN_ACTION (sph_particles_fetch_rung_cache_line);
HPX_PLAIN_ACTION (sph_particles_cache_free);

static array<std::unordered_map<line_id_type, hpx::shared_future<vector<array<fixed32, NDIM>>> , line_id_hash_hi>,PART_CACHE_SIZE> part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<sph_particle>>, line_id_hash_hi>, PART_CACHE_SIZE> sph_part_cache;
static array<std::unordered_map<line_id_type, hpx::shared_future<vector<pair<char, float>>> , line_id_hash_hi>, PART_CACHE_SIZE> rung_part_cache;
static array<spinlock_type, PART_CACHE_SIZE> mutexes;
static array<spinlock_type, PART_CACHE_SIZE> sph_mutexes;
static array<spinlock_type, PART_CACHE_SIZE> rung_mutexes;
static int group_cache_epoch = 0;

struct sph_sort_particle {
	part_int dm_index;
	float ent;
	float h;
};

struct sph_particle_ref {
	int index;
	bool operator<(const sph_particle_ref& other) const {
		return sph_particles_dm_index(index) < sph_particles_dm_index(other.index);
	}
	bool operator<(const sph_sort_particle& other) const {
		return sph_particles_dm_index(index) < other.dm_index;
	}
	operator sph_sort_particle() const {
		sph_sort_particle p;
		p.dm_index = sph_particles_dm_index(index);
		p.ent = sph_particles_ent(index);
		p.h = sph_particles_smooth_len(index);
		return p;
	}
	sph_particle_ref operator=(const sph_sort_particle& p) {
		sph_particles_dm_index(index) = p.dm_index;
		sph_particles_ent(index) = p.ent;
		sph_particles_smooth_len(index) = p.h;
		return *this;
	}
};

bool operator<(const sph_sort_particle& a, const sph_sort_particle& b) {
	return a.dm_index < b.dm_index;
}

void swap(sph_particle_ref a, sph_particle_ref b) {
	sph_sort_particle c = (sph_sort_particle) a;
	a = (sph_sort_particle) b;
	b = c;
}

struct sph_particle_iterator {
	using iterator_category = std::random_access_iterator_tag;
	using difference_type = int;
	using value_type = sph_sort_particle;
	using pointer = int;  // or also value_type*
	using reference = sph_particle_ref&;  // or also value_type&
	int index;
	sph_particle_ref operator*() const {
		sph_particle_ref ref;
		ref.index = index;
		return ref;
	}
	int operator-(const sph_particle_iterator& other) const {
		return index - other.index;
	}
	sph_particle_iterator operator+(int i) const {
		sph_particle_iterator j;
		j.index = index + i;
		return j;
	}
	sph_particle_iterator operator-(int i) const {
		sph_particle_iterator j;
		j.index = index - i;
		return j;
	}
	sph_particle_iterator& operator--() {
		index--;
		return *this;
	}
	sph_particle_iterator& operator--(int) {
		index--;
		return *this;
	}
	sph_particle_iterator& operator++() {
		index++;
		return *this;
	}
	sph_particle_iterator& operator++(int) {
		index++;
		return *this;
	}
	bool operator!=(const sph_particle_iterator& other) const {
		return index != other.index;
	}
	bool operator==(const sph_particle_iterator& other) const {
		return index == other.index;
	}
	bool operator<(const sph_particle_iterator& other) const {
		return index < other.index;
	}
};

void sph_particles_sort_by_particles(pair<part_int> rng) {
	sph_particle_iterator b;
	sph_particle_iterator e;
	b.index = rng.first;
	e.index = rng.second;
	std::sort(b, e);
}

part_int sph_particles_sort(pair<part_int> rng, fixed32 xmid, int xdim) {
	part_int begin = rng.first;
	part_int end = rng.second;
	part_int lo = begin;
	part_int hi = end;
	const bool do_groups = get_options().do_groups;
	const bool do_tracers = get_options().do_tracers;
	while (lo < hi) {
		if (sph_particles_pos(xdim, lo) >= xmid) {
			while (lo != hi) {
				hi--;
				if (sph_particles_pos(xdim, hi) < xmid) {
					std::swap(sph_particles_e[hi], sph_particles_e[lo]);
					std::swap(sph_particles_h[hi], sph_particles_h[lo]);
					std::swap(sph_particles_dm[hi], sph_particles_dm[lo]);
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

part_int sph_particles_size() {
	return size;
}

template<class T>
void sph_particles_array_resize(T*& ptr, part_int new_capacity, bool reg) {
	T* new_ptr;
#ifdef USE_CUDA
	if( reg ) {
		cudaMallocManaged(&new_ptr,sizeof(T) * new_capacity);
	} else {
		new_ptr = (T*) malloc(sizeof(T) * new_capacity);
	}
#else
	new_ptr = (T*) malloc(sizeof(T) * new_capacity);
#endif
	if (capacity > 0) {
		hpx_copy(PAR_EXECUTION_POLICY, ptr, ptr + size, new_ptr).get();
#ifdef USE_CUDA
		if( reg ) {
			cudaFree(ptr);
		} else {
			free(ptr);
		}
#else
		free(ptr);
#endif
	}
	ptr = new_ptr;

}

void sph_particles_resize(part_int sz) {
	if (sz > capacity) {
		part_int new_capacity = std::max(capacity, (part_int) 100);
		while (new_capacity < sz) {
			new_capacity = size_t(101) * new_capacity / size_t(100);
		}
		PRINT("Resizing sph_particles to %li from %li\n", new_capacity, capacity);
		sph_particles_array_resize(sph_particles_dm, new_capacity, false);
		sph_particles_array_resize(sph_particles_e, new_capacity, false);
		sph_particles_array_resize(sph_particles_h, new_capacity, false);
		sph_particles_array_resize(sph_particles_de, new_capacity, false);
		sph_particles_array_resize(sph_particles_sa, new_capacity, false);
		sph_particles_array_resize(sph_particles_fv, new_capacity, false);
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_array_resize(sph_particles_dv[dim], new_capacity, true);
		}
		capacity = new_capacity;
	}
	part_int new_parts = sz - size;
	part_int offset = particles_size();
	particles_resize(particles_size() + new_parts);
	int oldsz = size;
	size = sz;
	for (int i = 0; i < new_parts; i++) {
		particles_sph_index(offset + i) = oldsz + i;
		sph_particles_dm_index(oldsz + i) = offset + i;
		for (int dim = 0; dim < NDIM; dim++) {
			sph_particles_gforce(dim, oldsz + i) = 0.0f;
		}
	}
}

void sph_particles_free() {
	free(sph_particles_fv);
	free(sph_particles_dm);
	free(sph_particles_sa);
	free(sph_particles_h);
	free(sph_particles_e);
	free(sph_particles_de);
	for (int dim = 0; dim < NDIM; dim++) {
		free(sph_particles_dv[NDIM]);
	}
}

void sph_particles_resolve_with_particles() {
	const int nthread = hpx_hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthread; proc++) {
		futs.push_back(hpx::async([proc, nthread] {
			const part_int b = (size_t) proc * sph_particles_size() / nthread;
			const part_int e = (size_t) (proc + 1) * sph_particles_size() / nthread;
			for( part_int i = b; i < e; i++) {
				particles_sph_index(sph_particles_dm_index(i)) = i;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void sph_particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				x[j] = sph_particles_pos(XDIM, i);
				y[j] = sph_particles_pos(YDIM, i);
				z[j] = sph_particles_pos(ZDIM, i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_cache_read_line(line_id);
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

static const array<fixed32, NDIM>* sph_particles_cache_read_line(line_id_type line_id) {
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
			auto line_fut = hpx::async<sph_particles_fetch_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<array<fixed32, NDIM>> sph_particles_fetch_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<array<fixed32, NDIM>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			line[i - begin][dim] = sph_particles_pos(dim, i);
		}
	}
	return line;
}

void sph_particles_global_read_sph(particle_global_range range, vector<float>& ent, vector<float>& fvel, vector<float>& vx, vector<float>& vy,
		vector<float>& vz, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	const int sz = offset + range.range.second - range.range.first;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				ent[j] = sph_particles_ent(i);
				vx[j] = sph_particles_vel(XDIM, i);
				vy[j] = sph_particles_vel(YDIM, i);
				vz[j] = sph_particles_vel(ZDIM, i);
				fvel[j] = sph_particles_fvel(i);
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_sph_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					const sph_particle& part = ptr[src_index];
					ent[dest_index] = part.ent;
					vx[dest_index] = part.v[XDIM];
					vy[dest_index] = part.v[YDIM];
					vz[dest_index] = part.v[ZDIM];
					fvel[dest_index] = part.fvel;
					dest_index++;
				}
			}
		}
	}
}

static const sph_particle* sph_particles_sph_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(sph_mutexes[bin]);
	auto iter = sph_part_cache[bin].find(line_id);
	if (iter == sph_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<sph_particle>> >();
		sph_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_sph_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = sph_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<sph_particle> sph_particles_fetch_sph_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<sph_particle> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin] = sph_particles_get_particle(i);
	}
	return line;
}

void sph_particles_global_read_rungs_and_smoothlens(particle_global_range range, vector<char>& rungs, vector<float>& hs, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			for (part_int i = range.range.first; i < range.range.second; i++) {
				const int j = offset + i - range.range.first;
				if (rungs.size()) {
					rungs[j] = sph_particles_rung(i);
				}
				if (hs.size()) {
					hs[j] = sph_particles_smooth_len(i);
				}
			}
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = sph_particles_rung_cache_read_line(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					if (rungs.size()) {
						rungs[dest_index] = ptr[src_index].first;
					}
					if (hs.size()) {
						hs[dest_index] = ptr[src_index].second;
					}
					dest_index++;
				}
			}
		}
	}
}

static const pair<char, float>* sph_particles_rung_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(rung_mutexes[bin]);
	auto iter = rung_part_cache[bin].find(line_id);
	if (iter == rung_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<vector<pair<char, float>>> >();
		rung_part_cache[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<sph_particles_fetch_rung_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = rung_part_cache[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<pair<char, float>> sph_particles_fetch_rung_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<pair<char, float>> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		line[i - begin].first = sph_particles_rung(i);
		line[i - begin].second = sph_particles_smooth_len(i);
	}
	return line;
}

void sph_particles_cache_free() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_particles_cache_free_action>(HPX_PRIORITY_HI, c));
	}
	part_cache = decltype(part_cache)();
	sph_part_cache = decltype(sph_part_cache)();
	rung_part_cache = decltype(rung_part_cache)();
	hpx::wait_all(futs.begin(), futs.end());
}

