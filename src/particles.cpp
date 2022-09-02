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

constexpr bool verbose = true;
#define PARTICLES_CPP

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/flops.hpp>

#include <gsl/gsl_rng.h>

#include <unordered_map>

struct line_id_type;

static vector<group_int> particles_group_refresh_cache_line(part_int index);

static vector<group_particle> particles_group_fetch_cache_line(part_int index);
static const group_particle* particles_group_cache_read_line(line_id_type line_id);
void particles_group_cache_free();

struct particles_cache_entry {
	array<fixed32, NDIM> x;
	char type;
	float zeta1;
	float zeta2;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & type;
		arc & x;
		arc & zeta1;
		arc & zeta2;
	}
};

static vector<particles_cache_entry> particles_fetch_cache_line(part_int index);
static const particles_cache_entry* particles_cache_read_line(line_id_type line_id);
void particles_cache_free();

static vector<char> particles_fetch_cache_line_rungs(part_int index);
static const char* particles_cache_read_line_rungs(line_id_type line_id);

static void particles_set_global_offset(vector<size_t>);

static shared_mutex_type shared_mutex;

static part_int size = 0;
static part_int capacity = 0;
static vector<size_t> global_offsets;
static part_int rung_begin;
static std::vector<part_int> rung_begins;

HPX_PLAIN_ACTION (particles_cache_free);
HPX_PLAIN_ACTION (particles_inc_group_cache_epoch);
HPX_PLAIN_ACTION (particles_destroy);
HPX_PLAIN_ACTION (particles_fetch_cache_line_rungs);
HPX_PLAIN_ACTION (particles_fetch_cache_line);
HPX_PLAIN_ACTION (particles_group_refresh_cache_line);
HPX_PLAIN_ACTION (particles_group_fetch_cache_line);
HPX_PLAIN_ACTION (particles_random_init);
HPX_PLAIN_ACTION (particles_sample);
HPX_PLAIN_ACTION (particles_groups_init);
HPX_PLAIN_ACTION (particles_groups_destroy);
HPX_PLAIN_ACTION (particles_set_global_offset);
HPX_PLAIN_ACTION (particles_set_tracers);
HPX_PLAIN_ACTION (particles_get_tracers);
HPX_PLAIN_ACTION (particles_get_sample);
/*
 struct particle_ref {
 int index;
 bool operator<(const particle_ref& other) const {
 return particles_sph_index(index) < particles_sph_index(other.index);
 }
 bool operator<(const particle& other) const {
 return particles_sph_index(index) < other.sph_index;
 }
 operator particle() const {
 return particles_get_particle(index);
 }
 particle_ref operator=(const particle& other) {
 particles_set_particle(other, index);
 return *this;
 }
 };

 bool operator<(const particle& a, const particle& b) {
 return a.sph_index < b.sph_index;
 }

 void swap(particle_ref a, particle_ref b) {
 particle c = (particle) a;
 a = (particle) b;
 b = c;
 }

 struct particle_iterator {
 using iterator_category = std::random_access_iterator_tag;
 using difference_type = int;
 using value_type = particle;
 using pointer = int;  // or also value_type*
 using reference = particle_ref&;  // or also value_type&
 int index;
 particle_ref operator*() const {
 particle_ref ref;
 ref.index = index;
 return ref;
 }
 int operator-(const particle_iterator& other) const {
 return index - other.index;
 }
 particle_iterator operator+(int i) const {
 particle_iterator j;
 j.index = index + i;
 return j;
 }
 particle_iterator operator-(int i) const {
 particle_iterator j;
 j.index = index - i;
 return j;
 }
 particle_iterator& operator--() {
 index--;
 return *this;
 }
 particle_iterator& operator--(int) {
 index--;
 return *this;
 }
 particle_iterator& operator++() {
 index++;
 return *this;
 }
 particle_iterator& operator++(int) {
 index++;
 return *this;
 }
 bool operator!=(const particle_iterator& other) const {
 return index != other.index;
 }
 bool operator==(const particle_iterator& other) const {
 return index == other.index;
 }
 bool operator<(const particle_iterator& other) const {
 return index < other.index;
 }
 };
 */

HPX_PLAIN_ACTION (particles_pop_rungs);
HPX_PLAIN_ACTION (particles_push_rungs);
HPX_PLAIN_ACTION (particles_active_pct);

#ifdef TREEPM
HPX_PLAIN_ACTION (particles_active_count);

size_t particles_active_count() {
	vector < hpx::future < size_t >> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_active_count_action>(c));
	}
	auto rng = particles_current_range();
	size_t num_active = rng.second - rng.first;
	for (auto& f : futs) {
		num_active += f.get();
	}
	return num_active;
}
#endif

double particles_active_pct() {
	vector<hpx::future<double>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_active_pct_action>(c));
	}
	double num_active = particles_size() - rung_begin;
	for (auto& f : futs) {
		num_active += f.get();
	}
	if (hpx_rank() == 0) {
		num_active /= get_options().nparts;
	}
	return num_active;
}

HPX_PLAIN_ACTION (particles_displace);

void particles_displace(double dx, double dy, double dz) {
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_displace_action>(c, dx, dy, dz));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads,proc, dx, dy, dz]() {
			const part_int begin = (size_t) proc * particles_size() / nthreads;
			const part_int end = (size_t) (proc + 1) * particles_size() / nthreads;
			for( part_int i = begin; i < end; i++) {
				double x = particles_pos(XDIM,i).to_double();
				double y = particles_pos(YDIM,i).to_double();
				double z = particles_pos(ZDIM,i).to_double();
				x = fmod(x + dx, 1.0);
				y = fmod(y + dy, 1.0);
				z = fmod(z + dz, 1.0);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	for (auto& f : futs) {
		f.get();
	}
}

void particles_pop_rungs() {
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_pop_rungs_action>(c));
	}
	rung_begin = rung_begins.back();
	rung_begins.pop_back();
	hpx::wait_all(futs.begin(), futs.end());
}

void particles_push_rungs() {
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_push_rungs_action>(c));
	}
	rung_begins.push_back(rung_begin);
	hpx::wait_all(futs.begin(), futs.end());
}

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

static array<std::unordered_map<line_id_type, hpx::shared_future<vector<particles_cache_entry>>, line_id_hash_hi>, PART_CACHE_SIZE> part_cache;
static array<spinlock_type, PART_CACHE_SIZE> mutexes;
static int group_cache_epoch = 0;

static array<std::unordered_map<line_id_type, hpx::shared_future<vector<char>>, line_id_hash_hi>, PART_CACHE_SIZE> part_cache_rungs;
static array<spinlock_type, PART_CACHE_SIZE> mutexes_rungs;

struct group_cache_entry {
	hpx::shared_future<vector<group_particle>> data;
	int epoch;
};

static array<std::unordered_map<line_id_type, group_cache_entry, line_id_hash_hi>, PART_CACHE_SIZE> group_part_cache;
static array<spinlock_type, PART_CACHE_SIZE> group_mutexes;

/*void particles_sort_by_sph(pair<part_int> rng) {
 particle_iterator b;
 particle_iterator e;
 b.index = rng.first;
 e.index = rng.second;
 std::sort(b, e);
 }*/

vector<output_particle> particles_get_sample(const range<double>& box) {
	vector<hpx::future<vector<output_particle>>>futs;
	vector<output_particle> output;
	for( const auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_get_sample_action>(c, box));
	}
	for( part_int i = 0; i < particles_size(); i++) {
		array<double,NDIM> x;
		for( int dim = 0; dim < NDIM; dim++) {
			x[dim] = particles_pos(dim,i).to_double();
		}
		if( box.contains(x)) {
			output_particle data;
			for( int dim = 0; dim < NDIM; dim++) {
				data.x[dim] = particles_pos(dim,i);
				data.v[dim] = particles_vel(dim,i);
			}
			data.r = particles_rung(i);
			output.push_back(data);
		}
	}
	for( auto& f : futs) {
		auto vec = f.get();
		output.insert(output.end(), vec.begin(), vec.end());
	}
	return std::move(output);
}
vector<output_particle> particles_get_tracers() {
	vector<hpx::future<vector<output_particle>>>futs;
	vector<output_particle> output;
	for( const auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_get_tracers_action>(c));
	}
	for( part_int i = 0; i < particles_size(); i++) {
		if( particles_tracer(i) ) {
			output_particle data;
			for( int dim = 0; dim < NDIM; dim++) {
				data.x[dim] = particles_pos(dim,i);
				data.v[dim] = particles_vel(dim,i);
			}
			data.r = particles_rung(i);
			output.push_back(data);
		}
	}
	for( auto& f : futs) {
		auto vec = f.get();
		output.insert(output.end(), vec.begin(), vec.end());
	}
	return std::move(output);
}

void particles_set_tracers(size_t count) {
	hpx::future<void> fut;
	if (hpx_rank() < hpx_size() - 1) {
		fut = hpx::async<particles_set_tracers_action>(hpx_localities()[hpx_rank() + 1], count + particles_size());
	}

	double particles_per_tracer = get_options().nparts / get_options().tracer_count;
	size_t cycles = count / particles_per_tracer;
	double start = cycles * particles_per_tracer;
	start -= count;
	if (start < 0) {
		start += particles_per_tracer;
	}
	memset(&particles_tracer(0), 0, particles_size());
	for (double r = start; r < particles_size(); r += particles_per_tracer) {
		particles_tracer((part_int) r) = 1;
	}

	if (hpx_rank() < hpx_size() - 1) {
		fut.get();
	}

}

std::unordered_map<int, part_int> particles_groups_init() {
	vector<hpx::future<std::unordered_map<int, part_int>>>futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < particles_groups_init_action > (c));
	}

	ALWAYS_ASSERT(!particles_grp);
	particles_grp = new std::atomic<group_int>[size];
	hpx_fill(PAR_EXECUTION_POLICY, particles_grp, particles_grp + particles_size(), NO_GROUP).get();
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
		particles_set_global_offset(std::move(offsets));
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

static void particles_set_global_offset(vector<size_t> map) {
	particles_global_offset = map[hpx_rank()];
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_set_global_offset_action>(c, map));
	}
	particles_global_offset = map[hpx_rank()];
	global_offsets = std::move(map);
	hpx::wait_all(futs.begin(), futs.end());
}

void particles_groups_destroy() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_groups_destroy_action>(c));
	}
	const int nthreads = hpx_hardware_concurrency();
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
	profiler_enter(__FUNCTION__);
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_cache_free_action>(c));
	}
	part_cache = decltype(part_cache)();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

static std::function<void()> undo_last_memadvise = nullptr;

void particles_memadvise_gpu() {
#ifdef USE_CUDA
	cuda_set_device();
	if( undo_last_memadvise ) {
		undo_last_memadvise();
	}
	int deviceid = cuda_get_device();
	const auto rng = particles_current_range();
	const auto begin = rng.first;
	const auto count = rng.second - begin;
	CUDA_CHECK(cudaMemAdvise(particles_vel_data() + begin, count * sizeof(array<float, NDIM>), cudaMemAdviseSetAccessedBy, deviceid));
	CUDA_CHECK(cudaMemAdvise(&particles_rung(0) + begin, count * sizeof(char), cudaMemAdviseSetAccessedBy, deviceid));
	undo_last_memadvise = [begin, count, deviceid]() {
		CUDA_CHECK(cudaMemAdvise(particles_vel_data() + begin, count * sizeof(array<float, NDIM>), cudaMemAdviseUnsetAccessedBy, deviceid));
		CUDA_CHECK(cudaMemAdvise(&particles_rung(0) + begin, count * sizeof(char), cudaMemAdviseUnsetAccessedBy, deviceid));
	};
#endif
}

void particles_memadvise_cpu() {
#ifdef USE_CUDA
	cuda_set_device();
	int deviceid = cuda_get_device();
	if( undo_last_memadvise ) {
		undo_last_memadvise();
	}
	const auto rng = particles_current_range();
	const auto begin = rng.first;
	const auto count = rng.second - begin;
	CUDA_CHECK(cudaMemAdvise(particles_vel_data() + begin, count * sizeof(array<float, NDIM>), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	CUDA_CHECK(cudaMemAdvise(&particles_rung(0) + begin, count * sizeof(char), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	undo_last_memadvise = [begin, count]() {
		CUDA_CHECK(cudaMemAdvise(particles_vel_data() + begin, count * sizeof(array<float, NDIM>), cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
		CUDA_CHECK(cudaMemAdvise(&particles_rung(0) + begin, count * sizeof(char), cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
	};
	for( int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemPrefetchAsync(&particles_pos(dim,begin), count * sizeof(fixed32), cudaCpuDeviceId, 0));
	}
	CUDA_CHECK(cudaMemPrefetchAsync(&particles_vel(XDIM,begin), count * sizeof(array<float,NDIM>), cudaCpuDeviceId, 0));
	CUDA_CHECK(cudaMemPrefetchAsync(&particles_rung(begin), count * sizeof(char), cudaCpuDeviceId, 0));
#endif
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

static const group_particle* particles_group_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = group_part_cache[bin].find(line_id);
	const group_particle* ptr;
	if (iter == group_part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::promise<vector<group_particle>> >();
		auto& entry = group_part_cache[bin][line_id];
		entry.data = prms->get_future();
		entry.epoch = group_cache_epoch;
		lock.unlock();
		hpx::async(HPX_PRIORITY_HI, [prms,line_id]() {
			auto fut = hpx::async<particles_group_fetch_cache_line_action>(hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(fut.get());
			return 'a';
		});
		lock.lock();
		iter = group_part_cache[bin].find(line_id);
	} else if (iter->second.epoch < group_cache_epoch) {
		auto prms = std::make_shared<hpx::promise<vector<group_particle>> >();
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

HPX_PLAIN_ACTION (particles_rung_counts);

vector<size_t> particles_rung_counts() {
	profiler_enter(__FUNCTION__);
	vector<hpx::future<vector<size_t>>>futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_rung_counts_action>(c));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads]() {
			vector<size_t> counts;
			const part_int b = (size_t) proc * particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * particles_size() / nthreads;
			for( part_int i = b; i < e; i++) {
				const auto rung = particles_rung(i);
				if( counts.size() <= rung ) {
					counts.resize(rung + 1,0);
				}
				counts[rung]++;
			}
			return counts;
		}));
	}
	vector<size_t> counts;
	for (auto& f : futs) {
		auto these_counts = f.get();
		if (counts.size() < these_counts.size()) {
			counts.resize(these_counts.size(), 0);
		}
		for (int i = 0; i < these_counts.size(); i++) {
			counts[i] += these_counts[i];
		}
	}
	if (hpx_rank() == 0) {
		size_t tot = 0;
		for (int i = 0; i < counts.size(); i++) {
			tot += counts[i];
		}
		//	ALWAYS_ASSERT(tot == nparts);
	}
	profiler_exit();
	return counts;
}

HPX_PLAIN_ACTION (particles_set_minrung);

void particles_set_minrung(int minrung) {
	profiler_enter(__FUNCTION__);
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_set_minrung_action>(c, minrung));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,minrung]() {
			const part_int b = (size_t) proc *particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * particles_size() / nthreads;
			for( part_int i = b; i < e; i++) {
				auto& rung = particles_rung(i);
				rung = std::max((int)rung, minrung);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	profiler_exit();
}

void particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			std::memcpy(x + offset, &particles_pos(XDIM, range.range.first), sizeof(fixed32) * sz);
			std::memcpy(y + offset, &particles_pos(YDIM, range.range.first), sizeof(fixed32) * sz);
			std::memcpy(z + offset, &particles_pos(ZDIM, range.range.first), sizeof(fixed32) * sz);
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
					x[dest_index] = ptr[src_index].x[XDIM];
					y[dest_index] = ptr[src_index].x[YDIM];
					z[dest_index] = ptr[src_index].x[ZDIM];
					dest_index++;
				}
			}
		}
	}
}

static const particles_cache_entry* particles_cache_read_line(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = part_cache[bin].find(line_id);
	const pair<array<fixed32, NDIM>, char>* ptr;
	if (iter == part_cache[bin].end()) {
		auto prms = std::make_shared<hpx::promise<vector<particles_cache_entry>> >();
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

static vector<particles_cache_entry> particles_fetch_cache_line(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<particles_cache_entry> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		auto& ln = line[i - begin];
		for (int dim = 0; dim < NDIM; dim++) {
			ln.x[dim] = particles_pos(dim, i);
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
		futs.push_back(hpx::async<particles_inc_group_cache_epoch_action>(c));
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
		futs.push_back(hpx::async<particles_destroy_action>(c));
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
void particles_array_resize(T*& ptr, part_int new_capacity, bool reg) {
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

void particles_resize(part_int sz, bool lock) {
	if (sz > capacity) {
		if (lock) {
			shared_mutex.lock();
		}
		part_int new_capacity = std::max(capacity, (part_int) 100);
		while (new_capacity < sz) {
			new_capacity = size_t(105) * new_capacity / size_t(100);
		}
//		PRINT("%i: Resizing particles to %li from %li\n", hpx_rank(), new_capacity, capacity);
		for (int dim = 0; dim < NDIM; dim++) {
			particles_array_resize(particles_x[dim], new_capacity, true);
			particles_array_resize(particles_v, new_capacity, true);
		}
		particles_array_resize(particles_r, new_capacity, true);
		if (get_options().do_groups) {
			particles_array_resize(particles_lgrp, new_capacity, false);
			for (part_int i = capacity; i < new_capacity; i++) {
				particles_lgrp[i] = NO_GROUP;
			}
		}
		if (get_options().save_force) {
			for (int dim = 0; dim < NDIM; dim++) {
				particles_array_resize(particles_g[dim], new_capacity, true);
			}
			particles_array_resize(particles_p, new_capacity, true);
		}
		if (get_options().do_tracers) {
			particles_array_resize(particles_tr, new_capacity, false);
		}
		capacity = new_capacity;
		if (lock) {
			shared_mutex.unlock();
		}
	}
	int oldsz = size;
	size = sz;
//	PRINT( "Resized particles to %i on %i\n", sz, hpx_rank());
}

void particles_free() {
	for (int dim = 0; dim < NDIM; dim++) {
#ifdef USE_CUDA
		CUDA_CHECK(cudaFree(particles_x[dim]));
#else
		free(particles_x[dim]);
#endif
	}
#ifdef USE_CUDA
	CUDA_CHECK(cudaFree(particles_v));
#else
	free(particles_v[dim]);
#endif
#ifdef USE_CUDA
	CUDA_CHECK(cudaFree(particles_r));
#else
	free(particles_r);
#endif
	if (get_options().do_groups) {
		free(particles_lgrp);
	}
	if (get_options().save_force) {
		for (int dim = 0; dim < NDIM; dim++) {
#ifdef USE_CUDA
			CUDA_CHECK(cudaFree(particles_g[dim]));
#else
			free(particles_g[dim]);
#endif
		}
#ifdef USE_CUDA
		CUDA_CHECK(cudaFree(particles_p));
#else
		free(particles_p);
#endif
	}
	if (get_options().do_tracers) {
		free(particles_tr);
	}
}

void particles_random_init() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<particles_random_init_action>(c));
	}
	const size_t total_num_parts = get_options().nparts;
	const size_t begin = (size_t)(hpx_rank()) * total_num_parts / hpx_size();
	const size_t end = (size_t)(hpx_rank() + 1) * total_num_parts / hpx_size();
	const size_t my_num_parts = end - begin;
	particles_resize(my_num_parts);
	const int nthreads = hpx_hardware_concurrency();
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

pair<array<double, NDIM>, double> particles_enclosing_sphere(pair<part_int> rng) {
	auto tot_rng = particles_current_range();
	auto nparts_tot = tot_rng.second - tot_rng.first;
	const int nthreads = (rng.second - rng.first) * hpx_hardware_concurrency() / nparts_tot;

	const auto func1 = [nthreads,rng](int proc) {
		range<double> box;
		for (int dim = 0; dim < NDIM; dim++) {
			box.begin[dim] = 1.0;
			box.end[dim] = 0.0;
		}
		const auto begin = (size_t) (proc)* (rng.second-rng.first)*nthreads + rng.first;
		const auto end = (size_t) (proc+1)* (rng.second-rng.first)*nthreads + rng.first;
		for (part_int i = begin; i < end; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = particles_pos(dim, i).to_double();
				box.begin[dim] = std::min(box.begin[dim], x);
				box.end[dim] = std::max(box.end[dim], x);
			}
		}
		return box;
	};
	vector<hpx::future<range<double>>>futs1;
	for (int proc = 1; proc < nthreads; proc++) {
		futs1.push_back(hpx::async(func1, proc));
	}
	auto box = func1(0);
	for (auto& fut : futs1) {
		auto this_box = fut.get();
		for (int dim = 0; dim < NDIM; dim++) {
			box.begin[dim] = std::min(box.begin[dim], this_box.begin[dim]);
			box.end[dim] = std::max(box.end[dim], this_box.end[dim]);
		}
	}
	array<double, NDIM> xc;
	for (int dim = 0; dim < NDIM; dim++) {
		xc[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
	}
	const auto func2 = [nthreads,rng,xc](int proc) {
		double rmax = 0.0;
		const auto begin = (size_t) (proc)* (rng.second-rng.first)*nthreads + rng.first;
		const auto end = (size_t) (proc+1)* (rng.second-rng.first)*nthreads + rng.first;
		for (part_int i = begin; i < end; i++) {
			double r2 = 0.0;
			for( int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim,i).to_double();
				r2 += sqr(x-xc[dim]);
			}
			rmax = std::max(rmax, sqrt(r2));
		}
		return rmax;
	};
	vector<hpx::future<double>> futs2;
	for (int proc = 1; proc < nthreads; proc++) {
		futs2.push_back(hpx::async(func2, proc));
	}
	auto rmax = func2(0);
	for (auto& fut : futs2) {
		rmax = std::max(rmax, fut.get());
	}
	pair<array<double, NDIM>, double> rc;
	rc.first = xc;
	rc.second = rmax;
	return rc;
}

void particles_swap(part_int lo, part_int hi) {
	static const bool do_groups = get_options().do_groups;
	static const bool do_tracers = get_options().do_tracers;
	static const bool save_force = get_options().save_force;
	for (int dim = 0; dim < NDIM; dim++) {
		std::swap(particles_x[dim][hi], particles_x[dim][lo]);
	}
	std::swap(particles_v[hi], particles_v[lo]);
	std::swap(particles_r[hi], particles_r[lo]);
	if (do_groups) {
		std::swap(particles_lgrp[hi], particles_lgrp[lo]);
	}
	if (do_tracers) {
		std::swap(particles_tr[hi], particles_tr[lo]);
	}
	if( save_force ) {
		std::swap(particles_p[hi], particles_p[lo]);
		for (int dim = 0; dim < NDIM; dim++) {
			std::swap(particles_g[dim][hi], particles_g[dim][lo]);
		}
	}
}

part_int particles_sort(pair<part_int> rng, double xm, int xdim) {
	part_int begin = rng.first;
	part_int end = rng.second;
	part_int lo = begin;
	part_int hi = end;
	fixed32 xmid(xm);
	const bool do_groups = get_options().do_groups;
	const bool do_tracers = get_options().do_tracers;
	auto& xptr_dim = particles_x[xdim];
	auto& x = particles_x[XDIM];
	auto& y = particles_x[YDIM];
	auto& z = particles_x[ZDIM];
	int flops = 0;
	while (lo < hi) {
		flops++;
		if (xptr_dim[lo] >= xmid) {
			while (lo != hi) {
				hi--;
				flops++;
				if (xptr_dim[hi] < xmid) {
					particles_swap(hi, lo);
					break;
				}
			}
		}
		lo++;
	}
	return hi;
	add_cpu_flops(flops);
}

pair<part_int, part_int> particles_current_range() {
	pair<part_int, part_int> rc;
	rc.first = rung_begin;
	rc.second = particles_size();
	return rc;
}

HPX_PLAIN_ACTION (particles_sort_by_rung);

void particles_sort_by_rung(int minrung) {
	profiler_enter(__FUNCTION__);
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_sort_by_rung_action>(c, minrung));
	}
	if (minrung == 0) {
		rung_begin = 0;
	} else {
		part_int begin;
		part_int end;
		begin = rung_begin;
		end = particles_size();
		part_int lo = begin;
		part_int hi = end;
		while (lo < hi) {
			if (particles_rung(lo) >= minrung) {
				while (lo != hi) {
					hi--;
					if (particles_rung(hi) < minrung) {
						particles_swap(hi,lo);
						break;
					}
				}
			}
			lo++;
		}
		rung_begin = hi;
	}

	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void particles_global_read_rungs(particle_global_range range, char* r, part_int offset) {
	const part_int line_size = get_options().part_cache_line_size;
	if (range.range.first != range.range.second) {
		if (range.proc == hpx_rank()) {
			const part_int dif = offset - range.range.first;
			const part_int sz = range.range.second - range.range.first;
			std::memcpy(r + offset, &particles_rung(range.range.first), sizeof(char) * sz);
		} else {
			line_id_type line_id;
			line_id.proc = range.proc;
			const part_int start_line = (range.range.first / line_size) * line_size;
			const part_int stop_line = ((range.range.second - 1) / line_size) * line_size;
			part_int dest_index = offset;
			for (part_int line = start_line; line <= stop_line; line += line_size) {
				line_id.index = line;
				const auto* ptr = particles_cache_read_line_rungs(line_id);
				const auto begin = std::max(line_id.index, range.range.first);
				const auto end = std::min(line_id.index + line_size, range.range.second);
				for (part_int i = begin; i < end; i++) {
					const part_int src_index = i - line_id.index;
					r[dest_index] = ptr[src_index];
					dest_index++;
				}
			}
		}
	}
}

static const char* particles_cache_read_line_rungs(line_id_type line_id) {
	const part_int line_size = get_options().part_cache_line_size;
	const size_t bin = line_id_hash_lo()(line_id);
	std::unique_lock<spinlock_type> lock(mutexes[bin]);
	auto iter = part_cache_rungs[bin].find(line_id);
	if (iter == part_cache_rungs[bin].end()) {
		auto prms = std::make_shared<hpx::promise<vector<char>> >();
		part_cache_rungs[bin][line_id] = prms->get_future();
		lock.unlock();
		hpx::apply([prms,line_id]() {
			auto line_fut = hpx::async<particles_fetch_cache_line_rungs_action>(hpx_localities()[line_id.proc],line_id.index);
			prms->set_value(line_fut.get());
		});
		lock.lock();
		iter = part_cache_rungs[bin].find(line_id);
	}
	auto fut = iter->second;
	lock.unlock();
	return fut.get().data();
}

static vector<char> particles_fetch_cache_line_rungs(part_int index) {
	const part_int line_size = get_options().part_cache_line_size;
	vector<char> line(line_size);
	const part_int begin = (index / line_size) * line_size;
	const part_int end = std::min(particles_size(), begin + line_size);
	for (part_int i = begin; i < end; i++) {
		auto& ln = line[i - begin];
		ln = particles_rung(i);
	}
	return line;
}

vector<particle_sample> particles_sample(int cnt) {
	const bool save_force = get_options().save_force;
	vector<particle_sample> parts;
	vector<hpx::future<vector<particle_sample>>>futs;
	if (hpx_rank() == 0) {
		const auto& localities = hpx_localities();
		for (int i = 1; i < localities.size(); i++) {
			const part_int b = (size_t) i * cnt / localities.size();
			const part_int e = (size_t)(i + 1) * cnt / localities.size();
			const part_int this_cnt = e - b;
			futs.push_back(hpx::async<particles_sample_action>(localities[i], this_cnt));
		}
		const part_int b = 0;
		const part_int e = (size_t)(1) * cnt / localities.size();
		cnt = e - b;
	}
	const int seed = 4321 * hpx_rank() + 42;
	gsl_rng* rndgen = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rndgen, seed);
	PRINT("Selecting %i particles\n", cnt);
	for (part_int i = 0; i < cnt; i++) {
		particle_sample sample;
		const part_int index = ((size_t) gsl_rng_get(rndgen) * (size_t) gsl_rng_get(rndgen)) % particles_size();
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
	PRINT("Done\n");
	for (auto& f : futs) {
		const auto these_parts = f.get();
		parts.insert(parts.end(), these_parts.begin(), these_parts.end());
	}
	return std::move(parts);
}

void particles_load(FILE* fp) {
	part_int size;
	FREAD(&size, sizeof(part_int), 1, fp);
	PRINT("Reading %i total particles.\n", size);
	particles_resize(size);
	PRINT("particles_size  = %i\n", particles_size());
	FREAD(&particles_pos(XDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_pos(YDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(&particles_pos(ZDIM, 0), sizeof(fixed32), particles_size(), fp);
	FREAD(particles_vel_data(), sizeof(array<float,NDIM>), particles_size(), fp);
	FREAD(&particles_rung(0), sizeof(char), particles_size(), fp);
	if (get_options().do_groups) {
		FREAD(&particles_lastgroup(0), sizeof(group_int), particles_size(), fp);
	}
	if (get_options().do_tracers) {
		FREAD(&particles_tracer(0), sizeof(char), particles_size(), fp);
	}
	FREAD(&size, sizeof(int), 1, fp);
	rung_begins.resize(size);
	FREAD(&rung_begin, sizeof(int), 1, fp);
	FREAD(rung_begins.data(), sizeof(int), rung_begins.size(), fp);

}

void particles_save(FILE* fp) {
	part_int size = particles_size();
	fwrite(&size, sizeof(part_int), 1, fp);
	fwrite(&particles_pos(XDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(&particles_pos(YDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(&particles_pos(ZDIM, 0), sizeof(fixed32), particles_size(), fp);
	fwrite(particles_vel_data(), sizeof(array<float, NDIM> ), particles_size(), fp);
	fwrite(&particles_rung(0), sizeof(char), particles_size(), fp);
	if (get_options().do_groups) {
		fwrite(&particles_lastgroup(0), sizeof(group_int), particles_size(), fp);
	}
	if (get_options().do_tracers) {
		fwrite(&particles_tracer(0), sizeof(char), particles_size(), fp);
	}
	size = rung_begins.size();
	fwrite(&size, sizeof(int), 1, fp);
	fwrite(&rung_begin, sizeof(int), 1, fp);
	fwrite(rung_begins.data(), sizeof(int), rung_begins.size(), fp);

}

HPX_PLAIN_ACTION (particles_sum_energies);

energies_t particles_sum_energies() {
	profiler_enter(__FUNCTION__);
	std::vector<hpx::future<energies_t>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_sum_energies_action>(c));
	}
	energies_t energies;
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads,proc]() {
			energies_t energies;
			const part_int b = (size_t) proc * particles_size() / nthreads;
			const part_int e = (size_t) (proc + 1) * particles_size() / nthreads;
			for( part_int i = b; i != e; i++) {
				const float vx = particles_vel(XDIM,i);
				const float vy = particles_vel(YDIM,i);
				const float vz = particles_vel(ZDIM,i);
				if( get_options().save_force ) {
					energies.pot += 0.5 * particles_pot(i);
				}
				energies.kin += 0.5 * (sqr(vx)+sqr(vy)+sqr(vz));

			}
			return energies;
		}));
	}

	for (auto& f : futs) {
		energies += f.get();
	}
	profiler_exit();
	return energies;

}

shared_mutex_type& particles_shared_mutex() {
	return shared_mutex;
}

vector<int> particles_get_local(const vector<pair<part_int>>& ranges) {
	vector<int> rc;
	size_t sz = 0;
	for (const auto& r : ranges) {
		sz += r.second - r.first;
	}

	rc.resize(NDIM * sz);

	const auto func1 = [&ranges,&rc,sz]() {
		std::shared_lock<shared_mutex_type> lock(shared_mutex);
		part_int i = 0;
		for (int j = 0; j < ranges.size() / 2; j++) {
			const auto& r = ranges[j];
			const auto count = r.second - r.first;
			for (int dim = 0; dim < NDIM; dim++) {
				memcpy(&rc[i + dim * sz], &particles_pos(dim, r.first), count * sizeof(fixed32));
			}
			i += count;
		}
	};
	const auto func2 = [&ranges,&rc,sz]() {
		std::shared_lock<shared_mutex_type> lock(shared_mutex);
		part_int i = sz;
		for (int j = ranges.size() - 1; j >= (int) ranges.size() / 2; j--) {
			const auto& r = ranges[j];
			const auto count = r.second - r.first;
			i -= count;
			for (int dim = 0; dim < NDIM; dim++) {
				memcpy(&rc[i + dim * sz], &particles_pos(dim, r.first), count * sizeof(fixed32));
			}
		}
	};
	auto fut1 = hpx::async(func1);
	func2();
	fut1.get();
	return rc;
}

HPX_PLAIN_ACTION (particles_get_local);

hpx::future<vector<int>> particles_get(int rank, const vector<pair<part_int>>& ranges) {
	return hpx::async<particles_get_local_action>(hpx_localities()[rank], ranges);
}

HPX_PLAIN_ACTION (particles_read_gadget4);

gadget_io_header particles_read_gadget4(std::string fname_base) {
	PRINT("Reading gadget4 file\n");
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<particles_read_gadget4_action>(c, fname_base));
	}
	std::string fname = fname_base;
	if (hpx_size() > 1) {
		fname += std::string(".") + std::to_string(hpx_rank());
	}
	FILE* fp = fopen(fname.c_str(), "rb");
	if (!fp) {
		THROW_ERROR("Unable to open %s for writing.\n", fname.c_str());
	}
	int size;
	gadget_io_header header;

	FREAD(&size, sizeof(int), 1, fp);
	PRINT("%i\n", size);
	FREAD(&header, sizeof(gadget_io_header), 1, fp);
	FREAD(&size, sizeof(int), 1, fp);
	PRINT("%i\n", size);

	PRINT("Print nparts = %lli\n", header.npart[1]);
	const auto nparts = header.npart[1];
	particles_resize(nparts);

	FREAD(&size, sizeof(int), 1, fp);
	PRINT("%i\n", size);
	for (part_int i = 0; i < nparts; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			float pos;
			FREAD(&pos, sizeof(float), 1, fp);
			pos /= header.BoxSize;
			particles_pos(dim, i) = pos;
		}
	}
	FREAD(&size, sizeof(int), 1, fp);
	PRINT("%i\n", size);

	FREAD(&size, sizeof(int), 1, fp);
	PRINT("%i\n", size);
	for (part_int i = 0; i < nparts; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			float vel;
			FREAD(&vel, sizeof(float), 1, fp);
			particles_vel(dim, i) = vel;
		}
	}
	const auto a = 1.0 / (header.redshift + 1.0);
	for (int i = 0; i < nparts; i++) {
		//	PRINT( "%e %e %e\n", particles_pos(XDIM,i).to_double(), particles_pos(YDIM,i).to_double(), particles_pos(ZDIM,i).to_double());
		for (int dim = 0; dim < NDIM; dim++) {
			particles_vel(dim, i) *= sqrt(a) * a * 1e5 * get_options().code_to_s / get_options().code_to_cm;
		}
	}
	FREAD(&size, sizeof(int), 1, fp);
	PRINT("%i\n", size);

	fclose(fp);
	hpx::wait_all(futs.begin(), futs.end());
	return header;
}
