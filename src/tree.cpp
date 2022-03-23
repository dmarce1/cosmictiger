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

#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/tree.hpp>

#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

static vector<tree_node> tree_fetch_cache_line(int);
static void tree_allocate_nodes();

HPX_PLAIN_ACTION (tree_allocate_nodes);
HPX_PLAIN_ACTION (tree_create);
HPX_PLAIN_ACTION (tree_destroy);
HPX_PLAIN_ACTION (tree_fetch_cache_line);
//HPX_PLAIN_ACTION (tree_sort_particles_by_sph_particles);

class tree_allocator {
	int next;
	int last;
public:
	bool ready;
	void reset();
	bool is_ready();
	tree_allocator();
	~tree_allocator();
	tree_allocator(const tree_allocator&) = default;
	tree_allocator& operator=(const tree_allocator&) = default;
	tree_allocator(tree_allocator&&) = default;
	tree_allocator& operator=(tree_allocator&&) = default;
	int allocate();
};

static vector<tree_allocator*> allocator_list;
static thread_local tree_allocator allocator;
static vector<tree_node> nodes;
static vector<pair<part_int>> leaf_part_ranges;
static mutex_type leaf_part_range_mutex;
static std::atomic<int> num_threads(0);
static std::atomic<int> next_id;
static array<std::unordered_map<tree_id, hpx::shared_future<vector<tree_node>>, tree_id_hash_hi>, TREE_CACHE_SIZE> tree_cache;
static array<spinlock_type, TREE_CACHE_SIZE> mutex;
static std::atomic<int> allocator_mtx(0);

long long tree_nodes_size() {
	return nodes.size();
}

struct last_cache_entry_t;

static std::unordered_set<last_cache_entry_t*> last_cache_entries;
static std::atomic<int> last_cache_entry_mtx(0);

struct last_cache_entry_t {
	tree_id line;
	const tree_node* ptr;
	void reset() {
		line.proc = line.index = -1;
		ptr = nullptr;
	}
	last_cache_entry_t() {
		line.proc = line.index = -1;
		ptr = nullptr;
		while (last_cache_entry_mtx++ != 0) {
			last_cache_entry_mtx--;
		}
		last_cache_entries.insert(this);
		last_cache_entry_mtx--;
	}
	~last_cache_entry_t() {
		while (last_cache_entry_mtx++ != 0) {
			last_cache_entry_mtx--;
		}
		last_cache_entries.erase(this);
		last_cache_entry_mtx--;
	}
};

static thread_local last_cache_entry_t last_cache_entry;

static void reset_last_cache_entries() {
	for (auto i = last_cache_entries.begin(); i != last_cache_entries.end(); i++) {
		(*i)->reset();
	}
}

static const tree_node* tree_cache_read(tree_id id);

void tree_allocator::reset() {
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	next = (next_id += tree_cache_line_size);
	last = std::min(next + tree_cache_line_size, (int) nodes.size());
	if (next >= nodes.size()) {
		THROW_ERROR("%s\n", "Tree arena full");
	}
}

tree_allocator::tree_allocator() {
	ready = false;
	while (allocator_mtx++ != 0) {
		allocator_mtx--;
	}
	allocator_list.push_back(this);
	allocator_mtx--;
}

int tree_allocator::allocate() {
	if (next == last) {
		reset();
	}
	ASSERT(next < nodes.size());
	return next++;
}

tree_allocator::~tree_allocator() {
	while (allocator_mtx++ != 0) {
		allocator_mtx--;
	}
	for (int i = 0; i < allocator_list.size(); i++) {
		if (allocator_list[i] == this) {
			allocator_list[i] = allocator_list.back();
			allocator_list.pop_back();
			break;
		}
	}
	allocator_mtx--;
}

tree_create_params::tree_create_params(int min_rung_, double theta_, double hmax_) {
	theta = theta_;
	min_rung = min_rung_;
	hmax = hmax_;
	min_level = tree_min_level(theta, hmax);
}

int tree_min_level(double theta, double h) {
	int lev = 1;
	double dx;
	double r1, r2, r;
	do {
		int N = 1 << (lev / NDIM);
		dx = EWALD_DIST * N;
		double a2;
		constexpr double ffac = 1.01;
		if (lev % NDIM == 0) {
			r1 = 2.0f * std::sqrt(3) + ffac * N * h;
			a2 = std::sqrt(3);
		} else if (lev % NDIM == 1) {
			r1 = 2.0f * 1.5 + ffac * N * h;
			a2 = 1.5;
		} else {
			r1 = 2.0f * std::sqrt(1.5) + ffac * N * h;
			a2 = std::sqrt(1.5);
		}
		r2 = (1.0 + SINK_BIAS) * a2 / theta;
		lev++;
		r = std::max(r1, r2);
	} while (dx <= r);
	return 0;
}

fast_future<tree_create_return> tree_create_fork(tree_create_params params, size_t key, const pair<int, int>& proc_range, const pair<part_int>& part_range,
		const range<double>& box, const int depth, const bool local_root, bool threadme) {
	static std::atomic<int> nthreads(0);
	fast_future<tree_create_return> rc;
	bool remote = false;
	if (proc_range.first != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = part_range.second - part_range.first > MIN_SORT_THREAD_PARTS;
		if (threadme) {
			if (nthreads++ < SORT_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency() || proc_range.second - proc_range.first > 1) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		rc.set_value(tree_create(params, key, proc_range, part_range, box, depth, local_root));
	} else if (remote) {
//		PRINT( "%i calling local on %i at %li\n", hpx_rank(), proc_range.first, time(NULL));
		rc = hpx::async<tree_create_action>(HPX_PRIORITY_HI, hpx_localities()[proc_range.first], params, key, proc_range, part_range, box, depth, local_root);
	} else {
		rc = hpx::async([params,proc_range,key,part_range,depth,local_root, box]() {
			auto rc = tree_create(params,key,proc_range,part_range,box,depth,local_root);
			nthreads--;
			return rc;
		});
	}
	return rc;
}

static void tree_allocate_nodes() {
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	static const int bucket_size = BUCKET_SIZE;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<tree_allocate_nodes_action>(HPX_PRIORITY_HI, c));
	}
	next_id = -tree_cache_line_size;
	nodes.resize(std::max(size_t(size_t(TREE_NODE_ALLOCATION_SIZE) * particles_size() / bucket_size), (size_t) NTREES_MIN));
	while (allocator_mtx++ != 0) {
		allocator_mtx--;
	}
	for (int i = 0; i < allocator_list.size(); i++) {
		allocator_list[i]->ready = false;
	}
	allocator_mtx--;
	hpx::wait_all(futs.begin(), futs.end());
}

tree_create_return tree_create(tree_create_params params, size_t key, pair<int, int> proc_range, pair<part_int> part_range, range<double> box, int depth,
		bool local_root) {
	stack_trace_activate();
	const double h = get_options().hsoft;
	static const int bucket_size = BUCKET_SIZE;
	tree_create_return rc;
	const static bool sph = get_options().sph;
	if (depth >= MAX_DEPTH) {
		THROW_ERROR("%s\n", "Maximum depth exceeded\n");
	}
	if (depth == 0) {
		tree_allocate_nodes();
	}
#ifdef USE_CUDA
	cudaStream_t stream;
#endif
	if (local_root) {
//		PRINT("Sorting on %i at %li\n", hpx_rank(), time(NULL));
		part_range.first = 0;
		part_range.second = particles_size();
#ifdef USE_CUDA
		CUDA_CHECK(cudaStreamCreate(&stream));
		CUDA_CHECK(cudaMemPrefetchAsync(&particles_rung(0), particles_size() * sizeof(char), cudaCpuDeviceId, stream));
#endif
	}
	array<tree_id, NCHILD> children;
	array<fixed32, NDIM> x;
	array<double, NDIM> Xc;
	multipole<float> multi;
	size_t nactive;
	int flops = 0;
	double total_flops = 0.0;
	float radius;
	double r;
	int min_depth = depth;
	int max_depth = depth;
	if (!allocator.ready) {
		allocator.reset();
		allocator.ready = true;
	}
	size_t node_count;
	size_t active_nodes = 0;
	size_t leaf_nodes = 0;
	size_t active_leaf_nodes = 0;
	const int index = allocator.allocate();
	double hsoft_max = 0.0;

	if (proc_range.second - proc_range.first > 1 || part_range.second - part_range.first > bucket_size || depth < params.min_level) {
		auto left_box = box;
		auto right_box = box;
		auto left_range = proc_range;
		auto right_range = proc_range;
		auto left_parts = part_range;
		auto right_parts = part_range;
		bool left_local_root = false;
		bool right_local_root = false;
		if (proc_range.second - proc_range.first > 1) {
			const int mid = (proc_range.first + proc_range.second) / 2;
			left_box = domains_range(key << 1);
			right_box = domains_range((key << 1) + 1);
			left_range.second = right_range.first = mid;
			left_local_root = left_range.second - left_range.first == 1;
			right_local_root = right_range.second - right_range.first == 1;
			flops += 7;
		} else {
			const int xdim = box.longest_dim();
			double xmax = box.end[xdim];
			double xmin = box.begin[xdim];
			double nparts_inv = 1.0 / (part_range.second - part_range.first);
			part_int mid;
			double xmid;
			double error;
			double parts_above;
			double parts_below;
			do {
				xmid = 0.5 * (xmax + xmin);
				mid = particles_sort(part_range, xmid, xdim);
				parts_above = part_range.second - mid;
				parts_below = mid - part_range.first;
				error = fabs(parts_above - parts_below) * nparts_inv;
				if (parts_above > parts_below) {
					xmin = xmid;
				} else {
					xmax = xmid;
				}
			} while (error > 0.5);
			left_parts.second = right_parts.first = mid;
			left_box.end[xdim] = right_box.begin[xdim] = xmid;
			flops += 2;
		}
		auto futr = tree_create_fork(params, (key << 1) + 1, right_range, right_parts, right_box, depth + 1, right_local_root, true);
		auto futl = tree_create_fork(params, (key << 1), left_range, left_parts, left_box, depth + 1, left_local_root, false);
		const auto rcl = futl.get();
		const auto rcr = futr.get();
		const auto xl = rcl.pos;
		const auto xr = rcr.pos;
		const auto ml = rcl.multi;
		const auto mr = rcr.multi;
		const double Rl = rcl.radius;
		const double Rr = rcr.radius;
		hsoft_max = std::max(rcl.hsoft_max, rcr.hsoft_max);
		min_depth = std::min(rcl.min_depth, rcr.min_depth);
		max_depth = std::max(rcl.max_depth, rcr.max_depth);
		total_flops += rcl.flops + rcr.flops;
		nactive = rcl.nactive + rcr.nactive;
		double rr;
		double rl;
		array<double, NDIM> Xl;
		array<double, NDIM> Xr;
		array<double, NDIM> Xc;
		array<double, NDIM> N;
		float R;
		double norminv = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			Xl[dim] = xl[dim].to_double();
			Xr[dim] = xr[dim].to_double();
			N[dim] = Xl[dim] - Xr[dim];
			norminv += sqr(N[dim]);
			flops += 5;
		}
		norminv = 1.0 / std::sqrt(norminv);
		flops += 8;
		for (int dim = 0; dim < NDIM; dim++) {
			N[dim] *= norminv;
			flops += 1;
		}
		r = 0.0;
		flops += 2;
		if (mr[0] != 0.0 && ml[0.0] != 0.0) {
			for (int dim = 0; dim < NDIM; dim++) {
				double xmin, xmax;
				flops += 1;
				if (N[dim] > 0.0) {
					xmax = std::max(Xl[dim] + N[dim] * Rl, Xr[dim] + N[dim] * Rr);
					xmin = std::min(Xl[dim] - N[dim] * Rl, Xr[dim] - N[dim] * Rr);
				} else {
					xmax = std::max(Xl[dim] - N[dim] * Rl, Xr[dim] - N[dim] * Rr);
					xmin = std::min(Xl[dim] + N[dim] * Rl, Xr[dim] + N[dim] * Rr);
				}
				Xc[dim] = (xmax + xmin) * 0.5;
				r += sqr((xmax - xmin) * 0.5);
				flops += 14;
			}
		} else if (mr[0] == 0.0 && ml[0.0] == 0) {
			flops += 2;
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
				flops == 2;
			}
		} else if (mr[0] != 0.0) {
			flops += 4;
			Xc = Xr;
			r = Rr * Rr;
		} else {
			flops += 4;
			Xc = Xl;
			r = Rl * Rl;
		}
		radius = std::sqrt(r);
		r = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			r += sqr((box.begin[dim] - box.end[dim]) * 0.5);
		}
		r = std::sqrt(r);
		flops += 8 + 3 * NDIM;
		if (r < radius) {
			radius = r;
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
			}
			flops += 2 * NDIM;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = Xc[dim];
		}
		array<simd_double, NDIM> mdx;
		multipole<simd_double> simdM;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			simdM[i][LEFT] = ml[i];
			simdM[i][RIGHT] = mr[i];
		}
		for (int dim = 0; dim < NDIM; dim++) {
			mdx[dim][LEFT] = Xl[dim] - Xc[dim];
			mdx[dim][RIGHT] = Xr[dim] - Xc[dim];
		}
		flops += 2 * NDIM;
		simdM = M2M<simd_double>(simdM, mdx);
		flops += 1203 * NCHILD;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = simdM[i][LEFT] + simdM[i][RIGHT];
		}
		flops += MULTIPOLE_SIZE;
		children[LEFT] = rcl.id;
		children[RIGHT] = rcr.id;
		node_count = 1 + rcl.node_count + rcr.node_count;
		leaf_nodes = rcl.leaf_nodes + rcr.leaf_nodes;
		active_leaf_nodes = rcl.active_leaf_nodes + rcr.active_leaf_nodes;
		if (rcl.nactive || rcr.nactive) {
			active_nodes += 1 + rcl.active_nodes + rcr.active_nodes;
		}
	} else {
		children[LEFT].index = children[RIGHT].index = -1;
		multipole<double> M;
		array<double, NDIM> Xmax;
		array<double, NDIM> Xmin;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			M[i] = 0.0;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Xmax[dim] = box.begin[dim];
			Xmin[dim] = box.end[dim];
		}
		nactive = 0;
		array<double, NDIM> dx;
		for (part_int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				Xmax[dim] = std::max(Xmax[dim], x);
				Xmin[dim] = std::min(Xmin[dim], x);
			}
			flops += 3 * NDIM;
			if (particles_rung(i) >= params.min_rung) {
				nactive++;
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Xc[dim] = (Xmax[dim] + Xmin[dim]) * 0.5;
		}
		flops += 2 * NDIM;
		const part_int maxi = round_up(part_range.second - part_range.first, (part_int) SIMD_FLOAT_SIZE) + part_range.first;
		array<simd_int, NDIM> Y;
		for (int dim = 0; dim < NDIM; dim++) {
			Y[dim] = fixed32(Xc[dim]).raw();
		}
		const simd_float _2float = fixed2float;
		const double dm_mass = get_options().dm_mass;
		const double sph_mass = get_options().sph_mass;

		for (part_int i = part_range.first; i < maxi; i += SIMD_FLOAT_SIZE) {
			array<simd_int, NDIM> X;
			simd_float mask;
			const int maxj = std::min(i + SIMD_FLOAT_SIZE, part_range.second);
			for (part_int j = i; j < maxj; j++) {
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim][j - i] = particles_pos(dim, j).raw();
				}
				if (sph) {
					mask[j - i] = particles_type(j) == DARK_MATTER_TYPE ? dm_mass : sph_mass;
				} else {
					mask[j - i] = 1.0f;
				}
			}
			for (part_int j = maxj; j < i + SIMD_FLOAT_SIZE; j++) {
				mask[j - i] = 0.f;
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim][j - i] = particles_pos(dim, maxj - 1).raw();
				}
			}
			array < simd_float, NDIM > dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = simd_float(X[dim] - Y[dim]) * _2float;
			}
			flops += SIMD_FLOAT_SIZE * NDIM * 3;
			auto m = P2M(dx);
			flops += 211 * SIMD_FLOAT_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				m[j] *= mask;
			}
			flops += SIMD_FLOAT_SIZE * MULTIPOLE_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[j] += m[j].sum();
			}
			flops += MULTIPOLE_SIZE * (1 + 3);
		}
		r = 0.0;
		for (part_int i = part_range.first; i < part_range.second; i++) {
			double this_radius = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				this_radius += sqr(x - Xc[dim]);
			}
			r = std::max(r, this_radius);
			flops += 3 * NDIM + 1;
		}
		radius = std::sqrt(r);
		r = 0.0;
		for (part_int i = part_range.first; i < part_range.second; i++) {
			double this_radius = 0.0;
			double this_h;
			this_h = h;
			if (sph) {
				if (particles_type(i) == SPH_TYPE) {
					this_h = sph_particles_smooth_len(particles_cat_index(i));
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				this_radius += sqr(x - Xc[dim]);
			}
			r = std::max(r, sqrt(this_radius) + h);
		}
		hsoft_max = r - radius;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = M[i];
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = Xc[dim];
		}
		node_count = 1;
		leaf_nodes = 1;
		if (nactive) {
			active_leaf_nodes++;
			active_nodes++;
		}
	}
	tree_node node;
	node.node_count = node_count;
	node.radius = radius;
	node.children = children;
	node.local_root = local_root;
	node.part_range = part_range;
	node.sink_part_range = part_range;
	node.proc_range = proc_range;
	node.pos = x;
	node.multi = multi;
	node.hsoft_max = hsoft_max;
	node.nactive = nactive;
	node.active_nodes = active_nodes;
	node.depth = depth;
	const part_int nparts = part_range.second - part_range.first;
	const bool global = proc_range.second - proc_range.first > 1;
	node.leaf = !global && (depth >= params.min_level) && (nparts <= BUCKET_SIZE);
	/*	if (sph && SPH_BUCKET_SIZE < BUCKET_SIZE) {
	 if (node.leaf) {
	 std::lock_guard<mutex_type> lock(leaf_part_range_mutex);
	 leaf_part_ranges.push_back(part_range);
	 }
	 }*/
	if (index >= nodes.size()) {
		THROW_ERROR("%s\n", "Tree arena full\n");
	}
	nodes[index] = node;
	rc.active_nodes = active_nodes;
	rc.id.index = index;
	rc.id.proc = hpx_rank();
	rc.multi = node.multi;
	rc.pos = node.pos;
	rc.radius = node.radius;
	rc.hsoft_max = hsoft_max;
	rc.leaf_nodes = leaf_nodes;
	rc.active_leaf_nodes = active_leaf_nodes;
	rc.node_count = node.node_count;
	rc.nactive = nactive;
	total_flops += flops;
	rc.flops = total_flops;
	rc.min_depth = min_depth;
	rc.max_depth = max_depth;
	if (local_root) {
		if (sph) {
			particles_resolve_with_sph_particles();
		}
#ifdef USE_CUDA
		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaStreamDestroy(stream));
		particles_memadvise_gpu();
#endif
	}
	if (depth == 0) {
		PRINT("total nodes = %i\n", rc.node_count);
	}
//		PRINT("%i %e\n", index, nodes[index].radius);
	return rc;
}

void tree_destroy(bool free_tree) {
	profiler_enter(__FUNCTION__);

	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<tree_destroy_action>(HPX_PRIORITY_HI, c, free_tree));
	}
	if (free_tree) {
		nodes = decltype(nodes)();
	}
	tree_cache = decltype(tree_cache)();
	reset_last_cache_entries();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

const tree_node* tree_get_node(tree_id id) {
	if (id.proc == hpx_rank()) {
		ASSERT(id.index >= 0);
#ifndef NDEBUG
		if (id.index >= nodes.size()) {
			THROW_ERROR("id.index is %li but nodes.size() is %li\n", id.index, nodes.size());
		}
#endif
		return &nodes[id.index];
	} else {
		return tree_cache_read(id);
	}
}

static const tree_node* tree_cache_read(tree_id id) {
	const int line_size = get_options().tree_cache_line_size;
	tree_id line_id;
	const tree_node* ptr;
	line_id.proc = id.proc;
	ASSERT(line_id.proc >= 0 && line_id.proc < hpx_size());
	line_id.index = (id.index / line_size) * line_size;
	if (line_id != last_cache_entry.line) {
		const size_t bin = tree_id_hash_lo()(id);
		std::unique_lock<spinlock_type> lock(mutex[bin]);
		auto iter = tree_cache[bin].find(line_id);
		if (iter == tree_cache[bin].end()) {
			auto prms = std::make_shared<hpx::lcos::local::promise<vector<tree_node>>>();
			tree_cache[bin][line_id] = prms->get_future();
			lock.unlock();
			hpx::async(HPX_PRIORITY_HI, [prms,line_id]() {
				const tree_fetch_cache_line_action action;
				auto fut = hpx::async<tree_fetch_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
				prms->set_value(fut.get());
				return 'a';
			});
			lock.lock();
			iter = tree_cache[bin].find(line_id);
		}
		auto fut = iter->second;
		lock.unlock();
		ptr = fut.get().data();
	} else {
		ptr = last_cache_entry.ptr;
	}
	last_cache_entry.ptr = ptr;
	last_cache_entry.line = line_id;
	return ptr + id.index - line_id.index;
}

static vector<tree_node> tree_fetch_cache_line(int index) {
	const int line_size = get_options().tree_cache_line_size;
	vector<tree_node> line;
	line.reserve(line_size);
	const int begin = (index / line_size) * line_size;
	const int end = begin + line_size;
	for (int i = begin; i < end; i++) {
		line.push_back(nodes[i]);
	}
	return std::move(line);

}

/*void tree_sort_particles_by_sph_particles() {
 const int nthreads = hpx_hardware_concurrency();
 vector<hpx::future<void>> futs;
 for (auto c : hpx_children()) {
 futs.push_back(hpx::async<tree_sort_particles_by_sph_particles_action>(c));
 }
 for (int proc = 0; proc < nthreads; proc++) {
 futs.push_back(hpx::async([proc, nthreads]() {
 const int b = (size_t) proc * leaf_part_ranges.size() / nthreads;
 const int e = (size_t) (proc + 1) * leaf_part_ranges.size() / nthreads;
 for( int i = b; i < e; i++) {
 particles_sort_by_sph(leaf_part_ranges[i]);
 }
 }));
 }
 hpx::wait_all(futs.begin(), futs.end());
 leaf_part_ranges.resize(0);
 }*/
