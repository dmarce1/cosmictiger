/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <exception>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>

static vector<tree_node> tree_fetch_cache_line(int);
static void tree_allocate_nodes();

HPX_PLAIN_ACTION(tree_allocate_nodes);
HPX_PLAIN_ACTION(tree_create);
HPX_PLAIN_ACTION(tree_destroy);
HPX_PLAIN_ACTION(tree_fetch_cache_line);

class tree_allocator {
	int next;
	int last;

  public:
	bool ready;
	void reset();
	bool is_ready();
	tree_allocator();
	~tree_allocator();
	tree_allocator(tree_allocator const&) = default;
	tree_allocator& operator=(tree_allocator const&) = default;
	tree_allocator(tree_allocator&&) = default;
	tree_allocator& operator=(tree_allocator&&) = default;
	int allocate();
};

static vector<tree_allocator*> allocator_list;
static thread_local tree_allocator allocator;
static tree_node* nodes = nullptr;
static multi_pos* multis = nullptr;
// Cache lines span reserved allocator slots, including holes between threads.
static std::vector<unsigned char> populatedNodes;
static vector<pair<part_int>> leaf_part_ranges;
static mutex_type leaf_part_range_mutex;
static std::atomic<int> num_threads(0);
static std::atomic<int> next_id;
static std::atomic<int> next_mult_id;
static array<std::unordered_map<tree_id, std::shared_ptr<hpx::shared_future<vector<tree_node>>>, tree_id_hash_hi>, TREE_CACHE_SIZE> tree_cache;
static array<spinlock_type, TREE_CACHE_SIZE> mutex;
static std::atomic<int> allocator_mtx(0);
static size_t nodes_size;

static vector<short> leaf_sizes;
static spinlock_type leaf_sizes_mutex;

long long tree_nodes_size() { return nodes_size; }
struct last_cache_entry_t;

static std::unordered_set<last_cache_entry_t*> last_cache_entries;
static std::atomic<int> last_cache_entry_mtx(0);

struct last_cache_entry_t {
	tree_id line;
	tree_node const* ptr;
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

static tree_node const* tree_cache_read(tree_id id);

multi_pos* tree_add_multipole(multi_pos const& mpos) {
	int const tree_alloc_line_size = get_options().tree_alloc_line_size;
	auto* ptr = &multis[next_mult_id++ + tree_alloc_line_size];
	*ptr = mpos;
	return ptr;
}

void tree_allocator::reset() {
	int const tree_alloc_line_size = get_options().tree_alloc_line_size;
	next = (next_id += tree_alloc_line_size);
	next_mult_id += tree_alloc_line_size;
	last = std::min(next + tree_alloc_line_size, (int)nodes_size);
	if (next >= nodes_size) {
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
	ASSERT(next < nodes_size);
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
	min_level = 9;
	leaf_pushed = false;
	do_leaf_sizes = false;
	par_parts = 0;
}

fast_future<tree_create_return> tree_create_fork(tree_create_params params, size_t key, pair<int, int> const& proc_range, pair<part_int> const& part_range,
												 range<double> const& box, int const depth, bool const local_root, bool threadme) {
	fast_future<tree_create_return> rc;
	bool remote = false;
	if (proc_range.first != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = part_range.second - part_range.first >= params.par_parts;
	}
	if (!threadme) {
		rc.set_value(tree_create(params, key, proc_range, part_range, box, depth, local_root));
	} else if (remote) {
		rc = hpx::async<tree_create_action>(hpx_localities()[proc_range.first], params, key, proc_range, part_range, box, depth, local_root);
	} else {
		rc = hpx::async([params, proc_range, key, part_range, depth, local_root, box]() {
			auto rc = tree_create(params, key, proc_range, part_range, box, depth, local_root);
			return rc;
		});
	}
	return rc;
}

size_t tree_add_remote(tree_node const& remote) {
	int const tree_alloc_line_size = get_options().tree_alloc_line_size;
	if (next_id + tree_alloc_line_size >= nodes_size) {
		PRINT("TREE ALLOCATION EXCEEDED\n");
		abort();
	}
	size_t index = next_id++ + tree_alloc_line_size;
	nodes[index] = remote;
	populatedNodes[index] = true;
	return index;
}

tree_node* tree_data() { return nodes; }

static bool set_gpu = false;
static bool set_cpu = false;

void tree_2_cpu() {
	if (!set_cpu) {
		if (set_gpu) {
			cuda_set_device();
			CUDA_CHECK(cudaMemAdvise(nodes, nodes_size * sizeof(tree_node), cudaMemAdviseUnsetReadMostly, cuda_get_device()));
		}
		set_gpu = false;
		set_cpu = true;
	}
}

void tree_2_gpu() {
	if (!set_gpu) {
		cuda_set_device();
		CUDA_CHECK(cudaMemAdvise(nodes, nodes_size * sizeof(tree_node), cudaMemAdviseSetReadMostly, cuda_get_device()));
		set_gpu = true;
		set_cpu = false;
	}
}

static void tree_allocate_nodes() {
	int const tree_alloc_line_size = get_options().tree_alloc_line_size;
	static int const bucket_size = get_options().bucket_size;
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<tree_allocate_nodes_action>(c));
	}
	next_id = -tree_alloc_line_size;
	next_mult_id = -tree_alloc_line_size;
	size_t const sz = std::max(size_t(size_t(TREE_NODE_ALLOCATION_SIZE) * particles_size() / bucket_size), (size_t)NTREES_MIN);
	if (nodes_size < sz) {
		if (nodes != nullptr) {
			CUDA_CHECK(cudaFree(nodes));
			CUDA_CHECK(cudaFree(multis));
		}
		CUDA_CHECK(cudaMallocManaged(&nodes, sz * sizeof(tree_node)));
		CUDA_CHECK(cudaMallocManaged(&multis, sz * sizeof(multi_pos)));
	}
	nodes_size = sz;
	populatedNodes.assign(sz, 0);
	while (allocator_mtx++ != 0) {
		allocator_mtx--;
	}
	for (int i = 0; i < allocator_list.size(); i++) {
		allocator_list[i]->ready = false;
	}
	allocator_mtx--;
	hpx::wait_all(futs.begin(), futs.end());
	tree_2_cpu();
}

long long tree_nodes_next_index() {
	int const tree_alloc_line_size = get_options().tree_alloc_line_size;
	return next_id + tree_alloc_line_size;
}

tree_create_return tree_create(tree_create_params params, size_t key, pair<int, int> proc_range, pair<part_int> part_range, range<double> box, int depth,
							   bool local_root) {

	if (key == 1) {
		profiler_enter(__FUNCTION__);
	}
	int flops = 0;
	stack_trace_activate();
	double const h = get_options().hsoft;
	int bucket_size = get_options().bucket_size;
	tree_create_return rc;
	if (depth >= MAX_DEPTH) {
		THROW_ERROR("%s\n", "Maximum depth exceeded\n");
	}
	if (depth == 0) {
		tree_allocate_nodes();
	}
	if (local_root) {
		part_range = particles_current_range();
		size_t cnt = part_range.second - part_range.first;
		int nthreads = 1;
		while (nthreads < 4 * hpx_hardware_concurrency()) {
			nthreads *= 2;
			cnt /= 2;
		}
		params.par_parts = cnt;
	}
	array<tree_id, NCHILD> children;
	array<fixed32, NDIM>& x = rc.pos;
	multipole<float>& multi = rc.multi;
	auto& rbox = rc.box;
	float& radius = rc.radius;
	array<double, NDIM> Xc;
	double r;
	if (!allocator.ready) {
		allocator.reset();
		allocator.ready = true;
	}
	int const index = allocator.allocate();
	bool isleaf = true;
	auto const nparts = part_range.second - part_range.first;
	float box_r = 0.0f;
	for (int dim = 0; dim < NDIM; dim++) {
		box_r += sqr(0.5 * (box.end[dim] - box.begin[dim])); // 12
	}
	box_r = sqrt(box_r);																					   // 4
	bool ewald_satisfied = (box_r < 0.25 * (params.theta / (1.0 + params.theta)) && box_r < 0.125 - 0.25 * h); // 10
	double max_ratio = 1.0;
	flops += 26;
	static simd_float const _2float = fixed2float;
	if (proc_range.second - proc_range.first > 1 || nparts > bucket_size || (!ewald_satisfied && nparts > 0)) {
		isleaf = false;
		int const xdim = box.longest_dim();
		auto left_box = box;
		auto right_box = box;
		auto left_range = proc_range;
		auto right_range = proc_range;
		auto left_parts = part_range;
		auto right_parts = part_range;
		bool left_local_root = false;
		bool right_local_root = false;
		if (proc_range.second - proc_range.first > 1) {
			int const mid = (proc_range.first + proc_range.second) / 2;
			left_box = domains_range(key << 1);
			right_box = domains_range((key << 1) + 1);
			left_range.second = right_range.first = mid;
			left_local_root = left_range.second - left_range.first == 1;
			right_local_root = right_range.second - right_range.first == 1;
		} else {
			double const xmid = 0.5 * (box.end[xdim] + box.begin[xdim]);
			flops += 2;
			part_int const mid = particles_sort(part_range, xmid, xdim);
			left_parts.second = right_parts.first = mid;
			left_box.end[xdim] = right_box.begin[xdim] = xmid;
		}
		auto futr = tree_create_fork(params, (key << 1) + 1, right_range, right_parts, right_box, depth + 1, right_local_root, true);
		auto futl = tree_create_fork(params, (key << 1), left_range, left_parts, left_box, depth + 1, left_local_root, false);
		auto const rcl = futl.get();
		auto const rcr = futr.get();
		auto const& xl = rcl.pos;
		auto const& xr = rcr.pos;
		auto const& ml = rcl.multi;
		auto const& mr = rcr.multi;
		double const& Rl = rcl.radius;
		double const& Rr = rcr.radius;
		auto const& boxl = rcl.box;
		auto const& boxr = rcr.box;
		for (int dim = 0; dim < NDIM; dim++) {
			if (ml[0] != 0.0 && mr[0] != 0.0) {
				rbox.begin[dim] = std::min(boxl.begin[dim], boxr.begin[dim]);
				rbox.end[dim] = std::max(boxl.end[dim], boxr.end[dim]);
			} else if (ml[0] != 0.0) {
				rbox = boxl;
			} else {
				rbox = boxr;
			}
		}
		array<double, NDIM> Xl;
		array<double, NDIM> Xr;
		for (int dim = 0; dim < NDIM; dim++) {
			Xl[dim] = xl[dim].to_double(); // 3
			Xr[dim] = xr[dim].to_double(); // 3
		}
		array<double, NDIM> Xc1, Xc2;
		double radius1 = 0.0, radius2;
		bool tried1 = false;
		if (nparts <= 64 * get_options().bucket_size && (rbox.end[xdim] - rbox.begin[xdim] <= 0.5)) {
			tried1 = true;
			for (int dim = 0; dim < NDIM; dim++) {
				Xc1[dim] = 0.5 * (rbox.begin[dim] + rbox.end[dim]); // 6
			}
			array<simd_int, NDIM> xc;
			for (int dim = 0; dim < NDIM; dim++) {
				xc[dim] = fixed32(Xc1[dim]).raw();
			}
			flops += 6;
			simd_float r2_max(0.0);
			for (part_int i = part_range.first; i < part_range.second; i += SIMD_FLOAT_SIZE) {
				simd_int x;
				array<simd_float, NDIM> dx;
				int const maxj = std::min(part_range.second - i, SIMD_FLOAT_SIZE);
				for (int dim = 0; dim < NDIM; dim++) {
					for (int j = 0; j < maxj; j++) {
						x[j] = particles_pos(dim, i + j).raw();
					}
					for (int j = maxj; j < SIMD_FLOAT_SIZE; j++) {
						x[j] = xc[dim][0];
					}
					dx[dim] = simd_float(x - xc[dim]) * _2float; // 6
				}
				flops += 18 + maxj * 6;
				auto const r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
				r2_max = max(r2_max, r2);
			}
			float r2 = 0.0f;
			for (int i = 0; i < SIMD_FLOAT_SIZE; i++) {
				r2 = std::max(r2, r2_max[i]);
			}
			flops += 4 + SIMD_FLOAT_SIZE;
			radius1 = sqrt(r2);
			//	PRINT("%e\n", radius);
		}
		double rr;
		double rl;
		array<double, NDIM> N;
		float R;
		double norminv = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			N[dim] = Xl[dim] - Xr[dim]; // 3
			norminv += sqr(N[dim]);		// 6
		}
		norminv = 1.0 / std::sqrt(norminv); // 8
		for (int dim = 0; dim < NDIM; dim++) {
			N[dim] *= norminv;		   // 3
			N[dim] = std::abs(N[dim]); // 3
		}
		flops += 29;
		r = 0.0;
		if (mr[0] != 0.0) {
			if (ml[0] != 0.0) {
				for (int dim = 0; dim < NDIM; dim++) {
					double const xmax = std::max(Xl[dim] + N[dim] * Rl, Xr[dim] + N[dim] * Rr);
					double const xmin = std::min(Xl[dim] - N[dim] * Rl, Xr[dim] - N[dim] * Rr);
					Xc2[dim] = (xmax + xmin) * 0.5;
					r += sqr((xmax - xmin) * 0.5);
				}
				flops += 45;
			} else {
				Xc2 = Xr;
				r = Rr * Rr;
				flops++;
			}
		} else {
			if (ml[0] != 0.0) {
				Xc2 = Xl;
				r = Rl * Rl;
				flops++;
			} else {
				ALWAYS_ASSERT(false);
				flops += 6;
			}
		}
		radius2 = std::sqrt(r); // 4
		if (tried1 && radius1 < radius2) {
			Xc = Xc1;
			radius = radius1;
		} else {
			Xc = Xc2;
			radius = radius2;
		}
		r = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			double const span = (rbox.end[dim] - rbox.begin[dim]);
			ALWAYS_ASSERT(span >= 0.0);
			r += sqr(span * 0.5); // 12
		}
		r = std::sqrt(r); // 4
		if (r < radius) { // 1
			radius = r;
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (rbox.begin[dim] + rbox.end[dim]) * 0.5; // 6
			}
		}

		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = Xc[dim]; // 3
		}
		array<simd_double, NDIM> mdx;
		multipole<simd_double> simdM;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			simdM[i][LEFT] = ml[i];
			simdM[i][RIGHT] = mr[i];
		}
		for (int dim = 0; dim < NDIM; dim++) {
			mdx[dim][LEFT] = Xl[dim] - Xc[dim];	 // 3
			mdx[dim][RIGHT] = Xr[dim] - Xc[dim]; // 3
		}
		simdM = M2M<simd_double>(simdM, mdx);
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = simdM[i][LEFT] + simdM[i][RIGHT];
		}
		flops += 2421 + MULTIPOLE_SIZE * 2;
		children[LEFT] = rcl.id;
		children[RIGHT] = rcr.id;
	} else {
		array<double, NDIM>& Xmin = rbox.begin;
		array<double, NDIM>& Xmax = rbox.end;
		for (int dim = 0; dim < NDIM; dim++) {
			Xmin[dim] = box.end[dim];
			Xmax[dim] = box.begin[dim];
		}
		for (part_int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				double const x = particles_pos(dim, i).to_double(); // 3
				Xmax[dim] = std::max(Xmax[dim], x);					// 3
				Xmin[dim] = std::min(Xmin[dim], x);					// 3
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Xc[dim] = (Xmax[dim] + Xmin[dim]) * 0.5; // 6
		}
		r = 0.0;
		flops += 15;
		for (part_int i = part_range.first; i < part_range.second; i++) {
			double this_radius = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				double const x = particles_pos(dim, i).to_double(); // 3
				this_radius += sqr(x - Xc[dim]);					// 9
			}
			r = std::max(r, this_radius); // 2
			flops += 14;
		}
		radius = std::sqrt(r);
		children[LEFT].index = children[RIGHT].index = -1;
		multipole<double> M;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			M[i] = 0.0;
		}
		array<double, NDIM> dx;
		part_int const maxi = round_up(part_range.second - part_range.first, (part_int)SIMD_FLOAT_SIZE) + part_range.first;
		array<simd_int, NDIM> Y;
		for (int dim = 0; dim < NDIM; dim++) {
			Y[dim] = fixed32(Xc[dim]).raw();
		}

		for (part_int i = part_range.first; i < maxi; i += SIMD_FLOAT_SIZE) {
			array<simd_int, NDIM> X;
			simd_float mask;
			part_int const maxj = std::min(i + SIMD_FLOAT_SIZE, part_range.second);
			for (part_int j = i; j < maxj; j++) {
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim][j - i] = particles_pos(dim, j).raw();
				}
				mask[j - i] = particles_mass(j);
			}
			for (part_int j = maxj; j < i + SIMD_FLOAT_SIZE; j++) {
				mask[j - i] = 0.f;
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim][j - i] = particles_pos(dim, maxj - 1).raw();
				}
			}
			array<simd_float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = simd_float(X[dim] - Y[dim]) * _2float; // 3
			}
			auto m = P2M(dx); // 211
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				m[j] *= mask;		// MULTIPOLE_SIZE
				M[j] += m[j].sum(); // MULTIPOLE_SIZE * 6
			}
			flops += 214 + MULTIPOLE_SIZE * 7;
		}
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = M[i];
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = Xc[dim];
		}
	}
	tree_node node;
	node.radius = radius;
	node.children = children;
	node.local_root = local_root;
	node.part_range = part_range;
	node.proc_range = proc_range;
	node.pos = x;
	node.mpos = multis + index;
	node.mpos->pos = x;
	node.mpos->multi = multi;
	node.depth = depth;
	for (int dim = 0; dim < NDIM; dim++) {
		node.box.begin[dim] = rbox.begin[dim];
		node.box.end[dim] = rbox.end[dim];
	}
	bool const global = proc_range.second - proc_range.first > 1;
	node.leaf = isleaf;
	if (index >= nodes_size) {
		THROW_ERROR("%s\n", "Tree arena full\n");
	}
	nodes[index] = node;
	populatedNodes[index] = true;
	rc.id.index = index;
	rc.id.proc = hpx_rank();
	if (local_root && get_options().sph) {
		particles_resolve_with_sph_particles();
	}
	if (key == 1) {
		profiler_exit();
	}
	add_cpu_flops(flops);
	return rc;
}

HPX_PLAIN_ACTION(tree_reset);

void tree_reset() {

	vector<hpx::future<void>> futs;
	auto const children = hpx_children();
	for (auto const& c : children) {
		futs.push_back(hpx::async<tree_reset_action>(c));
	}
	tree_cache = decltype(tree_cache)();
	reset_last_cache_entries();
	hpx::wait_all(futs.begin(), futs.end());
}
vector<double> tree_get_leaf_sizes();
HPX_PLAIN_ACTION(tree_get_leaf_sizes);

vector<double> tree_get_leaf_sizes() {
	vector<hpx::future<vector<double>>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<tree_get_leaf_sizes_action>(c));
	}
	double my_avg = 0.0;
	for (int i = 0; i < leaf_sizes.size(); i++) {
		my_avg += leaf_sizes[i];
	}
	my_avg /= leaf_sizes.size();
	vector<double> avgs;
	avgs.push_back(my_avg);
	for (auto& f : futs) {
		auto tmp = f.get();
		for (auto const& m : tmp) {
			avgs.push_back(m);
		}
	}
	return avgs;
}

int tree_avg_leaf_size() {
	auto sizes = tree_get_leaf_sizes();
	double avg = 0.0;
	for (int i = 0; i < sizes.size(); i++) {
		avg += sizes[i];
	}
	avg /= sizes.size();
	return avg;
}

void tree_destroy(bool free_tree) {
	profiler_enter(__FUNCTION__);

	vector<hpx::future<void>> futs;
	auto const children = hpx_children();
	for (auto const& c : children) {
		futs.push_back(hpx::async<tree_destroy_action>(c, free_tree));
	}
	tree_cache = decltype(tree_cache)();
	reset_last_cache_entries();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

tree_node const* tree_get_node(tree_id id) {
	if (id.proc == hpx_rank()) {
		ASSERT(id.index >= 0);
#ifndef NDEBUG
		if (id.index >= nodes_size) {
			THROW_ERROR("id.index is %li but nodes.size() is %li\n", id.index, nodes_size);
		}
#endif
		return &nodes[id.index];
	} else {
		return tree_cache_read(id);
	}
}

static tree_node const* tree_cache_read(tree_id id) {
	int const line_size = get_options().tree_cache_line_size;
	tree_id line_id;
	tree_node const* ptr;
	line_id.proc = id.proc;
	ASSERT(line_id.proc >= 0 && line_id.proc < hpx_size());
	line_id.index = (id.index / line_size) * line_size;
	if (line_id != last_cache_entry.line) {
		size_t const bin = tree_id_hash_lo()(id);
		std::shared_ptr<hpx::shared_future<vector<tree_node>>> future;
		{
			std::lock_guard<spinlock_type> lock(mutex[bin]);
			auto const entry = tree_cache[bin].find(line_id);
			if (entry != tree_cache[bin].end()) {
				future = entry->second;
			}
		}
		if (!future) {
			auto promise = std::make_shared<hpx::promise<vector<tree_node>>>();
			auto candidate = std::make_shared<hpx::shared_future<vector<tree_node>>>(promise->get_future().share());
			bool inserted;
			{
				std::lock_guard<spinlock_type> lock(mutex[bin]);
				auto const entry = tree_cache[bin].emplace(line_id, candidate);
				inserted = entry.second;
				future = entry.first->second;
			}
			if (inserted) {
				try {
					hpx::post([promise, line_id]() {
						try {
							promise->set_value(hpx::async<tree_fetch_cache_line_action>(hpx_localities()[line_id.proc], line_id.index).get());
						} catch (...) {
							promise->set_exception(std::current_exception());
						}
					});
				} catch (...) {
					promise->set_exception(std::current_exception());
				}
			}
		}
		ptr = future->get().data();
	} else {
		ptr = last_cache_entry.ptr;
	}
	last_cache_entry.ptr = ptr;
	last_cache_entry.line = line_id;
	return ptr + id.index - line_id.index;
}

static vector<tree_node> tree_fetch_cache_line(int index) {
	int const line_size = get_options().tree_cache_line_size;
	ALWAYS_ASSERT(index >= 0 && size_t(index) < nodes_size);
	vector<tree_node> line;
	line.reserve(line_size);
	int const begin = (index / line_size) * line_size;
	int const end = std::min(size_t(begin) + size_t(line_size), nodes_size);
	static multi_pos emptyMultipole = [] {
		multi_pos empty;
		empty.multi = 0.0f;
		for (int dim = 0; dim < NDIM; dim++) {
			empty.pos[dim] = 0.0;
		}
		return empty;
	}();
	for (int i = begin; i < end; i++) {
		if (populatedNodes[i]) {
			line.push_back(nodes[i]);
		} else {
			// Unused slots preserve cache offsets but never dereference arena garbage.
			tree_node empty{};
			empty.mpos = &emptyMultipole;
			empty.leaf = true;
			empty.proc_range = pair<int>(hpx_rank(), hpx_rank() + 1);
			for (int child = 0; child < NCHILD; child++) {
				empty.children[child].proc = hpx_rank();
				empty.children[child].index = -1;
			}
			line.push_back(empty);
		}
	}
	return line;
}
