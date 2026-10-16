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

#include <exception>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/stack_trace.hpp>

constexpr bool verbose = true;

static vector<sph_tree_node> sph_tree_fetch_cache_line(int);
static void sph_tree_allocate_nodes();

HPX_PLAIN_ACTION(sph_tree_allocate_nodes);
HPX_PLAIN_ACTION(sph_tree_create);
HPX_PLAIN_ACTION(sph_tree_destroy);
HPX_PLAIN_ACTION(sph_tree_cache_clear);
HPX_PLAIN_ACTION(sph_tree_fetch_cache_line);
HPX_PLAIN_ACTION(sph_tree_free_neighbor_list);

class sph_tree_allocator {
	int next;
	int last;

  public:
	bool ready;
	void reset();
	bool is_ready();
	sph_tree_allocator();
	~sph_tree_allocator();
	sph_tree_allocator(sph_tree_allocator const&) = default;
	sph_tree_allocator& operator=(sph_tree_allocator const&) = default;
	sph_tree_allocator(sph_tree_allocator&&) = default;
	sph_tree_allocator& operator=(sph_tree_allocator&&) = default;
	int allocate();
};

static vector<sph_tree_allocator*> allocator_list;
static thread_local sph_tree_allocator allocator;
static vector<sph_tree_node> nodes;
static std::atomic<int> num_threads(0);
static std::atomic<int> next_id;
static array<std::unordered_map<tree_id, std::shared_ptr<hpx::shared_future<vector<sph_tree_node>>>, tree_id_hash_hi>, SPH_TREE_CACHE_SIZE> tree_cache;
static array<spinlock_type, SPH_TREE_CACHE_SIZE> mutex;
static std::atomic<int> allocator_mtx(0);
static vector<pair<part_int>> leaf_part_ranges;
static mutex_type leaf_part_range_mutex;
static vector<tree_id> neighbor_list;
static vector<int> leaflist;
static mutex_type leaflist_mutex;
static mutex_type neighbor_list_mutex;

void sph_tree_set_neighbor_range(tree_id id, pair<int, int> rng) { nodes[id.index].neighbor_range = rng; }

void sph_tree_clear_neighbor_ranges() {
	for (int i = 0; i < leaflist.size(); i++) {
		nodes[leaflist[i]].neighbor_range.first = -1;
		nodes[leaflist[i]].neighbor_range.second = -1;
	}
}

int sph_tree_leaflist_size() { return leaflist.size(); }

tree_id const sph_tree_get_leaf(int i) {
	tree_id id;
	id.index = leaflist[i];
	id.proc = hpx_rank();
	return id;
}

void sph_tree_free_neighbor_list() {
	//	PRINT("leaflist size = %i\n", leaflist.size());
	neighbor_list.resize(0);
}

int sph_tree_allocate_neighbor_list(vector<tree_id> const& values) {
	std::lock_guard<mutex_type> lock(neighbor_list_mutex);
	if (values.size()) {
		int i = neighbor_list.size();
		neighbor_list.resize(i + values.size());
		memcpy(&neighbor_list[i], values.data(), sizeof(tree_id) * values.size());
		return i;
	} else {
		return 0;
	}
}

tree_id& sph_tree_get_neighbor(int i) { return neighbor_list[i]; }

long long sph_tree_nodes_size() { return nodes.size(); }

struct last_cache_entry_t;

static std::unordered_set<last_cache_entry_t*> last_cache_entries;
static std::atomic<int> last_cache_entry_mtx(0);

struct last_cache_entry_t {
	tree_id line;
	sph_tree_node const* ptr;
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
	while (last_cache_entry_mtx++ != 0) {
		last_cache_entry_mtx--;
	}
	for (auto* entry : last_cache_entries) {
		entry->reset();
	}
	last_cache_entry_mtx--;
}

void sph_tree_cache_clear() {
	vector<hpx::future<void>> futures;
	for (auto const& child : hpx_children()) {
		futures.push_back(hpx::async<sph_tree_cache_clear_action>(child));
	}
	tree_cache = decltype(tree_cache)();
	reset_last_cache_entries();
	for (auto& future : futures) {
		future.get();
	}
}

static sph_tree_node const* tree_cache_read(tree_id id);

void sph_tree_allocator::reset() {
	int const tree_cache_line_size = get_options().tree_cache_line_size;
	next = (next_id += tree_cache_line_size);
	last = std::min(next + tree_cache_line_size, (int)nodes.size());
	if (next >= nodes.size()) {
		THROW_ERROR("%s\n", "Tree arena full");
	}
}

sph_tree_allocator::sph_tree_allocator() {
	ready = false;
	while (allocator_mtx++ != 0) {
		allocator_mtx--;
	}
	allocator_list.push_back(this);
	allocator_mtx--;
}

int sph_tree_allocator::allocate() {
	if (next == last) {
		reset();
	}
	ASSERT(next < nodes.size());
	return next++;
}

sph_tree_allocator::~sph_tree_allocator() {
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

fast_future<sph_tree_create_return> sph_tree_create_fork(sph_tree_create_params params, size_t key, pair<int, int> const& proc_range,
														 pair<part_int> const& part_range, range<double> const& box, int const depth, bool const local_root,
														 bool threadme) {
	static std::atomic<int> nthreads(0);
	fast_future<sph_tree_create_return> rc;
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
		rc.set_value(sph_tree_create(params, key, proc_range, part_range, box, depth, local_root));
	} else if (remote) {
		//		PRINT( "%i calling local on %i at %li\n", hpx_rank(), proc_range.first, time(NULL));
		rc = hpx::async<sph_tree_create_action>(hpx_localities()[proc_range.first], params, key, proc_range, part_range, box, depth, local_root);
	} else {
		rc = hpx::async([params, proc_range, key, part_range, depth, local_root, box]() {
			auto rc = sph_tree_create(params, key, proc_range, part_range, box, depth, local_root);
			nthreads--;
			return rc;
		});
	}
	return rc;
}

static void sph_tree_allocate_nodes() {
	int const tree_cache_line_size = get_options().tree_cache_line_size;
	static int const bucket_size = get_options().sph_bucket_size;
	vector<hpx::future<void>> futs;
	for (auto const& c : hpx_children()) {
		futs.push_back(hpx::async<sph_tree_allocate_nodes_action>(c));
	}
	next_id = -tree_cache_line_size;
	nodes.resize(std::max(size_t(size_t(SPH_TREE_NODE_ALLOCATION_SIZE) * sph_particles_size() / bucket_size), (size_t)NSPH_TREES_MIN));
	while (allocator_mtx++ != 0) {
		allocator_mtx--;
	}
	for (int i = 0; i < allocator_list.size(); i++) {
		allocator_list[i]->ready = false;
	}
	allocator_mtx--;
	hpx::wait_all(futs.begin(), futs.end());
}

sph_tree_create_return sph_tree_create(sph_tree_create_params params, size_t key, pair<int, int> proc_range, pair<part_int> part_range, range<double> box,
									   int depth, bool local_root) {
	// PRINT( "%i\n", depth);
	stack_trace_activate();
	static int const bucket_size = get_options().sph_bucket_size;
	sph_tree_create_return rc;
	if (depth >= MAX_DEPTH) {
		THROW_ERROR("%s\n", "Maximum depth exceeded\n");
	}
	if (depth == 0) {
		sph_tree_allocate_nodes();
	}
	if (local_root) {
		leaflist.resize(0);
		part_range.first = 0;
		part_range.second = sph_particles_size();
	}
	array<tree_id, NCHILD> children;
	array<fixed32, NDIM> x;
	array<double, NDIM> Xc;
	size_t nactive;
	int flops = 0;
	double total_flops = 0.0;
	float radius;
	double r;
	int min_depth = depth;
	int max_depth = depth;
	if (local_root) {
		//		PRINT("%i\n", allocator.ready);
	}
	if (!allocator.ready) {
		allocator.reset();
		allocator.ready = true;
	}
	size_t node_count;
	size_t active_nodes = 0;
	size_t leaf_nodes = 0;
	size_t active_leaf_nodes = 0;
	int const index = allocator.allocate();
	if (local_root) {
		//		PRINT("INDEX = %i\n", index);
	}
	fixed32_range inner_box;
	fixed32_range outer_box;
	if (proc_range.second - proc_range.first > 1 || part_range.second - part_range.first > bucket_size) {
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
			auto const dleft_box = domains_range(key << 1);
			auto const dright_box = domains_range((key << 1) + 1);
			left_range.second = right_range.first = mid;
			left_local_root = left_range.second - left_range.first == 1;
			right_local_root = right_range.second - right_range.first == 1;
			flops += 7;
		} else {
			double span_max = 0.0;
			int xdim;
			for (int dim = 0; dim < NDIM; dim++) {
				auto const span = box.end[dim] - box.begin[dim];
				if (span > span_max) {
					span_max = span;
					xdim = dim;
				}
			}
			double const xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
			part_int const mid = sph_particles_sort(part_range, xmid, xdim);
			left_parts.second = right_parts.first = mid;
			left_box.end[xdim] = right_box.begin[xdim] = xmid;
			flops += 2;
		}
		auto futr = sph_tree_create_fork(params, (key << 1) + 1, right_range, right_parts, right_box, depth + 1, right_local_root, true);
		auto futl = sph_tree_create_fork(params, (key << 1), left_range, left_parts, left_box, depth + 1, left_local_root, false);
		sph_tree_create_return const rcl = futl.get();
		sph_tree_create_return const rcr = futr.get();
		auto const iboxl = rcl.inner_box;
		auto const iboxr = rcr.inner_box;
		auto const oboxl = rcl.outer_box;
		auto const oboxr = rcr.outer_box;
		min_depth = std::min(rcl.min_depth, rcr.min_depth);
		max_depth = std::max(rcl.max_depth, rcr.max_depth);
		total_flops += rcl.flops + rcr.flops;
		nactive = rcl.nactive + rcr.nactive;
		children[LEFT] = rcl.id;
		children[RIGHT] = rcr.id;
		node_count = 1 + rcl.node_count + rcr.node_count;
		leaf_nodes = rcl.leaf_nodes + rcr.leaf_nodes;
		active_leaf_nodes = rcl.active_leaf_nodes + rcr.active_leaf_nodes;
		if (rcl.nactive || rcr.nactive) {
			active_nodes += 1 + rcl.active_nodes + rcr.active_nodes;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			inner_box.begin[dim] = std::min(iboxl.begin[dim], iboxr.begin[dim]);
			inner_box.end[dim] = std::max(iboxl.end[dim], iboxr.end[dim]);
			outer_box.begin[dim] = std::min(oboxl.begin[dim], oboxr.begin[dim]);
			outer_box.end[dim] = std::max(oboxl.end[dim], oboxr.end[dim]);
		}
		//		PRINT("%i %i %i %i | %e %e %e %e |%e %e %e %e \n", iboxr.valid, iboxl.valid, oboxr.valid, oboxl.valid, iboxl.begin[XDIM], iboxl.end[XDIM],
		// iboxr.begin[XDIM], iboxr.end[XDIM], oboxl.begin[XDIM], oboxl.end[XDIM], oboxr.begin[XDIM], oboxr.end[XDIM]);
	} else {
		children[LEFT].index = children[RIGHT].index = -1;
		children[LEFT].proc = children[RIGHT].proc = -1;
		nactive = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			inner_box.begin[dim] = 1.9;
			inner_box.end[dim] = -0.9;
			outer_box.begin[dim] = 1.9;
			outer_box.end[dim] = -0.9;
		}
		auto const tiny = 2.0 * range_fixed::min().to_double();
		for (part_int i = part_range.first; i < part_range.second; i++) {
			float const h = params.h_wt * sph_particles_smooth_len(i);
			array<fixed32, NDIM> X;
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = sph_particles_pos(dim, i);
			}
			for (int dim = 0; dim < NDIM; dim++) {
				double const x = X[dim].to_double();
				inner_box.begin[dim] = std::min(inner_box.begin[dim].to_double(), x - tiny);
				inner_box.end[dim] = std::max(inner_box.end[dim].to_double(), x + tiny);
				if (sph_particles_rung(i) >= params.min_rung) {
					outer_box.begin[dim] = std::min(outer_box.begin[dim].to_double(), x - h - tiny);
					outer_box.end[dim] = std::max(outer_box.end[dim].to_double(), x + h + tiny);
				}
			}
			if (sph_particles_rung(i) >= params.min_rung) {
				nactive++;
			}
		}

		/*	double small_span = 1e30;
		 double hmax = 0.0;
		 for( int dim = 0; dim < NDIM; dim++) {
		 double span = box.end[dim] - box.begin[dim];
		 small_span = std::min(small_span, span);
		 hmax += sqr(span);
		 }
		 //	hmax += 12.0 * sqr(small_span);
		 hmax = 1.5 * sqrt(hmax + 3.0 * sqr(small_span));
		 for( int dim = 0; dim < NDIM; dim++) {
		 inner_box.begin[dim] = box.begin[dim];
		 inner_box.end[dim] = box.end[dim];
		 outer_box.begin[dim] = box.begin[dim] - hmax;
		 outer_box.end[dim] = box.begin[dim] + hmax;
		 }*/

		node_count = 1;
		leaf_nodes = 1;
		if (nactive) {
			active_leaf_nodes++;
			active_nodes++;
		}
		std::lock_guard<mutex_type> lock(leaflist_mutex);
		leaflist.push_back(index);
	}
	sph_tree_node node;
	node = nodes[index];
	node.node_count = node_count;
	node.inner_box = inner_box;
	node.outer_box = outer_box;

	node.children = children;
	node.local_root = local_root;
	node.part_range = part_range;
	node.proc_range = proc_range;
	node.nactive = nactive;
	node.active_nodes = active_nodes;
	node.neighbor_range.first = -1;
	node.neighbor_range.second = -1;
	node.depth = depth;
	node.sink_part_range = part_range;
	part_int const nparts = part_range.second - part_range.first;
	bool const global = proc_range.second - proc_range.first > 1;
	node.leaf = !global && (nparts <= get_options().sph_bucket_size);
	/*	if ( BUCKET_SIZE <= get_options().sph_bucket_size) {
	 if (node.leaf) {
	 std::lock_guard<mutex_type> lock(leaf_part_range_mutex);
	 leaf_part_ranges.push_back(part_range);
	 }
	 }*/
	if (index >= nodes.size()) {
		THROW_ERROR("%s\n", "Tree arena full\n");
	}
	rc.active_nodes = active_nodes;
	rc.id.index = index;
	rc.id.proc = hpx_rank();
	rc.inner_box = node.inner_box;
	rc.outer_box = node.outer_box;
	for (int dim = 0; dim < NDIM; dim++) {
		node.box.begin[dim] = box.begin[dim];
		node.box.end[dim] = box.end[dim]; // == 1.0 ? fixed32::max() : fixed32(box.end[dim]);
										  //	PRINT( "---------%e %e\n", node.box.begin[dim].to_float(), node.box.end[dim].to_float());
	}
	nodes[index] = node;
	rc.leaf_nodes = leaf_nodes;
	rc.active_leaf_nodes = active_leaf_nodes;
	rc.node_count = node.node_count;
	rc.nactive = nactive;
	total_flops += flops;
	rc.flops = total_flops;
	rc.min_depth = min_depth;
	rc.max_depth = max_depth;
	if (local_root) {
		sph_particles_resolve_with_particles();
	}
	return rc;
}

void sph_tree_destroy(bool free_sph_tree) {
	profiler_enter(__FUNCTION__);

	vector<hpx::future<void>> futs;
	auto const children = hpx_children();
	for (auto const& c : children) {
		futs.push_back(hpx::async<sph_tree_destroy_action>(c, free_sph_tree));
	}
	if (free_sph_tree) {
		nodes = decltype(nodes)();
	} else {
		int const nthreads = hpx_hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(hpx::async([proc, nthreads]() {
				int const b = (size_t)proc * nodes.size() / nthreads;
				int const e = (size_t)(proc + 1) * nodes.size() / nthreads;
				for (int i = b; i < e; i++) {
					nodes[i].inner_box = fixed32_range();
					nodes[i].outer_box = fixed32_range();
				}
			}));
		}
	}
	tree_cache = decltype(tree_cache)();
	reset_last_cache_entries();
	hpx::wait_all(futs.begin(), futs.end());
	profiler_exit();
}

void sph_tree_set_nactive(tree_id id, part_int i) { nodes[id.index].nactive = i; }

void sph_tree_set_converged(tree_id id, bool c) { nodes[id.index].converged = c; }

sph_tree_node const* sph_tree_get_node(tree_id id) {
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

void sph_tree_set_boxes(tree_id id, fixed32_range const& ibox, fixed32_range const& obox) {
	nodes[id.index].outer_box = obox;
	nodes[id.index].inner_box = ibox;
}

static sph_tree_node const* tree_cache_read(tree_id id) {
	int const lineSize = get_options().tree_cache_line_size;
	tree_id lineId;
	lineId.proc = id.proc;
	ASSERT(lineId.proc >= 0 && lineId.proc < hpx_size());
	lineId.index = (id.index / lineSize) * lineSize;
	sph_tree_node const* ptr;
	if (lineId != last_cache_entry.line) {
		size_t const bin = tree_id_hash_lo()(lineId);
		std::shared_ptr<hpx::shared_future<vector<sph_tree_node>>> future;
		{
			std::lock_guard<spinlock_type> lock(mutex[bin]);
			auto const entry = tree_cache[bin].find(lineId);
			if (entry != tree_cache[bin].end()) {
				future = entry->second;
			}
		}
		if (!future) {
			auto promise = std::make_shared<hpx::promise<vector<sph_tree_node>>>();
			auto candidate = std::make_shared<hpx::shared_future<vector<sph_tree_node>>>(promise->get_future().share());
			bool inserted;
			{
				std::lock_guard<spinlock_type> lock(mutex[bin]);
				auto const entry = tree_cache[bin].emplace(lineId, candidate);
				inserted = entry.second;
				future = entry.first->second;
			}
			if (inserted) {
				try {
					hpx::apply([promise, lineId]() {
						try {
							auto remote = hpx::async<sph_tree_fetch_cache_line_action>(hpx_localities()[lineId.proc], lineId.index);
							promise->set_value(remote.get());
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
	last_cache_entry.line = lineId;
	return ptr + id.index - lineId.index;
}

static vector<sph_tree_node> sph_tree_fetch_cache_line(int index) {
	int const line_size = get_options().tree_cache_line_size;
	vector<sph_tree_node> line(line_size);
	int const begin = (index / line_size) * line_size;
	int const end = std::min(begin + line_size, int(nodes.size()));
	for (int i = begin; i < end; i++) {
		line[i - begin] = nodes[i];
	}
	return line;
}
