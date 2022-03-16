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
 Foundation, Inc., 51 Franklin Ssph_treet, Fifth Floor, Boston, MA  02110-1301, USA.
 */

constexpr bool verbose = true;

#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/sph_tree.hpp>

#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

static vector<sph_tree_node> sph_tree_fetch_cache_line(int);
static void sph_tree_allocate_nodes();

HPX_PLAIN_ACTION (sph_tree_allocate_nodes);
HPX_PLAIN_ACTION (sph_tree_create);
HPX_PLAIN_ACTION (sph_tree_destroy);
HPX_PLAIN_ACTION (sph_tree_fetch_cache_line);
HPX_PLAIN_ACTION (sph_tree_free_neighbor_list);

class sph_tree_allocator {
	int next;
	int last;
public:
	bool ready;
	void reset();
	bool is_ready();
	sph_tree_allocator();
	~sph_tree_allocator();
	sph_tree_allocator(const sph_tree_allocator&) = default;
	sph_tree_allocator& operator=(const sph_tree_allocator&) = default;
	sph_tree_allocator(sph_tree_allocator&&) = default;
	sph_tree_allocator& operator=(sph_tree_allocator&&) = default;
	int allocate();
};

static vector<sph_tree_allocator*> allocator_list;
static thread_local sph_tree_allocator allocator;
static vector<sph_tree_node> nodes;
static std::atomic<int> num_threads(0);
static std::atomic<int> next_id;
static array<std::unordered_map<tree_id, hpx::shared_future<vector<sph_tree_node>>, tree_id_hash_hi>, SPH_TREE_CACHE_SIZE> tree_cache;
static array<spinlock_type, SPH_TREE_CACHE_SIZE> mutex;
static std::atomic<int> allocator_mtx(0);
static vector<pair<part_int>> leaf_part_ranges;
static mutex_type leaf_part_range_mutex;
static vector<tree_id> neighbor_list;
static vector<int> leaflist;
static mutex_type leaflist_mutex;
static mutex_type neighbor_list_mutex;

void sph_tree_set_neighbor_range(tree_id id, pair<int, int> rng) {
	nodes[id.index].neighbor_range = rng;
}

void sph_tree_clear_neighbor_ranges() {
	for (int i = 0; i < leaflist.size(); i++) {
		nodes[leaflist[i]].neighbor_range.first = -1;
		nodes[leaflist[i]].neighbor_range.second = -1;
	}
}

int sph_tree_leaflist_size() {
	return leaflist.size();
}

const tree_id sph_tree_get_leaf(int i) {
	tree_id id;
	id.index = leaflist[i];
	id.proc = hpx_rank();
	return id;
}

void sph_tree_free_neighbor_list() {
//	PRINT("leaflist size = %i\n", leaflist.size());
	neighbor_list.resize(0);
}

int sph_tree_allocate_neighbor_list(const vector<tree_id>& values) {
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

tree_id& sph_tree_get_neighbor(int i) {
	return neighbor_list[i];
}

long long sph_tree_nodes_size() {
	return nodes.size();
}

struct last_cache_entry_t;

static std::unordered_set<last_cache_entry_t*> last_cache_entries;
static std::atomic<int> last_cache_entry_mtx(0);

struct last_cache_entry_t {
	tree_id line;
	const sph_tree_node* ptr;
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

static const sph_tree_node* tree_cache_read(tree_id id);

void sph_tree_allocator::reset() {
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	next = (next_id += tree_cache_line_size);
	last = std::min(next + tree_cache_line_size, (int) nodes.size());
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

fast_future<sph_tree_create_return> sph_tree_create_fork(sph_tree_create_params params, size_t key, const pair<int, int>& proc_range,
		const pair<part_int>& part_range, const range<double>& box, const int depth, const bool local_root, bool threadme) {
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
		rc = hpx::async<sph_tree_create_action>(HPX_PRIORITY_HI, hpx_localities()[proc_range.first], params, key, proc_range, part_range, box, depth, local_root);
	} else {
		rc = hpx::async([params,proc_range,key,part_range,depth,local_root, box]() {
			auto rc = sph_tree_create(params,key,proc_range,part_range,box,depth,local_root);
			nthreads--;
			return rc;
		});
	}
	return rc;
}

static void sph_tree_allocate_nodes() {
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	static const int bucket_size = get_options().sph_bucket_size;
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_tree_allocate_nodes_action>(HPX_PRIORITY_HI, c));
	}
	next_id = -tree_cache_line_size;
	nodes.resize(std::max(size_t(size_t(SPH_TREE_NODE_ALLOCATION_SIZE) * sph_particles_size() / bucket_size), (size_t) NSPH_TREES_MIN));
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
	//PRINT( "%i\n", depth);
	stack_trace_activate();
	static const int bucket_size = get_options().sph_bucket_size;
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
	const int index = allocator.allocate();
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
			const int mid = (proc_range.first + proc_range.second) / 2;
			const auto dleft_box = domains_range(key << 1);
			const auto dright_box = domains_range((key << 1) + 1);
			left_range.second = right_range.first = mid;
			left_local_root = left_range.second - left_range.first == 1;
			right_local_root = right_range.second - right_range.first == 1;
			flops += 7;
		} else {
			double span_max = 0.0;
			int xdim;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto span = box.end[dim] - box.begin[dim];
				if (span > span_max) {
					span_max = span;
					xdim = dim;
				}
			}
			const double xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
			const part_int mid = sph_particles_sort(part_range, xmid, xdim);
			left_parts.second = right_parts.first = mid;
			left_box.end[xdim] = right_box.begin[xdim] = xmid;
			flops += 2;
		}
		auto futr = sph_tree_create_fork(params, (key << 1) + 1, right_range, right_parts, right_box, depth + 1, right_local_root, true);
		auto futl = sph_tree_create_fork(params, (key << 1), left_range, left_parts, left_box, depth + 1, left_local_root, false);
		const sph_tree_create_return rcl = futl.get();
		const sph_tree_create_return rcr = futr.get();
		const auto iboxl = rcl.inner_box;
		const auto iboxr = rcr.inner_box;
		const auto oboxl = rcl.outer_box;
		const auto oboxr = rcr.outer_box;
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
		for( int dim = 0; dim < NDIM; dim++) {
			inner_box.begin[dim] = std::min(iboxl.begin[dim], iboxr.begin[dim]);
			inner_box.end[dim] = std::max(iboxl.end[dim], iboxr.end[dim]);
			outer_box.begin[dim] = std::min(oboxl.begin[dim], oboxr.begin[dim]);
			outer_box.end[dim] = std::max(oboxl.end[dim], oboxr.end[dim]);
		}
//		PRINT("%i %i %i %i | %e %e %e %e |%e %e %e %e \n", iboxr.valid, iboxl.valid, oboxr.valid, oboxl.valid, iboxl.begin[XDIM], iboxl.end[XDIM], iboxr.begin[XDIM], iboxr.end[XDIM], oboxl.begin[XDIM], oboxl.end[XDIM], oboxr.begin[XDIM], oboxr.end[XDIM]);
	} else {
		children[LEFT].index = children[RIGHT].index = -1;
		nactive = 0;
		for( int dim = 0; dim < NDIM; dim++) {
			inner_box.begin[dim] = 1.9;
			inner_box.end[dim] = -0.9;
			outer_box.begin[dim] = 1.9;
			outer_box.end[dim] = -0.9;
		}
		for (part_int i = part_range.first; i < part_range.second; i++) {
			sph_particles_semi_active(i) = false;
			const float h = params.h_wt * sph_particles_smooth_len(i);
			array<fixed32, NDIM> X;
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = sph_particles_pos(dim, i);
			}
			for( int dim = 0; dim < NDIM; dim++) {
				const double x = X[dim].to_double();
				inner_box.begin[dim] = std::min(inner_box.begin[dim].to_double(), x);
				inner_box.end[dim] = std::max(inner_box.end[dim].to_double(), x);
				outer_box.begin[dim] = std::min(outer_box.begin[dim].to_double(), x - h);
				outer_box.end[dim] = std::max(outer_box.end[dim].to_double(), x + h);
			}
			if (sph_particles_rung(i) >= params.min_rung) {
				nactive++;
			}
		}
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
	const part_int nparts = part_range.second - part_range.first;
	const bool global = proc_range.second - proc_range.first > 1;
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
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<sph_tree_destroy_action>(HPX_PRIORITY_HI, c, free_sph_tree));
	}
	if (free_sph_tree) {
		nodes = decltype(nodes)();
	} else {
		const int nthreads = hpx_hardware_concurrency();
		for (int proc = 0; proc < nthreads; proc++) {
			futs.push_back(hpx::async([proc, nthreads]() {
				const int b = (size_t) proc * nodes.size() / nthreads;
				const int e = (size_t) (proc+1) * nodes.size() / nthreads;
				for( int i = b; i < e; i++) {
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

void sph_tree_set_nactive(tree_id id, part_int i) {
	nodes[id.index].nactive = i;
}

void sph_tree_set_converged(tree_id id, bool c) {
	nodes[id.index].converged = c;
}

const sph_tree_node* sph_tree_get_node(tree_id id) {
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

void sph_tree_set_boxes(tree_id id, const fixed32_range& ibox, const fixed32_range& obox) {
	nodes[id.index].outer_box = obox;
	nodes[id.index].inner_box = ibox;
}

static const sph_tree_node* tree_cache_read(tree_id id) {
	const int line_size = get_options().tree_cache_line_size;
	tree_id line_id;
	const sph_tree_node* ptr;
	line_id.proc = id.proc;
	ASSERT(line_id.proc >= 0 && line_id.proc < hpx_size());
	line_id.index = (id.index / line_size) * line_size;
	if (line_id != last_cache_entry.line) {
		const size_t bin = tree_id_hash_lo()(id);
		std::unique_lock<spinlock_type> lock(mutex[bin]);
		auto iter = tree_cache[bin].find(line_id);
		if (iter == tree_cache[bin].end()) {
			auto prms = std::make_shared<hpx::lcos::local::promise<vector<sph_tree_node>>>();
			tree_cache[bin][line_id] = prms->get_future();
			lock.unlock();
			hpx::async(HPX_PRIORITY_HI, [prms,line_id]() {
				const sph_tree_fetch_cache_line_action action;
				auto fut = hpx::async<sph_tree_fetch_cache_line_action>(HPX_PRIORITY_HI, hpx_localities()[line_id.proc],line_id.index);
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

static vector<sph_tree_node> sph_tree_fetch_cache_line(int index) {
	const int line_size = get_options().tree_cache_line_size;
	vector<sph_tree_node> line;
	line.reserve(line_size);
	const int begin = (index / line_size) * line_size;
	const int end = begin + line_size;
	for (int i = begin; i < end; i++) {
		line.push_back(nodes[i]);
	}
	return std::move(line);

}


