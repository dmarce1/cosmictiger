constexpr bool verbose = true;

#include <tigerfmm/fast_future.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/safe_io.hpp>
#include <tigerfmm/tree.hpp>

#include <shared_mutex>
#include <unordered_map>

static vector<tree_node> tree_fetch_cache_line(int);

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
	tree_allocator(const tree_allocator&) = default;
	tree_allocator& operator=(const tree_allocator&) = default;
	tree_allocator(tree_allocator&&) = default;
	tree_allocator& operator=(tree_allocator&&) = default;
	int allocate();
};

static vector<tree_allocator*> allocator_list;
static vector<tree_node> nodes;
static std::atomic<int> num_threads(0);
static std::atomic<int> next_id;
static thread_local tree_allocator allocator;
static array<std::unordered_map<tree_id, hpx::shared_future<vector<tree_node>>, tree_id_hash_hi>, TREE_CACHE_SIZE> tree_cache;
static array<spinlock_type, TREE_CACHE_SIZE> mutex;
static thread_local tree_id last_line;
static thread_local const tree_node* last_ptr = nullptr;

static const tree_node* tree_cache_read(tree_id id);

void tree_allocator::reset() {
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	next = (next_id += tree_cache_line_size);
	last = next + tree_cache_line_size;
	if (last > nodes.size()) {
		THROW_ERROR("%s\n", "Tree arena full");
	}
}

tree_allocator::tree_allocator() {
	ready = false;
	std::lock_guard<spinlock_type> lock(mutex[0]);
	allocator_list.push_back(this);
}

int tree_allocator::allocate() {
	if (next == last) {
		reset();
	}
	return next++;
}

tree_allocator::~tree_allocator() {
	std::lock_guard<spinlock_type> lock(mutex[0]);
	for (int i = 0; i < allocator_list.size(); i++) {
		if (allocator_list[i] == this) {
			allocator_list[i] = allocator_list.back();
			allocator_list.pop_back();
			break;
		}
	}
}

tree_create_params::tree_create_params(int min_rung_, double theta_) {
	theta = theta_;
	min_rung = min_rung_;
	min_level = tree_min_level(theta);
}

int tree_min_level(double theta) {
	const double h = get_options().hsoft;
	int lev = 1;
	double dx;
	double r;
	do {
		int N = 1 << (lev / NDIM);
		dx = EWALD_DIST * N;
		double a;
		constexpr double ffac = 1.01;
		if (lev % NDIM == 0) {
			a = std::sqrt(3) + ffac * h;
		} else if (lev % NDIM == 1) {
			a = 1.5 + ffac * h;
		} else {
			a = std::sqrt(1.5) + ffac * h;
		}
		r = (1.0 + SINK_BIAS) * a / theta + h * N;
		lev++;
	} while (dx <= r);
	int i = 1;
	while (i < hpx_size()) {
		i *= 2;
	}
	return i == hpx_size() ? lev : lev + 1;
}

fast_future<tree_create_return> tree_create_fork(tree_create_params params, const pair<int, int>& proc_range, const pair<int, int>& part_range,
		const range<double>& box, const int depth, const bool local_root, bool threadme) {
	static std::atomic<int> nthreads(0);
	fast_future<tree_create_return> rc;
	bool remote = false;
	if (proc_range.first != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = part_range.second - part_range.first > MIN_SORT_THREAD_PARTS || proc_range.second - proc_range.first > 1;
		if (threadme) {
			if (nthreads++ < SORT_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency()) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		rc.set_value(tree_create(params, proc_range, part_range, box, depth, local_root));
	} else if (remote) {
		rc = hpx::async<tree_create_action>(hpx_localities()[proc_range.first], params, proc_range, part_range, box, depth, local_root);
	} else {
		rc = hpx::async([params,proc_range,part_range,depth,local_root, box]() {
			auto rc = tree_create(params,proc_range,part_range,box,depth,local_root);
			nthreads--;
			return rc;
		});
	}
	return rc;
}

tree_create_return tree_create(tree_create_params params, pair<int, int> proc_range, pair<int, int> part_range, range<double> box, int depth, bool local_root) {
	const double h = get_options().hsoft;
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	static const int bucket_size = std::min(SINK_BUCKET_SIZE, SOURCE_BUCKET_SIZE);
	tree_create_return rc;
	if (depth >= MAX_DEPTH) {
		THROW_ERROR("%s\n", "Maximum depth exceeded\n");
	}
	if (nodes.size() == 0) {
		last_ptr = nullptr;
		next_id = -tree_cache_line_size;
		nodes.resize(std::max(size_t(TREE_NODE_ALLOCATION_SIZE) * particles_size() / bucket_size, (size_t) NTREES_MIN));
//		PRINT("%i trees allocated\n", nodes.size());
		for (int i = 0; i < allocator_list.size(); i++) {
			allocator_list[i]->ready = false;
		}
	}
	if (local_root) {
		part_range.first = 0;
		part_range.second = particles_size();
	}
	array<tree_id, NCHILD> children;
	array<fixed32, NDIM> x;
	array<double, NDIM> Xc;
	multipole<float> multi;
	size_t nactive;
	float radius;
	double r;
	if (!allocator.ready) {
		allocator.reset();
		allocator.ready = true;
	}
	int node_count;
	int active_nodes = 0;
	const int index = allocator.allocate();
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
			const double wt = double(proc_range.second - mid) / double(proc_range.second - proc_range.first);
			const int dim = box.longest_dim();
			const double xmid = box.begin[dim] * wt + box.end[dim] * (1.0 - wt);
			left_box.end[dim] = right_box.begin[dim] = xmid;
			left_range.second = right_range.first = mid;
			left_local_root = left_range.second - left_range.first == 1;
			right_local_root = right_range.second - right_range.first == 1;
		} else {
			const int xdim = box.longest_dim();
			const double xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
			const int mid = particles_sort(part_range, xmid, xdim);
			left_parts.second = right_parts.first = mid;
			left_box.end[xdim] = right_box.begin[xdim] = xmid;
		}
		auto futl = tree_create_fork(params, left_range, left_parts, left_box, depth + 1, left_local_root, true);
		auto futr = tree_create_fork(params, right_range, right_parts, right_box, depth + 1, right_local_root, false);
		const auto rcl = futl.get();
		const auto rcr = futr.get();
		const auto xl = rcl.pos;
		const auto xr = rcr.pos;
		const auto ml = rcl.multi;
		const auto mr = rcr.multi;
		const double Rl = rcl.radius;
		const double Rr = rcr.radius;
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
		}
		norminv = 1.0 / std::sqrt(norminv);
		for (int dim = 0; dim < NDIM; dim++) {
			N[dim] *= norminv;
		}
		r = 0.0;
		if (mr[0] != 0.0 && ml[0.0] != 0.0) {
			for (int dim = 0; dim < NDIM; dim++) {
				double xmin, xmax;
				if (N[dim] > 0.0) {
					xmax = std::max(Xl[dim] + N[dim] * Rl, Xr[dim] + N[dim] * Rr);
					xmin = std::min(Xl[dim] - N[dim] * Rl, Xr[dim] - N[dim] * Rr);
				} else {
					xmax = std::max(Xl[dim] - N[dim] * Rl, Xr[dim] - N[dim] * Rr);
					xmin = std::min(Xl[dim] + N[dim] * Rl, Xr[dim] + N[dim] * Rr);
				}
				Xc[dim] = (xmax + xmin) * 0.5;
				r += sqr((xmax - xmin) * 0.5);
			}
		} else if (mr[0] == 0.0 && ml[0.0] == 0) {
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
			}
		} else if (mr[0] != 0.0) {
			Xc = Xr;
			r = Rr * Rr;
		} else {
			Xc = Xl;
			r = Rl * Rl;
		}
		radius = std::sqrt(r);
		r = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			r += sqr((box.begin[dim] - box.end[dim]) * 0.5);
		}
		r = std::sqrt(r);
		if (r < radius) {
			radius = r;
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
			}
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
		simdM = M2M < simd_double > (simdM, mdx);
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = simdM[i][LEFT] + simdM[i][RIGHT];
		}
		children[LEFT] = rcl.id;
		children[RIGHT] = rcr.id;
		node_count = 1 + rcl.node_count + rcr.node_count;
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
		for (int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				Xmax[dim] = std::max(Xmax[dim], x);
				Xmin[dim] = std::min(Xmin[dim], x);
			}
			if (particles_rung(i) >= params.min_rung) {
				nactive++;
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Xc[dim] = (Xmax[dim] + Xmin[dim]) * 0.5;
		}
		const int maxi = round_down(part_range.second - part_range.first, SIMD_FLOAT_SIZE) + part_range.first;
		array<simd_int, NDIM> Y;
		for (int dim = 0; dim < NDIM; dim++) {
			Y[dim] = fixed32(Xc[dim]).raw();
		}
		const simd_float _2float = fixed2float;
		for (int i = part_range.first; i < maxi; i += SIMD_FLOAT_SIZE) {
			array<simd_int, NDIM> X;
			for (int j = i; j < i + SIMD_FLOAT_SIZE; j++) {
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim][j - i] = particles_pos(dim, j).raw();
				}
			}
			array<simd_float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = simd_float(X[dim] - Y[dim]) * _2float;
			}
			const auto m = P2M(dx);
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[j] += m[j].sum();
			}
		}
		for (int i = maxi; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				dx[dim] = x - Xc[dim];
			}
			const auto m = P2M(dx);
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[j] += m[j];
			}
		}
		r = 0.0;
		for (int i = part_range.first; i < part_range.second; i++) {
			double this_radius = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				this_radius += sqr(x - Xc[dim]);
			}
			r = std::max(r, this_radius);
		}
		radius = std::sqrt(r);
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = M[i];
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = Xc[dim];
		}
		node_count = 1;
		if (nactive) {
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
	node.nactive = nactive;
	node.depth = depth;
	const int nparts = part_range.second - part_range.first;
	const bool global = proc_range.second - proc_range.first > 1;
	node.sink_leaf = !global && (depth >= params.min_level) && (nparts <= SINK_BUCKET_SIZE);
	node.source_leaf = !global && (depth >= params.min_level) && (nparts <= SOURCE_BUCKET_SIZE);
	nodes[index] = node;
	if (index > nodes.size()) {
		THROW_ERROR("%s\n", "Tree arena full\n");
	}
	rc.active_nodes = active_nodes;
	rc.id.index = index;
	rc.id.proc = hpx_rank();
	rc.multi = node.multi;
	rc.pos = node.pos;
	rc.radius = node.radius;
	rc.node_count = node.node_count;
	rc.nactive = nactive;
	if (local_root) {
//		PRINT("%i tree nodes remaining\n", nodes.size() - (int ) next_id);
	}
	//if (depth == 0)
//		PRINT("%i %e\n", index, nodes[index].radius);
	return rc;
}

void tree_destroy() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<tree_destroy_action>(c));
	}
	nodes = decltype(nodes)();
	tree_cache = decltype(tree_cache)();
	hpx::wait_all(futs.begin(), futs.end());
}

const tree_node* tree_get_node(tree_id id) {
	if (id.proc == hpx_rank()) {
		//	PRINT( "%i %e\n", id.index, nodes[id.index].radius);
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
	line_id.index = (id.index / line_size) * line_size;
	if (line_id != last_line || last_ptr == nullptr) {
		const size_t bin = tree_id_hash_lo()(id);
		std::unique_lock<spinlock_type> lock(mutex[bin]);
		auto iter = tree_cache[bin].find(line_id);
		if (iter == tree_cache[bin].end()) {
			auto prms = std::make_shared<hpx::lcos::local::promise<vector<tree_node>>>();
			tree_cache[bin][line_id] = prms->get_future();
			lock.unlock();
			hpx::apply([prms,line_id]() {
				auto line_fut = hpx::async<tree_fetch_cache_line_action>(hpx_localities()[line_id.proc],line_id.index);
				prms->set_value(line_fut.get());
			});
			lock.lock();
			iter = tree_cache[bin].find(line_id);
		}
		auto fut = iter->second;
		lock.unlock();
		ptr = fut.get().data();
	} else {
		ptr = last_ptr;
	}
	last_ptr = ptr;
	last_line = line_id;
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
