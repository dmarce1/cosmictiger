constexpr bool verbose = false;

#include <tigerfmm/fast_future.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/safe_io.hpp>
#include <tigerfmm/tree.hpp>

HPX_PLAIN_ACTION(tree_create);
HPX_PLAIN_ACTION(tree_destroy);

static vector<tree_node> nodes;
static shared_mutex_type mutex;
static std::atomic<int> num_threads(0);
static std::atomic<int> next_id;

fast_future<tree_create_return> tree_create_fork(const pair<int, int>& proc_range, const pair<int, int>& part_range, const range<double>& box,
		const size_t& morton_id, const bool& local_root, bool threadme) {
	static std::atomic<int> nthreads(0);
	fast_future<tree_create_return> rc;
	bool remote = false;
	if (proc_range.second - proc_range.first > 1 || proc_range.first != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = part_range.second - part_range.first > MIN_SORT_THREAD_PARTS;
		if (threadme) {
			if (nthreads++ < hpx::thread::hardware_concurrency()) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		rc.set_value(tree_create(proc_range, part_range, box, morton_id, local_root));
	} else if (remote) {
		rc = hpx::async<tree_create_action>(hpx_localities()[proc_range.first], proc_range, part_range, box, morton_id, local_root);
	} else {
		rc = hpx::async([proc_range,part_range,morton_id,local_root, box] {
			auto rc = tree_create(proc_range,part_range,box,morton_id,local_root);
			nthreads--;
			return rc;
		});
	}
	return rc;
}

tree_create_return tree_create(pair<int, int> proc_range, pair<int, int> part_range, range<double> box, size_t morton_id, bool local_root) {
	const double h = get_options().hsoft;
	tree_create_return rc;
	if (nodes.size() == 0) {
		next_id = 0;
		nodes.resize(4 * particles_size() / BUCKET_SIZE);
	}
	if (local_root) {
		part_range.first = 0;
		part_range.second = particles_size();
	}
	array<tree_id, NCHILD> children;
	array<fixed32, NDIM> x;
	multipole<float> multi;
	float radius;
	if (proc_range.second - proc_range.first > 1 || part_range.second - part_range.first > BUCKET_SIZE) {
		auto left_morton = morton_id << 1;
		auto right_morton = (morton_id << 1) | 1;
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
		auto futl = tree_create_fork(left_range, left_parts, left_box, left_morton, left_local_root, true);
		auto futr = tree_create_fork(right_range, right_parts, right_box, right_morton, right_local_root, false);
		const auto rcl = futl.get();
		const auto rcr = futr.get();
		const auto xl = rcl.pos;
		const auto xr = rcr.pos;
		const auto ml = rcl.multi;
		const auto mr = rcr.multi;
		const double Rl = rcl.radius;
		const double Rr = rcr.radius;
		double rr;
		double rl;
		array<double, NDIM> Xl;
		array<double, NDIM> Xr;
		array<double, NDIM> Xc;
		array<double, NDIM> N;
		multipole<double> Mr;
		multipole<double> Ml;
		float R;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			Mr[i] = mr[i];
			Ml[i] = ml[i];
		}
		double norminv = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			Xl[dim] = xl[dim].to_double();
			Xr[dim] = xr[dim].to_double();
			N[dim] = Xl[dim] - Xr[dim];
			norminv = sqr(N[dim]);
		}
		norminv = 1.0 / std::sqrt(norminv);
		for (int dim = 0; dim < NDIM; dim++) {
			N[dim] *= norminv;
		}
		rr = rl = 0.0;
		if (Mr[0] != 0.0 && Ml[0.0] != 0.0) {
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (Xl[dim] + Xr[dim] + N[dim] * (Rl - Rr)) * 0.5;
				rl += sqr(Xl[dim] - Xc[dim]);
				rr += sqr(Xr[dim] - Xc[dim]);
			}
		} else if (Mr[0] == 0.0 && Ml[0.0] == 0) {
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
			}
		} else if (Mr[0] != 0.0) {
			Xc = Xr;
		} else {
			Xc = Xl;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = Xc[dim];
		}
		rl = std::sqrt(rl) + Rl;
		rr = std::sqrt(rr) + Rr;
		radius = std::max(rl, rr);
		Mr = M2M<double>(Mr, Xr);
		Ml = M2M<double>(Ml, Xl);
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = Mr[i] + Ml[i];
		}
		children[LEFT] = rcl.id;
		children[RIGHT] = rcr.id;
	} else {
		children[LEFT].index = children[RIGHT].index = -1;
		multipole<double> M;
		array<double, NDIM> X;
		array<double, NDIM> Xmax;
		array<double, NDIM> Xmin;
		double r;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			M[i] = 0.0;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Xmax[dim] = box.begin[dim];
			Xmin[dim] = box.end[dim];
		}
		for (int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				X[dim] = x;
				Xmax[dim] = std::max(Xmax[dim], x);
				Xmin[dim] = std::min(Xmin[dim], x);
			}
			const auto m = P2M(X);
			for (int i = 0; i < MULTIPOLE_SIZE; i++) {
				M[i] += m[i];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = (Xmax[dim] + Xmin[dim]) * 0.5;
		}
		r = 0.0;
		for (int i = part_range.first; i < part_range.second; i++) {
			double this_radius = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				this_radius += sqr(x - X[dim]);
			}
			r = std::max(r, this_radius);
		}
		radius = std::sqrt(r);
		r = 0.0;
		r = std::max(r, sqr(box.begin[XDIM] - X[XDIM], box.begin[YDIM] - X[YDIM], box.begin[ZDIM] - X[ZDIM]));
		r = std::max(r, sqr(box.begin[XDIM] - X[XDIM], box.begin[YDIM] - X[YDIM], box.end[ZDIM] - X[ZDIM]));
		r = std::max(r, sqr(box.begin[XDIM] - X[XDIM], box.end[YDIM] - X[YDIM], box.begin[ZDIM] - X[ZDIM]));
		r = std::max(r, sqr(box.begin[XDIM] - X[XDIM], box.end[YDIM] - X[YDIM], box.end[ZDIM] - X[ZDIM]));
		r = std::max(r, sqr(box.end[XDIM] - X[XDIM], box.begin[YDIM] - X[YDIM], box.begin[ZDIM] - X[ZDIM]));
		r = std::max(r, sqr(box.end[XDIM] - X[XDIM], box.begin[YDIM] - X[YDIM], box.end[ZDIM] - X[ZDIM]));
		r = std::max(r, sqr(box.end[XDIM] - X[XDIM], box.end[YDIM] - X[YDIM], box.begin[ZDIM] - X[ZDIM]));
		r = std::max(r, sqr(box.end[XDIM] - X[XDIM], box.end[YDIM] - X[YDIM], box.end[ZDIM] - X[ZDIM]));
		radius = std::min((double) radius, std::sqrt(r)) + h;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			multi[i] = M[i];
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = X[dim];
		}
	}
	tree_node node;
	node.radius = radius;
	node.children = children;
	node.local_root = local_root;
	node.morton_id = morton_id;
	node.part_range = part_range;
	node.proc_range = proc_range;
	node.pos = x;
	node.multi = multi;
	node.index = next_id++;
	nodes[node.index] = node;
	if( node.index > nodes.size() ) {
		THROW_ERROR( "Tree arena full\n");
	}
	rc.id.index = node.index;
	rc.id.proc = hpx_rank();
	rc.multi = node.multi;
	rc.pos = node.pos;
	rc.radius = node.radius;
	return rc;
}

void tree_destroy() {
	vector<hpx::future<void>> futs;
	const auto children = hpx_children();
	for (const auto& c : children) {
		futs.push_back(hpx::async<tree_destroy_action>(c));
	}
	nodes.resize(0);
	hpx::wait_all(futs.begin(), futs.end());
}
