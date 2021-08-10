#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/group_tree.hpp>

HPX_PLAIN_ACTION (group_tree_create);

class group_tree_allocator {
	int next;
	int last;
public:
	bool ready;
	void reset();
	bool is_ready();
	group_tree_allocator();
	~group_tree_allocator();
	group_tree_allocator(const group_tree_allocator&) = default;
	group_tree_allocator& operator=(const group_tree_allocator&) = default;
	group_tree_allocator(group_tree_allocator&&) = default;
	group_tree_allocator& operator=(group_tree_allocator&&) = default;
	int allocate();
};

static vector<group_tree_allocator*> allocator_list;
static array<spinlock_type, TREE_CACHE_SIZE> mutex;
static thread_local group_tree_allocator allocator;
static std::atomic<int> next_id;
static vector<group_tree_node> nodes;

fast_future<tree_id> group_tree_create_fork(pair<int, int> proc_range, pair<part_int> part_range, group_range box, int depth, bool local_root, bool threadme) {
	static std::atomic<int> nthreads(0);
	fast_future<tree_id> rc;
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
		rc.set_value(group_tree_create(proc_range, part_range, box, depth, local_root));
	} else if (remote) {
		rc = hpx::async < group_tree_create_action > (hpx_localities()[proc_range.first], proc_range, part_range, box, depth, local_root);
	} else {
		rc = hpx::async([proc_range,part_range,depth,local_root, box]() {
			auto rc = group_tree_create(proc_range,part_range,box,depth,local_root);
			nthreads--;
			return rc;
		});
	}
	return rc;

}

tree_id group_tree_create(pair<int, int> proc_range, pair<part_int> part_range, group_range box, int depth, bool local_root) {
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	if (depth >= MAX_DEPTH) {
		THROW_ERROR("%s\n", "Maximum depth exceeded\n");
	}
	if (nodes.size() == 0) {
		next_id = -tree_cache_line_size;
		nodes.resize(std::max(size_t(size_t(TREE_NODE_ALLOCATION_SIZE) * particles_size() / GROUP_BUCKET_SIZE), (size_t) 1));
		for (int i = 0; i < allocator_list.size(); i++) {
			allocator_list[i]->ready = false;
		}
	}
	if (local_root) {
		part_range.first = 0;
		part_range.second = particles_size();
	}
	if (!allocator.ready) {
		allocator.reset();
		allocator.ready = true;
	}
	const int index = allocator.allocate();
	group_tree_node node;
	if (proc_range.second - proc_range.first > 1 || part_range.second - part_range.first > GROUP_BUCKET_SIZE) {
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
			const part_int mid = particles_sort(part_range, xmid, xdim);
			left_parts.second = right_parts.first = mid;
			left_box.end[xdim] = right_box.begin[xdim] = xmid;
		}
		auto futr = group_tree_create_fork(right_range, right_parts, right_box, depth + 1, right_local_root, true);
		auto futl = group_tree_create_fork(left_range, left_parts, left_box, depth + 1, left_local_root, false);
		node.children[LEFT] = futl.get();
		node.children[RIGHT] = futr.get();
	} else {
		node.children[LEFT].index = node.children[RIGHT].index = -1;
	}
	node.proc_range = proc_range;
	node.part_range = part_range;
	node.box = box;
	node.local_root = local_root;
	tree_id myid;
	myid.index = index;
	myid.proc = hpx_rank();
	return myid;
}

void group_tree_allocator::reset() {
	const int tree_cache_line_size = get_options().tree_cache_line_size;
	next = (next_id += tree_cache_line_size);
	last = next + tree_cache_line_size;
	if (last > nodes.size()) {
		THROW_ERROR("%s\n", "Tree arena full");
	}
}

group_tree_allocator::group_tree_allocator() {
	ready = false;
	std::lock_guard<spinlock_type> lock(mutex[0]);
	allocator_list.push_back(this);
}

int group_tree_allocator::allocate() {
	if (next == last) {
		reset();
	}
	return next++;
}

group_tree_allocator::~group_tree_allocator() {
	std::lock_guard<spinlock_type> lock(mutex[0]);
	for (int i = 0; i < allocator_list.size(); i++) {
		if (allocator_list[i] == this) {
			allocator_list[i] = allocator_list.back();
			allocator_list.pop_back();
			break;
		}
	}
}

