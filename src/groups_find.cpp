#include <cosmictiger/groups_find.hpp>
#include <cosmictiger/group_tree.hpp>
#include <cosmictiger/gravity.hpp>

#include <stack>

HPX_PLAIN_ACTION (groups_find);

void atomic_min(std::atomic<group_int>& min_value, group_int value) {
	group_int prev_value = min_value;
	while (prev_value > value && !min_value.compare_exchange_strong(prev_value, value)) {
	}
}

hpx::future<size_t> groups_find_fork(tree_id self, vector<tree_id> checklist, double link_len, bool threadme) {
	static std::atomic<int> nthreads(0);
	hpx::future < size_t > rc;
	const group_tree_node* self_ptr = group_tree_get_node(self);
	bool remote = false;
	if (self.proc != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = self_ptr->part_range.second - self_ptr->part_range.first > MIN_KICK_THREAD_PARTS;
		if (threadme) {
			if (nthreads++ < KICK_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency() || (self_ptr->proc_range.second - self_ptr->proc_range.first > 1)) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		rc = groups_find(self, std::move(checklist), link_len);
	} else if (remote) {
		ASSERT(self_ptr->proc_range.first >= 0);
		ASSERT(self_ptr->proc_range.first < hpx_size());
		rc = hpx::async < groups_find_action > (hpx_localities()[self_ptr->proc_range.first], self, std::move(checklist), link_len);
	} else {
		rc = hpx::async([self,link_len] (vector<tree_id> checklist) {
			auto rc = groups_find(self,std::move(checklist), link_len);
			nthreads--;
			return rc;
		}, std::move(checklist));
	}
	return rc;
}

struct leaf_workspace {
	vector<fixed32> X;
	vector<fixed32> Y;
	vector<fixed32> Z;
	vector<group_int> G;
};

static thread_local std::stack<leaf_workspace> leaf_workspaces;
static thread_local std::stack<vector<tree_id>> lists;

static vector<tree_id> get_list() {
	if (lists.empty()) {
		lists.push(vector<tree_id>());
	}
	auto list = std::move(lists.top());
	lists.pop();
	return std::move(list);
}

static void cleanup_list(vector<tree_id> && list) {
	lists.push(std::move(list));
}

static leaf_workspace get_leaf_workspace() {
	if (leaf_workspaces.empty()) {
		leaf_workspaces.push(leaf_workspace());
	}
	auto list = std::move(leaf_workspaces.top());
	leaf_workspaces.pop();
	return std::move(list);
}

static void cleanup_leaf_workspace(leaf_workspace && list) {
	leaf_workspaces.push(std::move(list));
}

hpx::future<size_t> groups_find(tree_id self, vector<tree_id> checklist, double link_len) {
	const group_tree_node* self_ptr = group_tree_get_node(self);
	bool thread_left = true;
	vector<tree_id> nextlist = get_list();
	vector<tree_id> leaflist = get_list();
	nextlist.resize(0);
	leaflist.resize(0);
	const auto self_box = self_ptr->box.pad(link_len * 1.001);
	const bool iamleaf = self_ptr->children[LEFT].index == -1;
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const group_tree_node* other_ptr = group_tree_get_node(checklist[ci]);
			if (other_ptr->active) {
				if (self_box.periodic_intersection(other_ptr->box).volume() > 0) {
					if (other_ptr->children[LEFT].index == -1) {
						leaflist.push_back(checklist[ci]);
					} else {
						nextlist.push_back(other_ptr->children[LEFT]);
						nextlist.push_back(other_ptr->children[RIGHT]);
					}
				}
			}
		}
		std::swap(nextlist, checklist);
		nextlist.resize(0);
	} while (iamleaf && nextlist.size());
	if (self_ptr->children[LEFT].index == -1) {
		leaf_workspace ws = get_leaf_workspace();
		vector<fixed32>& X = ws.X;
		vector<fixed32>& Y = ws.Y;
		vector<fixed32>& Z = ws.Z;
		vector<group_int>& G = ws.G;
		if (leaflist.size()) {
			const auto my_rng = self_ptr->part_range;
			const float link_len2 = sqr(link_len);
			bool found_any_link = false;
			int total_size = 0;
			for (int i = 0; i < leaflist.size(); i++) {
				const group_tree_node* other_ptr = group_tree_get_node(leaflist[i]);
				if (other_ptr != self_ptr) {
					const auto& other_rng = other_ptr->part_range;
					const int other_size = other_rng.second - other_rng.first;
					total_size += other_size;
				}
			}
			X.resize(total_size);
			Y.resize(total_size);
			Z.resize(total_size);
			G.resize(total_size);
			total_size = 0;
			for (int i = 0; i < leaflist.size(); i++) {
				const group_tree_node* other_ptr = group_tree_get_node(leaflist[i]);
				if (other_ptr != self_ptr) {
					const auto& other_rng = other_ptr->part_range;
					int other_size = other_rng.second - other_rng.first;
					particles_global_read_pos_and_group(other_ptr->global_part_range(), X.data(), Y.data(), Z.data(), G.data(), total_size);
					for (int i = total_size; i < total_size + other_size; i++) {
						array<double, NDIM> x;
						x[XDIM] = X[i].to_double();
						x[YDIM] = Y[i].to_double();
						x[ZDIM] = Z[i].to_double();
						if (!self_box.contains(x)) {
							const int j = total_size + other_size - 1;
							X[i] = X[j];
							Y[i] = Y[j];
							Z[i] = Z[j];
							G[i] = G[j];
							other_size--;
							i--;
						}
					}
					total_size += other_size;
				}
			}
			for (part_int k = my_rng.first; k < my_rng.second; k++) {
				const auto myx = particles_pos(XDIM, k);
				const auto myy = particles_pos(YDIM, k);
				const auto myz = particles_pos(ZDIM, k);
				for (int j = 0; j < total_size; j++) {
					const float x = distance(myx, X[j]);
					const float y = distance(myy, Y[j]);
					const float z = distance(myz, Z[j]);
					const float r2 = sqr(x, y, z);
					if (r2 < link_len2) {
						auto& grp = particles_group(k);
						const group_int start_group = particles_group(k);
						if ((group_int) grp == NO_GROUP) {
							grp = particles_group_init(k);
							found_any_link = true;
						}
						if (grp > G[j]) {
							grp = G[j];
							found_any_link = true;
						}
					}
				}
			}
			bool found_link;
			do {
				found_link = false;
				for (part_int j = my_rng.first; j < my_rng.second; j++) {
					for (part_int k = j + 1; k < my_rng.second; k++) {
						const float x = distance(particles_pos(XDIM, k), particles_pos(XDIM, j));
						const float y = distance(particles_pos(YDIM, k), particles_pos(YDIM, j));
						const float z = distance(particles_pos(ZDIM, k), particles_pos(ZDIM, j));
						if (sqr(x, y, z) < link_len2) {
							auto& grpa = particles_group(k);
							auto& grpb = particles_group(j);
							if ((group_int) grpa == NO_GROUP) {
								grpa = particles_group_init(k);
								found_link = true;
								found_any_link = true;
							}
							if ((group_int) grpb == NO_GROUP) {
								grpb = particles_group_init(j);
								found_link = true;
								found_any_link = true;
							}
							if (grpa != grpb) {
								if (grpa < grpb) {
									grpb = (group_int) grpa;
								} else {
									grpa = (group_int) grpb;
								}
								found_link = true;
								found_any_link = true;
							}
						}
					}
				}
			} while (found_link);
			cleanup_list(std::move(nextlist));
			cleanup_list(std::move(leaflist));
			cleanup_leaf_workspace(std::move(ws));
			if (found_any_link) {
				group_tree_set_active(self, true);
				return hpx::make_ready_future((size_t) 1);
			} else {
				group_tree_set_active(self, false);
				return hpx::make_ready_future((size_t) 0);
			}
		} else {
			cleanup_list(std::move(nextlist));
			cleanup_list(std::move(leaflist));
			cleanup_leaf_workspace(std::move(ws));
			return hpx::make_ready_future((size_t) 0);
		}
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		cleanup_list(std::move(nextlist));
		cleanup_list(std::move(leaflist));
		if (checklist.size()) {
			const group_tree_node* cl = group_tree_get_node(self_ptr->children[LEFT]);
			const group_tree_node* cr = group_tree_get_node(self_ptr->children[RIGHT]);
			std::array<hpx::future<size_t>, NCHILD> futs;
			futs[RIGHT] = groups_find_fork(self_ptr->children[RIGHT], checklist, link_len, true);
			futs[LEFT] = groups_find_fork(self_ptr->children[LEFT], std::move(checklist), link_len, false);
			if (futs[LEFT].is_ready() && futs[RIGHT].is_ready()) {
				auto rcl = futs[LEFT].get();
				auto rcr = futs[RIGHT].get();
				const auto tot = rcl + rcr;
				group_tree_set_active(self, tot != 0);
				return hpx::make_ready_future(tot);
			} else {
				return hpx::when_all(futs.begin(), futs.end()).then([self](hpx::future<std::vector<hpx::future<size_t>>> futfut) {
					auto futs = futfut.get();
					const auto tot = futs[LEFT].get() + futs[RIGHT].get();
					group_tree_set_active(self, tot != 0);
					return tot;
				});
			}
		} else {
			return hpx::make_ready_future((size_t) 0);
		}
	}
}

