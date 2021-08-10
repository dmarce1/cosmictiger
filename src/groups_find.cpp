#include <cosmictiger/groups_find.hpp>
#include <cosmictiger/group_tree.hpp>

HPX_PLAIN_ACTION (groups_find);

hpx::future<void> groups_find_fork(tree_id self, vector<tree_id> checklist, double link_len, bool threadme) {
	static std::atomic<int> nthreads(0);
	hpx::future<void> rc;
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

hpx::future<void> groups_find(tree_id self, vector<tree_id> checklist, double link_len) {
	const group_tree_node* self_ptr = group_tree_get_node(self);
	bool thread_left = true;
	vector<tree_id> nextlist;
	vector<tree_id> leaflist;
	array<const group_tree_node*, SIMD_FLOAT_SIZE> other_ptrs;
	const auto self_box = self_ptr->box.pad(link_len * 1.001);
	const bool iamleaf = self_ptr->children[LEFT].index == -1;
	do {
		for (int ci = 0; ci < checklist.size(); ci += SIMD_FLOAT_SIZE) {
			const group_tree_node* other_ptr = group_tree_get_node(checklist[ci]);
			if (self_box.intersection(other_ptr->box).volume() > 0) {
				if (other_ptr->children[LEFT].index == -1) {
					nextlist.push_back(checklist[ci]);
				} else {
					nextlist.push_back(other_ptr->children[LEFT]);
					nextlist.push_back(other_ptr->children[RIGHT]);
				}
			}
		}
		std::swap(nextlist, checklist);
		nextlist.resize(0);
	} while (iamleaf && nextlist.size());

	if (self_ptr->children[LEFT].index == -1) {
		const auto my_rng = self_ptr->part_range;
		const double R2 = sqr(link_len);
		for (int i = 0; i < leaflist.size(); i++) {
			const group_tree_node* other_ptr = group_tree_get_node(leaflist[i]);
			const auto& other_rng = other_ptr->part_range;
			for (part_int j = other_rng.first; j < other_rng.second; j++) {
				for (part_int k = my_rng.first; k < my_rng.second; k++) {
					const double x = (particles_pos(XDIM, k) - particles_pos(XDIM, j)).to_double();
					const double y = (particles_pos(YDIM, k) - particles_pos(YDIM, j)).to_double();
					const double z = (particles_pos(ZDIM, k) - particles_pos(ZDIM, j)).to_double();
					if (sqr(x, y, z) < R2) {

					}
				}
			}
		}

		return hpx::make_ready_future();
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		const group_tree_node* cl = group_tree_get_node(self_ptr->children[LEFT]);
		const group_tree_node* cr = group_tree_get_node(self_ptr->children[RIGHT]);
		std::array<hpx::future<void>, NCHILD> futs;
		futs[RIGHT] = groups_find_fork(self_ptr->children[RIGHT], checklist, link_len, true);
		futs[LEFT] = groups_find_fork(self_ptr->children[LEFT], std::move(checklist), link_len, false);
		if (futs[LEFT].is_ready() && futs[RIGHT].is_ready()) {
			futs[LEFT].get();
			futs[RIGHT].get();
			return hpx::make_ready_future();
		} else {
			return hpx::when_all(futs.begin(), futs.end());
		}
	}
}

