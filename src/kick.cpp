constexpr bool verbose = true;
#include <tigerfmm/fast_future.hpp>
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/safe_io.hpp>

HPX_PLAIN_ACTION(kick);

struct kick_workspace {
	vector<tree_id> nextlist;
	vector<tree_id> partlist;
	vector<tree_id> multlist;
	kick_workspace() = default;
	kick_workspace(const kick_workspace&) = delete;
	kick_workspace& operator=(const kick_workspace&) = delete;
	kick_workspace(kick_workspace&&) = default;
	kick_workspace& operator=(kick_workspace&&) = default;
};

static thread_local std::stack<kick_workspace> workspaces;

static kick_workspace get_workspace() {
	if (workspaces.empty()) {
		workspaces.push(kick_workspace());
	}
	kick_workspace workspace = std::move(workspaces.top());
	workspaces.pop();
	return std::move(workspace);
}

static void cleanup_workspace(kick_workspace&& workspace) {
	workspace.multlist.resize(0);
	workspace.partlist.resize(0);
	workspace.nextlist.resize(0);
	workspaces.push(std::move(workspace));
}

fast_future<kick_return> kick_fork(kick_params params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist,
		vector<tree_id> echecklist, bool threadme) {
	static std::atomic<int> nthreads(0);
	fast_future<kick_return> rc;
	const tree_node* self_ptr = tree_get_node(self);
	bool remote = false;
	if (self.proc != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = self_ptr->part_range.second - self_ptr->part_range.first > MIN_KICK_THREAD_PARTS
				|| self_ptr->proc_range.second - self_ptr->proc_range.first > 1;
		if (threadme) {
			if (nthreads++ < KICK_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency()) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		rc.set_value(kick(params, L, pos, self, std::move(dchecklist), std::move(echecklist)));
	} else if (remote) {
		rc = hpx::async<kick_action>(hpx_localities()[self_ptr->proc_range.first], params, L, pos, self, std::move(dchecklist), std::move(echecklist));
	} else {
		rc = hpx::async([params,self,L,pos] (vector<tree_id> dchecklist, vector<tree_id> echecklist) {
			auto rc = kick(params,L,pos,self,std::move(dchecklist),std::move(echecklist));
			nthreads--;
			return rc;
		}, std::move(dchecklist), std::move(echecklist));
	}
	return rc;
}

kick_return kick(kick_params params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist) {
	kick_return kr;

	auto workspace = get_workspace();
	vector<tree_id>& nextlist = workspace.nextlist;
	vector<tree_id>& partlist = workspace.partlist;
	vector<tree_id>& multlist = workspace.multlist;

	const float thetainv2 = 1.0 / sqr(params.theta);
	const float thetainv = 1.0 / params.theta;
	const tree_node* self_ptr = tree_get_node(self);

	for (int ci = 0; ci < echecklist.size(); ci++) {
		const tree_node* other_ptr = tree_get_node(echecklist[ci]);
		array<float, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = distance(self_ptr->pos[dim], other_ptr->pos[dim]);
		}
		const float R2 = std::max(EWALD_DIST2, sqr(dx[XDIM], dx[YDIM], dx[ZDIM]));
		const float r2 = sqr(SINK_BIAS * self_ptr->radius + other_ptr->radius) * thetainv2;
		if (R2 > r2) {
			multlist.push_back(echecklist[ci]);
		} else {
			nextlist.push_back(other_ptr->children[LEFT]);
			nextlist.push_back(other_ptr->children[RIGHT]);
		}
	}
	std::swap(echecklist, nextlist);
	gravity_cc_ewald(multlist);
	nextlist.resize(0);
	multlist.resize(0);

	int pass = 0;
	do {
		for (int ci = 0; ci < dchecklist.size(); ci++) {
			const tree_node* other_ptr = tree_get_node(dchecklist[ci]);
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self_ptr->pos[dim], other_ptr->pos[dim]);
			}
			const float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
			const bool far1 = R2 > sqr(SINK_BIAS * self_ptr->radius + other_ptr->radius) * thetainv2;
			const bool far2 = R2 > sqr(SINK_BIAS * self_ptr->radius * thetainv + other_ptr->radius);
			const bool far3 = R2 > sqr(self_ptr->radius + other_ptr->radius * thetainv);
			if (far1 || (pass > 0 && far3)) {
				multlist.push_back(dchecklist[ci]);
			} else if ((far2 || pass > 0) && other_ptr->is_leaf()) {
				partlist.push_back(dchecklist[ci]);
			} else if (other_ptr->is_leaf()) {
				nextlist.push_back(dchecklist[ci]);
			} else {
				const auto child_checks = other_ptr->children;
				nextlist.push_back(child_checks[LEFT]);
				nextlist.push_back(child_checks[RIGHT]);
			}
			if (pass == 0) {
				gravity_cc(multlist);
				gravity_cp(partlist);
				multlist.resize(0);
				partlist.resize(0);
			}
		}
		std::swap(dchecklist, nextlist);
		nextlist.resize(0);
		pass++;
	} while (dchecklist.size() && self_ptr->is_leaf());

	if (self_ptr->is_leaf()) {
		gravity_pc(multlist);
		gravity_pp(partlist);
		cleanup_workspace(std::move(workspace));
	} else {
		cleanup_workspace(std::move(workspace));
		auto futl = kick_fork(params, L, self_ptr->pos, self_ptr->children[LEFT], dchecklist, echecklist, true);
		auto futr = kick_fork(params, L, self_ptr->pos, self_ptr->children[RIGHT], std::move(dchecklist), std::move(echecklist), false);
		const auto rcl = futl.get();
		const auto rcr = futr.get();
		kr.max_rung = std::max(rcl.max_rung, rcr.max_rung);
	}

	return kr;
}
