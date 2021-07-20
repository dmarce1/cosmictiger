constexpr bool verbose = true;
#include <tigerfmm/fast_future.hpp>
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/safe_io.hpp>

HPX_PLAIN_ACTION (kick);

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

	const simd_float thetainv2(1.0 / sqr(params.theta));
	const simd_float thetainv(1.0 / params.theta);
	static const simd_float sink_bias(SINK_BIAS);
	static const simd_float ewald_dist2(EWALD_DIST2);
	const tree_node* self_ptr = tree_get_node(self);
	array<const tree_node*, SIMD_FLOAT_SIZE> other_ptrs;

	simd_float self_radius = self_ptr->radius;
	array<simd_int, NDIM> self_pos;
	for (int dim = 0; dim < NDIM; dim++) {
		self_pos[dim] = self_ptr->pos[dim].raw();
	}
	array<simd_int, NDIM> other_pos;
	array<simd_float, NDIM> dx;
	simd_float other_radius;
	simd_int other_leaf;
	for (int ci = 0; ci < echecklist.size(); ci += SIMD_FLOAT_SIZE) {
		const int maxci = std::min((int) echecklist.size(), ci + SIMD_FLOAT_SIZE);
		const int maxi = maxci - ci;
		for (int i = ci; i < maxci; i++) {
			other_ptrs[i - ci] = tree_get_node(echecklist[i]);
		}
		for (int i = 0; i < maxi; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				other_pos[dim][i] = other_ptrs[i]->pos[dim].raw();
			}
			other_radius[i] = other_ptrs[i]->radius;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = simd_float(self_pos[dim] - other_pos[dim]) * fixed2float;
		}
		const simd_float R2 = max(ewald_dist2, sqr(dx[XDIM], dx[YDIM], dx[ZDIM]));
		const simd_float r2 = sqr(sink_bias * self_radius + other_radius) * thetainv2;
		const simd_float far = R2 > r2;
		for (int i = 0; i < maxi; i++) {
			if (far[i]) {
				multlist.push_back(echecklist[ci + i]);
			} else {
				nextlist.push_back(other_ptrs[i]->children[LEFT]);
				nextlist.push_back(other_ptrs[i]->children[RIGHT]);
			}
		}
	}
	std::swap(echecklist, nextlist);
	gravity_cc_ewald(multlist);
	nextlist.resize(0);
	multlist.resize(0);

	int pass = 0;
	do {
		const simd_int passozero(pass > 0);
		for (int ci = 0; ci < dchecklist.size(); ci += SIMD_FLOAT_SIZE) {
			const int maxci = std::min((int) dchecklist.size(), ci + SIMD_FLOAT_SIZE);
			const int maxi = maxci - ci;
			for (int i = ci; i < maxci; i++) {
				other_ptrs[i - ci] = tree_get_node(dchecklist[i]);
			}
			for (int i = 0; i < maxi; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					other_pos[dim][i] = other_ptrs[i]->pos[dim].raw();
				}
				other_radius[i] = other_ptrs[i]->radius;
				other_leaf[i] = other_ptrs[i]->is_leaf();
			}
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = simd_float(self_pos[dim] - other_pos[dim]) * fixed2float;
			}
			const simd_float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
			const simd_float far1 = R2 > sqr(sink_bias * self_radius + other_radius) * thetainv2;
			const simd_float far2 = R2 > sqr(sink_bias * self_radius * thetainv + other_radius);
			const simd_float far3 = R2 > sqr(self_radius + other_radius * thetainv);
			const simd_float mult = far1 + (passozero * far3);
			const simd_float part = (far2 + passozero) * other_leaf;
			for (int i = 0; i < maxi; i++) {
				if (mult[i]) {
					multlist.push_back(dchecklist[ci + i]);
				} else if (part[i]) {
					partlist.push_back(dchecklist[ci + i]);
				} else if (other_leaf[i]) {
					nextlist.push_back(dchecklist[ci + i]);
				} else {
					const auto child_checks = other_ptrs[i]->children;
					nextlist.push_back(child_checks[LEFT]);
					nextlist.push_back(child_checks[RIGHT]);

				}
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
