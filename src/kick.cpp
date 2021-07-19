#include <tigerfmm/fast_future.hpp>
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/math.hpp>

HPX_PLAIN_ACTION(kick);

fast_future<kick_return> kick_fork(kick_params params, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist, bool threadme) {
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
		rc.set_value(kick(params, self, std::move(dchecklist), std::move(echecklist)));
	} else if (remote) {
		rc = hpx::async<kick_action>(hpx_localities()[self_ptr->proc_range.first], params, self, std::move(dchecklist), std::move(echecklist));
	} else {
		rc = hpx::async([params,self] (vector<tree_id> dchecklist, vector<tree_id> echecklist) {
			auto rc = kick(params,self,std::move(dchecklist),std::move(echecklist));
			nthreads--;
			return rc;
		}, std::move(dchecklist), std::move(echecklist));
	}
	return rc;
}

kick_return kick(kick_params params, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist) {
	kick_return kr;

	vector<tree_id> nextlist;
	vector<tree_id> partlist;
	vector<tree_id> multlist;

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
		if (R2 > sqr(SINK_BIAS * self_ptr->radius + other_ptr->radius) * thetainv2) {
			multlist.push_back(echecklist[ci]);
		} else {
			nextlist.push_back(echecklist[ci]);
		}
	}
	echecklist = std::move(nextlist);
	gravity_cc_ewald(std::move(multlist));

	int pass = 0;
	do {
		for (int ci = 0; ci < dchecklist.size(); ci++) {
			const tree_node* other_ptr = tree_get_node(dchecklist[ci]);
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self_ptr->pos[dim], other_ptr->pos[dim]);
			}
			const float R2 = std::max(EWALD_DIST2, sqr(dx[XDIM], dx[YDIM], dx[ZDIM]));
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
				gravity_cc(std::move(multlist));
				gravity_cp(std::move(partlist));
			}
		}
		dchecklist = std::move(nextlist);
		pass++;
	} while (dchecklist.size() && self_ptr->is_leaf());

	if (self_ptr->is_leaf()) {
		gravity_pc(std::move(multlist));
		gravity_pp(std::move(partlist));
	} else {
		auto futl = kick_fork(params, self_ptr->children[LEFT], dchecklist, echecklist, true);
		auto futr = kick_fork(params, self_ptr->children[RIGHT], std::move(dchecklist), std::move(echecklist), false);
		const auto rcl = futl.get();
		const auto rcr = futr.get();
		kr.max_rung = std::max(rcl.max_rung, rcr.max_rung);
	}

	return kr;
}
