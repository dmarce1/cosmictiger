/*
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
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

constexpr bool verbose = true;
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/flops.hpp>

#include <unistd.h>
#include <stack>

HPX_PLAIN_ACTION (kick);

simd_float box_intersects_sphere(const range<simd_double8>& box, const array<simd_int, NDIM>& x, simd_float r) {
	simd_float d2 = 0.0f;
	for (int dim = 0; dim < NDIM; dim++) {
		const simd_double8 X = simd_double8(x[dim]) * simd_double8(fixed32::to_float_factor);
		simd_double8 d2begin = X - box.begin[dim];
		simd_double8 d2beginp1 = d2begin + simd_double8(1.0);
		simd_double8 d2beginm1 = d2begin + simd_double8(-1.0);
		d2begin = sqr(d2begin);
		d2beginp1 = sqr(d2beginp1);
		d2beginm1 = sqr(d2beginm1);
		d2begin = min(d2begin, min(d2beginp1, d2beginm1));
		simd_double8 d2end = X - box.end[dim];
		simd_double8 d2endp1 = d2end + simd_double8(1.0);
		simd_double8 d2endm1 = d2end + simd_double8(-1.0);
		d2end = sqr(d2end);
		d2endp1 = sqr(d2endp1);
		d2endm1 = sqr(d2endm1);
		d2end = min(d2end, min(d2endp1, d2endm1));
		d2 += simd_float(((X < box.begin[dim]) + (X > box.end[dim])) * min(d2begin, d2end));
	}
	return d2 < sqr(r);
}

#define MAX_ACTIVE_WORKSPACES 1

struct workspace {
	vector<tree_id> nextlist;
	vector<tree_id> cplist;
	vector<tree_id> pclist;
	vector<tree_id> cclist;
	vector<tree_id> leaflist;
	workspace() = default;
	workspace(const workspace&) = delete;
	workspace& operator=(const workspace&) = delete;
	workspace(workspace&&) = default;
	workspace& operator=(workspace&&) = default;
};

static thread_local std::stack<workspace> local_workspaces;
static std::atomic<size_t> parts_covered;
static hpx::promise<kick_return> local_return_promise;
static hpx::future<kick_return> local_return_future;
static kick_return local_return;

static workspace get_workspace() {
	if (local_workspaces.empty()) {
		local_workspaces.push(workspace());
	}
	workspace workspace = std::move(local_workspaces.top());
	local_workspaces.pop();
	return std::move(workspace);
}

static void cleanup_workspace(workspace&& workspace) {
	workspace.cclist.resize(0);
	workspace.cplist.resize(0);
	workspace.pclist.resize(0);
	workspace.nextlist.resize(0);
	workspace.leaflist.resize(0);
	local_workspaces.push(std::move(workspace));
}

hpx::future<kick_return> kick_fork(kick_params params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist,
		vector<tree_id> echecklist, std::shared_ptr<kick_workspace> cuda_workspace, bool threadme) {
	static std::atomic<int> nthreads(1);
	hpx::future<kick_return> rc;
	const tree_node* self_ptr = tree_get_node(self);
	bool remote = false;
	bool all_local = true;
	for (const auto& i : dchecklist) {
		if (i.proc != hpx_rank()) {
			all_local = false;
			break;
		}
	}
	if (self.proc != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = self_ptr->part_range.second - self_ptr->part_range.first > MIN_KICK_THREAD_PARTS;
		const int maxthreads = std::max(1, hpx_size() == 1 ? (int) hpx_hardware_concurrency() * 2 : (int) hpx_hardware_concurrency() / 2);
		if (threadme) {
			if (nthreads++ < maxthreads || !self_ptr->is_local()) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		if (all_local) {
			hpx_yield();
		}
		rc = hpx::make_ready_future(kick(params, L, pos, self, std::move(dchecklist), std::move(echecklist), std::move(cuda_workspace)));
	} else if (remote) {
		rc = hpx::async<kick_action>(hpx_localities()[self_ptr->proc_range.first], params, L, pos, self, std::move(dchecklist), std::move(echecklist), nullptr);
	} else {
		if (all_local) {
			rc = hpx::async([params,self,L,pos, cuda_workspace] (vector<tree_id> dchecklist, vector<tree_id> echecklist) {
				auto rc = kick(params,L,pos,self,std::move(dchecklist),std::move(echecklist), cuda_workspace);
				nthreads--;
				return rc;
			}, std::move(dchecklist), std::move(echecklist));
		} else {
			rc = hpx::async(HPX_PRIORITY_HI, [params,self,L,pos, cuda_workspace] (vector<tree_id> dchecklist, vector<tree_id> echecklist) {
				auto rc = kick(params,L,pos,self,std::move(dchecklist),std::move(echecklist), cuda_workspace);
				nthreads--;
				return rc;
			}, std::move(dchecklist), std::move(echecklist));
		}
	}
	return rc;
}

kick_return kick(kick_params params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist,
		std::shared_ptr<kick_workspace> cuda_workspace) {
	flop_counter<int> flops = 0;
	if (self.proc == 0 && self.index == 0) {
		profiler_enter(__FUNCTION__);
	}
	stack_trace_activate();
	const tree_node* self_ptr = tree_get_node(self);
	timer tm;
	if (self_ptr->local_root) {
		tm.start();
		parts_covered = 0;
		local_return = kick_return();
		local_return_promise = decltype(local_return_promise)();
		local_return_future = local_return_promise.get_future();
	}
	ASSERT(self.proc == hpx_rank());
	bool thread_left = true;
	size_t cuda_mem_usage;
	if (get_options().cuda && params.gpu) {
		if (params.gpu && cuda_workspace == nullptr && self_ptr->is_local()) {
			cuda_workspace = std::make_shared < kick_workspace > (params, self_ptr->nparts());
		}
		size_t max_parts = CUDA_KICK_PARTS_MAX;
		const auto rng = particles_current_range();
		max_parts = std::min((size_t) max_parts,
				(size_t) std::max((part_int) (rng.second - rng.first) / kick_block_count(), (part_int) get_options().bucket_size));
		if (params.gpu && self_ptr->nparts() <= max_parts && self_ptr->is_local()) {
			/*	if (eligible && !self_ptr->leaf && self_ptr->nparts() > CUDA_KICK_PARTS_MAX / 8) {
			 const auto all_local = [](const vector<tree_id>& list) {
			 bool all = true;
			 for( int i = 0; i < list.size(); i++) {
			 if( !tree_get_node(list[i])->is_local_here() ) {
			 all = false;
			 break;
			 }
			 }
			 return all;
			 };
			 eligible = all_local(dchecklist) && all_local(echecklist);
			 }*/
			if (self_ptr->nparts()) {
				cuda_workspace->add_work(cuda_workspace, L, pos, self, std::move(dchecklist), std::move(echecklist));
				parts_covered += self_ptr->nparts();
			}
			if (self_ptr->local_root) {
				auto kr = local_return_future.get();
				return kr;
			} else {
				return kick_return();
			}
		}
		thread_left = cuda_workspace != nullptr;
	}
	kick_return kr;
	const float hfloat = params.h;
	const float GM = params.GM;
	const float eta = params.eta;
	const bool save_force = params.save_force;
	auto workspace = get_workspace();
	vector<tree_id>& nextlist = workspace.nextlist;
	vector<tree_id>& cclist = workspace.cclist;
	vector<tree_id>& cplist = workspace.cplist;
	vector<tree_id>& pclist = workspace.pclist;
	vector<tree_id>& leaflist = workspace.leaflist;
	const simd_float thetainv2(1.0 / sqr(params.theta));
	const simd_float thetainv(1.0 / params.theta);
	static const simd_float sink_bias(SINK_BIAS);
	array<const tree_node*, SIMD_FLOAT_SIZE> other_ptrs;
	array<float, NDIM> Ldx;
	simd_float self_radius = self_ptr->radius;
	array<simd_int, NDIM> self_pos;
	range<simd_double8> self_box;
	array<simd_int, NDIM> other_pos;
	array<simd_float, NDIM> dx;
	simd_float other_radius;
	simd_float other_leaf;
	range<simd_double8> other_box;
	for (int dim = 0; dim < NDIM; dim++) {
		self_pos[dim] = self_ptr->pos[dim].raw();
		self_box.begin[dim] = (double) self_ptr->box.begin[dim].raw();
		self_box.end[dim] = (double) self_ptr->box.end[dim].raw();
	}
	for (int dim = 0; dim < NDIM; dim++) {
		self_box.begin[dim] *= range_fixed::to_float_factor;
		self_box.end[dim] *= range_fixed::to_float_factor;
	}
	for (int dim = 0; dim < NDIM; dim++) {
		Ldx[dim] = distance(self_ptr->pos[dim], pos[dim]);
	}
	L = L2L(L, Ldx, params.do_phi);
	flops += 1511 + params.do_phi * 178;
	simd_float my_hsoft;
	my_hsoft = get_options().hsoft;
	const float hinv = 1.0f / get_options().hsoft;
	force_vectors forces;
	const part_int mynparts = self_ptr->nparts();
	if (self_ptr->leaf) {
		forces = force_vectors(mynparts);
		for (part_int i = 0; i < mynparts; i++) {
			forces.gx[i] = forces.gy[i] = forces.gz[i] = 0.0f;
			forces.phi[i] = -SELF_PHI * hinv;
		}
	}
	{
		nextlist.resize(0);
		cclist.resize(0);
		leaflist.resize(0);
		auto& checklist = echecklist;
		do {
			for (int ci = 0; ci < checklist.size(); ci += SIMD_FLOAT_SIZE) {
				const int maxci = std::min((int) checklist.size(), ci + SIMD_FLOAT_SIZE);
				const int maxi = maxci - ci;
				for (int i = ci; i < maxci; i++) {
					other_ptrs[i - ci] = tree_get_node(checklist[i]);
				}
				for (int i = 0; i < maxi; i++) {
					for (int dim = 0; dim < NDIM; dim++) {
						other_pos[dim][i] = other_ptrs[i]->pos[dim].raw();
					}
					other_radius[i] = other_ptrs[i]->radius;
					other_leaf[i] = other_ptrs[i]->leaf;
				}
				for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
					for (int dim = 0; dim < NDIM; dim++) {
						other_pos[dim][i] = 0.f;
					}
					other_radius[i] = 0.f;
					other_leaf[i] = 0;
					for (int dim = 0; dim < NDIM; dim++) {
						other_box.begin[dim][i] = other_box.end[dim][i] = 0.0;
					}
				}
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = simd_float(self_pos[dim] - other_pos[dim]) * fixed2float;                         // 3
				}
				simd_float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);                                       // 5
				R2 = max(R2, max(sqr(simd_float(0.5) - (self_radius + other_radius)), simd_float(0)));
				const auto hsoft = my_hsoft;
				const simd_float dcc = (self_radius + other_radius) * thetainv;
				const simd_float cc = R2 > sqr(dcc);
				flops += maxi * 15;
				for (int i = 0; i < maxi; i++) {
					if (cc[i]) {
						cclist.push_back(checklist[ci + i]);
					} else if (other_leaf[i]) {
						leaflist.push_back(checklist[ci + i]);
					} else {
						const auto child_checks = other_ptrs[i]->children;
						nextlist.push_back(child_checks[LEFT]);
						nextlist.push_back(child_checks[RIGHT]);
					}
					if (!cc[i]) {
						ALWAYS_ASSERT(!other_leaf[i] || !self_ptr->leaf);
					}
				}
			}
			std::swap(checklist, nextlist);
			nextlist.resize(0);
		} while (checklist.size() && self_ptr->leaf);
		cpu_gravity_cc(GRAVITY_EWALD, L, cclist, self, params.do_phi);
		if (!self_ptr->leaf) {
			checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		}
	}

	{
		nextlist.resize(0);
		cclist.resize(0);
		pclist.resize(0);
		cplist.resize(0);
		leaflist.resize(0);
		auto& checklist = dchecklist;
		do {
			for (int ci = 0; ci < checklist.size(); ci += SIMD_FLOAT_SIZE) {
				const int maxci = std::min((int) checklist.size(), ci + SIMD_FLOAT_SIZE);
				const int maxi = maxci - ci;
				for (int i = ci; i < maxci; i++) {
					other_ptrs[i - ci] = tree_get_node(checklist[i]);
				}
				for (int i = 0; i < maxi; i++) {
					for (int dim = 0; dim < NDIM; dim++) {
						other_pos[dim][i] = other_ptrs[i]->pos[dim].raw();
					}
					other_radius[i] = other_ptrs[i]->radius;
					other_leaf[i] = other_ptrs[i]->leaf;
				}
				for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
					for (int dim = 0; dim < NDIM; dim++) {
						other_pos[dim][i] = 0.f;
					}
					other_radius[i] = 0.f;
					other_leaf[i] = 0;
				}
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = simd_float(self_pos[dim] - other_pos[dim]) * fixed2float;                         // 3
				}
				simd_float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);                                       // 5
				const auto hsoft = my_hsoft;
				const auto mind = self_radius + other_radius + hsoft;
				const simd_float dcc = max((self_radius + other_radius) * thetainv, mind);
				const simd_float dcp = max((thetainv * self_radius + other_radius), mind);
				const simd_float dpc = max((self_radius + other_radius * thetainv), mind);
				const simd_float cc = R2 > sqr(dcc);
				flops += maxi * 20;
				simd_float pc, cp;
				cp = simd_float(0);
				pc = simd_float(0);
				if (self_ptr->leaf) {
					for (int i = 0; i < maxi; i++) {
						for (int dim = 0; dim < NDIM; dim++) {
							other_box.begin[dim][i] = other_ptrs[i]->box.begin[dim].raw();
							other_box.end[dim][i] = other_ptrs[i]->box.end[dim].raw();
						}
					}
					for (int dim = 0; dim < NDIM; dim++) {
						other_box.begin[dim] *= range_fixed::to_float_factor;
						other_box.end[dim] *= range_fixed::to_float_factor;
					}
					for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
						for (int dim = 0; dim < NDIM; dim++) {
							other_box.begin[dim][i] = other_box.end[dim][i] = 0.0;
						}
					}
					const auto nocc = (simd_float(1) - cc) * other_leaf;
					pc = nocc * ((R2 > sqr(dpc)) + box_intersects_sphere(self_box, other_pos, other_radius)) * (dpc > dcp);
					cp = nocc * ((R2 > sqr(dcp)) + box_intersects_sphere(other_box, self_pos, self_radius)) * (dcp > dpc);
					flops += maxi * 34;
				}
				for (int i = 0; i < maxi; i++) {
					if (cc[i]) {
						cclist.push_back(checklist[ci + i]);
					} else if (cp[i]) {
						cplist.push_back(checklist[ci + i]);
					} else if (pc[i]) {
						pclist.push_back(checklist[ci + i]);
					} else if (other_leaf[i]) {
						leaflist.push_back(checklist[ci + i]);
					} else {
						const auto child_checks = other_ptrs[i]->children;
						nextlist.push_back(child_checks[LEFT]);
						nextlist.push_back(child_checks[RIGHT]);
					}
				}
			}
			std::swap(checklist, nextlist);
			nextlist.resize(0);
		} while (checklist.size() && self_ptr->leaf);
		cpu_gravity_cc(GRAVITY_DIRECT, L, cclist, self, params.do_phi);
		if (self_ptr->leaf) {
			if (self_ptr->nparts() > 0) {
				if (cuda_workspace != nullptr) {
					cuda_workspace->add_parts(cuda_workspace, self_ptr->nparts());
				}
				cpu_gravity_cp(L, cplist, self, params.do_phi);
				cpu_gravity_pc(forces, params.do_phi, self, pclist);
				cpu_gravity_pp(forces, params.do_phi, self, leaflist, params.h);
			}
		} else {
			ALWAYS_ASSERT(cplist.size() == 0);
			checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		}
	}

	if (self_ptr->leaf) {
		cleanup_workspace(std::move(workspace));
		const auto rng = self_ptr->part_range;
		for (part_int i = rng.first; i < rng.second; i++) {
			if (particles_rung(i) >= params.min_rung) {
				const part_int j = i - rng.first;
				array<float, NDIM> dx;
				for (part_int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(particles_pos(dim, i), self_ptr->pos[dim]);
				}
				const auto L2 = L2P(L, dx, true);
				float hsoft = get_options().hsoft;
				forces.phi[j] += SCALE_FACTOR1 * L2(0, 0, 0);
				forces.gx[j] -= SCALE_FACTOR2 * L2(1, 0, 0);
				forces.gy[j] -= SCALE_FACTOR2 * L2(0, 1, 0);
				forces.gz[j] -= SCALE_FACTOR2 * L2(0, 0, 1);
				forces.gx[j] *= GM;
				forces.gy[j] *= GM;
				forces.gz[j] *= GM;
				forces.phi[j] *= GM;
				if (save_force) {
					particles_gforce(XDIM, i) = forces.gx[j];
					particles_gforce(YDIM, i) = forces.gy[j];
					particles_gforce(ZDIM, i) = forces.gz[j];
					particles_pot(i) = forces.phi[j];
				}
				auto& vx = particles_vel(XDIM, i);
				auto& vy = particles_vel(YDIM, i);
				auto& vz = particles_vel(ZDIM, i);
				auto& rung = particles_rung(i);
				float g2;
				float kin0 = 0.5 * sqr(vx, vy, vz);
				const float sgn = params.top ? 1 : -1;
				if (params.ascending) {
					const float dt = 0.5f * rung_dt[params.min_rung] * params.t0;
					if (!params.first_call) {
						vx = fmaf(sgn * forces.gx[j], dt, vx);
						vy = fmaf(sgn * forces.gy[j], dt, vy);
						vz = fmaf(sgn * forces.gz[j], dt, vz);
						flops += 9;
					}
				}
				kr.kin += 0.5 * sqr(vx, vy, vz);
				kr.xmom += vx;
				kr.ymom += vy;
				kr.zmom += vz;
				kr.nmom += sqrt(sqr(vx, vy, vz));
				if (params.descending) {
					g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]) + 1e-35f; // 6
					const float factor = eta * sqrtf(params.a);                  // 5
					float dt = std::min(factor * sqrtf(hsoft / sqrtf(g2)), (float) params.t0);      // 14
					rung = std::max(params.min_rung + int((int) ceilf(log2f(params.t0 / dt)) > params.min_rung), (int) (rung - 1)); //13
					kr.max_rung = std::max((int) rung, kr.max_rung);
					ALWAYS_ASSERT(rung >= 0);ALWAYS_ASSERT(rung < MAX_RUNG);
					dt = 0.5f * rung_dt[params.min_rung] * params.t0;                                                            // 2
					vx = fmaf(sgn * forces.gx[j], dt, vx);                              // 3
					vy = fmaf(sgn * forces.gy[j], dt, vy);                              // 3
					vz = fmaf(sgn * forces.gz[j], dt, vz);                              // 3
					flops += 49;
				}
				float kin1= 0.5 * sqr(vx, vy, vz);
				kr.pot += 0.5 * forces.phi[j];
				kr.dkin += kin1 - kin0;
				flops += 570 + params.do_phi * 178;
			}
		}
		parts_covered += self_ptr->nparts();
		local_return += kr;
		const auto goal = rng.second - rng.first;
		if (goal == parts_covered) {
			local_return_promise.set_value(local_return);
		}
		add_cpu_flops(flops);
		if (self_ptr->local_root) {
			return local_return;
		} else {
			return kick_return();
		}
	} else {
		cleanup_workspace(std::move(workspace));
		const tree_node* cl = tree_get_node(self_ptr->children[LEFT]);
		const tree_node* cr = tree_get_node(self_ptr->children[RIGHT]);
		std::array<hpx::future<kick_return>, NCHILD> futs;
		futs[RIGHT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[RIGHT], dchecklist, echecklist, cuda_workspace, thread_left);
		futs[LEFT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[LEFT], std::move(dchecklist), std::move(echecklist), cuda_workspace, false);
		if (self_ptr->proc_range.second - self_ptr->proc_range.first > 1) {
			const auto rcl = futs[LEFT].get();
			const auto rcr = futs[RIGHT].get();
			kr += rcl;
			kr += rcr;
			flops += 12;
		} else {
			if (self_ptr->local_root) {
				kr = local_return_future.get();
			}
		}
		if (self.proc == 0 && self.index == 0) {
			profiler_exit();
		}
		flops += 12;

		add_cpu_flops(flops);
		return kr;
	}
}

void kick_set_rc(kick_return kr) {
	local_return += kr;
	local_return_promise.set_value(local_return);
}
