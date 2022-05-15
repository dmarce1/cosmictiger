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

#include <unistd.h>
#include <stack>

HPX_PLAIN_ACTION (kick);

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
static part_int cuda_workspace_max_parts;
static part_int cuda_branch_max_parts;

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
	static std::atomic<int> nthreads(0);
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
		if (threadme) {
			if (nthreads++ < KICK_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency() || !self_ptr->is_local()) {
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
		rc = kick(params, L, pos, self, std::move(dchecklist), std::move(echecklist), std::move(cuda_workspace));
	} else if (remote) {
		rc = hpx::async<kick_action>(HPX_PRIORITY_HI, hpx_localities()[self_ptr->proc_range.first], params, L, pos, self, std::move(dchecklist), std::move(echecklist), nullptr);
	} else {
		const auto thread_priority = all_local ? HPX_PRIORITY_LO : HPX_PRIORITY_NORMAL;
		rc = hpx::async(thread_priority, [params,self,L,pos, cuda_workspace] (vector<tree_id> dchecklist, vector<tree_id> echecklist) {
			auto rc = kick(params,L,pos,self,std::move(dchecklist),std::move(echecklist), cuda_workspace);
			nthreads--;
			return rc;
		}, std::move(dchecklist), std::move(echecklist));
	}
	return rc;
}

hpx::future<kick_return> kick(kick_params params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist,
		vector<tree_id> echecklist, std::shared_ptr<kick_workspace> cuda_workspace) {
	//params.gpu = false;
	stack_trace_activate();
	const static bool sph = get_options().sph;
	const static float dm_mass = get_options().dm_mass;
	const static float sph_mass = get_options().sph_mass;
	const tree_node* self_ptr = tree_get_node(self);
	timer tm;
	if (self_ptr->local_root) {
		tm.start();
	}
	ASSERT(self.proc == hpx_rank());
	bool thread_left = true;
	size_t cuda_mem_usage;
	if (get_options().cuda && params.gpu) {
		if (params.gpu && cuda_workspace == nullptr && self_ptr->is_local()) {
			cuda_mem_usage = kick_estimate_cuda_mem_usage(params.theta, self_ptr->nparts(), dchecklist.size() + echecklist.size());
			if (cuda_total_mem() * CUDA_MAX_MEM > cuda_mem_usage) {
				cuda_workspace = std::make_shared < kick_workspace > (params, self_ptr->nparts());
			}
		}
		bool eligible;
		size_t max_parts = CUDA_KICK_PARTS_MAX;
		const auto rng = particles_current_range();
		max_parts = std::min((size_t) max_parts, (size_t) std::max((rng.second - rng.first) / kick_block_count(), BUCKET_SIZE));
		eligible = params.gpu && self_ptr->nparts() <= max_parts && self_ptr->is_local();
		if (eligible && !self_ptr->leaf && self_ptr->nparts() > CUDA_KICK_PARTS_MAX / 8) {
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
		}
		if (eligible && self_ptr->nparts() > 0) {
			return cuda_workspace->add_work(cuda_workspace, L, pos, self, std::move(dchecklist), std::move(echecklist));
		}
		thread_left = cuda_workspace != nullptr;
	}
	kick_return kr;
	const int glass = get_options().glass;
//	const simd_float h = params.h;
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
	for (int dim = 0; dim < NDIM; dim++) {
		self_pos[dim] = self_ptr->pos[dim].raw();
	}
	array<simd_int, NDIM> other_pos;
	array<simd_float, NDIM> dx;
	simd_float other_radius;
	simd_float other_leaf;
	for (int dim = 0; dim < NDIM; dim++) {
		Ldx[dim] = distance(self_ptr->pos[dim], pos[dim]);
	}
	L = L2L(L, Ldx, params.do_phi);
	const bool vsoft = get_options().vsoft;
	simd_float my_hsoft;
	my_hsoft = get_options().hsoft;
	force_vectors forces;
	const part_int mynparts = self_ptr->nparts();
	if (self_ptr->leaf) {
		forces = force_vectors(mynparts);
		for (part_int i = 0; i < mynparts; i++) {
			forces.gx[i] = forces.gy[i] = forces.gz[i] = 0.0f;
			forces.phi[i] = 0.0;
		}
	}
	for (int gtype = GRAVITY_DIRECT; gtype <= GRAVITY_EWALD; gtype++) {
		nextlist.resize(0);
		cclist.resize(0);
		pclist.resize(0);
		cplist.resize(0);
		leaflist.resize(0);
		auto& these_flops = kr.node_flops;
		auto& checklist = gtype == GRAVITY_DIRECT ? dchecklist : echecklist;
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
				if (gtype == GRAVITY_EWALD) {
					R2 = max(R2, max(sqr(simd_float(0.5) - (self_radius + other_radius)), simd_float(0)));
				}
				const auto hsoft = my_hsoft;
				const auto mind = self_radius + other_radius + hsoft;
				const simd_float dcc = max((self_radius + other_radius) * thetainv, mind);
				const simd_float dcp = max((thetainv * self_radius + other_radius), mind);
				const simd_float dpc = max((self_radius + other_radius * thetainv), mind);
				const simd_float cc = R2 > sqr(dcc);
				simd_float pc, cp;
				cp = simd_float(0);
				pc = simd_float(0);
				if (self_ptr->leaf) {
					pc = (simd_float(1) - cc) * other_leaf * (R2 > sqr(dpc)) * (dpc > dcp);
					cp = (simd_float(1) - cc) * other_leaf * (R2 > sqr(dcp)) * (dcp > dpc);
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
				these_flops += maxi * 22;
			}
			std::swap(checklist, nextlist);
			nextlist.resize(0);
		} while (checklist.size() && self_ptr->leaf);
		these_flops += cpu_gravity_cc(gtype, L, cclist, self, params.do_phi);
		if (self_ptr->leaf) {
			if (self_ptr->nparts() > 0) {
				if (cuda_workspace != nullptr && gtype == GRAVITY_EWALD) {
					cuda_workspace->add_parts(cuda_workspace, self_ptr->nparts());
				}
				these_flops += cpu_gravity_cp(gtype, L, cplist, self, params.do_phi);
				kr.part_flops += cpu_gravity_pc(gtype, forces, params.do_phi, self, pclist);
				kr.part_flops += cpu_gravity_pp(gtype, forces, params.do_phi, self, leaflist, params.h);
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
				forces.phi[j] += L2(0, 0, 0);
				forces.gx[j] -= L2(1, 0, 0);
				forces.gy[j] -= L2(0, 1, 0);
				forces.gz[j] -= L2(0, 0, 1);
				forces.gx[j] *= GM;
				forces.gy[j] *= GM;
				forces.gz[j] *= GM;
				forces.phi[j] *= GM;
				if (glass) {
					forces.gx[j] *= -1.f;
					forces.gy[j] *= -1.f;
					forces.gz[j] *= -1.f;
				}
				auto& vx = particles_vel(XDIM, i);
				auto& vy = particles_vel(YDIM, i);
				auto& vz = particles_vel(ZDIM, i);
				auto& rung = particles_rung(i);
				float g2;
				const float sgn = params.top ? 1 : -1;
				ALWAYS_ASSERT(!sph);
				ALWAYS_ASSERT(!vsoft);
				if (params.ascending) {
					const float dt = 0.5f * rung_dt[params.min_rung] * params.t0;
					if (!params.first_call) {
						vx = fmaf(sgn * forces.gx[j], dt, vx);
						vy = fmaf(sgn * forces.gy[j], dt, vy);
						vz = fmaf(sgn * forces.gz[j], dt, vz);
					}
				}
				kr.kin += 0.5 * sqr(vx, vy, vz);
				kr.xmom += vx;
				kr.ymom += vy;
				kr.zmom += vz;
				kr.nmom += sqrt(sqr(vx, vy, vz));
				if (params.descending) {
					g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]) + 1e-35f;
					const float factor = eta * sqrtf(params.a);
					float dt = std::min(factor * sqrtf(hsoft / sqrtf(g2)), (float) params.t0);
					rung = params.min_rung + int((int) ceilf(log2f(params.t0) - log2f(dt)) > params.min_rung);
					kr.max_rung = std::max(rung, kr.max_rung);
					ALWAYS_ASSERT(rung >= 0);
					ALWAYS_ASSERT(rung < MAX_RUNG);
					dt = 0.5f * rung_dt[params.min_rung] * params.t0;
					vx = fmaf(sgn * forces.gx[j], dt, vx);
					vy = fmaf(sgn * forces.gy[j], dt, vy);
					vz = fmaf(sgn * forces.gz[j], dt, vz);
				}
				kr.pot += 0.5 * forces.phi[j];
			}
		}
		return hpx::make_ready_future(kr);
	} else {
		cleanup_workspace(std::move(workspace));
		const tree_node* cl = tree_get_node(self_ptr->children[LEFT]);
		const tree_node* cr = tree_get_node(self_ptr->children[RIGHT]);
		std::array<hpx::future<kick_return>, NCHILD> futs;
		futs[RIGHT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[RIGHT], dchecklist, echecklist, cuda_workspace, thread_left);
		futs[LEFT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[LEFT], std::move(dchecklist), std::move(echecklist), cuda_workspace, false);
		if (futs[LEFT].is_ready() && futs[RIGHT].is_ready()) {
			const auto rcl = futs[LEFT].get();
			const auto rcr = futs[RIGHT].get();
			kr += rcl;
			kr += rcr;
			if (self_ptr->local_root) {
				tm.stop();
				char hostname[33];
				gethostname(hostname, 32);
//				PRINT("Kick took %e s on %s\n", tm.read(), hostname);
			}
			return hpx::make_ready_future(kr);
		} else {
			return hpx::when_all(futs.begin(), futs.end()).then([tm,self_ptr](hpx::future<std::vector<hpx::future<kick_return>>> futsfut) {
				auto futs = futsfut.get();
				kick_return kr;
				const auto rcl = futs[LEFT].get();
				const auto rcr = futs[RIGHT].get();
				kr += rcl;
				kr += rcr;
				if( self_ptr->local_root) {
					timer tm1 = tm;
					tm1.stop();
					char hostname[33];
					gethostname(hostname,32);
//					PRINT( "Kick took %e s on %s\n", tm1.read(), hostname);
				}
				return kr;
			});
		}
	}
}

