/*
 CosmicTiger - A cosmological N-Body code
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
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stars.hpp>

#include <unistd.h>
#include <stack>

HPX_PLAIN_ACTION (kick);

#define MAX_ACTIVE_WORKSPACES 1

struct workspace {
	vector<tree_id> nextlist;
	vector<tree_id> partlist;
	vector<tree_id> multlist;
	vector<tree_id> leaflist;
	workspace() = default;
	workspace(const workspace&) = delete;
	workspace& operator=(const workspace&) = delete;
	workspace(workspace&&) = default;
	workspace& operator=(workspace&&) = default;
};

static float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };
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
	workspace.multlist.resize(0);
	workspace.partlist.resize(0);
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
#ifdef USE_CUDA
	size_t cuda_mem_usage;
	if (get_options().cuda && params.gpu) {
		if( self_ptr->local_root) {
			const double max_load = self_ptr->node_count * params.node_load + (self_ptr->part_range.second - self_ptr->part_range.first);
			const double load = self_ptr->active_nodes * params.node_load + self_ptr->nactive;
			if( load / max_load < GPU_MIN_LOAD) {
				params.gpu = false;
			}
		}
		if (params.gpu && cuda_workspace == nullptr && self_ptr->is_local()) {
			cuda_mem_usage = kick_estimate_cuda_mem_usage(params.theta, self_ptr->nparts(), dchecklist.size() + echecklist.size());
			if (cuda_total_mem() * CUDA_MAX_MEM > cuda_mem_usage) {
				cuda_workspace = std::make_shared<kick_workspace>(params, self_ptr->nparts());
			}
		}
		bool eligible = params.gpu && self_ptr->nparts() <= CUDA_KICK_PARTS_MAX && self_ptr->is_local();
		if( eligible ) {
			if( self_ptr->children[LEFT].index != -1) {
				const part_int active_left = tree_get_node(self_ptr->children[LEFT])->nactive;
				const part_int active_right = tree_get_node(self_ptr->children[RIGHT])->nactive;
				if( active_left == 0 || active_right == 0 ) {
					eligible = false;
				}
			}
		}
		if( eligible && !self_ptr->leaf && self_ptr->nparts() > CUDA_KICK_PARTS_MAX / 8) {
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
		if (eligible) {
			return cuda_workspace->add_work(cuda_workspace, L, pos, self, std::move(dchecklist), std::move(echecklist));
		}
		thread_left = cuda_workspace != nullptr;
	}
#endif
	kick_return kr;
	const bool vsoft = sph && get_options().sph;
	const int glass = get_options().glass;
	const simd_float h = params.h;
	const float hfloat = params.h;
	const float GM = params.GM;
	const float eta = params.eta;
	const bool save_force = params.save_force;
	auto workspace = get_workspace();
	vector<tree_id>& nextlist = workspace.nextlist;
	vector<tree_id>& partlist = workspace.partlist;
	vector<tree_id>& multlist = workspace.multlist;
	vector<tree_id>& leaflist = workspace.leaflist;
	const simd_float thetainv2(1.0 / sqr(params.theta));
	const simd_float thetainv(1.0 / params.theta);
	static const simd_float sink_bias(SINK_BIAS);
	static const simd_float ewald_dist2(EWALD_DIST2);
	array<const tree_node*, SIMD_FLOAT_SIZE> other_ptrs;
	const bool do_phi = params.min_rung == 0;
	array<float, NDIM> Ldx;
	simd_float self_radius = self_ptr->radius;
	array<simd_int, NDIM> self_pos;
	for (int dim = 0; dim < NDIM; dim++) {
		self_pos[dim] = self_ptr->pos[dim].raw();
	}
	array<simd_int, NDIM> other_pos;
	array<simd_float, NDIM> dx;
	simd_float other_radius;
	simd_int other_leaf;
	for (int dim = 0; dim < NDIM; dim++) {
		Ldx[dim] = distance(self_ptr->pos[dim], pos[dim]);
	}
	L = L2L(L, Ldx, do_phi);
	multlist.resize(0);
	simd_float other_hsoft;
	simd_float hsoft;
	if (vsoft) {
		hsoft = self_ptr->hsoft_max;
	} else {
		hsoft = h;
	}
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
			if (vsoft) {
				other_hsoft[i] = other_ptrs[i]->hsoft_max;
			} else {
				other_hsoft[i] = h[0];
			}
		}
		for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				other_pos[dim][i] = 0.f;
			}
			other_radius[i] = 0.f;
			other_hsoft[i] = 1.f;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = simd_float(self_pos[dim] - other_pos[dim]) * fixed2float;              // 3
		}
		const simd_float R2 = max(ewald_dist2, sqr(dx[XDIM], dx[YDIM], dx[ZDIM]));          // 6
		const simd_float r2 = sqr((sink_bias * self_radius + other_radius) * thetainv); // 5
		const simd_float soft_sep = sqr(self_radius + other_radius + max(hsoft, other_hsoft)) < R2;
		const simd_float far = (R2 > r2) * soft_sep;                                                     // 1
		for (int i = 0; i < maxi; i++) {
			if (far[i]) {
				multlist.push_back(echecklist[ci + i]);
			} else {
				nextlist.push_back(other_ptrs[i]->children[LEFT]);
				nextlist.push_back(other_ptrs[i]->children[RIGHT]);
			}
		}
		kr.node_flops += maxi * 15;
	}
	std::swap(echecklist, nextlist);
	kr.node_flops += cpu_gravity_cc(L, multlist, self, GRAVITY_CC_EWALD, params.min_rung == 0);
	nextlist.resize(0);
	multlist.resize(0);
	partlist.resize(0);
	leaflist.resize(0);
	auto& these_flops = kr.node_flops;
	do {
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
				if (vsoft) {
					other_hsoft[i] = other_ptrs[i]->hsoft_max;
				} else {
					other_hsoft[i] = h[0];
				}
				other_radius[i] = other_ptrs[i]->radius;
				other_leaf[i] = other_ptrs[i]->leaf;
			}
			for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					other_pos[dim][i] = 0.f;
				}
				other_radius[i] = 0.f;
				other_hsoft[i] = 1.f;
				other_leaf[i] = 0;
			}
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = simd_float(self_pos[dim] - other_pos[dim]) * fixed2float;                         // 3
			}
			const simd_float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);                                       // 5
			const simd_float soft_sep = sqr(self_radius + other_radius + max(hsoft, other_hsoft)) < R2;
			const simd_float far1 = soft_sep * (R2 > sqr((sink_bias * self_radius + other_radius) * thetainv));     // 5
			const simd_float far2 = soft_sep * (R2 > sqr(sink_bias * self_radius * thetainv + other_radius));       // 5
			const simd_float mult = far1;                                                                  // 4
			const simd_float part = far2 * other_leaf * simd_float((self_ptr->part_range.second - self_ptr->part_range.first) > MIN_CP_PARTS);
			for (int i = 0; i < maxi; i++) {
				if (mult[i]) {
					multlist.push_back(dchecklist[ci + i]);
				} else if (part[i]) {
					partlist.push_back(dchecklist[ci + i]);
				} else if (other_leaf[i]) {
					leaflist.push_back(dchecklist[ci + i]);
				} else {
					const auto child_checks = other_ptrs[i]->children;
					nextlist.push_back(child_checks[LEFT]);
					nextlist.push_back(child_checks[RIGHT]);

				}
			}
			these_flops += maxi * 22;
		}
		std::swap(dchecklist, nextlist);
		nextlist.resize(0);
	} while (dchecklist.size() && self_ptr->leaf);
	these_flops += cpu_gravity_cc(L, multlist, self, GRAVITY_CC_DIRECT, params.min_rung == 0);
	these_flops += cpu_gravity_cp(L, partlist, self, params.min_rung == 0);
	if (self_ptr->leaf) {
		if (cuda_workspace != nullptr) {
			cuda_workspace->add_parts(cuda_workspace, self_ptr->nparts());
		}
		partlist.resize(0);
		multlist.resize(0);
		const part_int mynparts = self_ptr->nparts();
		force_vectors forces(mynparts);
		for (part_int i = 0; i < mynparts; i++) {
			forces.gx[i] = forces.gy[i] = forces.gz[i] = 0.0f;
			forces.phi[i] = 0.0;
		}
		for (int i = 0; i < leaflist.size(); i++) {
			const tree_node* other_ptr = tree_get_node(leaflist[i]);
			if (other_ptr->part_range.second - other_ptr->part_range.first >= MIN_PC_PARTS) {
				other_radius = other_ptr->radius;
				for (int dim = 0; dim < NDIM; dim++) {
					other_pos[dim] = other_ptr->pos[dim].raw();
				}
				const auto myrange = self_ptr->part_range;
				bool pp = false;
				for (part_int j = myrange.first; j < myrange.second; j += SIMD_FLOAT_SIZE) {
					j = std::min(j, myrange.second - SIMD_FLOAT_SIZE);
					for (int dim = 0; dim < NDIM; dim++) {
						for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
							self_pos[dim][k] = particles_pos(dim, j + k).raw();
						}
					}
					for (int dim = 0; dim < NDIM; dim++) {
						dx[dim] = simd_float(self_pos[dim] - other_pos[dim]) * fixed2float;        // 3
					}
					const simd_float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);                      // 5
					const simd_float rhs = sqr(h + other_radius * thetainv);                      // 3
					const simd_float near = R2 <= rhs;                                            // 1
					kr.part_flops += SIMD_FLOAT_SIZE * 12;
					if (near.sum()) {
						pp = true;
						break;
					}
				}
				if (pp) {
					partlist.push_back(leaflist[i]);
				} else {
					multlist.push_back(leaflist[i]);
				}
			} else {
				partlist.push_back(leaflist[i]);
			}
		}
		kr.part_flops += cpu_gravity_pc(forces, params.min_rung, self, multlist);
		kr.part_flops += cpu_gravity_pp(forces, params.min_rung, self, partlist, params.h);
		cleanup_workspace(std::move(workspace));

		const auto rng = self_ptr->part_range;
		for (part_int i = rng.first; i < rng.second; i++) {
			if (particles_rung(i) >= params.min_rung) {
				const part_int j = i - rng.first;
				array<float, NDIM> dx;
				for (part_int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(particles_pos(dim, i), self_ptr->pos[dim]);
				}
				const auto L2 = L2P(L, dx, params.min_rung == 0);
				float m = 1.f;
				int type = DARK_MATTER_TYPE;
				if (sph) {
					type = particles_type(i);
					m = type == DARK_MATTER_TYPE ? dm_mass : sph_mass;
				}
				forces.phi[j] += L2(0, 0, 0);
				if (!sph) {
					forces.phi[j] -= SELF_PHI * m / (2.f * params.h);
				}
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
				auto dt = 0.5f * rung_dt[rung] * params.t0;
				if (type == SPH_TYPE) {
					const int k = particles_cat_index(i);
					sph_particles_gforce(XDIM, k) = forces.gx[j];
					sph_particles_gforce(YDIM, k) = forces.gy[j];
					sph_particles_gforce(ZDIM, k) = forces.gz[j];
				} else {
					if (!params.first_call) {
						vx = fmaf(forces.gx[j], dt, vx);
						vy = fmaf(forces.gy[j], dt, vy);
						vz = fmaf(forces.gz[j], dt, vz);
					}
				}
				const float g2 = sqr(forces.gx[j], forces.gy[j], forces.gz[j]);
				if (type != SPH_TYPE) {
					const float factor = eta * sqrtf(params.a * hfloat);
					dt = std::min(factor / sqrtf(sqrtf(g2)), (float) params.t0);
					rung = std::max(std::max((int) ceilf(log2f(params.t0) - log2f(dt)), std::max(rung - 1, params.min_rung)), 1);
					kr.max_rung = std::max(rung, kr.max_rung);
					if (rung < 0 || rung >= MAX_RUNG) {
						PRINT("Rung out of range %i\n", rung);
					} else {
						dt = 0.5f * rung_dt[rung] * params.t0;
					}
					vx = fmaf(forces.gx[j], dt, vx);
					vy = fmaf(forces.gy[j], dt, vy);
					vz = fmaf(forces.gz[j], dt, vz);
				}
				kr.pot += m * forces.phi[j];
				kr.fx += forces.gx[j];
				kr.fy += forces.gy[j];
				kr.fz += forces.gz[j];
				kr.fnorm += g2;
			}
		}
		return hpx::make_ready_future(kr);
	} else {
		dchecklist.insert(dchecklist.end(), leaflist.begin(), leaflist.end());
		cleanup_workspace(std::move(workspace));
		const tree_node* cl = tree_get_node(self_ptr->children[LEFT]);
		const tree_node* cr = tree_get_node(self_ptr->children[RIGHT]);
		const bool exec_left = cl->nactive > 0 || !cl->is_local();
		const bool exec_right = cr->nactive > 0 || !cr->is_local();
		std::array<hpx::future<kick_return>, NCHILD> futs;
		if (exec_left && exec_right) {
			futs[RIGHT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[RIGHT], dchecklist, echecklist, cuda_workspace, thread_left);
			futs[LEFT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[LEFT], std::move(dchecklist), std::move(echecklist), cuda_workspace, false);
		} else if (exec_left) {
#ifdef USE_CUDA
			if (cuda_workspace != nullptr) {
				cuda_workspace->add_parts(cuda_workspace, cr->nparts());
			}
#endif
			futs[RIGHT] = hpx::make_ready_future(kick_return());
			futs[LEFT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[LEFT], std::move(dchecklist), std::move(echecklist), cuda_workspace, false);
		} else {
#ifdef USE_CUDA
			if (cuda_workspace != nullptr) {
				cuda_workspace->add_parts(cuda_workspace, cl->nparts());
			}
#endif
			futs[RIGHT] = kick_fork(params, L, self_ptr->pos, self_ptr->children[RIGHT], std::move(dchecklist), std::move(echecklist), cuda_workspace, false);
			futs[LEFT] = hpx::make_ready_future(kick_return());
		}
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

