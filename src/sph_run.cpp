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
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/sph_run.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>

#include <unistd.h>
#include <stack>

HPX_PLAIN_ACTION (sph_run);

#define MAX_ACTIVE_WORKSPACES 1

static float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

struct workspace {
	vector<tree_id> nextlist;
	vector<tree_id> leaflist;
	workspace() = default;
	workspace(const workspace&) = delete;
	workspace& operator=(const workspace&) = delete;
	workspace(workspace&&) = default;
	workspace& operator=(workspace&&) = default;
};
static thread_local std::stack<workspace> local_workspaces;

inline bool range_intersect(const range<fixed32>& a, const range<fixed32>& b) {
	bool intersect = true;
	for (int dim = 0; dim < NDIM; dim++) {
		if (distance(b.end[dim], a.begin[dim]) > 0.0 && distance(a.end[dim], b.begin[dim]) > 0.0) {
		} else {
			intersect = false;
			break;
		}
	}
	return intersect;
}

inline bool range_contains(const range<fixed32>& a, const array<fixed32, NDIM> x) {
	bool contains = true;
	for (int dim = 0; dim < NDIM; dim++) {
		if (distance(x[dim], a.begin[dim]) >= 0.0 && distance(a.end[dim], x[dim]) >= 0.0) {
		} else {
			contains = false;
			break;
		}
	}
	return contains;
}

static workspace get_workspace() {
	if (local_workspaces.empty()) {
		local_workspaces.push(workspace());
	}
	workspace workspace = std::move(local_workspaces.top());
	local_workspaces.pop();
	return std::move(workspace);
}

static void cleanup_workspace(workspace&& workspace) {
	workspace.nextlist.resize(0);
	workspace.leaflist.resize(0);
	local_workspaces.push(std::move(workspace));
}

hpx::future<sph_run_return> sph_run_fork(sph_run_params params, tree_id self, vector<tree_id> checklist, bool threadme) {
	static std::atomic<int> nthreads(0);
	hpx::future<sph_run_return> rc;
	const tree_node* self_ptr = tree_get_node(self);
	bool remote = false;
	bool all_local = true;
	for (const auto& i : checklist) {
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
		rc = sph_run(params, self, std::move(checklist));
	} else if (remote) {
		rc = hpx::async<sph_run_action>(HPX_PRIORITY_HI, hpx_localities()[self_ptr->proc_range.first], params, self, std::move(checklist));
	} else {
		const auto thread_priority = all_local ? HPX_PRIORITY_LO : HPX_PRIORITY_NORMAL;
		rc = hpx::async(thread_priority, [params,self] (vector<tree_id> checklist) {
			auto rc = sph_run(params,self,std::move(checklist));
			nthreads--;
			return rc;
		}, std::move(checklist));
	}
	return rc;
}

hpx::future<sph_run_return> sph_run(sph_run_params params, tree_id self, vector<tree_id> checklist) {
	bool do_outer, do_inner, active_only;
	switch (params.run_type) {
	case SPH_RUN_SMOOTH_LEN:
		do_inner = params.set1 != SPH_SET_SEMIACTIVE;
		do_outer = params.set1 = SPH_SET_SEMIACTIVE;
		active_only = true;
		break;
	case SPH_RUN_MARK_SEMIACTIVE:
		do_inner = false;
		do_outer = true;
		break;
	case SPH_RUN_FIND_BOXES:
		do_inner = false;
		do_outer = false;
		active_only = false;
		break;
	case SPH_RUN_COURANT:
		do_inner = true;
		do_outer = false;
		active_only = true;
		break;
	case SPH_RUN_GRAVITY:
		do_inner = false;
		do_outer = false;
		active_only = true;
		break;
	case SPH_RUN_FVELS:
		do_inner = true;
		do_outer = true;
		active_only = false;
		break;
	case SPH_RUN_HYDRO:
		do_inner = true;
		do_outer = true;
		active_only = false;
		break;
	case SPH_RUN_UPDATE:
		do_inner = false;
		do_outer = false;
		active_only = true;
		break;

	}
	stack_trace_activate();
	const sph_tree_node* self_ptr = sph_tree_get_node(self);
	if ((active_only && self_ptr->nactive == 0) || (checklist.size() == 0) || (params.run_type == SPH_RUN_FIND_BOXES && !self_ptr->box_active)) {
		return hpx::make_ready_future(sph_run_return());
	}
	timer tm;
	if (self_ptr->local_root) {
		tm.start();
	}
	ASSERT(self.proc == hpx_rank());
	bool thread_left = true;
	sph_run_return kr;
	auto workspace = get_workspace();
	vector<tree_id>& nextlist = workspace.nextlist;
	vector<tree_id>& leaflist = workspace.leaflist;
	range<fixed32> box;

	do {
		nextlist.resize(0);
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto* other = sph_tree_get_node(checklist[ci]);
			if (other->nactive > 0) {
				const bool test1 = do_inner && range_intersect(self_ptr->outer_box, other->inner_box);
				const bool test2 = do_outer && range_intersect(self_ptr->inner_box, other->outer_box);
				const bool test3 = (!do_outer && !do_inner) && (self_ptr == other);
				if (test1 || test2 || test3) {
					if (other->leaf) {
						leaflist.push_back(checklist[ci]);
					} else {
						nextlist.push_back(checklist[ci]);
					}
				}
			}
		}
		checklist = std::move(nextlist);
	} while (self_ptr->leaf && checklist.size());

	vector<fixed32> xs;
	vector<fixed32> ys;
	vector<fixed32> zs;
	vector<char> rungs;
	vector<float> hs;
	vector<float> ents;
	vector<float> vxs;
	vector<float> vys;
	vector<float> vzs;
	vector<float> fvels;
	static const float m = get_options().sph_mass;
	const auto load_data =
			[&xs,&ys,&zs,&rungs,&fvels, &hs,&leaflist,self_ptr,&ents,&vxs,&vys,&vzs](bool do_rungs, bool do_smoothlens, bool do_sph, bool check_inner, bool check_outer) {
				part_int offset;
				for (int ci = 0; ci < leaflist.size(); ci++) {
					const auto* other = sph_tree_get_node(leaflist[ci]);
					const auto this_sz = other->part_range.second - other->part_range.first;
					offset = xs.size();
					const int new_sz = xs.size() + this_sz;
					xs.resize(new_sz);
					ys.resize(new_sz);
					zs.resize(new_sz);
					if( do_rungs) {
						rungs.resize(new_sz);
					}
					if( do_smoothlens) {
						hs.resize(new_sz);
					}
					if( do_sph ) {
						ents.resize(new_sz);
						vxs.resize(new_sz);
						vys.resize(new_sz);
						vzs.resize(new_sz);
						fvels.resize(new_sz);
					}
					sph_particles_global_read_pos(other->global_part_range(), xs.data(), ys.data(), zs.data(), offset);
					if( do_rungs || do_smoothlens) {
						sph_particles_global_read_rungs_and_smoothlens(other->global_part_range(), rungs, hs, offset);
					}
					if( do_sph ) {
						sph_particles_global_read_sph(other->global_part_range(), ents, fvels, vxs, vys, vzs, offset);
					}
					int i = offset;
					while (i < xs.size()) {
						array<fixed32, NDIM> X;
						X[XDIM] = xs[i];
						X[YDIM] = ys[i];
						X[ZDIM] = zs[i];
						const bool test1 = check_inner && !range_contains(self_ptr->outer_box, X);
						bool test2 = false;
						if( check_outer && !test1) {
							assert(do_rungs);
							test2 = true;
							const auto& box = self_ptr->inner_box;
							for( int dim = 0; dim < NDIM; dim++) {
								if( distance(box.begin[dim], X[dim]) + hs[i] >= 0.0 && distance(X[dim], box.end[dim]) + hs[i] ) {
								} else {
									test2 = false;
									break;
								}
							}
						}
						if (test1 || test2) {
							xs[i] = xs.back();
							ys[i] = ys.back();
							zs[i] = zs.back();
							xs.pop_back();
							ys.pop_back();
							zs.pop_back();
							if( do_rungs ) {
								rungs[i] = rungs.back();
								rungs.pop_back();
								hs[i] = hs.back();
								hs.pop_back();
							}
							if( do_sph) {
								ents[i] = ents.back();
								vxs[i] = vxs.back();
								vys[i] = vys.back();
								vzs[i] = vzs.back();
								ents.pop_back();
								vxs.pop_back();
								vys.pop_back();
								vzs.pop_back();
							}
						} else {
							i++;
						}
					}
				}
			};

	switch (params.run_type) {
	case SPH_RUN_SMOOTH_LEN: {
		const bool test = params.set1 = SPH_SET_SEMIACTIVE;
		load_data(false, test, test, !test, test);
		const int self_nparts = self_ptr->part_range.second - self_ptr->part_range.first;
		float f, dfdh;
		float max_dlogh;
		float max_h;
		bool box_xceeded = false;
		do {
			max_dlogh = 0.0;
			max_h = 0.0;
			for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
				const bool test1 = params.set1 == SPH_SET_ACTIVE && sph_particles_rung(i) >= params.min_rung;
				const bool test2 = params.set1 == SPH_SET_SEMIACTIVE && (sph_particles_semi_active(i) && sph_particles_rung(i) < params.min_rung);
				const bool test3 = params.set1 == SPH_SET_ALL;
				if (test1 || test2 || test3) {
					float& h = sph_particles_smooth_len(i);
					const float h2 = sqr(h);
					const float hinv = 1.0f / h;
					const auto myx = sph_particles_pos(XDIM, i);
					const auto myy = sph_particles_pos(YDIM, i);
					const auto myz = sph_particles_pos(ZDIM, i);
					const int k = i - self_ptr->part_range.first;
					f = 0.0;
					dfdh = 0.0;
					for (int j = 0; j < xs.size(); j++) {
						const float dx = distance(myx, xs[j]);
						const float dy = distance(myy, ys[j]);
						const float dz = distance(myz, zs[j]);
						const float r2 = sqr(dx, dy, dz);
						if (r2 < h2) {
							const float r = sqrt(r);
							const float w = sph_W(r, hinv, 1.0f);
							const float dwdh = sph_dWdh(r, hinv, 1.0f);
							constexpr float c0 = float(4.0 / 3.0 * M_PI);
							f += c0 * w;
							dfdh += c0 * dwdh;
						}
						dfdh += 3.0f * f * hinv;
						f -= SPH_NEIGHBOR_COUNT;
						const float dh = -f / dfdh;
						max_dlogh = std::max(max_dlogh, fabs(dh) * hinv);
						h += dh;
					}
				}
				if (box_xceeded) {
					break;
				}
			}
		} while (max_dlogh > SPH_SMOOTHLEN_TOLER && !box_xceeded);
		if (box_xceeded) {
			kr.rc1 = true;
		}
		kr.rc2 = true;
	}
		break;

	case SPH_RUN_COURANT: {
		load_data(true, false, true, true, false);
		const float ainv = 1.0f / params.a;
		for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			const bool test1 = sph_particles_rung(i) >= params.min_rung;
			if (test1) {
				const float myh = sph_particles_smooth_len(i);
				const float myh2 = sqr(myh);
				const float myhinv = 1.0f / myh;
				const float myhinv3 = myhinv * sqr(myhinv);
				const float myrho = sph_den(myhinv3);
				const auto myx = sph_particles_pos(XDIM, i);
				const auto myy = sph_particles_pos(YDIM, i);
				const auto myz = sph_particles_pos(ZDIM, i);
				const auto myent = sph_particles_ent(i);
				const auto myvx = sph_particles_vel(XDIM, i);
				const auto myvy = sph_particles_vel(YDIM, i);
				const auto myvz = sph_particles_vel(ZDIM, i);
				const auto myc = sqrtf(SPH_GAMMA * pow(myrho, SPH_GAMMA - 1.0f) * myent);
				const int k = i - self_ptr->part_range.first;
				float max_c = 0.0f;
				for (int j = 0; j < xs.size(); j++) {
					const float dx = distance(myx, xs[j]);
					const float dy = distance(myy, ys[j]);
					const float dz = distance(myz, zs[j]);
					const float r2 = sqr(dx, dy, dz);
					if (r2 < myh2 && r2 != 0.0f) {
						const float h = hs[j];
						const float h2 = sqr(h);
						const float hinv = 1.0f / h;
						const float hinv3 = hinv * sqr(hinv);
						const float rho = sph_den(hinv3);
						const auto c = sqrtf(SPH_GAMMA * pow(rho, SPH_GAMMA - 1.0f) * ents[j]);
						const float dx = distance(myx, xs[j]);
						const float dy = distance(myy, ys[j]);
						const float dz = distance(myz, zs[j]);
						const float rinv = 1.0f / sqrt(sqr(dx, dy, dz));
						const float dvx = myvx - vxs[j];
						const float dvy = myvy - vys[j];
						const float dvz = myvz - vzs[j];
						const float w = std::min(0.0f, (dvy * dx + dvy * dy + dvz * dz) * rinv);
						max_c = std::max(max_c, c + myc - 3.0f * w);
					}
				}
				float dthydro = max_c / (params.a * myh);
				if (dthydro > 1.0e-99) {
					dthydro = 1.0 / dthydro;
				} else {
					dthydro = 1.0e99;
				}
				static const float eta = get_options().eta;
				static const float hgrav = get_options().hsoft;
				const float gx = sph_particles_gforce(XDIM, i);
				const float gy = sph_particles_gforce(YDIM, i);
				const float gz = sph_particles_gforce(ZDIM, i);
				char& rung = sph_particles_rung(i);
				const float g2 = sqr(gx, gy, gz);
				const float factor = eta * sqrtf(params.a * hgrav);
				float dt = std::min(factor / sqrtf(sqrtf(g2)), (float) params.t0);
				dt = std::min(dt, dthydro);
				rung = std::max((int) ceilf(log2f(params.t0) - log2f(dt)), std::max(rung - 1, params.min_rung));
				kr.max_rung = std::max(kr.max_rung, rung);
			}
		}
	}
		break;

	case SPH_RUN_FVELS: {
		load_data(true, false, true, true, false);
		const float ainv = 1.0f / params.a;
		for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			const bool test1 = sph_particles_rung(i) >= params.min_rung;
			const bool test2 = sph_particles_semi_active(i);
			if (test1 || test2) {
				const float h = sph_particles_smooth_len(i);
				const float h2 = sqr(h);
				const float hinv = 1.0f / h;
				const float hinv3 = hinv * sqr(hinv);
				const float rho = sph_den(hinv3);
				const float rhoinv = 1.0f / rho;
				const auto myx = sph_particles_pos(XDIM, i);
				const auto myy = sph_particles_pos(YDIM, i);
				const auto myz = sph_particles_pos(ZDIM, i);
				const int k = i - self_ptr->part_range.first;
				float dvx_dx = 0.0f;
				float dvx_dy = 0.0f;
				float dvx_dz = 0.0f;
				float dvy_dx = 0.0f;
				float dvy_dy = 0.0f;
				float dvy_dz = 0.0f;
				float dvz_dx = 0.0f;
				float dvz_dy = 0.0f;
				float dvz_dz = 0.0f;
				for (int j = 0; j < xs.size(); j++) {
					const float dx = distance(myx, xs[j]);
					const float dy = distance(myy, ys[j]);
					const float dz = distance(myz, zs[j]);
					const float r2 = sqr(dx, dy, dz);
					if (r2 < h2) {
						const float r = sqrt(r2);
						const float rinv = 1.0f / r;
						const float dWdr = sph_dWdr(r, hinv, hinv3) * rinv;
						const float tmp = m * dWdr * rhoinv;
						const float dWdr_x = dx * tmp;
						const float dWdr_y = dy * tmp;
						const float dWdr_z = dz * tmp;
						dvx_dx += vxs[j] * dWdr_x;
						dvx_dy += vxs[j] * dWdr_y;
						dvx_dz += vxs[j] * dWdr_z;
						dvy_dx += vys[j] * dWdr_x;
						dvy_dy += vys[j] * dWdr_y;
						dvy_dz += vys[j] * dWdr_z;
						dvz_dx += vzs[j] * dWdr_x;
						dvz_dy += vzs[j] * dWdr_y;
						dvz_dz += vzs[j] * dWdr_z;
					}
				}
				const float abs_div_v = fabs(dvx_dx + dvy_dy + dvz_dz);
				const float curl_vx = dvz_dy - dvy_dz;
				const float curl_vy = -dvz_dx + dvx_dz;
				const float curl_vz = dvy_dx - dvx_dy;
				const float abs_curl_v = sqrt(sqr(curl_vx, curl_vy, curl_vz));
				const float fvel = abs_div_v / (abs_div_v + abs_curl_v);
				sph_particles_fvel(i) = fvel;
			}
		}
	}
		break;

	case SPH_RUN_HYDRO: {
		load_data(true, true, true, true, true);
		for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			const bool test1 = sph_particles_rung(i) >= params.min_rung;
			const bool test2 = sph_particles_semi_active(i);
			if (test1 || test2) {
				const float myh = sph_particles_smooth_len(i);
				const char myrung = sph_particles_rung(i);
				const float myh2 = sqr(myh);
				const float myhinv = 1.0f / myh;
				const float myh3inv = myhinv * sqr(myhinv);
				const float myrho = sph_den(myh3inv);
				const float myrhoinv = 1.f / myrho;
				const auto myx = sph_particles_pos(XDIM, i);
				const auto myy = sph_particles_pos(YDIM, i);
				const auto myz = sph_particles_pos(ZDIM, i);
				const auto myent = sph_particles_ent(i);
				const auto myvx = sph_particles_vel(XDIM, i);
				const auto myvy = sph_particles_vel(YDIM, i);
				const auto myvz = sph_particles_vel(ZDIM, i);
				const auto myfvel = sph_particles_fvel(i);
				const auto myp = pow(myrho, SPH_GAMMA) * myent;
				const auto myc = sqrtf(SPH_GAMMA * myp * myrhoinv * myent);
				const int k = i - self_ptr->part_range.first;
				float max_c = 0.0f;
				for (int j = 0; j < xs.size(); j++) {
					const float dx = distance(myx, xs[j]);
					const float dy = distance(myy, ys[j]);
					const float dz = distance(myz, zs[j]);
					const float h = hs[j];
					const float h2 = sqr(h);
					const float r2 = sqr(dx, dy, dz);
					constexpr float alpha = 0.75;
					constexpr float beta = 2.0f * alpha;
					if (r2 < std::max(myh2, h2) && r2 != 0.0f) {
						const float hinv = 1.0f / h;
						const float h3inv = hinv * sqr(hinv);
						const float rho = sph_den(h3inv);
						const float rhoinv = 1.0f / rho;
						const float p = ents[j] * pow(rho, SPH_GAMMA);
						const float c = sqrtf(SPH_GAMMA * p * rhoinv * ents[j]);
						const float cij = 0.5f * (myc + c);
						const float hij = 0.5f * (h + myh);
						const float rho_ij = 0.5f * (rho + myrho);
						const float dvx = myvx - vxs[j];
						const float dvy = myvy - vys[j];
						const float dvz = myvz - vzs[j];
						const float r = sqrt(sqr(dx, dy, dz));
						const float rinv = 1.0f / r;
						const float r2inv = sqr(rinv);
						const float uij = std::min(0.f, hij * (dvx * dx + dvy * dy + dvz * dz) * r2inv / (c + myc)) * (myfvel + fvels[j]);
						const float Pfac = -alpha * uij + beta * sqr(uij);
						const float dWdri = sph_dWdr(r, myhinv, myh3inv) * rinv;
						const float dWdrj = sph_dWdr(r, hinv, h3inv) * rinv;
						const float dWdri_x = dx * dWdri;
						const float dWdri_y = dy * dWdri;
						const float dWdri_z = dz * dWdri;
						const float dWdrj_x = dx * dWdrj;
						const float dWdrj_y = dy * dWdrj;
						const float dWdrj_z = dz * dWdrj;
						const float dWdrij_x = 0.5f * (dWdri_x + dWdrj_x);
						const float dWdrij_y = 0.5f * (dWdri_y + dWdrj_y);
						const float dWdrij_z = 0.5f * (dWdri_z + dWdrj_z);
						const float Pfacp1 = Pfac + 1.0f;
						const float Prho2i = myp * myrhoinv * myrhoinv;
						const float Prho2j = p * rhoinv * rhoinv;
						const float dpx = (Prho2j * dWdrj_x + Prho2i * dWdri_x);
						const float dpy = (Prho2j * dWdrj_y + Prho2i * dWdri_y);
						const float dpz = (Prho2j * dWdrj_z + Prho2i * dWdri_z);
						const float tmp = m * Pfacp1;
						float dvxdt = -dpx * tmp;
						float dvydt = -dpy * tmp;
						float dvzdt = -dpz * tmp;
						const float dt = std::max(rung_dt[rungs[j]], rung_dt[myrung]) * params.t0;
						float dAdt = (dpx * dvx + dpy * dvy + dpz * dvz) * Pfac;
						dAdt *= 0.5 * m * (SPH_GAMMA - 1.f) * pow(myrho, 1.0f - SPH_GAMMA);
						sph_particles_dvel(XDIM, i) += dvxdt * dt;
						sph_particles_dvel(YDIM, i) += dvydt * dt;
						sph_particles_dvel(ZDIM, i) += dvzdt * dt;
						sph_particles_dent(i) += dAdt * dt;
					}
				}
				for (int j = 0; j < xs.size(); j++) {
					const float dx = distance(myx, xs[j]);
					const float dy = distance(myy, ys[j]);
					const float dz = distance(myz, zs[j]);
					const float h = hs[j];
					const float h2 = sqr(h);
					const float r2 = sqr(dx, dy, dz);
					constexpr float alpha = 0.75;
					constexpr float beta = 2.0f * alpha;
					if (r2 < std::max(myh2, h2) && r2 != 0.0f) {
						const float hinv = 1.0f / h;
						const float h3inv = hinv * sqr(hinv);
						const float rho = sph_den(h3inv);
						const float rhoinv = 1.0f / rho;
						const float p = ents[j] * pow(rho, SPH_GAMMA);
						const float c = sqrtf(SPH_GAMMA * p * rhoinv * ents[j]);
						const float cij = 0.5f * (myc + c);
						const float hij = 0.5f * (h + myh);
						const float rho_ij = 0.5f * (rho + myrho);
						const float dvx = myvx - vxs[j];
						const float dvy = myvy - vys[j];
						const float dvz = myvz - vzs[j];
						const float dx = distance(myx, xs[j]);
						const float dy = distance(myy, ys[j]);
						const float dz = distance(myz, zs[j]);
						const float r = sqrt(sqr(dx, dy, dz));
						const float rinv = 1.0f / r;
						const float r2inv = sqr(rinv);
						const float uij = std::min(0.f, hij * (dvx * dx + dvy * dy + dvz * dz) * r2inv);
						const float Piij = -alpha * cij * uij + beta * sqr(uij) / rho_ij;
						const float dWdri = sph_dWdr(r, myhinv, myh3inv) * rinv;
						const float dWdrj = sph_dWdr(r, hinv, h3inv) * rinv;
						const float dWdri_x = dx * dWdri;
						const float dWdri_y = dy * dWdri;
						const float dWdri_z = dz * dWdri;
						const float dWdrj_x = dx * dWdrj;
						const float dWdrj_y = dy * dWdrj;
						const float dWdrj_z = dz * dWdrj;
						const float dWdrij_x = 0.5f * (dWdri_x + dWdrj_x);
						const float dWdrij_y = 0.5f * (dWdri_y + dWdrj_y);
						const float dWdrij_z = 0.5f * (dWdri_z + dWdrj_z);
						float dvxdt = -0.5f * (p * rhoinv * rhoinv * dWdrj_x + myp * myrhoinv * myrhoinv * dWdri_x);
						float dvydt = -0.5f * (p * rhoinv * rhoinv * dWdrj_y + myp * myrhoinv * myrhoinv * dWdri_y);
						float dvzdt = -0.5f * (p * rhoinv * rhoinv * dWdrj_z + myp * myrhoinv * myrhoinv * dWdri_z);
						const float dt = std::max(rung_dt[rungs[j]], rung_dt[myrung]) * params.t0;
						dvxdt -= Piij * dWdrij_x;
						dvydt -= Piij * dWdrij_y;
						dvzdt -= Piij * dWdrij_z;
						float dAdt = Piij * (dvx * dWdrij_x + dvy * dWdrij_y + dvz * dWdrij_z);
						dAdt *= 0.5 * (SPH_GAMMA - 1.f) * pow(rho, 1.0f - SPH_GAMMA);
						sph_particles_dvel(XDIM, i) += dvxdt * dt;
						sph_particles_dvel(YDIM, i) += dvydt * dt;
						sph_particles_dvel(ZDIM, i) += dvzdt * dt;
						sph_particles_dent(i) += dAdt * dt;
					}
				}
			}
		}
		kr.rc1 = true;
	}
		break;

	case SPH_RUN_UPDATE: {
		for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			const bool test1 = sph_particles_rung(i) >= params.min_rung;
			const bool test2 = sph_particles_semi_active(i);
			if (test1 || test2) {
				for (int dim = 0; dim < NDIM; dim++) {
					sph_particles_vel(dim, i) += sph_particles_dvel(dim, i);
					sph_particles_dvel(dim, i) = 0.0f;
				}
				sph_particles_ent(i) += sph_particles_dent(i);
				sph_particles_dent(i) = 0.0f;
			}

		}
	}
		break;

	case SPH_RUN_GRAVITY: {
		for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			auto& rung = sph_particles_rung(i);
			const bool test1 = rung >= params.min_rung;
			if (test1) {
				float dt;
				float& vx = sph_particles_vel(XDIM, i);
				float& vy = sph_particles_vel(YDIM, i);
				float& vz = sph_particles_vel(ZDIM, i);
				float& gx = sph_particles_gforce(XDIM, i);
				float& gy = sph_particles_gforce(YDIM, i);
				float& gz = sph_particles_gforce(ZDIM, i);
				if (rung < 0 || rung >= MAX_RUNG) {
					PRINT("Rung out of range %i\n", rung);
				} else {
					dt = 0.5f * rung_dt[rung] * params.t0;
				}
				vx = fmaf(gx, dt, vx);
				vy = fmaf(gy, dt, vy);
				vz = fmaf(gz, dt, vz);
				gx = gy = gz = 0.f;
			}
		}
	}
		break;

	case SPH_RUN_MARK_SEMIACTIVE: {
		load_data(true, true, false, false, true);
		for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			const auto myx = sph_particles_pos(XDIM, i);
			const auto myy = sph_particles_pos(YDIM, i);
			const auto myz = sph_particles_pos(ZDIM, i);
			sph_particles_semi_active(i) = false;
			for (int j = 0; j < xs.size(); j++) {
				const float h = hs[j];
				const float h2 = sqr(h);
				const float dx = distance(myx, xs[j]);
				const float dy = distance(myy, ys[j]);
				const float dz = distance(myz, zs[j]);
				const float r2 = sqr(dx, dy, dz);
				if (r2 < h2 && r2 != 0.0) {
					sph_particles_semi_active(i) = true;
				}
			}

		}
	}
		break;

	case SPH_RUN_FIND_BOXES: {
		range<fixed32> ibox, obox;
		for (int dim = 0; dim < NDIM; dim++) {
			ibox.begin[dim] = fixed32::max();
			ibox.end[dim] = 0.0f;
		}
		obox = ibox;
		for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			const bool otest1 = params.set2 == SPH_SET_ACTIVE && sph_particles_rung(i) >= params.min_rung;
			const bool otest2 = params.set2 == SPH_SET_SEMIACTIVE && sph_particles_semi_active(i);
			const bool otest3 = params.set2 == SPH_SET_ALL;
			const bool itest1 = params.set1 == SPH_SET_ACTIVE && sph_particles_rung(i) >= params.min_rung;
			const bool itest2 = params.set1 == SPH_SET_SEMIACTIVE && sph_particles_semi_active(i);
			const bool itest3 = params.set1 == SPH_SET_ALL;
			const bool otest = otest1 || otest2 || otest3;
			const bool itest = itest1 || itest2 || itest3;
			if (otest || itest) {
				const float& h = sph_particles_smooth_len(i);
				const float h2 = sqr(h);
				const float hinv = 1.0f / h;
				const auto myx = sph_particles_pos(XDIM, i);
				const auto myy = sph_particles_pos(YDIM, i);
				const auto myz = sph_particles_pos(ZDIM, i);
				const int k = i - self_ptr->part_range.first;
				array<fixed32, NDIM> X;
				X[XDIM] = myx;
				X[YDIM] = myy;
				X[ZDIM] = myz;
				if (otest) {
					for (int dim = 0; dim < NDIM; dim++) {
						const float myh = params.h_wt * h;
						if (distance(obox.begin[dim], X[dim]) + myh >= 0.0) {
							obox.begin[dim] = fixed<int32_t>(X[dim]) - fixed<int32_t>(myh);
						}
						if (distance(X[dim], obox.end[dim]) + myh >= 0.0) {
							obox.end[dim] = fixed<int32_t>(X[dim]) + fixed<int32_t>(myh);
						}
					}
				}
				if (itest) {
					for (int dim = 0; dim < NDIM; dim++) {
						const float myh = params.h_wt * h;
						if (distance(ibox.begin[dim], X[dim]) >= 0.0) {
							ibox.begin[dim] = X[dim];
						}
						if (distance(X[dim], ibox.end[dim]) >= 0.0) {
							ibox.end[dim] = X[dim];
						}
					}
				}
			}
		}
		kr.inner_box = ibox;
		kr.outer_box = obox;
	}
		break;
	}
	if (self_ptr->leaf) {
		cleanup_workspace(std::move(workspace));
		return hpx::make_ready_future(kr);
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		cleanup_workspace(std::move(workspace));
		const sph_tree_node* cl = sph_tree_get_node(self_ptr->children[LEFT]);
		const sph_tree_node* cr = sph_tree_get_node(self_ptr->children[RIGHT]);
		std::array<hpx::future<sph_run_return>, NCHILD> futs;
		futs[RIGHT] = sph_run_fork(params, self_ptr->children[RIGHT], checklist, thread_left);
		futs[LEFT] = sph_run_fork(params, self_ptr->children[LEFT], std::move(checklist), false);

		const auto finish = [self,params](hpx::future<sph_run_return>& fl, hpx::future<sph_run_return>& fr) {
			sph_run_return kr;
			const auto rcl = fl.get();
			const auto rcr = fr.get();
			kr += rcl;
			kr += rcr;
			if( (params.run_type == SPH_RUN_FIND_BOXES) ) {
				sph_tree_set_boxes(self,kr.inner_box, kr.outer_box);
			}
			if( params.run_type == SPH_RUN_SMOOTH_LEN) {
				sph_tree_set_box_active(self,kr.rc2);
			} else {
				sph_tree_set_box_active(self,false);
			}
			if( params.run_type == SPH_RUN_HYDRO) {
				sph_tree_set_box_active(self,kr.rc1);
			}
			return kr;
		};
		if (futs[LEFT].is_ready() && futs[RIGHT].is_ready()) {
			return hpx::make_ready_future(finish(futs[LEFT], futs[RIGHT]));
		} else {
			return hpx::when_all(futs.begin(), futs.end()).then([finish,tm,self_ptr](hpx::future<std::vector<hpx::future<sph_run_return>>> futsfut) {
				auto futs = futsfut.get();
				return finish(futs[LEFT], futs[RIGHT]);
			});
		}
	}

}

