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
#include <cosmictiger/sph.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>

#include <unistd.h>
#include <stack>

HPX_PLAIN_ACTION (sph_tree_neighbor);

#define MAX_ACTIVE_WORKSPACES 1

static float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

inline bool range_intersect(const fixed32_range& a, const fixed32_range& b) {
	bool intersect = a.valid && b.valid;
	if (intersect) {
		for (int dim = 0; dim < NDIM; dim++) {
			//	print( "%i %e %e | %e %e\n", dim, a.begin[dim].to_float(), a.end[dim].to_float(), b.begin[dim].to_float(), b.end[dim].to_float());
			if (distance(b.end[dim], a.begin[dim]) >= 0.0 && distance(a.end[dim], b.begin[dim]) >= 0.0) {
			} else {
//			PRINT( "false\n");
				intersect = false;
				break;
			}
		}
	}
	return intersect;
}

inline bool range_contains(const fixed32_range& a, const array<fixed32, NDIM> x) {
	bool contains = a.valid;
	if (contains) {
		for (int dim = 0; dim < NDIM; dim++) {
			if (distance(x[dim], a.begin[dim]) >= 0.0 && distance(a.end[dim], x[dim]) >= 0.0) {
			} else {
				contains = false;
				break;
			}
		}
	}
	return contains;
}

hpx::future<sph_tree_neighbor_return> sph_tree_neighbor_fork(sph_tree_neighbor_params params, tree_id self, vector<tree_id> checklist, int level,
		bool threadme) {
	static std::atomic<int> nthreads(0);
	hpx::future<sph_tree_neighbor_return> rc;
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
		rc = sph_tree_neighbor(params, self, std::move(checklist), level + 1);
	} else if (remote) {
		rc = hpx::async<sph_tree_neighbor_action>(HPX_PRIORITY_HI, hpx_localities()[self_ptr->proc_range.first], params, self, std::move(checklist), level + 1);
	} else {
		const auto thread_priority = all_local ? HPX_PRIORITY_LO : HPX_PRIORITY_NORMAL;
		rc = hpx::async(thread_priority, [self,level, params] (vector<tree_id> checklist) {
			auto rc = sph_tree_neighbor(params, self,std::move(checklist), level + 1);
			nthreads--;
			return rc;
		}, std::move(checklist));
	}
	return rc;
}

hpx::future<sph_tree_neighbor_return> sph_tree_neighbor(sph_tree_neighbor_params params, tree_id self, vector<tree_id> checklist, int level) {
///	PRINT( "%i %i\n", level, checklist.size());
	stack_trace_activate();
	const sph_tree_node* self_ptr = sph_tree_get_node(self);
	sph_tree_neighbor_return kr;
	if (params.run_type == SPH_TREE_NEIGHBOR_NEIGHBORS && self_ptr->local_root) {
		sph_tree_free_neighbor_list();
		sph_tree_clear_neighbor_ranges();
	}
	if (params.run_type == SPH_TREE_NEIGHBOR_NEIGHBORS && checklist.size() == 0) {
		return hpx::make_ready_future(kr);
	}
	ASSERT(self.proc == hpx_rank());
	bool thread_left = true;
	vector<tree_id> nextlist;
	vector<tree_id> leaflist;
	fixed32_range box;

	if (params.run_type == SPH_TREE_NEIGHBOR_NEIGHBORS) {
		do {
			nextlist.resize(0);
			for (int ci = 0; ci < checklist.size(); ci++) {
				const auto* other = sph_tree_get_node(checklist[ci]);
				const bool test1 = range_intersect(self_ptr->outer_box, other->inner_box);
			/*	if (level > 6) {
					if (!test1 && self_ptr == other && self_ptr->nactive > 0) {
						PRINT("!\n");
						static mutex_type mutex;
						std::lock_guard<mutex_type> lock(mutex);
						PRINT( "%i %i\n", self_ptr->outer_box.valid, self_ptr->inner_box.valid);
					}
				}*/
				const bool test2 = range_intersect(self_ptr->inner_box, other->outer_box) && (other->nactive > 0);
				const bool test3 = level <= 6;
				if (test1 || test2 || test3) {
					if (other->leaf) {
						leaflist.push_back(checklist[ci]);
					} else {
						nextlist.push_back(other->children[LEFT]);
						nextlist.push_back(other->children[RIGHT]);
					}
				}
			}
			checklist = std::move(nextlist);
		} while (self_ptr->leaf && checklist.size());
	}
	if (self_ptr->leaf) {
		switch (params.run_type) {
		case SPH_TREE_NEIGHBOR_NEIGHBORS: {
			pair<int> rng;
			rng.first = sph_tree_allocate_neighbor_list(leaflist);
			rng.second = leaflist.size() + rng.first;
			//	if( leaflist.size() == 0) {
			//		PRINT( "zero\n");
			//	}
			sph_tree_set_neighbor_range(self, rng);
		}
			break;
		case SPH_TREE_NEIGHBOR_BOXES: {
			fixed32_range ibox, obox;
			for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
				const bool active = sph_particles_rung(i) >= params.min_rung;
				const bool semiactive = !active && sph_particles_semi_active(i);
				const float h = params.h_wt * sph_particles_smooth_len(i);
				const auto myx = sph_particles_pos(XDIM, i);
				const auto myy = sph_particles_pos(YDIM, i);
				const auto myz = sph_particles_pos(ZDIM, i);
				const int k = i - self_ptr->part_range.first;
				array<fixed32, NDIM> X;
				X[XDIM] = myx;
				X[YDIM] = myy;
				X[ZDIM] = myz;
				if ((active && (params.set & SPH_SET_ACTIVE)) || (semiactive && (params.set & SPH_SET_SEMIACTIVE))) {
					obox.accumulate(X, h);
				}
				ibox.accumulate(X);
			}
			kr.inner_box = ibox;
			kr.outer_box = obox;
			sph_tree_set_boxes(self, kr.inner_box, kr.outer_box);
		}
			break;
		}
		return hpx::make_ready_future(kr);
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		const sph_tree_node* cl = sph_tree_get_node(self_ptr->children[LEFT]);
		const sph_tree_node* cr = sph_tree_get_node(self_ptr->children[RIGHT]);
		std::array<hpx::future<sph_tree_neighbor_return>, NCHILD> futs;
		futs[RIGHT] = sph_tree_neighbor_fork(params, self_ptr->children[RIGHT], checklist, level, thread_left);
		futs[LEFT] = sph_tree_neighbor_fork(params, self_ptr->children[LEFT], std::move(checklist), level, false);

		const auto finish = [self,params](hpx::future<sph_tree_neighbor_return>& fl, hpx::future<sph_tree_neighbor_return>& fr) {
			sph_tree_neighbor_return kr;
			const auto rcl = fl.get();
			const auto rcr = fr.get();
			kr += rcl;
			kr += rcr;
			if( params.run_type == SPH_TREE_NEIGHBOR_BOXES ) {
				sph_tree_set_boxes(self, kr.inner_box, kr.outer_box);
			}
			return kr;
		};
		if (futs[LEFT].is_ready() && futs[RIGHT].is_ready()) {
			return hpx::make_ready_future(finish(futs[LEFT], futs[RIGHT]));
		} else {
			return hpx::when_all(futs.begin(), futs.end()).then([finish,self_ptr](hpx::future<std::vector<hpx::future<sph_tree_neighbor_return>>> futsfut) {
				auto futs = futsfut.get();
				return finish(futs[LEFT], futs[RIGHT]);
			});
		}
	}

}

HPX_PLAIN_ACTION (sph_run);

bool has_active_neighbors(const sph_tree_node* self) {
	bool rc = false;
	for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
		const auto id = sph_tree_get_neighbor(i);
		if (sph_tree_get_node(id)->nactive > 0) {
			rc = true;
			break;
		}
	}
	return rc;
}

struct sph_data_vecs {
	vector<fixed32> xs;
	vector<fixed32> ys;
	vector<fixed32> zs;
	vector<char> rungs;
	vector<float> fvels;
	vector<float> hs;
	vector<float> ents;
	vector<float> vxs;
	vector<float> vys;
	vector<float> vzs;
	void clear() {
		xs.clear();
		ys.clear();
		zs.clear();
		rungs.clear();
		fvels.clear();
		hs.clear();
		ents.clear();
		vxs.clear();
		vys.clear();
		vzs.clear();
	}
};

template<bool do_rungs, bool do_smoothlens, bool do_ent, bool do_vel, bool do_fvel, bool check_inner, bool check_outer, bool active_only>
void load_data(const sph_tree_node* self_ptr, const vector<tree_id>& neighborlist, sph_data_vecs& d, int min_rung) {
	part_int offset;
	d.clear();
	for (int ci = 0; ci < neighborlist.size(); ci++) {
		const auto* other = sph_tree_get_node(neighborlist[ci]);
		const auto this_sz = other->part_range.second - other->part_range.first;
		offset = d.xs.size();
		const int new_sz = d.xs.size() + this_sz;
		d.xs.resize(new_sz);
		d.ys.resize(new_sz);
		d.zs.resize(new_sz);
		if (do_rungs) {
			d.rungs.resize(new_sz);
		}
		if (do_smoothlens) {
			d.hs.resize(new_sz);
		}
		if (do_ent) {
			d.ents.resize(new_sz);
		}
		if (do_vel) {
			d.vxs.resize(new_sz);
			d.vys.resize(new_sz);
			d.vzs.resize(new_sz);
		}
		if (do_fvel) {
			d.fvels.resize(new_sz);
		}
		sph_particles_global_read_pos(other->global_part_range(), d.xs.data(), d.ys.data(), d.zs.data(), offset);
		if (do_rungs || do_smoothlens) {
			sph_particles_global_read_rungs_and_smoothlens(other->global_part_range(), d.rungs, d.hs, offset);
		}
		if (do_ent || do_vel) {
			sph_particles_global_read_sph(other->global_part_range(), d.ents, d.vxs, d.vys, d.vzs, offset);
		}
		if (do_fvel) {
			sph_particles_global_read_fvels(other->global_part_range(), d.fvels, offset);
		}
		int i = offset;
		while (i < d.xs.size()) {
			array<fixed32, NDIM> X;
			X[XDIM] = d.xs[i];
			X[YDIM] = d.ys[i];
			X[ZDIM] = d.zs[i];
			bool test0 = !(active_only && d.rungs[i] < min_rung);
			const bool test1 = check_inner && !range_contains(self_ptr->outer_box, X);
			bool test2 = true;
			if (check_outer && test1) {
				assert(do_rungs);
				test2 = true;
				const auto& box = self_ptr->inner_box;
				for (int dim = 0; dim < NDIM; dim++) {
					if (distance(box.begin[dim], X[dim]) + d.hs[i] >= 0.0 && distance(X[dim], box.end[dim]) + d.hs[i]) {
					} else {
						test2 = false;
						break;
					}
				}
			}
			if (test0 && test1 && test2) {
				d.xs[i] = d.xs.back();
				d.ys[i] = d.ys.back();
				d.zs[i] = d.zs.back();
				d.xs.pop_back();
				d.ys.pop_back();
				d.zs.pop_back();
				if (do_rungs) {
					d.rungs[i] = d.rungs.back();
					d.rungs.pop_back();
				}
				if (do_smoothlens) {
					d.hs[i] = d.hs.back();
					d.hs.pop_back();
				}
				if (do_ent) {
					d.ents[i] = d.ents.back();
					d.ents.pop_back();
				}
				if (do_vel) {
					d.vxs[i] = d.vxs.back();
					d.vys[i] = d.vys.back();
					d.vzs[i] = d.vzs.back();
					d.vxs.pop_back();
					d.vys.pop_back();
					d.vzs.pop_back();
				}
				if (do_fvel) {
					d.fvels[i] = d.fvels.back();
					d.fvels.pop_back();
				}
			} else {
				i++;
			}
		}
	}
}

static sph_run_return sph_smoothlens(const sph_tree_node* self_ptr, const vector<fixed32>& xs, const vector<fixed32>& ys, const vector<fixed32>& zs,
		int min_rung, bool active, bool semiactive, int nactive, int nneighbor) {
	sph_run_return rc;
	const int self_nparts = self_ptr->part_range.second - self_ptr->part_range.first;
	float f, dfdh;
	simd_float f_simd, dfdh_simd;
	bool box_xceeded = false;
	float max_h = 0.0;
	float min_h = std::numeric_limits<float>::max();
	int max_cnt = 0;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		const bool test1 = active && (sph_particles_rung(i) >= min_rung);
		const bool test2 = semiactive && (sph_particles_semi_active(i) && (sph_particles_rung(i) < min_rung));
		if (test1 || test2) {
			float error;
			float& h = sph_particles_smooth_len(i);
			const simd_int myx = (sph_particles_pos(XDIM, i).raw());
			const simd_int myy = (sph_particles_pos(YDIM, i).raw());
			const simd_int myz = (sph_particles_pos(ZDIM, i).raw());
			int cnt = 0;
			do {
				const simd_float h2 = sqr(h);
				const simd_float hinv = 1.0f / h;
				//PRINT( "%e %e %e \n", h, h2[0], hinv[0]);
				f_simd = 0.0;
				dfdh_simd = 0.0;
				const int maxj = round_up((int) xs.size(), SIMD_FLOAT_SIZE);
				for (int j = 0; j < maxj; j += SIMD_FLOAT_SIZE) {
					const int maxk = std::min((int) xs.size(), j + SIMD_FLOAT_SIZE);
					simd_int X, Y, Z;
					simd_float mask;
					for (int k = j; k < maxk; k++) {
						const int kmj = k - j;
						X[kmj] = xs[k].raw();
						Y[kmj] = ys[k].raw();
						Z[kmj] = zs[k].raw();
						mask[kmj] = 1.0f;
					}
					for (int k = maxk; k < j + SIMD_FLOAT_SIZE; k++) {
						const int kmj = k - j;
						X[kmj] = Y[kmj] = Z[kmj] = 0.0;
						mask[kmj] = 0.0f;
					}
					static const simd_float _2float(fixed2float);
					const simd_float dx = simd_float(myx - X) * _2float;
					const simd_float dy = simd_float(myy - Y) * _2float;
					const simd_float dz = simd_float(myz - Z) * _2float;
					const simd_float r2 = sqr(dx, dy, dz);
					mask *= (r2 < h2);
					const simd_float r = sqrt(r2);
					const simd_float q = r * hinv;
					const simd_float q2 = sqr(q);
					static const simd_float one = simd_float(1);
					static const simd_float four = simd_float(4);
					const simd_float _1mq = one - q;
					const simd_float _1mq2 = sqr(_1mq);
					const simd_float _1mq3 = _1mq * _1mq2;
					const simd_float _1mq4 = _1mq * _1mq3;
					static const simd_float A = simd_float(float(21.0 * 2.0 / 3.0));
					static const simd_float B = simd_float(float(840.0 / 3.0));
					const simd_float w = A * _1mq4 * (one + four * q);
					const simd_float dwdh = B * _1mq3 * q2 * hinv;
					f_simd += w * mask;
					dfdh_simd += dwdh * mask;
				}
				f = f_simd.sum();
				dfdh = dfdh_simd.sum();
				f -= SPH_NEIGHBOR_COUNT;
				float dh = -f / dfdh;
				error = fabs(log(h + dh) - log(h));
				h += dh;
				//if( cnt > 5 )
//					PRINT( "%i %e %e\n", cnt, h, dh);
				array<fixed32, NDIM> X;
				X[XDIM] = sph_particles_pos(XDIM, i);
				X[YDIM] = sph_particles_pos(YDIM, i);
				X[ZDIM] = sph_particles_pos(ZDIM, i);
				for (int dim = 0; dim < NDIM; dim++) {
					if (distance(self_ptr->outer_box.end[dim], X[dim]) + h < 0.0) {
						box_xceeded = true;
						break;
					}
					if (distance(X[dim], self_ptr->outer_box.begin[dim]) + h < 0.0) {
						box_xceeded = true;
						break;
					}
				}
				cnt++;
				if (cnt > 20) {
					PRINT("density solver failed to converge %e %e %i %i %i\n", h, dh, nactive, self_ptr->outer_box.valid, nneighbor);
					abort();
				}
			} while (error > SPH_SMOOTHLEN_TOLER);
			max_cnt = std::max(max_cnt, cnt);
			max_h = std::max(max_h, h);
			min_h = std::min(min_h, h);
		}
		if (box_xceeded) {
			break;
		}
	}
	rc.rc = box_xceeded;
	rc.hmin = min_h;
	rc.hmax = max_h;
	return rc;
}

static sph_run_return sph_mark_semiactive(const sph_tree_node* self_ptr, const vector<fixed32>& xs, const vector<fixed32>& ys, const vector<fixed32>& zs,
		const vector<char>& rungs, const vector<float>& hs, int min_rung) {
	sph_run_return rc;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		if (sph_particles_rung(i) < min_rung) {
			const simd_int myx = sph_particles_pos(XDIM, i).raw();
			const simd_int myy = sph_particles_pos(YDIM, i).raw();
			const simd_int myz = sph_particles_pos(ZDIM, i).raw();
			sph_particles_semi_active(i) = false;
			for (int j = 0; j < xs.size(); j += SIMD_FLOAT_SIZE) {
				simd_int x, y, z;
				simd_float mask;
				for (int k = j; k < j + SIMD_FLOAT_SIZE; k++) {
					const int kmj = k - j;
					if (k < xs.size()) {
						x[kmj] = xs[j].raw();
						y[kmj] = ys[j].raw();
						z[kmj] = zs[j].raw();
						mask[kmj] = 1.0f;
					} else {
						x[kmj] = myx[0];
						y[kmj] = myy[0];
						z[kmj] = myz[0];
						mask[kmj] = 0.f;
					}
				}
				const simd_float h = hs[j];
				const simd_float h2 = sqr(h);
				static const simd_float _2float(fixed2float);
				const simd_float dx = simd_float(myx - x) * _2float;
				const simd_float dy = simd_float(myy - y) * _2float;
				const simd_float dz = simd_float(myz - z) * _2float;
				const simd_float r2 = sqr(dx, dy, dz);
				mask *= (r2 < h2) * (r2 > 0.0);
				const bool semi = mask.sum();
				if (semi) {
					sph_particles_semi_active(i) = semi;
					break;
				}
			}
		} else {
			sph_particles_semi_active(i) = true;
		}
	}
	return rc;
}

sph_run_return sph_courant(const sph_tree_node* self_ptr, const vector<fixed32>& main_xs, const vector<fixed32>& main_ys, const vector<fixed32>& main_zs,
		const vector<float>& main_hs, const vector<float>& main_ents, const vector<float>& main_vxs, const vector<float>& main_vys, const vector<float>& main_vzs,
		int min_rung, float ascale, float t0) {
	sph_run_return rc;
	static thread_local vector<simd_int> xs;
	static thread_local vector<simd_int> ys;
	static thread_local vector<simd_int> zs;
	static thread_local vector<simd_float> hs;
	static thread_local vector<simd_float> ents;
	static thread_local vector<simd_float> vxs;
	static thread_local vector<simd_float> vys;
	static thread_local vector<simd_float> vzs;
	static thread_local vector<simd_float> fvels;
	static thread_local vector<simd_float> masks;
	const auto rung2dt = [](simd_int rung) {
		simd_float dt;
		for( int k = 0; k < SIMD_FLOAT_SIZE; k++) {
			dt[k] = rung_dt[rung[k]];
		}
		return dt;
	};
	const simd_float m = get_options().sph_mass;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		if (sph_particles_rung(i) >= min_rung) {
			xs.resize(0);
			ys.resize(0);
			zs.resize(0);
			hs.resize(0);
			ents.resize(0);
			fvels.resize(0);
			vxs.resize(0);
			vys.resize(0);
			vzs.resize(0);
			masks.resize(0);
			const simd_float myh = sph_particles_smooth_len(i);
			const simd_int myrung = sph_particles_rung(i);
			const simd_float myh2 = sqr(myh);
			const simd_float myhinv = simd_float(1.0) / myh;
			const simd_float myh3inv = myhinv * sqr(myhinv);
			const simd_float myrho = sph_den(myh3inv);
			const simd_float myrhoinv = simd_float(1.0) / myrho;
			const simd_int myx = sph_particles_pos(XDIM, i).raw();
			const simd_int myy = sph_particles_pos(YDIM, i).raw();
			const simd_int myz = sph_particles_pos(ZDIM, i).raw();
			const simd_float myvx = sph_particles_vel(XDIM, i);
			const simd_float myvy = sph_particles_vel(YDIM, i);
			const simd_float myvz = sph_particles_vel(ZDIM, i);
			const simd_float myent = sph_particles_ent(i);
			const simd_float myp = pow(myrho, simd_float(SPH_GAMMA)) * myent;
			const simd_float myc = sqrt(simd_float(SPH_GAMMA) * myp * myrhoinv);
			int base = -1;
			int offset = SIMD_FLOAT_SIZE;
			static const simd_float _2float(fixed2float);

			for (int j = 0; j < main_xs.size(); j += SIMD_FLOAT_SIZE) {
				simd_int x, y, z;
				simd_float mask, h;
				const int maxj = std::min((int) main_xs.size(), j + SIMD_FLOAT_SIZE);
				for (int k = j; k < j + SIMD_FLOAT_SIZE; k++) {
					const int kmj = k - j;
					if (k < main_xs.size()) {
						x[kmj] = main_xs[j].raw();
						y[kmj] = main_ys[j].raw();
						z[kmj] = main_zs[j].raw();
						h[kmj] = main_hs[j];
						mask[kmj] = 1.0f;
					} else {
						h[kmj] = 1.0f;
						x[kmj] = y[kmj] = z[kmj] = mask[kmj] = 0.0f;
					}
				}
				const simd_float dx = simd_float(myx - x) * _2float;
				const simd_float dy = simd_float(myy - y) * _2float;
				const simd_float dz = simd_float(myz - z) * _2float;
				const simd_float h2 = sqr(h);
				const simd_float r2 = sqrt(sqr(dx, dy, dz));
				mask *= simd_float(r2 > 0.0) * (simd_float(r2 < h2) + simd_float(r2 < myh));
				for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
					if (mask[k] > 0.0) {
						if (offset == SIMD_FLOAT_SIZE) {
							xs.push_back(myx);
							ys.push_back(myy);
							zs.push_back(myz);
							hs.push_back(simd_float(1.0));
							ents.push_back(simd_float(1.0));
							vxs.push_back(simd_float(0.0));
							vys.push_back(simd_float(0.0));
							vzs.push_back(simd_float(0.0));
							fvels.push_back(simd_float(0.0));
							masks.push_back(simd_float(0.0));
							base++;
							offset = 0;
						}
						const int jpk = j + k;
						xs.back()[offset] = main_xs[jpk].raw();
						ys.back()[offset] = main_ys[jpk].raw();
						zs.back()[offset] = main_zs[jpk].raw();
						hs.back()[offset] = main_hs[jpk];
						ents.back()[offset] = main_ents[jpk];
						vxs.back()[offset] = main_vxs[jpk];
						vys.back()[offset] = main_vys[jpk];
						vzs.back()[offset] = main_vzs[jpk];
						masks.back()[offset] = 1.0f;
						offset++;
					}
				}
			}
			static const simd_float tiny = simd_float(1e-15);
			static const simd_float one(1.0f);
			static const simd_float zero(0.0f);
			simd_float max_vsig(zero);
			for (int j = 0; j < xs.size(); j++) {
				const simd_float dx = simd_float(myx - xs[j]) * _2float;
				const simd_float dy = simd_float(myy - ys[j]) * _2float;
				const simd_float dz = simd_float(myz - zs[j]) * _2float;
				const simd_float h = hs[j];
				const simd_float h2 = sqr(h);
				const simd_float hinv = one / h;
				const simd_float hinv3 = hinv * sqr(hinv);
				const simd_float rho = sph_den(hinv3);
				const simd_float c = sqrt(simd_float(SPH_GAMMA) * pow(rho, simd_float(SPH_GAMMA - 1.0f)) * ents[j]);
				const simd_float rinv = one / (tiny + sqrt(sqr(dx, dy, dz)));
				const simd_float dvx = myvx - vxs[j];
				const simd_float dvy = myvy - vys[j];
				const simd_float dvz = myvz - vzs[j];
				const simd_float w = min(zero, (dvy * dx + dvy * dy + dvz * dz) * rinv);
				const simd_float sig = masks[j] * (c + myc - 3.0f * w);
				max_vsig = max(max_vsig, sig);
			}
			const float max_c = max_vsig.sum();
			rc.max_vsig = max_c;
			float dthydro = max_c / (ascale * myh[0]);
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
			const float factor = eta * sqrtf(ascale * hgrav);
			const float dt_grav = std::min(factor / sqrtf(sqrtf(g2)), (float) t0);
			const float dt = std::min(dt_grav, dthydro);
			const int rung1 = ceilf(log2f(t0) - log2f(dt));
			rung = std::max((int) rung1, std::max(rung - 1, min_rung));
//			PRINT( "%i %e %e %e %e\n", rung, dt_grav, gx, gy, gz);
			if (rung < 0 || rung >= MAX_RUNG) {
				PRINT("Rung out of range %e %i %i %e %e %e %e %e %e %e\n", sph_particles_smooth_len(i), rung1, rung, myc, sqrt(sqr(myvx, myvy, myvz)), gx, gy, gz,
						dt_grav, dthydro);
			}
			rc.max_rung = std::max(rc.max_rung, (int) rung);

		}
	}
	return rc;
}

sph_run_return sph_gravity(const sph_tree_node* self_ptr, int min_rung, float t0) {
	sph_run_return rc;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		auto& rung = sph_particles_rung(i);
		const bool test1 = rung >= min_rung;
		if (test1) {
			float dt;
			float& vx = sph_particles_vel(XDIM, i);
			float& vy = sph_particles_vel(YDIM, i);
			float& vz = sph_particles_vel(ZDIM, i);
			float& gx = sph_particles_gforce(XDIM, i);
			float& gy = sph_particles_gforce(YDIM, i);
			float& gz = sph_particles_gforce(ZDIM, i);
			dt = 0.5f * rung_dt[rung] * t0;
			vx = fmaf(gx, dt, vx);
			vy = fmaf(gy, dt, vy);
			vz = fmaf(gz, dt, vz);
			gx = gy = gz = 0.f;
		}
	}
	return rc;
}

sph_run_return sph_fvels(const sph_tree_node* self_ptr, const vector<fixed32>& main_xs, const vector<fixed32>& main_ys, const vector<fixed32>& main_zs,
		const vector<float>& main_hs, const vector<float>& main_vxs, const vector<float>& main_vys, const vector<float>& main_vzs) {
	sph_run_return rc;
	static thread_local vector<simd_int> xs;
	static thread_local vector<simd_int> ys;
	static thread_local vector<simd_int> zs;
	static thread_local vector<simd_float> hs;
	static thread_local vector<simd_float> vxs;
	static thread_local vector<simd_float> vys;
	static thread_local vector<simd_float> vzs;
	static thread_local vector<simd_float> masks;
	const auto rung2dt = [](simd_int rung) {
		simd_float dt;
		for( int k = 0; k < SIMD_FLOAT_SIZE; k++) {
			dt[k] = rung_dt[rung[k]];
		}
		return dt;
	};
	const simd_float m = get_options().sph_mass;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		if (sph_particles_semi_active(i)) {
			xs.resize(0);
			ys.resize(0);
			zs.resize(0);
			hs.resize(0);
			vxs.resize(0);
			vys.resize(0);
			vzs.resize(0);
			masks.resize(0);
			int base = -1;
			int offset = SIMD_FLOAT_SIZE;
			static const simd_float _2float(fixed2float);
			const simd_int myx = sph_particles_pos(XDIM, i).raw();
			const simd_int myy = sph_particles_pos(YDIM, i).raw();
			const simd_int myz = sph_particles_pos(ZDIM, i).raw();
			for (int j = 0; j < main_xs.size(); j += SIMD_FLOAT_SIZE) {
				simd_int x, y, z;
				simd_float mask, h;
				const int maxj = std::min((int) main_xs.size(), j + SIMD_FLOAT_SIZE);
				for (int k = j; k < j + SIMD_FLOAT_SIZE; k++) {
					const int kmj = k - j;
					if (k < main_xs.size()) {
						x[kmj] = main_xs[j].raw();
						y[kmj] = main_ys[j].raw();
						z[kmj] = main_zs[j].raw();
						h[kmj] = main_hs[j];
						mask[kmj] = 1.0f;
					} else {
						h[kmj] = 1.0f;
						x[kmj] = y[kmj] = z[kmj] = mask[kmj] = 0.0f;
					}
				}
				const simd_float dx = simd_float(myx - x) * _2float;
				const simd_float dy = simd_float(myy - y) * _2float;
				const simd_float dz = simd_float(myz - z) * _2float;
				const simd_float r2 = sqrt(sqr(dx, dy, dz));
				const simd_float myh = sph_particles_smooth_len(i);
				const simd_float myh2 = sqr(myh);
				mask *= simd_float(r2 > 0.0) * (simd_float(r2 < myh2));
				for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
					if (mask[k] > 0.0) {
						if (offset == SIMD_FLOAT_SIZE) {
							xs.push_back(myx);
							ys.push_back(myy);
							zs.push_back(myz);
							hs.push_back(simd_float(1.0));
							vxs.push_back(simd_float(0.0));
							vys.push_back(simd_float(0.0));
							vzs.push_back(simd_float(0.0));
							masks.push_back(simd_float(0.0));
							base++;
							offset = 0;
						}
						const int jpk = j + k;
						xs.back()[offset] = main_xs[jpk].raw();
						ys.back()[offset] = main_ys[jpk].raw();
						zs.back()[offset] = main_zs[jpk].raw();
						hs.back()[offset] = main_hs[jpk];
						vxs.back()[offset] = main_vxs[jpk];
						vys.back()[offset] = main_vys[jpk];
						vzs.back()[offset] = main_vzs[jpk];
						masks.back()[offset] = 1.0f;
						offset++;
					}
				}
			}
			static const simd_float one(1.0f);
			simd_float dvx_dx = 0.0f;
			simd_float dvx_dy = 0.0f;
			simd_float dvx_dz = 0.0f;
			simd_float dvy_dx = 0.0f;
			simd_float dvy_dy = 0.0f;
			simd_float dvy_dz = 0.0f;
			simd_float dvz_dx = 0.0f;
			simd_float dvz_dy = 0.0f;
			simd_float dvz_dz = 0.0f;
			static const simd_float tiny = simd_float(1e-15);
			for (int j = 0; j < xs.size(); j++) {
				const simd_float dx = simd_float(myx - xs[j]) * _2float;
				const simd_float dy = simd_float(myy - ys[j]) * _2float;
				const simd_float dz = simd_float(myz - zs[j]) * _2float;
				const simd_float r2 = sqr(dx, dy, dz);
				const simd_float r = sqrt(r2);
				const simd_float rinv = one / (r + tiny);
				const simd_float hinv = one / (hs[j]);
				const simd_float hinv3 = sqr(hinv) * hinv;
				const simd_float rho = sph_den(hinv3);
				const simd_float rhoinv = one / rho;
				const simd_float dWdr = sph_dWdr_rinv(r, hinv, hinv3);
				const simd_float tmp = m * dWdr * rhoinv * masks[j];
				const simd_float dWdr_x = dx * tmp;
				const simd_float dWdr_y = dy * tmp;
				const simd_float dWdr_z = dz * tmp;
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
			const float dvx_dx_sum = dvx_dx.sum();
			const float dvx_dy_sum = dvx_dy.sum();
			const float dvx_dz_sum = dvx_dz.sum();
			const float dvy_dx_sum = dvy_dx.sum();
			const float dvy_dy_sum = dvy_dy.sum();
			const float dvy_dz_sum = dvy_dz.sum();
			const float dvz_dx_sum = dvz_dx.sum();
			const float dvz_dy_sum = dvz_dy.sum();
			const float dvz_dz_sum = dvz_dz.sum();
			const float abs_div_v = fabs(dvx_dx_sum + dvy_dy_sum + dvz_dz_sum);
			const float curl_vx = dvz_dy_sum - dvy_dz_sum;
			const float curl_vy = -dvz_dx_sum + dvx_dz_sum;
			const float curl_vz = dvy_dx_sum - dvx_dy_sum;
			const float abs_curl_v = sqrt(sqr(curl_vx, curl_vy, curl_vz));
			const float fvel = abs_div_v / (abs_div_v + abs_curl_v);
			sph_particles_fvel(i) = fvel;
		}
	}
	return rc;
}

sph_run_return sph_hydro(const sph_tree_node* self_ptr, const vector<fixed32>& main_xs, const vector<fixed32>& main_ys, const vector<fixed32>& main_zs,
		const vector<char>& main_rungs, const vector<float>& main_hs, const vector<float>& main_ents, const vector<float>& main_vxs,
		const vector<float>& main_vys, const vector<float>& main_vzs, const vector<float>& main_fvels, float t0) {
	sph_run_return rc;
	static thread_local vector<simd_int> xs;
	static thread_local vector<simd_int> ys;
	static thread_local vector<simd_int> zs;
	static thread_local vector<simd_int> rungs;
	static thread_local vector<simd_float> hs;
	static thread_local vector<simd_float> ents;
	static thread_local vector<simd_float> vxs;
	static thread_local vector<simd_float> vys;
	static thread_local vector<simd_float> vzs;
	static thread_local vector<simd_float> fvels;
	static thread_local vector<simd_float> masks;
	const auto rung2dt = [](simd_int rung) {
		simd_float dt;
		for( int k = 0; k < SIMD_FLOAT_SIZE; k++) {
			dt[k] = rung_dt[rung[k]];
		}
		return dt;
	};
	const simd_float m = get_options().sph_mass;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		if (sph_particles_semi_active(i)) {
			sph_particles_dvel(XDIM, i) = 0.0f;
			sph_particles_dvel(YDIM, i) = 0.0f;
			sph_particles_dvel(ZDIM, i) = 0.0f;
			sph_particles_dent(i) = 0.0f;
			xs.resize(0);
			ys.resize(0);
			zs.resize(0);
			hs.resize(0);
			rungs.resize(0);
			ents.resize(0);
			fvels.resize(0);
			vxs.resize(0);
			vys.resize(0);
			vzs.resize(0);
			masks.resize(0);
			const simd_float myh = sph_particles_smooth_len(i);
			const simd_int myrung = sph_particles_rung(i);
			const simd_float myh2 = sqr(myh);
			const simd_float myhinv = simd_float(1.0) / myh;
			const simd_float myh3inv = myhinv * sqr(myhinv);
			const simd_float myrho = sph_den(myh3inv);
			const simd_float myrhoinv = simd_float(1.0) / myrho;
			const simd_int myx = sph_particles_pos(XDIM, i).raw();
			const simd_int myy = sph_particles_pos(YDIM, i).raw();
			const simd_int myz = sph_particles_pos(ZDIM, i).raw();
			const simd_float myent = sph_particles_ent(i);
			const simd_float myvx = sph_particles_vel(XDIM, i);
			const simd_float myvy = sph_particles_vel(YDIM, i);
			const simd_float myvz = sph_particles_vel(ZDIM, i);
			const simd_float myfvel = sph_particles_fvel(i);
			const simd_float myp = pow(myrho, simd_float(SPH_GAMMA)) * myent;
			const simd_float myc = sqrt(simd_float(SPH_GAMMA) * myp * myrhoinv);
			float max_c = 0.0f;
			int base = -1;
			int offset = SIMD_FLOAT_SIZE;
			static const simd_float _2float(fixed2float);

			for (int j = 0; j < main_xs.size(); j += SIMD_FLOAT_SIZE) {
				simd_int x, y, z;
				simd_float mask, h;
				const int maxj = std::min((int) main_xs.size(), j + SIMD_FLOAT_SIZE);
				for (int k = j; k < j + SIMD_FLOAT_SIZE; k++) {
					const int kmj = k - j;
					if (k < main_xs.size()) {
						x[kmj] = main_xs[j].raw();
						y[kmj] = main_ys[j].raw();
						z[kmj] = main_zs[j].raw();
						h[kmj] = main_hs[j];
						mask[kmj] = 1.0f;
					} else {
						h[kmj] = 1.0f;
						x[kmj] = y[kmj] = z[kmj] = mask[kmj] = 0.0f;
					}
				}
				const simd_float dx = simd_float(myx - x) * _2float;
				const simd_float dy = simd_float(myy - y) * _2float;
				const simd_float dz = simd_float(myz - z) * _2float;
				const simd_float h2 = sqr(h);
				const simd_float r2 = sqrt(sqr(dx, dy, dz));
				mask *= simd_float(r2 > 0.0) * (simd_float(r2 < h2) + simd_float(r2 < myh));
				for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
					if (mask[k] > 0.0) {
						if (offset == SIMD_FLOAT_SIZE) {
							xs.push_back(myx);
							ys.push_back(myy);
							zs.push_back(myz);
							rungs.push_back(simd_int(0));
							hs.push_back(simd_float(1.0));
							ents.push_back(simd_float(1.0));
							vxs.push_back(simd_float(0.0));
							vys.push_back(simd_float(0.0));
							vzs.push_back(simd_float(0.0));
							fvels.push_back(simd_float(0.0));
							masks.push_back(simd_float(0.0));
							base++;
							offset = 0;
						}
						const int jpk = j + k;
						xs.back()[offset] = main_xs[jpk].raw();
						ys.back()[offset] = main_ys[jpk].raw();
						zs.back()[offset] = main_zs[jpk].raw();
						rungs.back()[offset] = main_rungs[jpk];
						hs.back()[offset] = main_hs[jpk];
						ents.back()[offset] = main_ents[jpk];
						vxs.back()[offset] = main_vxs[jpk];
						vys.back()[offset] = main_vys[jpk];
						vzs.back()[offset] = main_vzs[jpk];
						fvels.back()[offset] = main_fvels[jpk];
						masks.back()[offset] = 1.0f;
						offset++;
					}
				}
			}
			for (int j = 0; j < xs.size(); j++) {
				const simd_float dx = simd_float(myx - xs[j]) * _2float;
				const simd_float dy = simd_float(myy - ys[j]) * _2float;
				const simd_float dz = simd_float(myz - zs[j]) * _2float;
				const simd_float h = hs[j];
				const simd_float h2 = sqr(h);
				const simd_float r2 = sqr(dx, dy, dz);
				static const simd_float alpha = 0.75;
				static const simd_float beta = 2.0f * alpha;
				static const simd_float one = simd_float(1.f);
				static const simd_float zero = simd_float(0.f);
				static const simd_float tiny = simd_float(1e-15);
				static const simd_float gamma = SPH_GAMMA;
				const simd_float hinv = one / h;
				const simd_float h3inv = hinv * sqr(hinv);
				const simd_float rho = sph_den(h3inv);
				const simd_float rhoinv = one / rho;
				const simd_float p = ents[j] * pow(rho, gamma);
				const simd_float c = sqrt(gamma * p * rhoinv);
				const simd_float cij = 0.5f * (myc + c);
				const simd_float hij = 0.5f * (h + myh);
				const simd_float rho_ij = 0.5f * (rho + myrho);
				const simd_float dvx = myvx - vxs[j];
				const simd_float dvy = myvy - vys[j];
				const simd_float dvz = myvz - vzs[j];
				const simd_float r = sqrt(r2);
				const simd_float rinv = one / (r + tiny);
				const simd_float r2inv = sqr(rinv);
				const simd_float uij = min(zero, hij * (dvx * dx + dvy * dy + dvz * dz) * r2inv);
				const simd_float Piij = (-alpha * uij * cij + beta * sqr(uij)) * rhoinv;
				const simd_float dWdri = (r < myh) * sph_dWdr_rinv(r, myhinv, myh3inv);
				const simd_float dWdrj = (r < h) * sph_dWdr_rinv(r, hinv, h3inv);
				const simd_float dWdri_x = dx * dWdri;
				const simd_float dWdri_y = dy * dWdri;
				const simd_float dWdri_z = dz * dWdri;
				const simd_float dWdrj_x = dx * dWdrj;
				const simd_float dWdrj_y = dy * dWdrj;
				const simd_float dWdrj_z = dz * dWdrj;
				const simd_float dWdrij_x = 0.5f * (dWdri_x + dWdrj_x);
				const simd_float dWdrij_y = 0.5f * (dWdri_y + dWdrj_y);
				const simd_float dWdrij_z = 0.5f * (dWdri_z + dWdrj_z);
				const simd_float Prho2i = myp * myrhoinv * myrhoinv;
				const simd_float Prho2j = p * rhoinv * rhoinv;
				const simd_float dviscx = Piij * dWdrij_x;
				const simd_float dviscy = Piij * dWdrij_y;
				const simd_float dviscz = Piij * dWdrij_z;
				const simd_float dpx = (Prho2j * dWdrj_x + Prho2i * dWdri_x) + dviscx;
				const simd_float dpy = (Prho2j * dWdrj_y + Prho2i * dWdri_y) + dviscy;
				const simd_float dpz = (Prho2j * dWdrj_z + Prho2i * dWdri_z) + dviscz;
				const simd_float dvxdt = -dpx * m;
				const simd_float dvydt = -dpy * m;
				const simd_float dvzdt = -dpz * m;
				const simd_float dt = min(rung2dt(rungs[j]), rung2dt(myrung)) * simd_float(t0);
				simd_float dAdt = (dviscx * dvx + dviscy * dvy + dviscz * dvz);
				dAdt *= simd_float(0.5) * m * (SPH_GAMMA - 1.f) * pow(myrho, 1.0f - SPH_GAMMA);
				sph_particles_dvel(XDIM, i) += (dvxdt * dt * masks[j]).sum();
				sph_particles_dvel(YDIM, i) += (dvydt * dt * masks[j]).sum();
				sph_particles_dvel(ZDIM, i) += (dvzdt * dt * masks[j]).sum();
				sph_particles_dent(i) += (dAdt * dt * masks[j]).sum();
			}
		}
	}
	return rc;
}

sph_run_return sph_update(const sph_tree_node* self_ptr) {
	sph_run_return rc;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		if (sph_particles_semi_active(i)) {
			for (int dim = 0; dim < NDIM; dim++) {
				sph_particles_vel(dim, i) += sph_particles_dvel(dim, i);
				sph_particles_dvel(dim, i) = 0.0f;
			}
			sph_particles_ent(i) += sph_particles_dent(i);
			sph_particles_dent(i) = 0.0f;
			sph_particles_semi_active(i) = false;
		}

	}
	return rc;
}

sph_run_return sph_run(sph_run_params params) {
	sph_run_return rc;
	vector<hpx::future<sph_run_return>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_run_action>(c, params));
	}
	int nthreads = hpx_hardware_concurrency();
	if (hpx_size() > 1) {
		nthreads *= 4;
	}
	static std::atomic<int> next;
	next = 0;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,params]() {
			sph_run_return rc;
			int index = next++;
			sph_data_vecs data;
			while( index < sph_tree_leaflist_size()) {
				data.clear();
				const auto* self = sph_tree_get_leaf(index);
				bool test;
				switch(params.run_type) {

					case SPH_RUN_SMOOTHLEN:
					test = (params.set & SPH_SET_ACTIVE) && (self->nactive > 0);
					if( !test && (params.set & SPH_SET_SEMIACTIVE) ) {
						test = has_active_neighbors(self);
					}
					break;

					case SPH_RUN_FVELS:
					case SPH_RUN_HYDRO:
					case SPH_RUN_MARK_SEMIACTIVE:
					test = (self->nactive > 0);
					if( !test ) {
						test = has_active_neighbors(self);
					}
					break;

					case SPH_RUN_COURANT:
					case SPH_RUN_GRAVITY:
					test = self->nactive > 0;
					case SPH_RUN_UPDATE:
					test = test || has_active_neighbors(self);
					break;
				}
				if(test) {
					vector<tree_id> neighbors;
					switch(params.run_type) {

						case SPH_RUN_SMOOTHLEN:
						case SPH_RUN_MARK_SEMIACTIVE:
						case SPH_RUN_COURANT:
						case SPH_RUN_FVELS:
						case SPH_RUN_HYDRO:
						for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
							const auto id = sph_tree_get_neighbor(i);
							neighbors.push_back(id);
						}
						break;

						case SPH_RUN_GRAVITY:
						case SPH_RUN_UPDATE:
						break;

					}
					//		PRINT( "neighbors_size = %i\n",neighbors.size());
				switch(params.run_type) {
					case SPH_RUN_SMOOTHLEN:
					load_data<false, false, false, false, false, true, false, false>(self, neighbors, data, params.min_rung);
					break;
					case SPH_RUN_MARK_SEMIACTIVE:
					load_data<true, true, false, false, false, false, true, true>(self, neighbors, data, params.min_rung);
					break;
					case SPH_RUN_COURANT:
					load_data<false, true, true, true, false, true, false, false>(self, neighbors, data, params.min_rung);
					break;
					case SPH_RUN_FVELS:
					load_data<false, true, false, true, false, true, false, false>(self, neighbors, data, params.min_rung);
					break;
					case SPH_RUN_HYDRO:
					load_data<true, true, true, true, true, true, false, false>(self, neighbors, data, params.min_rung);
					break;
					case SPH_RUN_GRAVITY:
					case SPH_RUN_UPDATE:
					break;
				}
				sph_run_return this_rc;
				switch(params.run_type) {
					case SPH_RUN_SMOOTHLEN:
					if( neighbors.size()==0) {
//						PRINT( "%i\n", neighbors.size());
					}

					this_rc = sph_smoothlens(self,data.xs, data.ys, data.zs, params.min_rung, params.set & SPH_SET_ACTIVE, params.set & SPH_SET_SEMIACTIVE, self->nactive, neighbors.size());
					break;
					case SPH_RUN_MARK_SEMIACTIVE:
					this_rc = sph_mark_semiactive(self,data.xs, data.ys, data.zs, data.rungs, data.hs, params.min_rung);
					break;
					case SPH_RUN_COURANT:
					this_rc = sph_courant(self,data.xs, data.ys, data.zs, data.hs,data.ents,data.vxs,data.vys,data.vzs, params.min_rung, params.a, params.t0);
					break;
					case SPH_RUN_GRAVITY:
					this_rc = sph_gravity(self, params.min_rung, params.t0);
					break;
					case SPH_RUN_FVELS:
					this_rc = sph_fvels(self, data.xs, data.ys, data.zs, data.hs, data.vxs, data.vys, data.vzs);
					break;
					case SPH_RUN_HYDRO:
					this_rc = sph_hydro(self, data.xs, data.ys, data.zs, data.rungs, data.hs, data.ents, data.vxs, data.vys, data.vzs, data.fvels, params.t0);
					break;
					case SPH_RUN_UPDATE:
					this_rc = sph_update(self);
					break;
				}
				rc += this_rc;
			}
			index = next++;
		}
		return rc;
	}));
	}
	for (auto& f : futs) {
		rc += f.get();
	}
	return rc;
}
