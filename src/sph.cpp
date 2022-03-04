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
#include <cosmictiger/sph_cuda.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/constants.hpp>

#include <fenv.h>
#include <unistd.h>
#include <stack>

HPX_PLAIN_ACTION (sph_tree_neighbor);

#define MAX_ACTIVE_WORKSPACES 1

static float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

struct sph_tree_id_hash {
	inline size_t operator()(tree_id id) const {
		const int i = id.index;
		return i * hpx_size() + id.proc;
	}
};

inline bool range_intersect(const fixed32_range& a, const fixed32_range& b) {
	return a.periodic_intersection(b).volume() != range_fixed(0.0);
}

struct sph_run_workspace {
	sph_run_params params;
	vector<fixed32, pinned_allocator<fixed32>> host_x;
	vector<fixed32, pinned_allocator<fixed32>> host_y;
	vector<fixed32, pinned_allocator<fixed32>> host_z;
	vector<float, pinned_allocator<float>> host_gx;
	vector<float, pinned_allocator<float>> host_gy;
	vector<float, pinned_allocator<float>> host_gz;
	vector<float, pinned_allocator<float>> host_vx;
	vector<float, pinned_allocator<float>> host_vy;
	vector<float, pinned_allocator<float>> host_vz;
	vector<float, pinned_allocator<float>> host_eint;
	vector<float, pinned_allocator<float>> host_T;
	vector<float, pinned_allocator<float>> host_colog;
	vector<float, pinned_allocator<float>> host_kappa;
	vector<float, pinned_allocator<float>> host_lambda_e;
	vector<float, pinned_allocator<float>> host_mmw;
	vector<float, pinned_allocator<float>> host_alpha;
	vector<float, pinned_allocator<float>> host_difco;
	vector<char, pinned_allocator<char>> host_oldrung;
	vector<dif_vector, pinned_allocator<dif_vector>> host_difvec;
	vector<float, pinned_allocator<float>> host_fvel;
	vector<float, pinned_allocator<float>> host_f0;
	vector<float, pinned_allocator<float>> host_fpot;
	vector<float, pinned_allocator<float>> host_h;
	vector<float, pinned_allocator<float>> host_gamma;
	vector<char, pinned_allocator<char>> host_rungs;
	vector<sph_tree_node, pinned_allocator<sph_tree_node>> host_trees;
	vector<int, pinned_allocator<int>> host_neighbors;
	vector<int> host_selflist;
	std::unordered_map<tree_id, int, sph_tree_id_hash> tree_map;
	std::unordered_map<int, pair<int>> neighbor_ranges;
	mutex_type mutex;
	void add_work(tree_id selfid);
	sph_run_return to_gpu();
	sph_run_workspace(sph_run_params p) {
		params = p;
	}
};

vector<sph_values> sph_values_at(vector<double> x, vector<double> y, vector<double> z) {
	int max_rung = 0;
	sph_tree_create_params tparams;
	PRINT("Doing values at\n");
	tparams.h_wt = 2.0;
	tparams.min_rung = 0;
	tree_id root_id;
	root_id.proc = 0;
	root_id.index = 0;
	sph_tree_create_return sr;
	vector<tree_id> checklist;
	checklist.push_back(root_id);
	sph_tree_neighbor_params tnparams;

	tnparams.h_wt = 2.0;
	tnparams.min_rung = 0;
	tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;

	sph_run_params sparams;
	sparams.a = 1.0f;
	sparams.t0 = 0.0;
	sparams.min_rung = 0;
	bool cont;
	sph_run_return kr;
	sparams.set = SPH_SET_ACTIVE;
	sparams.phase = 0;
	timer tm;
	tm.start();
	PRINT("starting sph_tree_create = %e\n", tm.read());
	sr = sph_tree_create(tparams);
	tm.stop();
	tm.reset();
	PRINT("sph_tree_create time = %e %i\n", tm.read(), sr.nactive);

	tm.start();
	sph_tree_neighbor(tnparams, root_id, checklist).get();
	tm.stop();
	PRINT("sph_tree_neighbor(SPH_TREE_NEIGHBOR_NEIGHBORS): %e\n", tm.read());
	tm.reset();

//	do {
	sparams.set = SPH_SET_ACTIVE;
	sparams.run_type = SPH_RUN_SMOOTHLEN;
//		timer tm;
	tm.start();
//		kr = sph_run(sparams);
	tm.stop();
	//	PRINT("sph_run(SPH_RUN_SMOOTHLEN (active)): tm = %e min_h = %e max_h = %e\n", tm.read(), kr.hmin, kr.hmax);
	tm.reset();
	cont = kr.rc;
	tnparams.h_wt = cont ? 2.0 : 1.0;
	tnparams.run_type = SPH_TREE_NEIGHBOR_BOXES;
	tnparams.set = cont ? SPH_SET_ACTIVE : SPH_SET_ALL;
	tm.start();
	sph_tree_neighbor(tnparams, root_id, vector<tree_id>()).get();
	tm.stop();
	PRINT("sph_tree_neighbor(SPH_TREE_NEIGHBOR_BOXES): %e\n", tm.read());
	tm.reset();
	tm.start();
	tnparams.run_type = SPH_TREE_NEIGHBOR_NEIGHBORS;
	sph_tree_neighbor(tnparams, root_id, checklist).get();
	tm.stop();
	PRINT("sph_tree_neighbor(SPH_TREE_NEIGHBOR_NEIGHBORS): %e\n", tm.read());
	tm.reset();
//	} while (cont);
	tnparams.run_type = SPH_TREE_NEIGHBOR_VALUE_AT;

	vector<sph_values> values(x.size());
	for (int i = 0; i < x.size(); i++) {
		tnparams.x = x[i];
		tnparams.y = y[i];
		tnparams.z = z[i];
		auto rc = sph_tree_neighbor(tnparams, root_id, checklist).get();
		values[i] = rc.value_at;
	}
	sph_tree_destroy(true);
	sph_particles_cache_free();
	return values;
}

inline bool range_contains(const fixed32_range& a, const array<fixed32, NDIM> x) {
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

hpx::future<sph_tree_neighbor_return> sph_tree_neighbor_fork(sph_tree_neighbor_params params, tree_id self, vector<tree_id> checklist, int level,
		bool threadme) {
	static std::atomic<int> nthreads(0);
	hpx::future<sph_tree_neighbor_return> rc;
	const sph_tree_node* self_ptr = sph_tree_get_node(self);
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
	vector<float> f0s;
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
		f0s.clear();
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
			d.f0s.resize(new_sz);
		}
		sph_particles_global_read_pos(other->global_part_range(), d.xs.data(), d.ys.data(), d.zs.data(), offset);
		if (do_rungs || do_smoothlens) {
			sph_particles_global_read_rungs_and_smoothlens(other->global_part_range(), d.rungs.data(), d.hs.data(), offset);
		}
		if (do_ent || do_vel) {
			//		sph_particles_global_read_sph(other->global_part_range(), params.a, d.ents.data(), d.vxs.data(), d.vys.data(), d.vzs.data(), nullptr, offset);
		}
		if (do_fvel) {
//			sph_particles_global_read_fvels(other->global_part_range(), d.fvels.data(), d.f0s.data(), offset);
		}
		int i = offset;
		while (i < d.xs.size()) {
			array<fixed32, NDIM> X;
			X[XDIM] = d.xs[i];
			X[YDIM] = d.ys[i];
			X[ZDIM] = d.zs[i];
			bool test0 = (active_only && d.rungs[i] < min_rung);
			bool test1 = false;
			bool test2 = true;
			if (!test0) {
				test1 = check_inner && !range_contains(self_ptr->outer_box, X);
				if (check_outer && test1) {
					assert(do_rungs);
					test2 = true;
					const auto& box = self_ptr->inner_box;
					for (int dim = 0; dim < NDIM; dim++) {
						if (distance(box.begin[dim], X[dim]) + d.hs[i] > 0.0 || distance(X[dim], box.end[dim]) + d.hs[i] > 0.0) {
						} else {
							test2 = false;
							break;
						}
					}
				}
			}
			if (test0 || (test1 && test2)) {
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
					d.f0s[i] = d.f0s.back();
					d.fvels.pop_back();
					d.f0s.pop_back();
				}
			} else {
				i++;
			}
		}
	}
}

hpx::future<sph_tree_neighbor_return> sph_tree_neighbor(sph_tree_neighbor_params params, tree_id self, vector<tree_id> checklist, int level) {
///	PRINT( "%i %i\n", level, checklist.size());
	sph_tree_neighbor_return kr;
	if (params.run_type == SPH_TREE_NEIGHBOR_BOXES) {
//		return hpx::make_ready_future(kr);
	}
	stack_trace_activate();
	const sph_tree_node* self_ptr = sph_tree_get_node(self);
	if (params.run_type == SPH_TREE_NEIGHBOR_VALUE_AT) {
		array<fixed32, NDIM> x;
		x[XDIM] = params.x;
		x[YDIM] = params.y;
		x[ZDIM] = params.z;
		const bool test = self_ptr->box.contains(x);
		//PRINT("%i\n", test);
		//PRINT("%e %e %e\n", x[XDIM].to_float(), self_ptr->box.begin[0].to_float(), self_ptr->box.end[0].to_float());
		//PRINT("%e %e %e\n", x[YDIM].to_float(), self_ptr->box.begin[1].to_float(), self_ptr->box.end[1].to_float());
		//PRINT("%e %e %e\n", x[ZDIM].to_float(), self_ptr->box.begin[2].to_float(), self_ptr->box.end[2].to_float());
		if (!test && level > 3) {
			return hpx::make_ready_future(kr);
		}
	}
	if (params.run_type == SPH_TREE_NEIGHBOR_NEIGHBORS && self_ptr->local_root) {
		sph_tree_free_neighbor_list();
		sph_tree_clear_neighbor_ranges();
	}
	if (params.run_type == SPH_TREE_NEIGHBOR_NEIGHBORS && checklist.size() == 0) {
		return hpx::make_ready_future(kr);
	}
	ASSERT(self.proc == hpx_rank());
	bool thread_left = params.run_type != SPH_TREE_NEIGHBOR_VALUE_AT;
	vector<tree_id> nextlist;
	vector<tree_id> leaflist;
	fixed32_range box;

	if (params.run_type == SPH_TREE_NEIGHBOR_NEIGHBORS || params.run_type == SPH_TREE_NEIGHBOR_VALUE_AT) {
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
				const bool test2 = range_intersect(params.run_type == SPH_TREE_NEIGHBOR_VALUE_AT ? self_ptr->box : self_ptr->inner_box, other->outer_box);
				//				const bool test2 = range_intersect(params.run_type == SPH_TREE_NEIGHBOR_VALUE_AT ? self_ptr->box : self_ptr->inner_box, other->outer_box);
				const bool test3 = level <= 9;
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
			//		PRINT( "%i %i\n", level, leaflist.size());
			sph_tree_set_neighbor_range(self, rng);
		}
			break;
		case SPH_TREE_NEIGHBOR_BOXES: {
			fixed32_range ibox, obox;
			float maxh = 0.f;
			for (int dim = 0; dim < NDIM; dim++) {
				ibox.begin[dim] = obox.begin[dim] = 1.9;
				ibox.end[dim] = obox.end[dim] = -.9;
			}
			for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
				const bool active = sph_particles_rung(i) >= params.min_rung;
				const bool semiactive = !active && sph_particles_semi_active(i);
				const float h = params.h_wt * sph_particles_smooth_len(i);
//				PRINT( "%e\n", params.h_wt);
				const auto myx = sph_particles_pos(XDIM, i);
				const auto myy = sph_particles_pos(YDIM, i);
				const auto myz = sph_particles_pos(ZDIM, i);
				const int k = i - self_ptr->part_range.first;
				maxh = std::max(maxh, h);
				array<fixed32, NDIM> X;
				X[XDIM] = myx;
				X[YDIM] = myy;
				X[ZDIM] = myz;
				//			if ((params.set & SPH_SET_ALL) || (active && (params.set & SPH_SET_ACTIVE)) || (semiactive && (params.set & SPH_SET_SEMIACTIVE))) {
				for (int dim = 0; dim < NDIM; dim++) {
					const double x = X[dim].to_double();
					obox.begin[dim] = std::min(obox.begin[dim].to_double(), x - h);
					obox.end[dim] = std::max(obox.end[dim].to_double(), x + h);
				}
				//			}
				for (int dim = 0; dim < NDIM; dim++) {
					const double x = X[dim].to_double();
					ibox.begin[dim] = std::min(ibox.begin[dim].to_double(), x);
					ibox.end[dim] = std::max(ibox.end[dim].to_double(), x);
				}
			}
			kr.inner_box = ibox;
			kr.outer_box = obox;
			sph_tree_set_boxes(self, kr.inner_box, kr.outer_box);
		}
			break;

		case SPH_TREE_NEIGHBOR_VALUE_AT: {
			float h = 0.f;
			float r2min = std::numeric_limits<float>::max();
			sph_data_vecs dat;
			load_data<true, true, true, true, true, true, true, false>(self_ptr, leaflist, dat, params.min_rung);
			sph_values values;
			values.vx = 0.0;
			values.vy = 0.0;
			values.vz = 0.0;
			values.rho = 0.0;
			values.p = 0.0;
			for (int i = 0; i < dat.xs.size(); i++) {
				const auto myx = dat.xs[i];
				const auto myy = dat.ys[i];
				const auto myz = dat.zs[i];
				const float dx = distance(fixed32(params.x), myx);
				const float dy = distance(fixed32(params.y), myy);
				const float dz = distance(fixed32(params.z), myz);
				const float r2 = sqr(dx, dy, dz);
				if (r2 < r2min) {
					r2min = r2;
					h = dat.hs[i];
				}
			}
			const float m = get_options().sph_mass;
			float one = 0.f;
			for (int i = 0; i < dat.xs.size(); i++) {
				const auto x = dat.xs[i];
				const auto y = dat.ys[i];
				const auto z = dat.zs[i];
				const float dx = distance(fixed32(params.x), x);
				const float dy = distance(fixed32(params.y), y);
				const float dz = distance(fixed32(params.z), z);
				const float r2 = sqr(dx, dy, dz);
				if (r2 < h * h) {
					const float r = sqrt(r2);
					const float hinv = 1.0f / h;
					const float h3inv = sqr(hinv) * hinv;
					const float rho = sph_den(h3inv);
					const float rhoinv = 1.0 / rho;
					const float q = r * hinv;
					const float w = h3inv * kernelW(q) * rhoinv * m;
					const float p = dat.ents[i] * pow(rho, get_options().gamma);
					one += w;
					values.vx += w * dat.vxs[i];
					values.vy += w * dat.vys[i];
					values.vz += w * dat.vzs[i];
					values.rho += rho * w;
					values.p += p * w;
					//		PRINT( "%e %e %e \n", dat.ents[i], rho, p);
				}
			}
			//	PRINT( "%e\n", one );
			if (one > 0.0) {
				const float oneinv = 1.f / one;
				values.vx *= oneinv;
				values.vy *= oneinv;
				values.vz *= oneinv;
				values.rho *= oneinv;
				values.p *= oneinv;
			}
			kr.has_value_at = true;
//				PRINT("%e %e\n", params.x, values.vx);
			kr.value_at = values;
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
				simd_float count(simd_float(0.0));
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
					count += mask;
					static const simd_float _2float(fixed2float);
					const simd_float dx = simd_float(myx - X) * _2float;
					const simd_float dy = simd_float(myy - Y) * _2float;
					const simd_float dz = simd_float(myz - Z) * _2float;
					const simd_float r2 = sqr(dx, dy, dz);
					mask *= (r2 < h2);
					const simd_float r = sqrt(r2);
					const simd_float q = r * hinv;
					const simd_float w = kernelW(q);
					const simd_float dwdh = -q * hinv * dkernelW_dq(q);
					f_simd += w * mask;
					dfdh_simd += dwdh * mask;
				}
				float dh = 0.1 * h;
				if (count.sum() > 1.0) {
					f = f_simd.sum();
					dfdh = dfdh_simd.sum();
					f -= get_options().neighbor_number * 3.0 / (4.0 * M_PI);
					/*	for( int l = 0; l < SIMD_FLOAT_SIZE; l++) {
					 if( abs(dfdh[l]) < 1e-20) {
					 PRINT( "!!!!!!!!!!!! %e\n", f[l]);
					 }
					 }*/
					dh = -f / dfdh;
					if (dh > 0.5 * h) {
						dh = 0.5 * h;
					} else if (dh < -0.5 * h) {
						dh = -0.5 * h;
					}
					error = fabs(log(h + dh) - log(h));
					h += dh;
				} else {
					h *= 1.1;
					error = 1.0;
				}
				//if( cnt > 5 )
				//	PRINT("%i %e %e %e\n", cnt, h - dh, dh, f);
				array<fixed32, NDIM> X;
				X[XDIM] = sph_particles_pos(XDIM, i);
				X[YDIM] = sph_particles_pos(YDIM, i);
				X[ZDIM] = sph_particles_pos(ZDIM, i);
				for (int dim = 0; dim < NDIM; dim++) {
					if (distance(self_ptr->outer_box.end[dim], X[dim]) - h < 0.0) {
						box_xceeded = true;
						break;
					}
					if (distance(X[dim], self_ptr->outer_box.begin[dim]) - h < 0.0) {
						box_xceeded = true;
						break;
					}
				}
				cnt++;
				if (cnt > 100) {
//					PRINT("density solver failed to converge %e %e %i %i %i %i\n", h, dh, nactive, self_ptr->outer_box.valid, self_ptr->inner_box.valid, nneighbor);
					abort();
				}
				//			PRINT( "%e\n", h);
			} while (error > SPH_SMOOTHLEN_TOLER);
			max_cnt = std::max(max_cnt, cnt);
			max_h = std::max(max_h, h);
			min_h = std::min(min_h, h);
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
			const float myh = sph_particles_smooth_len(i);
			const simd_float myh2 = sqr(myh);
			sph_particles_semi_active(i) = false;
			for (int j = 0; j < xs.size(); j += SIMD_FLOAT_SIZE) {
				simd_int x, y, z;
				simd_float mask;
				for (int k = j; k < j + SIMD_FLOAT_SIZE; k++) {
					const int kmj = k - j;
					if (k < xs.size()) {
						x[kmj] = xs[k].raw();
						y[kmj] = ys[k].raw();
						z[kmj] = zs[k].raw();
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
				mask *= ((((r2 < h2) + (r2 < myh2)) * (r2 > 0.0)) > 0.0);
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
	/*const simd_float m = get_options().sph_mass;
	 const simd_float minv = simd_float(1.0f) / m;
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
	 const simd_float myp = pow(myrho, simd_float(get_options().gamma)) * myent;
	 const simd_float myc = sqrt(simd_float(get_options().gamma) * myp * myrhoinv);
	 int base = -1;
	 int offset = SIMD_FLOAT_SIZE;
	 static const simd_float _2float(fixed2float);

	 int count = 0;
	 for (int j = 0; j < main_xs.size(); j += SIMD_FLOAT_SIZE) {
	 simd_int x, y, z;
	 simd_float mask, h;
	 const int maxj = std::min((int) main_xs.size(), j + SIMD_FLOAT_SIZE);
	 for (int k = j; k < j + SIMD_FLOAT_SIZE; k++) {
	 const int kmj = k - j;
	 if (k < main_xs.size()) {
	 x[kmj] = main_xs[k].raw();
	 y[kmj] = main_ys[k].raw();
	 z[kmj] = main_zs[k].raw();
	 h[kmj] = main_hs[k];
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
	 const simd_float r2 = sqr(dx, dy, dz);
	 mask *= simd_float(r2 < myh2);
	 for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
	 if (mask[k] > 0.0) {
	 count++;
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
	 //			if( xs.size() == 0 ) {
	 //				PRINT( "%i\n", count);
	 //				PRINT( "ZERO %i %i %e %e\n", self_ptr->neighbor_range.second - self_ptr->neighbor_range.first, main_xs.size(), myh[0], myx[0]*_2float[0]);
	 //				abort();
	 //			}
	 static const simd_float tiny = simd_float(1e-15);
	 static const simd_float one(1.0f);
	 static const simd_float zero(0.0f);
	 static const simd_float half(0.5f);
	 simd_float dvx_dx = 0.0f;
	 simd_float dvx_dy = 0.0f;
	 simd_float dvx_dz = 0.0f;
	 simd_float dvy_dx = 0.0f;
	 simd_float dvy_dy = 0.0f;
	 simd_float dvy_dz = 0.0f;
	 simd_float dvz_dx = 0.0f;
	 simd_float dvz_dy = 0.0f;
	 simd_float dvz_dz = 0.0f;
	 simd_float drho_dh = 0.f;
	 simd_float max_vsig(zero);
	 simd_float vsig(0.f);
	 for (int j = 0; j < xs.size(); j++) {
	 const simd_float dx = simd_float(myx - xs[j]) * _2float;
	 const simd_float dy = simd_float(myy - ys[j]) * _2float;
	 const simd_float dz = simd_float(myz - zs[j]) * _2float;
	 const simd_float h = hs[j];
	 const simd_float hinv = one / h;
	 const simd_float hinv3 = hinv * sqr(hinv);
	 const simd_float rho = sph_den(hinv3);
	 const simd_float rhoinv = one / rho;
	 const simd_float dvx = myvz - vxs[j];
	 const simd_float dvy = myvy - vys[j];
	 const simd_float dvz = myvz - vzs[j];
	 const simd_float c = sqrt(simd_float(get_options().gamma) * pow(rho, simd_float(SPH_GAMMA - 1.0f)) * ents[j]);
	 const simd_float r = sqrt(sqr(dx, dy, dz));
	 static const simd_float tiny = simd_float(1e-15);
	 const simd_float rinv = one / (r + tiny);
	 const simd_float dv = (dvx * dx + dvy * dy + dvz * dz) * rinv;
	 const simd_float this_vsig = (c + myc + simd_float(3) * max(-dv, simd_float(0))) * masks[j];
	 max_vsig = max(max_vsig, this_vsig);
	 const simd_float q = r * myhinv;
	 const simd_float dWdr = dkernelW_dq(q) * hinv * hinv3;
	 const simd_float tmp = m * dWdr * rhoinv * masks[j];
	 const simd_float dWdr_x = dx * tmp * rinv;
	 const simd_float dWdr_y = dy * tmp * rinv;
	 const simd_float dWdr_z = dz * tmp * rinv;
	 dvx_dx -= dvx * dWdr_x;
	 dvx_dy -= dvx * dWdr_y;
	 dvx_dz -= dvx * dWdr_z;
	 dvy_dx -= dvy * dWdr_x;
	 dvy_dy -= dvy * dWdr_y;
	 dvy_dz -= dvy * dWdr_z;
	 dvz_dx -= dvz * dWdr_x;
	 dvz_dy -= dvz * dWdr_y;
	 dvz_dz -= dvz * dWdr_z;
	 drho_dh += -q * dkernelW_dq(q) * masks[j];
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
	 const float sw = 1e-4 * myc[0] / myh[0];
	 const float abs_curl_v = sqrt(sqr(curl_vx, curl_vy, curl_vz));
	 const float fvel = abs_div_v / (abs_div_v + abs_curl_v + sw);
	 const float c0 = drho_dh.sum() * 4.0 * M_PI / (9.0 * get_options().neighbor_number);
	 const float pre = 1.0f / (1.0f + c0);
	 sph_particles_fpre(i) = pre;
	 sph_particles_fvel(i) = fvel;
	 const float vsig_max = max_vsig.max();
	 rc.max_vsig = vsig_max;
	 float dthydro = rc.max_vsig / (ascale * myh[0]);
	 if (dthydro > 1.0e-99) {
	 //				dthydro = SPH_CFL / dthydro;
	 } else {
	 dthydro = 1.0e99;
	 }
	 static const float eta = get_options().eta;
	 static const float hgrav = myh[0];
	 const float gx = sph_particles_gforce(XDIM, i);
	 const float gy = sph_particles_gforce(YDIM, i);
	 const float gz = sph_particles_gforce(ZDIM, i);
	 char& rung = sph_particles_rung(i);
	 const float g2 = sqr(gx, gy, gz);
	 const float factor = eta * sqrtf(ascale * hgrav);
	 const float dt_grav = std::min(factor / sqrtf(sqrtf(g2 + tiny[0])), (float) t0);
	 const float dt = std::min(dt_grav, dthydro);
	 const int rung_hydro = ceilf(log2f(t0) - log2f(dthydro));
	 const int rung_grav = ceilf(log2f(t0) - log2f(dt_grav));
	 rung = std::max(std::max((int) std::max(rung_hydro, rung_grav), std::max(min_rung, (int) rung - 1)), 1);
	 //			PRINT( "%i %e %e %e %e\n", rung, dt_grav, gx, gy, gz);
	 if (rung < 0 || rung >= MAX_RUNG) {
	 PRINT("Rung out of range \n");
	 }
	 rc.max_rung_hydro = std::max(rc.max_rung_hydro, (int) rung_hydro);
	 rc.max_rung_grav = std::max(rc.max_rung_grav, (int) rung_grav);
	 rc.max_rung = std::max(rc.max_rung, (int) rung);

	 }
	 }*/
	return rc;
}

sph_run_return sph_hydro(const sph_tree_node* self_ptr, const vector<fixed32>& main_xs, const vector<fixed32>& main_ys, const vector<fixed32>& main_zs,
		const vector<char>& main_rungs, const vector<float>& main_hs, const vector<float>& main_ents, const vector<float>& main_vxs,
		const vector<float>& main_vys, const vector<float>& main_vzs, const vector<float>& main_fvels, const vector<float>& main_f0s, int min_rung, float t0,
		int phase, float a) {
	sph_run_return rc;
	/*	static thread_local vector<simd_int> xs;
	 static thread_local vector<simd_int> ys;
	 static thread_local vector<simd_int> zs;
	 static thread_local vector<simd_int> rungs;
	 static thread_local vector<simd_float> hs;
	 static thread_local vector<simd_float> ents;
	 static thread_local vector<simd_float> vxs;
	 static thread_local vector<simd_float> vys;
	 static thread_local vector<simd_float> vzs;
	 static thread_local vector<simd_float> fvels;
	 static thread_local vector<simd_float> f0s;
	 static thread_local vector<simd_float> masks;
	 const auto rung2dt = [](simd_int rung) {
	 simd_float dt;
	 for( int k = 0; k < SIMD_FLOAT_SIZE; k++) {
	 dt[k] = rung_dt[rung[k]];
	 }
	 return dt;
	 };
	 const simd_float ainv = 1.0f / a;
	 const simd_float m = get_options().sph_mass;
	 const simd_float minv = simd_float(1.f) / m;
	 for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
	 const bool active = sph_particles_rung(i) >= min_rung;
	 const bool semi = !active && sph_particles_semi_active(i);
	 const bool test = phase == 0 ? (semi || active) : active;
	 if (phase == 1 && test) {
	 sph_particles_dvel(XDIM, i) = 0.0f;
	 sph_particles_dvel(YDIM, i) = 0.0f;
	 sph_particles_dvel(ZDIM, i) = 0.0f;
	 sph_particles_dent(i) = 0.0f;
	 }
	 if (test) {
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
	 f0s.resize(0);
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
	 const simd_float myfvel = sph_particles_fvel(i);
	 static const simd_float half = simd_float(0.5f);
	 #ifdef SPH_TOTAL_ENERGY
	 const float myekin = m[0] * sqr(myvx[0], myvy[0], myvz[0]) * half[0];
	 const float myetherm = std::max(sph_particles_ent(i) - myekin, 0.0f);
	 const float myp = float(SPH_GAMMA - 1.0) * minv[0] * myrho[0] * myetherm;
	 #else
	 const float myent = sph_particles_ent(i);
	 const float myp = pow(myrho[0], SPH_GAMMA) * myent;
	 #endif
	 const simd_float myc = sqrt(SPH_GAMMA * myp * myrhoinv[0]);
	 const simd_float myf0 = sph_particles_fpre(i);
	 const simd_float Prho2i = myp * myrhoinv * myrhoinv * myf0;
	 #ifdef SPH_TOTAL_ENERGY
	 const simd_float Pvxrho2i = Prho2i * myvx;
	 const simd_float Pvyrho2i = Prho2i * myvy;
	 const simd_float Pvzrho2i = Prho2i * myvz;
	 #endif
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
	 x[kmj] = main_xs[k].raw();
	 y[kmj] = main_ys[k].raw();
	 z[kmj] = main_zs[k].raw();
	 h[kmj] = main_hs[k];
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
	 const simd_float r2 = sqr(dx, dy, dz);
	 mask *= simd_float(r2 > 0.0) * (simd_float(r2 < h2) + simd_float(r2 < myh2));
	 for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
	 const int jpk = j + k;
	 if (mask[k] > 0.0 && !(semi && main_rungs[jpk] < min_rung)) {
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
	 f0s.push_back(simd_float(1));
	 base++;
	 offset = 0;
	 }
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
	 f0s.back()[offset] = main_f0s[jpk];
	 offset++;
	 }
	 }
	 }
	 simd_float divv(0.f);
	 for (int j = 0; j < xs.size(); j++) {
	 static const simd_float one = simd_float(1.f);
	 static const simd_float zero = simd_float(0.f);
	 static const simd_float tiny = simd_float(1e-15);
	 static const simd_float gamma = SPH_GAMMA;
	 const simd_float dx = simd_float(myx - xs[j]) * _2float;
	 const simd_float dy = simd_float(myy - ys[j]) * _2float;
	 const simd_float dz = simd_float(myz - zs[j]) * _2float;
	 const simd_float h = hs[j];
	 const simd_float r2 = sqr(dx, dy, dz);
	 const simd_float hinv = one / h;
	 const simd_float h3inv = hinv * sqr(hinv);
	 const simd_float rho = sph_den(h3inv);
	 const simd_float rhoinv = one / rho;
	 #ifdef SPH_TOTAL_ENERGY
	 const simd_float ekin = half * m * sqr(vxs[j], vys[j], vzs[j]);
	 const simd_float etherm = max(ents[j] - ekin, zero);
	 const simd_float p = simd_float(SPH_GAMMA - 1.0) * rho * etherm * minv;
	 #else
	 const simd_float p = ents[j] * pow(rho, gamma);
	 #endif
	 const simd_float c = sqrt(gamma * p * rhoinv);
	 const simd_float cij = 0.5f * (myc + c);
	 const simd_float hij = 0.5f * (h + myh);
	 const simd_float rho_ij = 0.5f * (rho + myrho);
	 const simd_float dvx = myvx - vxs[j];
	 const simd_float dvy = myvy - vys[j];
	 const simd_float dvz = myvz - vzs[j];
	 const simd_float r = sqrt(r2);
	 const simd_float r2inv = one / (sqr(r) + 1e-2 * sqr(hij));
	 const simd_float uij = min(zero, hij * (dvx * dx + dvy * dy + dvz * dz) * r2inv);
	 //		PRINT( "%e %e %e %e \n", myvx[0], vxs[j][0], uij[0], c[0]);
	 const simd_float Piij = (uij * (-simd_float(SPH_ALPHA) * cij + simd_float(SPH_BETA) * uij)) * half * (myfvel + fvels[j]) / rho_ij;
	 const simd_float qi = r * myhinv;
	 const simd_float qj = r * hinv;
	 const simd_float rinv = one / (r + tiny);
	 const simd_float dWdri = (r < myh) * dkernelW_dq(qi) * myhinv * myh3inv;
	 const simd_float dWdrj = (r < h) * dkernelW_dq(qj) * hinv * h3inv;
	 const simd_float dWdri_x = dx * dWdri * rinv;
	 const simd_float dWdri_y = dy * dWdri * rinv;
	 const simd_float dWdri_z = dz * dWdri * rinv;
	 const simd_float dWdrj_x = dx * dWdrj * rinv;
	 const simd_float dWdrj_y = dy * dWdrj * rinv;
	 const simd_float dWdrj_z = dz * dWdrj * rinv;
	 const simd_float dWdrij_x = 0.5f * (dWdri_x + dWdrj_x);
	 const simd_float dWdrij_y = 0.5f * (dWdri_y + dWdrj_y);
	 const simd_float dWdrij_z = 0.5f * (dWdri_z + dWdrj_z);
	 const simd_float Prho2j = p * rhoinv * rhoinv * f0s[j];
	 //		PRINT( "%e\n", f0s[j][0]);
	 #ifdef SPH_TOTAL_ENERGY
	 const simd_float Pvxrho2j = Prho2j * vxs[j];
	 const simd_float Pvyrho2j = Prho2j * vys[j];
	 const simd_float Pvzrho2j = Prho2j * vzs[j];
	 #endif
	 const simd_float dviscx = Piij * dWdrij_x;
	 const simd_float dviscy = Piij * dWdrij_y;
	 const simd_float dviscz = Piij * dWdrij_z;
	 const simd_float dpx = (Prho2j * dWdrj_x + Prho2i * dWdri_x) + dviscx;
	 const simd_float dpy = (Prho2j * dWdrj_y + Prho2i * dWdri_y) + dviscy;
	 const simd_float dpz = (Prho2j * dWdrj_z + Prho2i * dWdri_z) + dviscz;
	 const simd_float dvxdt = -dpx * m;
	 const simd_float dvydt = -dpy * m;
	 const simd_float dvzdt = -dpz * m;
	 divv -= myf0 * ainv * m * rhoinv * (dvx * dWdri_x + dvy * dWdri_y + dvz * dWdri_z);
	 simd_float dt;
	 if (phase == 0) {
	 dt = 0.5f * min(rung2dt(rungs[j]), rung2dt(myrung)) * simd_float(t0);
	 } else if (phase == 1) {
	 dt = rung2dt(myrung) * simd_float(t0);
	 }
	 #ifdef SPH_TOTAL_ENERGY
	 simd_float dedt = m * (Pvxrho2j * dWdrj_x + Pvyrho2j * dWdrj_y + Pvzrho2j * dWdrj_z);
	 dedt += -sqr(m) * (Pvxrho2i * dWdri_x + Pvyrho2i * dWdri_y + Pvzrho2i * dWdri_z);
	 sph_particles_dent(i) += (dedt * dt).sum();
	 #else
	 simd_float dAdt = (dviscx * dvx + dviscy * dvy + dviscz * dvz);
	 dAdt *= simd_float(0.5) * m * (SPH_GAMMA - 1.f) * pow(myrho, 1.0f - SPH_GAMMA);
	 //			if( dAdt[0] > 1.0e-4 )
	 //			PRINT( "%e\n", dAdt[0]);
	 sph_particles_dent(i) += (dAdt * dt * masks[j]).sum();
	 #endif
	 sph_particles_dvel(XDIM, i) += (dvxdt * dt * masks[j]).sum();
	 sph_particles_dvel(YDIM, i) += (dvydt * dt * masks[j]).sum();
	 sph_particles_dvel(ZDIM, i) += (dvzdt * dt * masks[j]).sum();
	 sph_particles_divv(i) = divv.sum();
	 }
	 }
	 }*/
	return rc;
}

sph_run_return sph_update(const sph_tree_node* self_ptr, int min_rung, int phase) {
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);

	sph_run_return rc;
	/*if (phase == 0) {
	 rc.vol = rc.momx = rc.momy = rc.momz = rc.etherm = rc.ekin = rc.ent = 0.0;
	 static const float m = get_options().sph_mass;
	 for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
	 if (sph_particles_rung(i) >= min_rung) {
	 constexpr float SNZ = 0.02;
	 constexpr float SNHe = 0.16;
	 const float dchem = sph_particles_dchem(i);
	 if (dchem > 0.f) {
	 const float dZ = SNZ * dchem;
	 const float dHe = SNHe * dchem;
	 const float factor = 1.0f / (1.0f + dZ + dHe);
	 sph_particles_He0(i) = sph_particles_He0(i) * factor;
	 sph_particles_Hep(i) = sph_particles_Hep(i) * factor;
	 sph_particles_Hepp(i) = sph_particles_Hepp(i) * factor;
	 sph_particles_Hp(i) = sph_particles_Hp(i) * factor;
	 sph_particles_Hn(i) = sph_particles_Hn(i) * factor;
	 sph_particles_H2(i) = sph_particles_H2(i) * factor;
	 sph_particles_Z(i) = sph_particles_Z(i) * factor;
	 sph_particles_He0(i) += dHe;
	 sph_particles_Z(i) += dZ;
	 sph_particles_dchem(i) = 0.f;
	 }
	 for (int dim = 0; dim < NDIM; dim++) {
	 sph_particles_vel(dim, i) += sph_particles_dvel(dim, i);
	 sph_particles_dvel(dim, i) = 0.0f;
	 }
	 const auto original = sph_particles_ent(i);
	 sph_particles_ent(i) += sph_particles_dent(i);
	 if (sph_particles_ent(i) < 0.f) {
	 PRINT("Negative entropy %e %e %e %s %i\n", original, sph_particles_ent(i), sph_particles_dent(i), __FILE__, __LINE__);
	 }
	 sph_particles_dent(i) = 0.0f;
	 const float h = sph_particles_smooth_len(i);
	 const float vx = sph_particles_vel(XDIM, i);
	 const float vy = sph_particles_vel(YDIM, i);
	 const float vz = sph_particles_vel(ZDIM, i);
	 const float A = sph_particles_ent(i);
	 if (h <= 0.f) {
	 PRINT("Less than ZERO H! sph.cpp %e\n", h);
	 }
	 const float rho = sph_den(1.0 / (h * h * h));
	 if (std::isnan(A) || std::isnan(rho)) {
	 PRINT("ENTY bad! %e %e\n", A, rho);
	 }
	 const float p = A * pow(rho, SPH_GAMMA);
	 const float vol = (4.0 * h * h * h * M_PI / 3.0) / get_options().neighbor_number;
	 rc.momx += vx * m;
	 rc.momy += vy * m;
	 rc.momz += vz * m;
	 rc.ekin += 0.5f * m * sqr(vx, vy, vz);
	 rc.etherm += p / (SPH_GAMMA - 1.0f) * vol;
	 rc.ent += A;
	 rc.vol += vol;
	 }

	 }
	 } else {
	 for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
	 if (sph_particles_rung(i) >= min_rung) {
	 for (int dim = 0; dim < NDIM; dim++) {
	 sph_particles_vel(dim, i) += 0.5f * sph_particles_dvel(dim, i);
	 sph_particles_dvel(dim, i) *= -.5f;
	 }
	 const auto original = sph_particles_ent(i);
	 if (sph_particles_ent(i) < -0.5f * sph_particles_dent(i)) {
	 PRINT("Negative entropy %e %e %e %s %i\n", original, sph_particles_ent(i), 0.5f * sph_particles_dent(i), __FILE__, __LINE__);
	 }
	 sph_particles_ent(i) += 0.5f * sph_particles_dent(i);
	 sph_particles_dent(i) *= -0.5f;
	 sph_particles_semi_active(i) = false;
	 }

	 }
	 }*/
	return rc;
}
/*
 static sph_run_return sph_rungs(const sph_tree_node* self_ptr, const vector<fixed32>& xs, const vector<fixed32>& ys, const vector<fixed32>& zs,
 const vector<char>& rungs, const vector<float>& hs, int min_rung) {
 sph_run_return rc;
 for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
 auto& rung = sph_particles_rung(i);
 if (rung > min_rung) {
 const simd_int myx = sph_particles_pos(XDIM, i).raw();
 const simd_int myy = sph_particles_pos(YDIM, i).raw();
 const simd_int myz = sph_particles_pos(ZDIM, i).raw();
 const float myh = sph_particles_smooth_len(i);
 const simd_float myh2 = sqr(myh);
 for (int j = 0; j < xs.size(); j += SIMD_FLOAT_SIZE) {
 simd_int x, y, z;
 simd_float mask;
 for (int k = j; k < j + SIMD_FLOAT_SIZE; k++) {
 const int kmj = k - j;
 if (k < xs.size()) {
 x[kmj] = xs[k].raw();
 y[kmj] = ys[k].raw();
 z[kmj] = zs[k].raw();
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
 mask *= ((((r2 < h2) + (r2 < myh2)) * (r2 > 0.0)) > 0.0);
 for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
 if (mask[k]) {
 const int kpj = k + j;
 if (kpj < xs.size()) {
 if (rung < rungs[kpj] - 1) {
 PRINT( "1\n");
 rc.rc = true;
 rung = rungs[kpj] - 1;
 }
 }
 }
 }
 }
 }
 }
 return rc;
 }
 */
sph_run_return sph_gravity(const sph_tree_node* self_ptr, int min_rung, float t0) {
	sph_run_return rc;
	for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
		const int rung = sph_particles_rung(i);
		if (rung >= min_rung) {
			const float dt = rung_dt[rung] * t0 * 0.5;
			for (int dim = 0; dim < NDIM; dim++) {
				sph_particles_vel(dim, i) += dt * sph_particles_gforce(dim, i);
			}
		}
	}
	return rc;
}

sph_run_return sph_run(sph_run_params params, bool cuda) {
	std::string profile_name = "sph_run:" + std::to_string(params.run_type);
	profiler_enter(profile_name.c_str());

	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);
	params.cfl = get_options().cfl;
	if (get_options().cuda == false) {
		cuda = false;
	}
//	cuda = false;
	sph_run_return rc;
	vector<hpx::future<sph_run_return>> futs;
	vector<hpx::future<sph_run_return>> futs2;
	std::shared_ptr<sph_run_workspace> workspace_ptr = std::make_shared < sph_run_workspace > (params);
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_run_action>(c, params, cuda));
	}
	int nthreads = hpx_hardware_concurrency();
	if (hpx_size() > 1) {
		nthreads *= 4;
	}
	static std::atomic<int> next;
	next = 0;
	static std::atomic<int> gpu_work;
	gpu_work = 0;
//	PRINT( "sph_tree_list_size = %i\n", sph_tree_size());
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(
				hpx::async([proc,nthreads,params,cuda,workspace_ptr]() {
					sph_run_return rc;
					int index = next++;
					sph_data_vecs data;
					while( index < sph_tree_leaflist_size()) {
						data.clear();
						const auto selfid = sph_tree_get_leaf(index);
						const auto* self = sph_tree_get_node(selfid);
						bool test;
						switch(params.run_type) {

							case SPH_RUN_DIFFUSION:
							test = (self->nactive > 0);
							if( !test) {
								test = has_active_neighbors(self);
							}
							test = test && !self->converged;
							break;

							case SPH_RUN_SMOOTHLEN:
							test = (params.set & SPH_SET_ACTIVE) && (self->nactive > 0);
							if( !test && params.phase == 1) {
								test = has_active_neighbors(self);
							}
							break;

							case SPH_RUN_MARK_SEMIACTIVE:
							test = (self->nactive > 0);
							if( !test ) {
								test = has_active_neighbors(self);
							}
							break;

							case SPH_RUN_HYDRO:
							case SPH_RUN_RUNGS:
							test = (self->nactive > 0);
							break;

							case SPH_RUN_AUX:
							case SPH_RUN_COURANT:
							case SPH_RUN_UPDATE:
							case SPH_RUN_GRAVITY:
							test = self->nactive > 0;
							break;
						}
						if(test) {
							if( cuda ) {
								gpu_work++;
								workspace_ptr->add_work(selfid);
							} else {

								vector<tree_id> neighbors;
								switch(params.run_type) {

									case SPH_RUN_SMOOTHLEN:
									case SPH_RUN_MARK_SEMIACTIVE:
									case SPH_RUN_COURANT:
									case SPH_RUN_RUNGS:
									case SPH_RUN_HYDRO:
									case SPH_RUN_AUX:
									for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
										const auto id = sph_tree_get_neighbor(i);
										neighbors.push_back(id);
									}
									break;

									case SPH_RUN_UPDATE:
									case SPH_RUN_GRAVITY:
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
							case SPH_RUN_HYDRO:
							load_data<true, true, true, true, true, true, true, false>(self, neighbors, data, params.min_rung);
							break;
							case SPH_RUN_UPDATE:
							case SPH_RUN_GRAVITY:
							break;
						}
						sph_run_return this_rc;
						switch(params.run_type) {
							case SPH_RUN_SMOOTHLEN:
							this_rc = sph_smoothlens(self,data.xs, data.ys, data.zs, params.min_rung, params.set & SPH_SET_ACTIVE, params.set & SPH_SET_SEMIACTIVE, self->nactive, neighbors.size());
							break;
							case SPH_RUN_MARK_SEMIACTIVE:
							this_rc = sph_mark_semiactive(self,data.xs, data.ys, data.zs, data.rungs, data.hs, params.min_rung);
							break;
							case SPH_RUN_COURANT:
							this_rc = sph_courant(self,data.xs, data.ys, data.zs, data.hs,data.ents,data.vxs,data.vys,data.vzs, params.min_rung, params.a, params.t0);
							break;
							case SPH_RUN_HYDRO:
							this_rc = sph_hydro(self, data.xs, data.ys, data.zs, data.rungs, data.hs, data.ents, data.vxs, data.vys, data.vzs, data.fvels, data.f0s, params.min_rung, params.t0, params.phase, params.a);
							break;
							case SPH_RUN_UPDATE:
							this_rc = sph_update(self, params.min_rung, params.phase);
							break;
							case SPH_RUN_GRAVITY:
							this_rc = sph_gravity(self, params.min_rung, params.t0);
							break;
						}
						rc += this_rc;
					}
				}

				index = next++;
			}
			return rc;
		}));
	}
	for (auto& f : futs2) {
		rc += f.get();
	}
	if (cuda && gpu_work) {
		rc += workspace_ptr->to_gpu();
	}
	for (auto& f : futs) {
		rc += f.get();
	}
	profiler_exit();
	return rc;
}

void sph_run_workspace::add_work(tree_id selfid) {
	const auto* self = sph_tree_get_node(selfid);
	std::unique_lock<mutex_type> lock(mutex);
	std::unordered_map<tree_id, int, sph_tree_id_hash>::iterator iter;
	iter = tree_map.find(selfid);
	if (iter == tree_map.end()) {
		int index = host_trees.size();
		host_trees.resize(index + 1);
		tree_map[selfid] = index;
		host_trees[index] = *self;
	}
	for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
		const auto nid = sph_tree_get_neighbor(i);
		iter = tree_map.find(nid);
		if (iter == tree_map.end()) {
			int index = host_trees.size();
			host_trees.resize(index + 1);
			tree_map[nid] = index;
			lock.unlock();
			const auto* node = sph_tree_get_node(nid);
			lock.lock();
			host_trees[index] = *node;
		}
	}
	int neighbor_begin = host_neighbors.size();
	for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
		const auto nid = sph_tree_get_neighbor(i);
		host_neighbors.push_back(tree_map[nid]);
	}
	int neighbor_end = host_neighbors.size();
	const int myindex = tree_map[selfid];
	auto& r = neighbor_ranges[host_selflist.size()];
	host_selflist.push_back(myindex);
	r.first = neighbor_begin;
	r.second = neighbor_end;
}

sph_run_return sph_run_workspace::to_gpu() {
	size_t parts_size = 0;
	const bool chem = get_options().chem;
	const bool stars = get_options().stars;
	for (auto& node : host_trees) {
		parts_size += node.part_range.second - node.part_range.first;
	}
	host_x.resize(parts_size);
	host_y.resize(parts_size);
	host_z.resize(parts_size);
	host_rungs.resize(parts_size);
	switch (params.run_type) {
	case SPH_RUN_DIFFUSION:
		host_kappa.resize(parts_size);
		host_difco.resize(parts_size);
		host_oldrung.resize(parts_size);
		host_difvec.resize(parts_size);
		break;
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_COURANT:
	case SPH_RUN_RUNGS:
	case SPH_RUN_MARK_SEMIACTIVE:
	case SPH_RUN_DIFFUSION:
		host_h.resize(parts_size);
		break;
	}
	if (params.run_type == SPH_RUN_AUX) {
		host_h.resize(parts_size);
		host_gamma.resize(parts_size);
		host_vx.resize(parts_size);
		host_vy.resize(parts_size);
		host_vz.resize(parts_size);
		host_lambda_e.resize(parts_size);
		host_T.resize(parts_size);
		host_colog.resize(parts_size);
		host_mmw.resize(parts_size);
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_COURANT:
		host_eint.resize(parts_size);
		host_vx.resize(parts_size);
		host_vy.resize(parts_size);
		host_vz.resize(parts_size);
	case SPH_RUN_DIFFUSION:
		if (chem) {
			host_gamma.resize(parts_size);
		}
		break;
	}
	if (params.run_type == SPH_RUN_HYDRO) {
		host_gx.resize(parts_size);
		host_gy.resize(parts_size);
		host_gz.resize(parts_size);
	}
	switch (params.run_type) {
	case SPH_RUN_COURANT:
		if (stars) {
			host_gx.resize(parts_size);
			host_gy.resize(parts_size);
			host_gz.resize(parts_size);
		}
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
		host_fpot.resize(parts_size);
	case SPH_RUN_COURANT:
		host_f0.resize(parts_size);
		host_fvel.resize(parts_size);
		host_alpha.resize(parts_size);
		break;
	}
	vector<hpx::future<void>> futs;
	const int nthreads = 8 * hpx_hardware_concurrency();
	std::atomic<int> index(0);
	std::atomic<part_int> part_index(0);
	for (int i = 0; i < host_selflist.size(); i++) {
		host_trees[host_selflist[i]].neighbor_range = neighbor_ranges[i];
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(
				hpx::async(
						[&index,proc,nthreads,this,&part_index, chem, stars]() {
							int this_index = index++;
							while( this_index < host_trees.size()) {
								auto& node = host_trees[this_index];
								const part_int size = node.part_range.second - node.part_range.first;
								const part_int offset = (part_index += size) - size;
								sph_particles_global_read_pos(node.global_part_range(), host_x.data(), host_y.data(), host_z.data(), offset);
								switch (params.run_type) {
									case SPH_RUN_DIFFUSION:
									sph_particles_global_read_difcos(node.global_part_range(), host_difco.data(), host_kappa.data(), host_oldrung.data(), offset);
									sph_particles_global_read_difvecs(node.global_part_range(), host_difvec.data(), offset);
									break;
								}
								switch(params.run_type) {
									case SPH_RUN_SMOOTHLEN:
									sph_particles_global_read_rungs_and_smoothlens(node.global_part_range(), host_rungs.data(), nullptr, offset);
									break;
									case SPH_RUN_COURANT:
									case SPH_RUN_HYDRO:
									case SPH_RUN_RUNGS:
									case SPH_RUN_MARK_SEMIACTIVE:
									case SPH_RUN_DIFFUSION:
									sph_particles_global_read_rungs_and_smoothlens(node.global_part_range(), host_rungs.data(), host_h.data(), offset);
									break;
									case SPH_RUN_AUX:
									sph_particles_global_read_rungs_and_smoothlens(node.global_part_range(),host_rungs.data(), host_h.data(), offset);
									break;
								}
								const bool cond = get_options().conduction;
								switch(params.run_type) {
									case SPH_RUN_DIFFUSION:
									sph_particles_global_read_sph(node.global_part_range(), params.a, nullptr,nullptr,nullptr,nullptr, chem ? host_gamma.data() : nullptr, nullptr, nullptr, nullptr, nullptr,nullptr,offset);
									break;
									case SPH_RUN_HYDRO:
									sph_particles_global_read_sph(node.global_part_range(), params.a, host_eint.data(), host_vx.data(), host_vy.data(), host_vz.data(), chem ? host_gamma.data() : nullptr, nullptr, nullptr, nullptr,nullptr, host_alpha.data(), offset);
									break;
									case SPH_RUN_COURANT:
									sph_particles_global_read_sph(node.global_part_range(), params.a, host_eint.data(), host_vx.data(), host_vy.data(), host_vz.data(), chem ? host_gamma.data() : nullptr, nullptr, nullptr, nullptr, nullptr,host_alpha.data(), offset);
									break;
									case SPH_RUN_AUX:
									sph_particles_global_read_sph(node.global_part_range(), params.a, nullptr, host_vx.data(), host_vy.data(), host_vz.data(),chem ? host_gamma.data() : nullptr, cond ? host_T.data() : nullptr, cond ? host_lambda_e.data() : nullptr, cond ? host_mmw.data() : nullptr,cond ? host_colog.data() : nullptr,nullptr, offset);
									break;
								}
								switch(params.run_type) {
									case SPH_RUN_COURANT:
									if( stars ) {
										sph_particles_global_read_gforce(node.global_part_range(), host_gx.data(), host_gy.data(), host_gz.data(), offset);
										sph_particles_global_read_fvels(node.global_part_range(), host_fvel.data(), host_f0.data(), nullptr, offset);
									}
									break;
								}
								switch(params.run_type) {
									case SPH_RUN_HYDRO:
										sph_particles_global_read_fvels(node.global_part_range(), host_fvel.data(), host_f0.data(), host_fpot.data(), offset);
									sph_particles_global_read_gforce(node.global_part_range(), host_gx.data(), host_gy.data(), host_gz.data(), offset);
									break;
								}
								node.part_range.first = offset;
								node.part_range.second = offset + size;
								this_index = index++;
							}
						}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	if (host_x.size() != part_index) {
		PRINT("%i %i %i\n", host_x.size(), (int ) part_index, parts_size);
		PRINT("BROKEN\n");
		abort();
	}
	sph_run_cuda_data cuda_data;
	CUDA_CHECK(cudaMalloc(&cuda_data.selfs, sizeof(int) * host_selflist.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.x, sizeof(fixed32) * host_x.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.y, sizeof(fixed32) * host_y.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.z, sizeof(fixed32) * host_z.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.rungs, sizeof(char) * host_rungs.size()));
	switch (params.run_type) {
	case SPH_RUN_DIFFUSION:
		CUDA_CHECK(cudaMalloc(&cuda_data.difco, sizeof(float) * host_difco.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.kappa, sizeof(float) * host_kappa.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.oldrung, sizeof(char) * host_oldrung.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.dif_vec, sizeof(dif_vector) * host_difvec.size()));
		break;
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_COURANT:
	case SPH_RUN_DIFFUSION:
	case SPH_RUN_RUNGS:
	case SPH_RUN_MARK_SEMIACTIVE:
		CUDA_CHECK(cudaMalloc(&cuda_data.h, sizeof(float) * host_h.size()));
		break;
	}
	if (params.run_type == SPH_RUN_AUX) {
		CUDA_CHECK(cudaMalloc(&cuda_data.gamma, sizeof(float) * host_gamma.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.h, sizeof(float) * host_h.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vx, sizeof(float) * host_vx.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vy, sizeof(float) * host_vy.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vz, sizeof(float) * host_vz.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.colog, sizeof(float) * host_colog.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.T, sizeof(float) * host_T.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.lambda_e, sizeof(float) * host_lambda_e.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.mmw, sizeof(float) * host_mmw.size()));
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_COURANT:
		CUDA_CHECK(cudaMalloc(&cuda_data.eint, sizeof(float) * host_eint.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vx, sizeof(float) * host_vx.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vy, sizeof(float) * host_vy.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vz, sizeof(float) * host_vz.size()));
	case SPH_RUN_DIFFUSION:
		if (chem) {
			CUDA_CHECK(cudaMalloc(&cuda_data.gamma, sizeof(float) * host_gamma.size()));
		} else {
			cuda_data.gamma = nullptr;
		}
		break;
	}
	if (params.run_type == SPH_RUN_HYDRO) {
		CUDA_CHECK(cudaMalloc(&cuda_data.gx, sizeof(float) * host_gx.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.gy, sizeof(float) * host_gy.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.gz, sizeof(float) * host_gz.size()));
	}
	switch (params.run_type) {
	case SPH_RUN_COURANT:
		if (stars) {
			CUDA_CHECK(cudaMalloc(&cuda_data.gx, sizeof(float) * host_gx.size()));
			CUDA_CHECK(cudaMalloc(&cuda_data.gy, sizeof(float) * host_gy.size()));
			CUDA_CHECK(cudaMalloc(&cuda_data.gz, sizeof(float) * host_gz.size()));
		} else {
			cuda_data.gx = cuda_data.gy = cuda_data.gz = nullptr;
		}
		break;
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
		CUDA_CHECK(cudaMalloc(&cuda_data.fpot, sizeof(float) * host_fpot.size()));
	case SPH_RUN_COURANT:
		CUDA_CHECK(cudaMalloc(&cuda_data.fvel, sizeof(float) * host_fvel.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.f0, sizeof(float) * host_f0.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.alpha, sizeof(float) * host_alpha.size()));
		break;
	}
	CUDA_CHECK(cudaMalloc(&cuda_data.trees, sizeof(sph_tree_node) * host_trees.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.neighbors, sizeof(int) * host_neighbors.size()));
	auto stream = cuda_get_stream();
	switch (params.run_type) {
	case SPH_RUN_DIFFUSION:
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.kappa, host_kappa.data(), sizeof(float) * host_kappa.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.difco, host_difco.data(), sizeof(float) * host_difco.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.oldrung, host_oldrung.data(), sizeof(char) * host_oldrung.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.dif_vec, host_difvec.data(), sizeof(dif_vector) * host_difvec.size(), cudaMemcpyHostToDevice, stream));
		break;
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_COURANT:
	case SPH_RUN_RUNGS:
	case SPH_RUN_MARK_SEMIACTIVE:
	case SPH_RUN_DIFFUSION:
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.h, host_h.data(), sizeof(float) * host_h.size(), cudaMemcpyHostToDevice, stream));
		break;
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_COURANT:
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.eint, host_eint.data(), sizeof(float) * host_eint.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vx, host_vx.data(), sizeof(float) * host_vx.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vy, host_vy.data(), sizeof(float) * host_vy.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vz, host_vz.data(), sizeof(float) * host_vz.size(), cudaMemcpyHostToDevice, stream));
	case SPH_RUN_DIFFUSION:
		if (chem) {
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.gamma, host_gamma.data(), sizeof(float) * host_gamma.size(), cudaMemcpyHostToDevice, stream));
		}
		break;
	}
	if (params.run_type == SPH_RUN_AUX) {
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.gamma, host_gamma.data(), sizeof(float) * host_gamma.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.h, host_h.data(), sizeof(float) * host_h.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vx, host_vx.data(), sizeof(float) * host_vx.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vy, host_vy.data(), sizeof(float) * host_vy.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vz, host_vz.data(), sizeof(float) * host_vz.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.T, host_T.data(), sizeof(float) * host_T.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.colog, host_colog.data(), sizeof(float) * host_colog.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.lambda_e, host_lambda_e.data(), sizeof(float) * host_lambda_e.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.mmw, host_mmw.data(), sizeof(float) * host_mmw.size(), cudaMemcpyHostToDevice, stream));
	}
	if (params.run_type == SPH_RUN_HYDRO) {
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.gx, host_gx.data(), sizeof(float) * host_gx.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.gy, host_gy.data(), sizeof(float) * host_gy.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.gz, host_gz.data(), sizeof(float) * host_gz.size(), cudaMemcpyHostToDevice, stream));
	}
	switch (params.run_type) {
	case SPH_RUN_COURANT:
		if (stars) {
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.gx, host_gx.data(), sizeof(float) * host_gx.size(), cudaMemcpyHostToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.gy, host_gy.data(), sizeof(float) * host_gy.size(), cudaMemcpyHostToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.gz, host_gz.data(), sizeof(float) * host_gz.size(), cudaMemcpyHostToDevice, stream));
		}
		break;
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.fpot, host_fpot.data(), sizeof(float) * host_fvel.size(), cudaMemcpyHostToDevice, stream));
	case SPH_RUN_COURANT:
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.f0, host_f0.data(), sizeof(float) * host_f0.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.fvel, host_fvel.data(), sizeof(float) * host_fvel.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.alpha, host_alpha.data(), sizeof(float) * host_alpha.size(), cudaMemcpyHostToDevice, stream));
		break;
	}
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.x, host_x.data(), sizeof(fixed32) * host_x.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.y, host_y.data(), sizeof(fixed32) * host_y.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.z, host_z.data(), sizeof(fixed32) * host_z.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.rungs, host_rungs.data(), sizeof(char) * host_rungs.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.trees, host_trees.data(), sizeof(sph_tree_node) * host_trees.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.selfs, host_selflist.data(), sizeof(int) * host_selflist.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.neighbors, host_neighbors.data(), sizeof(int) * host_neighbors.size(), cudaMemcpyHostToDevice, stream));
	cuda_data.def_gamma = get_options().gamma;
	cuda_data.nselfs = host_selflist.size();
	cuda_data.h_snk = &sph_particles_smooth_len(0);
	cuda_data.chem = get_options().chem;
	cuda_data.gravity = get_options().gravity;
	cuda_data.conduction = get_options().conduction;
	cuda_data.tcool_snk = &sph_particles_tcool(0);
	cuda_data.dvec_snk = &sph_particles_d_dif_vec(0);
	cuda_data.kappa_snk = &sph_particles_kappa(0);
	cuda_data.deint_pred = &sph_particles_deint_pred(0);
	cuda_data.dalpha_pred = &sph_particles_dalpha_pred(0);
	cuda_data.dvx_pred = &sph_particles_dvel_pred(XDIM, 0);
	cuda_data.dvy_pred = &sph_particles_dvel_pred(YDIM, 0);
	cuda_data.dvz_pred = &sph_particles_dvel_pred(ZDIM, 0);
	cuda_data.deint_con = &sph_particles_deint_con(0);
	cuda_data.dalpha_con = &sph_particles_dalpha_con(0);
	cuda_data.dvx_con = &sph_particles_dvel_con(XDIM, 0);
	cuda_data.dvy_con = &sph_particles_dvel_con(YDIM, 0);
	cuda_data.dvz_con = &sph_particles_dvel_con(ZDIM, 0);
	cuda_data.code_dif_to_cgs = sqr(get_options().code_to_cm) / get_options().code_to_s;
	cuda_data.sa_snk = &sph_particles_semi_active(0);
	cuda_data.difco_snk = &sph_particles_difco(0);
	cuda_data.gx_snk = &sph_particles_gforce(XDIM, 0);
	cuda_data.gy_snk = &sph_particles_gforce(YDIM, 0);
	cuda_data.gz_snk = &sph_particles_gforce(ZDIM, 0);
	cuda_data.alpha_snk = &sph_particles_alpha(0);
	cuda_data.gcentral = get_options().gcentral;
	cuda_data.hcentral = get_options().hcentral;
	cuda_data.fvel_snk = &sph_particles_fvel(0);
	cuda_data.fpot_snk = &sph_particles_fpot(0);
	cuda_data.Z_snk = &sph_particles_Z(0);
	cuda_data.h0 = 2.0 * get_options().hsoft;
	cuda_data.f0_snk = &sph_particles_fpre(0);
	cuda_data.tdyn_snk = &sph_particles_tdyn(0);
//	cuda_data.Yform_snk = &sph_particles_formY(0);
//	cuda_data.Zform_snk = &sph_particles_formZ(0);
	cuda_data.eint_snk = &sph_particles_eint(0);
	cuda_data.vec0_snk = &sph_particles_dif_vec0(0);
	cuda_data.G = get_options().GM;
	cuda_data.Y0 = get_options().Y0;
	cuda_data.rho0_c = get_options().rho0_c;
	cuda_data.rho0_b = get_options().rho0_b;
	cuda_data.t0 = params.t0;
	const double rho_star_phys_cgs = STAR_N0 / get_options().Y0 / constants::avo;
	const double rho_star_phys_code = rho_star_phys_cgs * (std::pow(get_options().code_to_cm, 3) / get_options().code_to_g);
	const double rho_star_co_code = rho_star_phys_code * pow(params.a, 3);
	//cuda_data.hstar0 = powf(get_options().sph_mass * get_options().neighbor_number / (3.0/(4.0*M_PI)) / rho_star_co_code, (1./3.));
	if (1 / params.a - 1 < 20 && get_options().test == "") {
		cuda_data.hstar0 = get_options().hsoft / params.a;
	} else {
		cuda_data.hstar0 = 0;
	}
	PRINT("HSTAR = %e %e  %e  %e  \n", cuda_data.hstar0, rho_star_phys_cgs, rho_star_phys_code, rho_star_co_code);
	cuda_data.m = get_options().sph_mass;
	cuda_data.N = get_options().neighbor_number;
	cuda_data.kappa0 = 1.31 * pow(3.0, 1.5) * pow(constants::kb, 3.5) / 4.0 / sqrt(M_PI) / pow(constants::e, 4) / sqrt(constants::me);
//	cuda_data.dchem_snk = &sph_particles_dchem(0);
	cuda_data.eta = get_options().eta;
	cuda_data.divv_snk = &sph_particles_divv(0);
	cuda_data.hsoft_min = get_options().hsoft;
	PRINT("Running with %i nodes\n", host_trees.size());
	auto rc = sph_run_cuda(params, cuda_data, stream);
	cuda_stream_synchronize(stream);
	const bool courant = (params.run_type == SPH_RUN_HYDRO && params.phase == 1) || (params.run_type == SPH_RUN_COURANT) || params.run_type == SPH_RUN_RUNGS;
	if (courant) {
		CUDA_CHECK(cudaMemcpyAsync(host_rungs.data(), cuda_data.rungs, sizeof(char) * host_rungs.size(), cudaMemcpyDeviceToHost, stream));
	}
	cuda_end_stream(stream);
	if (courant) {
		for (int i = 0; i < host_selflist.size(); i++) {
			const auto& node = host_trees[host_selflist[i]];
			const part_int offset = node.sink_part_range.first - node.part_range.first;
			for (part_int i = node.part_range.first; i < node.part_range.second; i++) {
				sph_particles_rung(i + offset) = host_rungs[i];
			}
		}
	}
	switch (params.run_type) {
	case SPH_RUN_DIFFUSION:
		CUDA_CHECK(cudaFree(cuda_data.kappa));
		CUDA_CHECK(cudaFree(cuda_data.difco));
		CUDA_CHECK(cudaFree(cuda_data.oldrung));
		CUDA_CHECK(cudaFree(cuda_data.dif_vec));
		break;
	}

	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_DIFFUSION:
	case SPH_RUN_COURANT:
	case SPH_RUN_RUNGS:
	case SPH_RUN_MARK_SEMIACTIVE:
		CUDA_CHECK(cudaFree(cuda_data.h));
		break;
	}
	if (params.run_type == SPH_RUN_AUX) {
		CUDA_CHECK(cudaFree(cuda_data.gamma));
		CUDA_CHECK(cudaFree(cuda_data.h));
		CUDA_CHECK(cudaFree(cuda_data.vx));
		CUDA_CHECK(cudaFree(cuda_data.vy));
		CUDA_CHECK(cudaFree(cuda_data.vz));
		CUDA_CHECK(cudaFree(cuda_data.colog));
		CUDA_CHECK(cudaFree(cuda_data.T));
		CUDA_CHECK(cudaFree(cuda_data.lambda_e));
		CUDA_CHECK(cudaFree(cuda_data.mmw));
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
	case SPH_RUN_COURANT:
		CUDA_CHECK(cudaFree(cuda_data.eint));
		CUDA_CHECK(cudaFree(cuda_data.vx));
		CUDA_CHECK(cudaFree(cuda_data.vy));
		CUDA_CHECK(cudaFree(cuda_data.vz));
	case SPH_RUN_DIFFUSION:
		if (chem) {
			CUDA_CHECK(cudaFree(cuda_data.gamma));
		}
		break;
	}
	if (params.run_type == SPH_RUN_HYDRO) {
		CUDA_CHECK(cudaFree(cuda_data.gx));
		CUDA_CHECK(cudaFree(cuda_data.gy));
		CUDA_CHECK(cudaFree(cuda_data.gz));
	}
	switch (params.run_type) {
	case SPH_RUN_COURANT:
		if (stars) {
			CUDA_CHECK(cudaFree(cuda_data.gx));
			CUDA_CHECK(cudaFree(cuda_data.gy));
			CUDA_CHECK(cudaFree(cuda_data.gz));
		}
		break;
	}
	switch (params.run_type) {
	case SPH_RUN_HYDRO:
		CUDA_CHECK(cudaFree(cuda_data.fpot));
	case SPH_RUN_COURANT:
		CUDA_CHECK(cudaFree(cuda_data.f0));
		CUDA_CHECK(cudaFree(cuda_data.fvel));
		CUDA_CHECK(cudaFree(cuda_data.alpha));
		break;
	}
	CUDA_CHECK(cudaFree(cuda_data.x));
	CUDA_CHECK(cudaFree(cuda_data.y));
	CUDA_CHECK(cudaFree(cuda_data.z));
	CUDA_CHECK(cudaFree(cuda_data.rungs));
	CUDA_CHECK(cudaFree(cuda_data.trees));
	CUDA_CHECK(cudaFree(cuda_data.selfs));
	CUDA_CHECK(cudaFree(cuda_data.neighbors));
	return rc;
}

HPX_PLAIN_ACTION (sph_apply_diffusion_update);
HPX_PLAIN_ACTION (sph_init_diffusion);

float sph_apply_diffusion_update(int minrung, float toler) {
	profiler_enter(__FUNCTION__);
	vector<hpx::future<float>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_apply_diffusion_update_action>(c, minrung, toler));
	}
	float error = 0.f;
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc, toler, minrung]() {
			float error = 0.f;
			const part_int b = (size_t) proc * sph_tree_leaflist_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_tree_leaflist_size() / nthreads;
			const bool chem = get_options().chem;
			for( int i = b; i < e; i++) {
				const auto sid = sph_tree_get_leaf(i);
				auto* self = sph_tree_get_node(sid);
				if( !self->converged ) {
					float this_error = 0.f;
					if( self->nactive || has_active_neighbors(self)) {
						for( part_int j = self->part_range.first; j < self->part_range.second; j++) {
							const part_int k = sph_particles_dm_index(j);
							const auto rung = particles_rung(k);
							const bool sa = sph_particles_semi_active(j);
							if( rung >= minrung || sa) {
								auto dfrac = sph_particles_d_dif_vec(j);
								auto& frac = sph_particles_dif_vec(j);
								auto& frac0 = sph_particles_dif_vec0(j);
								for( int fi = chem ? 0 : NCHEMFRACS - 1; fi < DIFCO_COUNT; fi++) {
									if( dfrac[fi] < -frac[fi]*0.99999 ) {
										dfrac[fi] = -frac[fi]*0.99999;
									}
									this_error = std::max(this_error,fabs(dfrac[fi] / (0.5f * frac0[fi] + 0.5f * frac[fi])));
									if( fi != NCHEMFRACS) {
										this_error = std::min(this_error, fabs(dfrac[fi]) / (1e-6f));
									}
									frac[fi] += dfrac[fi];
									if( frac[fi] == INFINITY) {
										PRINT( "frac infinity with dfrac = %e\n", dfrac[fi]);
										abort();
									}
								}
							}
						}
					}
					if( this_error < toler ) {
						sph_tree_set_converged(sid);
						for( part_int j = self->part_range.first; j < self->part_range.second; j++) {
							const part_int k = sph_particles_dm_index(j);
							const auto rung = particles_rung(k);
							const bool sa = sph_particles_semi_active(j);
							auto vec = sph_particles_dif_vec(j);
							if( rung >= minrung || sa) {
								const float h = sph_particles_smooth_len(j);
								const float gamma = sph_particles_gamma(j);
								const float rho = sph_den(1/h/h/h);
								sph_particles_eint(j) = vec[NCHEMFRACS];
								double total = 0.0;
								for( int l = 0; l < NCHEMFRACS; l++) {
									total += vec[l];
								}
								if( total > 0.999999 ) {
									total = 1.0 / total;
									total *= 0.999999;
									for( int l = 0; l < NCHEMFRACS; l++) {
										vec[l] *= total;
									}
								}
								if( chem ) {
									sph_particles_He0(j) = vec[0];
									sph_particles_Hep(j) = vec[1];
									sph_particles_Hepp(j) = vec[2];
									sph_particles_Hp(j) = vec[3];
									sph_particles_Hn(j) = vec[4];
									sph_particles_H2(j) = vec[5];
									sph_particles_Z(j) = vec[6];
								}
							}
						}
					}
					error = std::max(error, this_error);
				}
			}
			return error;
		}));
	}
	for (auto& f : futs) {
		error = std::max(error, f.get());
	}
	profiler_exit();
	return error;
}

void sph_init_diffusion() {
	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<sph_init_diffusion_action>(c));
	}
	float error = 0.f;
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads, proc]() {
			const bool chem = get_options().chem;
			part_int b = (size_t) proc * sph_tree_leaflist_size() / nthreads;
			part_int e = (size_t) (proc+1) * sph_tree_leaflist_size() / nthreads;
			for( int i = b; i < e; i++) {
				const auto sid = sph_tree_get_leaf(i);
				sph_tree_set_converged(sid, false);
			}
			b = (size_t) proc * sph_particles_size() / nthreads;
			e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			for( int i = b; i < e; i++) {
				sph_particles_old_rung(i) = particles_rung(sph_particles_dm_index(i));
				dif_vector vec;
				const float h = sph_particles_smooth_len(i);
				const float gamma = sph_particles_gamma(i);
				const float rho = sph_den(1/h/h/h);
				vec[NCHEMFRACS] = sph_particles_eint(i);
				if( chem ) {
					vec[0] = std::max(sph_particles_He0(i),1e-30f);
					vec[1] = std::max(sph_particles_Hep(i),1e-30f);
					vec[2] = std::max(sph_particles_Hepp(i),1e-30f);
					vec[3] = std::max(sph_particles_Hp(i),1e-30f);
					vec[4] = std::max(sph_particles_Hn(i),1e-30f);
					vec[5] = std::max(sph_particles_H2(i),1e-30f);
					vec[6] = std::max(sph_particles_Z(i),1e-30f);
				} else {
					for( int fi = 0; fi < NCHEMFRACS; fi++) {
						vec[fi] = 1.0;
					}
				}
				sph_particles_dif_vec0(i) = vec;
				sph_particles_dif_vec(i) = vec;
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
}
