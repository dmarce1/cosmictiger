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
	return a.periodic_intersects(b);
}

struct sph_run_workspace {
	sph_run_params params;
	vector<fixed32, pinned_allocator<fixed32>> host_x;
	vector<fixed32, pinned_allocator<fixed32>> host_y;
	vector<fixed32, pinned_allocator<fixed32>> host_z;
	vector<array<float, NCHEMFRACS>, pinned_allocator<array<float, NCHEMFRACS>>> host_chem;
	vector<float, pinned_allocator<float>> host_cold_frac;
	vector<float, pinned_allocator<float>> host_divv;
	vector<float, pinned_allocator<float>> host_vx;
	vector<float, pinned_allocator<float>> host_vy;
	vector<float, pinned_allocator<float>> host_vz;
	vector<float, pinned_allocator<float>> host_entr;
	vector<float, pinned_allocator<float>> host_alpha;
	vector<float, pinned_allocator<float>> host_shearv;
	vector<float, pinned_allocator<float>> host_fpre1;
	vector<float, pinned_allocator<float>> host_fpre2;
	vector<float, pinned_allocator<float>> host_pre;
	vector<float, pinned_allocator<float>> host_h;
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
	return vector<sph_values>();
	/*int max_rung = 0;
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
	 return values;*/
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

bool is_converged(const sph_tree_node* self, int minrung) {
	bool converged = true;
	for (int i = self->part_range.first; i < self->part_range.second; i++) {
		if (sph_particles_rung(i) >= minrung) {
			if (!sph_particles_converged(i)) {
				converged = false;
				break;
			}
		}
	}
	return converged;
}

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

hpx::future<sph_tree_neighbor_return> sph_tree_neighbor(sph_tree_neighbor_params params, tree_id self, vector<tree_id> checklist, int level) {
	ALWAYS_ASSERT(params.seti);
//	ALWAYS_ASSERT(params.seto);
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
				bool test2 = false;
				bool test1 = false;
				if (params.seti | SPH_INTERACTIONS_I) {
					test1 = range_intersect(self_ptr->outer_box, other->inner_box);
				}
				if (params.seti | SPH_INTERACTIONS_J) {
					test2 = range_intersect(self_ptr->inner_box, other->outer_box);
				}
				if (test1 || test2) {
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
			/*			for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
			 if (sph_particles_id(i) == 591) {
			 PRINT("--------->>> %i %i %i\n", level, leaflist.size(),  self_ptr->part_range.second- self_ptr->part_range.first);
			 break;
			 }
			 }*/
			sph_tree_set_neighbor_range(self, rng);
		}
			break;
		case SPH_TREE_NEIGHBOR_BOXES: {
			fixed32_range ibox, obox, pbox;
			float maxh = 0.f;
			for (int dim = 0; dim < NDIM; dim++) {
				pbox.begin[dim] = ibox.begin[dim] = obox.begin[dim] = 1.9;
				pbox.end[dim] = ibox.end[dim] = obox.end[dim] = -.9;
			}
			bool show = false;
			for (part_int i = self_ptr->part_range.first; i < self_ptr->part_range.second; i++) {
				const bool active = sph_particles_rung(i) >= params.min_rung;
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
				const auto tiny = 10.0 * range_fixed::min().to_double();
				/*				if (sph_particles_id(i) == 591) {
				 show = true;
				 PRINT("??????????? %e\n", h / params.h_wt);
				 for (int dim = 0; dim < NDIM; dim++) {
				 const double x = X[dim].to_double();
				 pbox.begin[dim] = std::min(pbox.begin[dim].to_double(), x - h - tiny);
				 pbox.end[dim] = std::max(pbox.end[dim].to_double(), x + h + tiny);
				 PRINT("%e %e %e\n", pbox.begin[dim].to_double(), x, pbox.end[dim].to_double());
				 }
				 }*/

				if ((params.seto & SPH_SET_ALL) || (active && (params.seto & SPH_SET_ACTIVE))) {
					for (int dim = 0; dim < NDIM; dim++) {
						const double x = X[dim].to_double();
						obox.begin[dim] = std::min(obox.begin[dim].to_double(), x - h - tiny);
						obox.end[dim] = std::max(obox.end[dim].to_double(), x + h + tiny);
					}
				}
				if ((params.seti & SPH_SET_ALL) || (active && (params.seti & SPH_SET_ACTIVE))) {
					for (int dim = 0; dim < NDIM; dim++) {
						const double x = X[dim].to_double();
						ibox.begin[dim] = std::min(ibox.begin[dim].to_double(), x - tiny);
						ibox.end[dim] = std::max(ibox.end[dim].to_double(), x + tiny);
					}
				}
			}
			if (show) {
				for (int dim = 0; dim < NDIM; dim++) {
					PRINT("%e %e | %e %e\n", ibox.begin[dim].to_double(), ibox.end[dim].to_double(), obox.begin[dim].to_double(), obox.end[dim].to_double());
				}
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

sph_run_return sph_run(sph_run_params params, bool cuda) {
//	PRINT("SPHRUN = %i\n", params.run_type);
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
		futs2.push_back(hpx::async([proc,nthreads,params,cuda,workspace_ptr]() {
			sph_run_return rc;
			int index = next++;
			sph_data_vecs data;
			while( index < sph_tree_leaflist_size()) {
				data.clear();
				const auto selfid = sph_tree_get_leaf(index);
				const auto* self = sph_tree_get_node(selfid);
				bool test;
				switch(params.run_type) {

					case SPH_RUN_SMOOTHLEN:
					test = self->nactive > 0 && !is_converged(self, params.min_rung);
					break;

					case SPH_RUN_RUNGS:
					test = has_active_neighbors(self);
					break;

					case SPH_RUN_HYDRO:
					test = self->nactive > 0;
					break;

					case SPH_RUN_AUX:
					test = self->nactive > 0;
					break;
				}
				if(test) {
					if( cuda ) {
						gpu_work++;
						workspace_ptr->add_work(selfid);
					} else {
						ALWAYS_ASSERT(false);
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
	const bool gravity = get_options().gravity;
	const bool conduction = get_options().conduction;
	const bool diffusion = get_options().diffusion;
	for (auto& node : host_trees) {
		parts_size += node.part_range.second - node.part_range.first;
	}
	host_x.resize(parts_size);
	host_y.resize(parts_size);
	host_z.resize(parts_size);
	if (params.run_type == SPH_RUN_RUNGS) {
		host_rungs.resize(parts_size);
	} else if (params.run_type == SPH_RUN_COND_INIT) {
		host_rungs.resize(parts_size);
		host_h.resize(parts_size);
		host_entr.resize(parts_size);
		if (stars) {
			host_cold_frac.resize(parts_size);
		}
	} else if (params.run_type == SPH_RUN_SMOOTHLEN) {
		host_vx.resize(parts_size);
		host_vy.resize(parts_size);
		host_vz.resize(parts_size);
		host_entr.resize(parts_size);
		if (stars) {
			host_cold_frac.resize(parts_size);
		}
	} else if (params.run_type == SPH_RUN_HYDRO) {
		host_h.resize(parts_size);
		host_entr.resize(parts_size);
		host_vx.resize(parts_size);
		host_vy.resize(parts_size);
		host_vz.resize(parts_size);
		host_alpha.resize(parts_size);
		host_fpre1.resize(parts_size);
		host_fpre2.resize(parts_size);
		host_pre.resize(parts_size);
		if (stars) {
			host_cold_frac.resize(parts_size);
		}
		if (diffusion) {
			host_shearv.resize(parts_size);
			if (chem) {
				host_chem.resize(parts_size);
			}
		}
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
						[&index,proc,nthreads,this,&part_index, chem, stars, conduction, diffusion]() {
							int this_index = index++;
							while( this_index < host_trees.size()) {
								auto& node = host_trees[this_index];
								const part_int size = node.part_range.second - node.part_range.first;
								const part_int offset = (part_index += size) - size;
								sph_particles_global_read_pos(node.global_part_range(), host_x.data(), host_y.data(), host_z.data(), offset);
								if( params.run_type == SPH_RUN_SMOOTHLEN) {
									sph_particles_global_read_sph(node.global_part_range(),stars ? host_cold_frac.data() : nullptr, host_entr.data(), host_vx.data(), host_vy.data(), host_vz.data(), offset);
								} else if( params.run_type == SPH_RUN_HYDRO) {
									sph_particles_global_read_sph(node.global_part_range(),stars ? host_cold_frac.data() : nullptr, host_entr.data(),host_vx.data(), host_vy.data(), host_vz.data(), offset);
									sph_particles_global_read_aux(node.global_part_range(), host_h.data(), host_alpha.data(), host_pre.data(), host_fpre1.data(), host_fpre2.data(), diffusion ? host_shearv.data() : nullptr, chem ? host_chem.data() :nullptr, offset);
								} else if( params.run_type == SPH_RUN_RUNGS) {
									sph_particles_global_read_rungs(node.global_part_range(), host_rungs.data(), offset);
								} else if( params.run_type == SPH_RUN_COND_INIT) {
									sph_particles_global_read_rungs(node.global_part_range(), host_rungs.data(), offset);
									sph_particles_global_read_sph(node.global_part_range(),stars ? host_cold_frac.data() : nullptr, host_entr.data(),nullptr, nullptr, nullptr, offset);
									sph_particles_global_read_aux(node.global_part_range(), host_h.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, offset);
								}
								node.part_range.first = offset;
								node.part_range.second = offset + size;
								this_index = index++;
							}
						}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	sph_run_cuda_data cuda_data;
	CUDA_CHECK(cudaMalloc(&cuda_data.selfs, sizeof(int) * host_selflist.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.x, sizeof(fixed32) * host_x.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.y, sizeof(fixed32) * host_y.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.z, sizeof(fixed32) * host_z.size()));
	if (params.run_type == SPH_RUN_RUNGS) {
		CUDA_CHECK(cudaMalloc(&cuda_data.rungs, sizeof(char) * host_rungs.size()));
	} else if (params.run_type == SPH_RUN_COND_INIT) {
		CUDA_CHECK(cudaMalloc(&cuda_data.rungs, sizeof(char) * host_rungs.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.h, sizeof(float) * host_h.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.entr, sizeof(float) * host_entr.size()));
		if (stars) {
			CUDA_CHECK(cudaMalloc(&cuda_data.cold_frac, sizeof(float) * host_cold_frac.size()));
		}
	} else if (params.run_type == SPH_RUN_SMOOTHLEN) {
		CUDA_CHECK(cudaMalloc(&cuda_data.vx, sizeof(float) * host_vx.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vy, sizeof(float) * host_vy.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vz, sizeof(float) * host_vz.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.entr, sizeof(float) * host_entr.size()));
		if (stars) {
			CUDA_CHECK(cudaMalloc(&cuda_data.cold_frac, sizeof(float) * host_cold_frac.size()));
		}
	} else if (params.run_type == SPH_RUN_HYDRO) {
		CUDA_CHECK(cudaMalloc(&cuda_data.h, sizeof(float) * host_h.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.entr, sizeof(float) * host_entr.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vx, sizeof(float) * host_vx.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vy, sizeof(float) * host_vy.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.vz, sizeof(float) * host_vz.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.fpre1, sizeof(float) * host_fpre1.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.fpre2, sizeof(float) * host_fpre2.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.pre, sizeof(float) * host_pre.size()));
		CUDA_CHECK(cudaMalloc(&cuda_data.alpha, sizeof(float) * host_alpha.size()));
		if (stars) {
			CUDA_CHECK(cudaMalloc(&cuda_data.cold_frac, sizeof(float) * host_cold_frac.size()));
		}
		if (diffusion) {
			CUDA_CHECK(cudaMalloc(&cuda_data.shearv, sizeof(float) * host_shearv.size()));
			if (chem) {
				CUDA_CHECK(cudaMalloc(&cuda_data.chem, NCHEMFRACS * sizeof(float) * host_chem.size()));
			}
		}
	}
	CUDA_CHECK(cudaMalloc(&cuda_data.trees, sizeof(sph_tree_node) * host_trees.size()));
	CUDA_CHECK(cudaMalloc(&cuda_data.neighbors, sizeof(int) * host_neighbors.size()));
	auto stream = cuda_get_stream();
	if (params.run_type == SPH_RUN_SMOOTHLEN) {
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vx, host_vx.data(), sizeof(float) * host_vx.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vy, host_vy.data(), sizeof(float) * host_vy.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vz, host_vz.data(), sizeof(float) * host_vz.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.entr, host_entr.data(), sizeof(float) * host_entr.size(), cudaMemcpyHostToDevice, stream));
		if (stars) {
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.cold_frac, host_cold_frac.data(), sizeof(float) * host_cold_frac.size(), cudaMemcpyHostToDevice, stream));
		}
	} else if (params.run_type == SPH_RUN_HYDRO) {
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.h, host_h.data(), sizeof(float) * host_h.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.entr, host_entr.data(), sizeof(float) * host_entr.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vx, host_vx.data(), sizeof(float) * host_vx.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vy, host_vy.data(), sizeof(float) * host_vy.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.vz, host_vz.data(), sizeof(float) * host_vz.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.fpre1, host_fpre1.data(), sizeof(float) * host_fpre1.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.fpre2, host_fpre2.data(), sizeof(float) * host_fpre2.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.pre, host_pre.data(), sizeof(float) * host_pre.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.alpha, host_alpha.data(), sizeof(float) * host_alpha.size(), cudaMemcpyHostToDevice, stream));
		if (diffusion) {
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.shearv, host_shearv.data(), sizeof(float) * host_shearv.size(), cudaMemcpyHostToDevice, stream));
			if (chem) {
				CUDA_CHECK(cudaMemcpyAsync(cuda_data.chem, host_chem.data(), NCHEMFRACS * sizeof(float) * host_chem.size(), cudaMemcpyHostToDevice, stream));
			}
		}
		if (stars) {
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.cold_frac, host_cold_frac.data(), sizeof(float) * host_cold_frac.size(), cudaMemcpyHostToDevice, stream));
		}
	} else if (params.run_type == SPH_RUN_RUNGS) {
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.rungs, host_rungs.data(), sizeof(char) * host_rungs.size(), cudaMemcpyHostToDevice, stream));
	} else if (params.run_type == SPH_RUN_COND_INIT) {
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.h, host_h.data(), sizeof(float) * host_h.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.rungs, host_rungs.data(), sizeof(char) * host_rungs.size(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cuda_data.entr, host_entr.data(), sizeof(float) * host_entr.size(), cudaMemcpyHostToDevice, stream));
		if (stars) {
			CUDA_CHECK(cudaMemcpyAsync(cuda_data.cold_frac, host_cold_frac.data(), sizeof(float) * host_cold_frac.size(), cudaMemcpyHostToDevice, stream));
		}
	}
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.x, host_x.data(), sizeof(fixed32) * host_x.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.y, host_y.data(), sizeof(fixed32) * host_y.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.z, host_z.data(), sizeof(fixed32) * host_z.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.trees, host_trees.data(), sizeof(sph_tree_node) * host_trees.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.selfs, host_selflist.data(), sizeof(int) * host_selflist.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(cuda_data.neighbors, host_neighbors.data(), sizeof(int) * host_neighbors.size(), cudaMemcpyHostToDevice, stream));
	cuda_data.dm_index_snk = &sph_particles_dm_index(0);
	cuda_data.sa_snk = &sph_particles_semiactive(0);
	cuda_data.rungs_snk = &particles_rung(0);
	cuda_data.rec1_snk = &sph_particles_rec1(0);
	cuda_data.rec3_snk = &sph_particles_rec3(0);
	cuda_data.rec4_snk = &sph_particles_rec4(0);
	cuda_data.rec2_snk = &sph_particles_rec2(0);
	cuda_data.kap_snk = &sph_particles_kappa(0);
	cuda_data.dcold_mass = &sph_particles_dcold_mass(0);
	cuda_data.h_snk = &sph_particles_smooth_len(0);
	cuda_data.oldrung_snk = &sph_particles_oldrung(0);
	cuda_data.def_gamma = get_options().gamma;
	cuda_data.dalpha = &sph_particles_dalpha(0);
	cuda_data.curlv_snk = &sph_particles_curlv(0);
	cuda_data.gsoft = get_options().hsoft;
	cuda_data.nselfs = host_selflist.size();
	cuda_data.chemistry = get_options().chem;
	cuda_data.converged_snk = &sph_particles_converged(0);
	cuda_data.gravity = get_options().gravity;
	cuda_data.conduction = get_options().conduction;
	cuda_data.rec5_snk = &sph_particles_rec5(0);
	cuda_data.rec6_snk = &sph_particles_rec6(0);
	cuda_data.code_dif_to_cgs = sqr(get_options().code_to_cm) / get_options().code_to_s;
	cuda_data.gcentral = get_options().gcentral;
	cuda_data.hcentral = get_options().hcentral;
	cuda_data.G = get_options().GM;
	cuda_data.Y0 = get_options().Y0;
	cuda_data.rho0_c = get_options().rho0_c;
	cuda_data.rho0_b = get_options().rho0_b;
	cuda_data.t0 = params.t0;
//	PRINT("HSTAR = %e %e  %e  %e  \n", cuda_data.hstar0, rho_star_phys_cgs, rho_star_phys_code, rho_star_co_code);
	cuda_data.m = get_options().sph_mass;
	cuda_data.N = get_options().neighbor_number;
	cuda_data.kappa0 = 1.31 * pow(3.0, 1.5) * pow(constants::kb, 3.5) / 4.0 / sqrt(M_PI) / pow(constants::e, 4) / sqrt(constants::me);
//	cuda_data.dchem_snk = &sph_particles_dchem(0);
	cuda_data.eta = get_options().eta;
//	PRINT("Running with %i nodes\n", host_trees.size());
	PRINT("Sending %i\n", host_selflist.size());
	auto rc = sph_run_cuda(params, cuda_data, stream);
	cuda_stream_synchronize(stream);
	if (params.run_type == SPH_RUN_RUNGS) {
		CUDA_CHECK(cudaMemcpyAsync(host_rungs.data(), cuda_data.rungs, sizeof(char) * host_rungs.size(), cudaMemcpyDeviceToHost, stream));
	}
	cuda_end_stream(stream);
	if (params.run_type == SPH_RUN_RUNGS) {
		for (int i = 0; i < host_selflist.size(); i++) {
			const auto& node = host_trees[host_selflist[i]];
			const part_int offset = node.sink_part_range.first - node.part_range.first;
			for (part_int j = node.part_range.first; j < node.part_range.second; j++) {
				sph_particles_rung(j + offset) = host_rungs[j];
			}
		}
	}
	if (params.run_type == SPH_RUN_RUNGS) {
		CUDA_CHECK(cudaFree(cuda_data.rungs));
	} else if (params.run_type == SPH_RUN_COND_INIT) {
		CUDA_CHECK(cudaFree(cuda_data.rungs));
		CUDA_CHECK(cudaFree(cuda_data.h));
		CUDA_CHECK(cudaFree(cuda_data.entr));
		if (stars) {
			CUDA_CHECK(cudaFree(cuda_data.cold_frac));
		}
	} else if (params.run_type == SPH_RUN_SMOOTHLEN) {
		CUDA_CHECK(cudaFree(cuda_data.vx));
		CUDA_CHECK(cudaFree(cuda_data.vy));
		CUDA_CHECK(cudaFree(cuda_data.vz));
		CUDA_CHECK(cudaFree(cuda_data.entr));
		if (stars) {
			CUDA_CHECK(cudaFree(cuda_data.cold_frac));
		}
	} else if (params.run_type == SPH_RUN_HYDRO) {
		CUDA_CHECK(cudaFree(cuda_data.h));
		CUDA_CHECK(cudaFree(cuda_data.entr));
		CUDA_CHECK(cudaFree(cuda_data.vx));
		CUDA_CHECK(cudaFree(cuda_data.vy));
		CUDA_CHECK(cudaFree(cuda_data.vz));
		CUDA_CHECK(cudaFree(cuda_data.fpre1));
		CUDA_CHECK(cudaFree(cuda_data.fpre2));
		CUDA_CHECK(cudaFree(cuda_data.pre));
		CUDA_CHECK(cudaFree(cuda_data.alpha));
		if (stars) {
			CUDA_CHECK(cudaFree(cuda_data.cold_frac));
		}
		if (diffusion) {
			CUDA_CHECK(cudaFree(cuda_data.shearv));
			if (chem) {
				CUDA_CHECK(cudaFree(cuda_data.chem));
			}
		}
	}
	CUDA_CHECK(cudaFree(cuda_data.x));
	CUDA_CHECK(cudaFree(cuda_data.y));
	CUDA_CHECK(cudaFree(cuda_data.z));
	CUDA_CHECK(cudaFree(cuda_data.trees));
	CUDA_CHECK(cudaFree(cuda_data.selfs));
	CUDA_CHECK(cudaFree(cuda_data.neighbors));
	return rc;
}

