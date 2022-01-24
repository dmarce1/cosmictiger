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

#ifndef SPH_RUN_HPP_
#define SPH_RUN_HPP_

#include <cosmictiger/cuda_vector.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/tree.hpp>

#include <atomic>

struct sph_run_return {
	range<fixed32> inner_box;
	range<fixed32> outer_box;
	bool rc1;
	bool rc2;
	char max_rung;CUDA_EXPORT
	sph_run_return() {
		rc2 = false;
		rc1 = false;
		max_rung = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			inner_box.begin[dim] = outer_box.begin[dim] = fixed32::max();
			inner_box.end[dim] = outer_box.end[dim] = 0.0;
		}
	}
	CUDA_EXPORT
	sph_run_return& operator+=(const sph_run_return& other) {
		range<fixed32> rbox;
		for (int dim = 0; dim < NDIM; dim++) {
			rbox.end[dim] = distance(inner_box.end[dim], other.inner_box.end[dim]) > 0.0 ? inner_box.end[dim] : other.inner_box.end[dim];
			rbox.begin[dim] = distance(inner_box.begin[dim], other.inner_box.end[dim]) > 0.0 ? other.inner_box.begin[dim] : inner_box.begin[dim];
		}
		inner_box = rbox;
		for (int dim = 0; dim < NDIM; dim++) {
			rbox.end[dim] = distance(outer_box.end[dim], other.outer_box.end[dim]) > 0.0 ? outer_box.end[dim] : other.outer_box.end[dim];
			rbox.begin[dim] = distance(outer_box.begin[dim], other.outer_box.end[dim]) > 0.0 ? other.outer_box.begin[dim] : outer_box.begin[dim];
		}
		outer_box = rbox;
		rc1 = other.rc1 || rc1;
		rc2 = other.rc2 || rc2;
		max_rung = std::max(max_rung, other.max_rung);
		return *this;

	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & inner_box;
		arc & outer_box;
		arc & rc1;
		arc & rc2;
		arc & max_rung;
	}
};

#define SPH_RUN_SMOOTH_LEN 0
#define SPH_RUN_MARK_SEMIACTIVE 1
#define SPH_RUN_FIND_BOXES 2
#define SPH_RUN_COURANT 3
#define SPH_RUN_GRAVITY 4
#define SPH_RUN_HYDRO 5
#define SPH_RUN_UPDATE 6

#define SPH_SET_ACTIVE 0
#define SPH_SET_SEMIACTIVE 1
#define SPH_SET_ALL 2

struct sph_run_params {
	int run_type;
	int set1;
	int set2;
	int min_rung;
	float h_wt;
	float t0;
	float a;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & h_wt;
		arc & run_type;
		arc & min_rung;
		arc & set1;
		arc & set2;
		arc & t0;
		arc & a;
	}
};

struct sph_run_workitem {
	expansion<float> L;
	array<fixed32, NDIM> pos;
	tree_id self;
	vector<tree_id> dchecklist;
	vector<tree_id> echecklist;
};

struct sph_run_workspace;

#ifndef __CUDACC__
hpx::future<sph_run_return> sph_run(sph_run_params, tree_id self, vector<tree_id> checklist);
#endif

#endif /* SPH_RUN_HPP_ */
