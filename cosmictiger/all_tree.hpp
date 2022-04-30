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

#pragma once

#include <cosmictiger/tree.hpp>
#include <cosmictiger/hpx.hpp>

struct softlens_return {
	float hmax;
	float hmin;
	bool fail;
	double flops;
	template<class A>
	void serialize(A&&arc, unsigned) {
		arc & hmax;
		arc & hmin;
		arc & fail;
		arc & flops;
	}
};

struct all_tree_data {
	float a;
	float* softlen_snk;
	char* rung_snk;
	char* converged_snk;
	float* zeta1_snk;
	float* zeta2_snk;
	float* divv_snk;
	float* gx_snk;
	int pass;
	float* gy_snk;
	float* gz_snk;
	float* sph_h_snk;
	char* types;
	float hmin;
	float hmax;
	part_int* cat_snk;
	char* type_snk;
	int minrung;
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* vx;
	float* vy;
	float* vz;
	float* h;
	float dm_mass;
	float sph_mass;
	bool sph;
	int* selfs;
	tree_node* trees;
	int* neighbors;
	int nselfs;
	char* sa_snk;
	float N;
};

struct all_tree_range_return {
	fixed32_range ibox;
	fixed32_range obox;
	float hmax;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & ibox;
		arc & obox;
		arc & hmax;
	}
};

#ifndef __CUDACC__
hpx::future<void> all_tree_find_neighbors(tree_id self_id, vector<tree_id> checklist);
hpx::future<all_tree_range_return> all_tree_find_ranges(tree_id self_id, int, double = 1.01);
#endif
softlens_return all_tree_softlens(int minrung, float a);
softlens_return all_tree_derivatives(int minrung);
softlens_return all_tree_divv(int minrung, float a);
softlens_return all_tree_softlens_cuda(all_tree_data params, cudaStream_t stream);
softlens_return all_tree_derivatives_cuda(all_tree_data params, cudaStream_t stream);
softlens_return all_tree_divv_cuda(all_tree_data params, cudaStream_t stream);

