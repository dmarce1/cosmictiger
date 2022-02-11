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

#ifndef GRAVITY_HPP_
#define GRAVITY_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/kick.hpp>

struct force_vectors {
	vector<float> phi;
	vector<float> gx;
	vector<float> gy;
	vector<float> gz;
	force_vectors() = default;
	force_vectors(int sz) {
		phi.resize(sz);
		gx.resize(sz);
		gy.resize(sz);
		gz.resize(sz);
	}
};

enum gravity_cc_type {
	GRAVITY_CC_DIRECT, GRAVITY_CC_EWALD
};

size_t cpu_gravity_cc(expansion<float>&, const vector<tree_id>&, tree_id, gravity_cc_type, bool do_phi);
size_t cpu_gravity_cp(expansion<float>&, const vector<tree_id>&, tree_id, bool do_phi);
size_t cpu_gravity_pc(force_vectors&, int, tree_id, const vector<tree_id>&);
size_t cpu_gravity_pp(force_vectors&, int, tree_id, const vector<tree_id>&, float h);

#ifdef __CUDACC__
__device__
int cuda_gravity_cc(const cuda_kick_data&, expansion<float>&, const tree_node&, const fixedcapvec<int, MULTLIST_SIZE>&, gravity_cc_type, bool do_phi);
__device__
int cuda_gravity_cp(const cuda_kick_data&, expansion<float>&, const tree_node&, const fixedcapvec<int, PARTLIST_SIZE>&, bool do_phi);
__device__
int cuda_gravity_pc(const cuda_kick_data& data, const tree_node&, const fixedcapvec<int, MULTLIST_SIZE>&, int, bool);
__device__
int cuda_gravity_pp(const cuda_kick_data& data, const tree_node&, const fixedcapvec<int, PARTLIST_SIZE>&, int, float h, bool);
#endif
#endif /* GRAVITY_HPP_ */
