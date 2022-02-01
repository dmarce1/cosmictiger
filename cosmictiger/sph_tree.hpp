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
 Foundation, Inc., 51 Franklin Ssph_treet, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef SPH_TREE_HPP_
#define SPH_TREE_HPP_

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/tree.hpp>

struct sph_tree_node {
	array<tree_id, NCHILD> children;
	pair<int, int> proc_range;
	pair<part_int> part_range;
	pair<part_int> sink_part_range;
	fixed32_range inner_box;
	fixed32_range outer_box;
	fixed32_range box;
	size_t nactive;
	bool local_root;
	bool leaf;
	size_t node_count;
	size_t active_nodes;
	bool box_active;
	int depth;
	pair<int,int> neighbor_range;
	CUDA_EXPORT
	sph_tree_node() {
		box_active = false;
	}
	inline part_int nparts() const {
		return part_range.second - part_range.first;
	}
	inline particle_global_range global_part_range() const {
		particle_global_range r;
		r.proc = proc_range.first;
		r.range = part_range;
		return r;
	}
	inline bool is_local() const {
		return proc_range.second - proc_range.first == 1;
	}
	inline bool is_local_here() const {
		return is_local() && proc_range.first == hpx_rank();
	}
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & neighbor_range;
		arc & box_active;
		arc & box;
		arc & sink_part_range;
		arc & inner_box;
		arc & outer_box;
		arc & children;
		arc & proc_range;
		arc & part_range;
		arc & nactive;
		arc & local_root;
		arc & node_count;
		arc & depth;
	}
};

struct sph_tree_create_return {
	fixed32_range inner_box;
	fixed32_range outer_box;
	tree_id id;
	size_t nactive;
	size_t active_nodes;
	size_t node_count;
	size_t active_leaf_nodes;
	size_t leaf_nodes;
	int max_depth;
	int min_depth;
	double flops;
	template<class A>
	void serialize(A&& a, unsigned) {
		a & active_leaf_nodes;
		a & leaf_nodes;
		a & active_nodes;
		a & inner_box;
		a & outer_box;
		a & id;
		a & nactive;
		a & node_count;
		a & flops;
		a & max_depth;
		a & min_depth;
	}
};

struct sph_tree_create_params {
	int min_rung;
	float h_wt;
	sph_tree_create_params() = default;
	sph_tree_create_params(int min_rung, float h_wt);
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & min_rung;
		arc & h_wt;
	}
};

sph_tree_create_return sph_tree_create(sph_tree_create_params params, size_t key = 1, pair<int, int> proc_range = pair<int>(0, hpx_size()), pair<part_int> part_range =
		pair<part_int>(-1, -1), range<double> box = unit_box<double>(), int depth = 0, bool local_root = (hpx_size() == 1));
void sph_tree_destroy(bool free_sph_tree = false);
const sph_tree_node* sph_tree_get_node(tree_id);
void sph_tree_sort_sph_particles_by_particles();
void sph_tree_set_box_active(tree_id id, bool rc);
void sph_tree_set_nactive(tree_id id, part_int i);
void sph_tree_set_boxes(tree_id, const fixed32_range& , const fixed32_range& );
void sph_tree_free_neighbor_list();
int sph_tree_allocate_neighbor_list(const vector<tree_id>&);
tree_id& sph_tree_get_neighbor(int i);
void sph_tree_set_neighbor_range(tree_id id, pair<int,int> rng);
int sph_tree_leaflist_size();
const sph_tree_node* sph_tree_get_leaf(int i);
void sph_tree_clear_neighbor_ranges();


#endif /* SPH_TREE_HPP_ */
