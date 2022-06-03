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

#ifndef TREE_HPP_
#define TREE_HPP_

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>

struct multipole_pos {
	multipole<float> m;
	array<fixed32, NDIM> pos;
};

struct tree_id {
	int proc;
	int index;
	inline bool operator==(tree_id other) const {
		return proc == other.proc && index == other.index;
	}
	inline bool operator<(tree_id other) const {
		if (proc < other.proc) {
			return true;
		} else if (proc > other.proc) {
			return false;
		} else if (index < other.index) {
			return true;
		} else {
			return false;
		}
	}
	inline bool operator!=(tree_id other) const {
		return proc != other.proc || index != other.index;
	}
	template<class A>
	void serialize(A&& a, unsigned) {
		a & proc;
		a & index;
	}
};

struct tree_id_hash {
	inline size_t operator()(tree_id id) const {
		const int line_size = get_options().tree_cache_line_size;
		const int i = id.index / line_size;
		return i * (hpx_size() - 1) + ((id.proc < hpx_rank()) ? id.proc : id.proc - 1);
	}
};

struct tree_id_hash_lo {
	inline size_t operator()(tree_id id) const {
		tree_id_hash hash;
		return hash(id) % TREE_CACHE_SIZE;
	}
};

struct tree_id_hash_hi {
	inline size_t operator()(tree_id id) const {
		tree_id_hash hash;
		return hash(id) / TREE_CACHE_SIZE;
	}
};

struct multi_pos {
	multipole<float> multi;
	array<fixed32, NDIM> pos;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & multi;
		arc & pos;
	}
};

struct tree_node {
	multi_pos mpos;
	array<tree_id, NCHILD> children;
	pair<int, int> proc_range;
	pair<part_int> part_range;
	float radius;
	struct {
		unsigned short depth :14;
		unsigned short local_root :1;
		unsigned short leaf :1;
	};CUDA_EXPORT

	CUDA_EXPORT
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
		arc & mpos;
		arc & children;
		arc & proc_range;
		arc & part_range;
		arc & radius;
		int local_root0;
		int leaf0;
		int depth0;
		local_root0 = local_root;
		leaf0 = leaf;
		depth0 = depth;
		arc & local_root0;
		arc & leaf0;
		arc & depth0;
		local_root = local_root0;
		depth = depth0;
		leaf = leaf0;
	}
};

struct tree_create_return {
	multipole<float> multi;
	array<fixed32, NDIM> pos;
	tree_id id;
	float radius;
	range<double> box;
	template<class A>
	void serialize(A&& a, unsigned) {
		a & multi;
		a & id;
		a & pos;
		a & radius;
	}
};

struct tree_create_params {
	int min_rung;
	double theta;
//	double hmax;
	int min_level;
	bool leaf_pushed;
	bool do_leaf_sizes;
	tree_create_params() {
		leaf_pushed = false;
		do_leaf_sizes = false;
	}
	tree_create_params(int min_rung, double theta, double hmax);
	template<class A>
	void serialize(A&& arc, unsigned) {
//		arc & hmax;
		arc & do_leaf_sizes;
		arc & leaf_pushed;
		arc & min_rung;
		arc & theta;
		arc & min_level;
	}
};

tree_create_return tree_create(tree_create_params params, size_t key = 1, pair<int, int> proc_range = pair<int>(0, hpx_size()), pair<part_int> part_range =
		pair<part_int>(-1, -1), range<double> box = unit_box<double>(), int depth = 0, bool local_root = (hpx_size() == 1));
void tree_reset();
void tree_destroy(bool free_tree = false);
int tree_min_level(double theta, double hsoft);
const tree_node* tree_get_node(tree_id);
void tree_sort_particles_by_sph_particles();
void tree_free_neighbor_list();
long long tree_nodes_size();
long long tree_nodes_next_index();
void tree_clear_neighbor_ranges();
int tree_avg_leaf_size();
int tree_allocate_neighbor_list(const vector<tree_id>& values);
void tree_set_neighbor_range(tree_id id, pair<int, int> rng);
void tree_set_boxes(tree_id id, const fixed32_range& ibox, const fixed32_range& obox, float hmax);
const tree_id tree_get_leaf(int i);
size_t tree_add_remote(const tree_node& remote);
tree_node* tree_data();
tree_id& tree_get_neighbor(int i);
void tree_2_cpu();
void tree_2_gpu();

#endif /* TREE_HPP_ */
