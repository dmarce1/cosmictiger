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

struct tree_node {
	multipole<float> multi;
	array<fixed32, NDIM> pos;
	array<tree_id, NCHILD> children;
	pair<int, int> proc_range;
	pair<part_int> part_range;
	pair<part_int> sink_part_range;
	float radius;
	bool local_root;
	bool leaf;
	size_t node_count;
	int depth;
	bool valid;
	int index;

	CUDA_EXPORT
	inline const multipole_pos* get_multipole_ptr() const {
		return (multipole_pos*) &multi;
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
		arc & multi;
		arc & children;
		arc & pos;
		arc & proc_range;
		arc & part_range;
		arc & radius;
		arc & local_root;
		arc & leaf;
		arc & valid;
		arc & index;
		arc & node_count;
		arc & sink_part_range;
		arc & depth;
	}
};

struct tree_create_return {
	multipole<float> multi;
	array<fixed32, NDIM> pos;
	tree_id id;
	float radius;
	size_t node_count;
	double flops;
	template<class A>
	void serialize(A&& a, unsigned) {
		a & multi;
		a & id;
		a & pos;
		a & radius;
		a & node_count;
		a & flops;
	}
};

struct tree_create_params {
	int min_rung;
	double theta;
	bool htime;
//	double hmax;
	int min_level;
	tree_create_params() {
		htime = false;
	}
	tree_create_params(int min_rung, double theta, double hmax);
	template<class A>
	void serialize(A&& arc, unsigned) {
//		arc & hmax;
		arc & htime;
		arc & min_rung;
		arc & theta;
		arc & min_level;
	}
};


struct tree_sort_global_params {
	int *index;
	int N;
	float theta;
	int* next_alloc;
	tree_node* tree_nodes;
	float h;
	int alloc_line_size;
	array<fixed32*, NDIM> X;
	array<float*, NDIM> V;
	char* rungs;
	int bucket_size;
	int rank;
};

struct tree_sort_local_params {
	pair<part_int> part_range;
	range<double> box;
	int depth;
};

struct tree_sort_return {
	multipole<float> M;
	array<fixed32, NDIM> pos;
	float r;
	int node_count;
	int index;
};

cudaStream_t cuda_tree_sort(tree_sort_local_params* local_params, tree_sort_return* returns, tree_sort_global_params global_params);

tree_create_return tree_create(tree_create_params params, size_t key = 1, pair<int, int> proc_range = pair<int>(0, hpx_size()), pair<part_int> part_range =
		pair<part_int>(-1, -1), range<double> box = unit_box<double>(), int depth = 0, bool local_root = (hpx_size() == 1));
void tree_reset();
void tree_destroy(bool free_tree = false);
int tree_min_level(double theta, double hsoft);
const tree_node* tree_get_node(tree_id);
void tree_sort_particles_by_sph_particles();
void tree_free_neighbor_list();
long long tree_nodes_size() ;
long long tree_nodes_next_index();
void tree_clear_neighbor_ranges();
int tree_allocate_neighbor_list(const vector<tree_id>& values);
void tree_set_neighbor_range(tree_id id, pair<int, int> rng);
void tree_set_boxes(tree_id id, const fixed32_range& ibox, const fixed32_range& obox, float hmax);
int tree_leaflist_size();
const tree_id tree_get_leaf(int i);
size_t tree_add_remote(const tree_node& remote);
tree_node* tree_data();
tree_id& tree_get_neighbor(int i);
void tree_2_cpu();
void tree_2_gpu();

#endif /* TREE_HPP_ */
