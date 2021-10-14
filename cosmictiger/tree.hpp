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

size_t tree_nodes_size();

struct tree_id {
	int proc;
	int index;
	inline bool operator==(tree_id other) const {
		return proc == other.proc && index == other.index;
	}
	inline bool operator!=(tree_id other) const {
		return proc != other.proc || index != other.index;
	}
#ifndef __CUDA_ARCH__
	inline tree_id operator=(const tree_id& other) {
		if (proc == hpx_rank())
			ASSERT(index < tree_nodes_size());
		proc = other.proc;
		index = other.index;
		return *this;
	}
	inline tree_id operator=(tree_id&& other) {
		if (proc == hpx_rank())
			ASSERT(index < tree_nodes_size());
		proc = other.proc;
		index = other.index;
		return *this;
	}
	inline tree_id(const tree_id& other) {
		if (proc == hpx_rank())
			ASSERT(index < tree_nodes_size());
		proc = other.proc;
		index = other.index;
	}
	inline tree_id(tree_id&& other) {
		if (proc == hpx_rank())
			ASSERT(index < tree_nodes_size());
		proc = other.proc;
		index = other.index;
	}
	inline tree_id() = default;
	template<class A>
	void serialize(A&& a, unsigned) {
		if (proc == hpx_rank())
			ASSERT(index < tree_nodes_size());
		a & proc;
		a & index;
	}
#endif
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
	size_t nactive;
	float radius;
	bool local_root;
	bool sink_leaf;
	bool source_leaf;
	size_t node_count;
	size_t active_nodes;
	int depth;CUDA_EXPORT
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
		arc & nactive;
		arc & radius;
		arc & local_root;
		arc & sink_leaf;
		arc & source_leaf;
		arc & node_count;
		arc & sink_part_range;
		arc & depth;
	}
};

struct tree_create_return {
	multipole<float> multi;
	array<fixed32, NDIM> pos;
	tree_id id;
	size_t nactive;
	size_t active_nodes;
	float radius;
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
		a & multi;
		a & id;
		a & pos;
		a & nactive;
		a & radius;
		a & node_count;
		a & flops;
		a & max_depth;
		a & min_depth;
	}
};

struct tree_create_params {
	int min_rung;
	double theta;
	int min_level;
	tree_create_params() = default;
	tree_create_params(int min_rung, double theta);
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & min_rung;
		arc & theta;
		arc & min_level;
	}
};

tree_create_return tree_create(tree_create_params params, size_t key = 1, pair<int, int> proc_range = pair<int>(0, hpx_size()), pair<part_int> part_range =
		pair<part_int>(-1, -1), range<double> box = unit_box<double>(), int depth = 0, bool local_root = (hpx_size() == 1));
void tree_destroy();
int tree_min_level(double theta);
const tree_node* tree_get_node(tree_id);

#endif /* TREE_HPP_ */
