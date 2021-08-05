/*
 * tree.hpp
 *
 *  Created on: Jul 18, 2021
 *      Author: dmarce1
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
	array<fixed32,NDIM> pos;
};

struct tree_id {
	int proc;
	int index;
	inline bool operator==(tree_id other) const {
		return proc == other.proc && index == other.index;
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
	array<fixed32,NDIM> pos;
	array<tree_id, NCHILD> children;
	pair<int, int> proc_range;
	pair<int, int> part_range;
	pair<int, int> sink_part_range;
	size_t nactive;
	float radius;
	bool local_root;
	bool sink_leaf;
	bool source_leaf;
	size_t node_count;
	int depth;
	CUDA_EXPORT
	inline const multipole_pos* get_multipole_ptr() const {
		return (multipole_pos*) &multi;
	}
	inline int nparts() const {
		return part_range.second - part_range.first;
	}
	inline particle_global_range global_part_range() const {
		particle_global_range r;
		r.proc = proc_range.first;
		r.range = part_range;
		return r;
	}
	bool is_local() const {
		return proc_range.second - proc_range.first == 1;
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
	double flops;
	template<class A>
	void serialize(A&& a, unsigned) {
		a & flops;
		a & active_nodes;
		a & multi;
		a & id;
		a & pos;
		a & nactive;
		a & radius;
		a & node_count;
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

tree_create_return tree_create(tree_create_params params, pair<int, int> proc_range = pair<int>(0, hpx_size()), pair<int, int> part_range = pair<int>(-1, -1),
		range<double> box = unit_box<double>(), int depth = 0, bool local_root = (hpx_size() == 1));
void tree_destroy();
int tree_min_level(double theta);
const tree_node* tree_get_node(tree_id);

#endif /* TREE_HPP_ */
