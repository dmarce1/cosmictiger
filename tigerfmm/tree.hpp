/*
 * tree.hpp
 *
 *  Created on: Jul 18, 2021
 *      Author: dmarce1
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <tigerfmm/fixed.hpp>
#include <tigerfmm/fmm_kernels.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/range.hpp>

struct tree_id {
	int proc;
	int index;
	inline bool operator==(tree_id other) const {
		return proc == other.proc && index == other.index;
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
	array<tree_id, NCHILD> children;
	array<fixed32, NDIM> pos;
	pair<int, int> proc_range;
	pair<int, int> part_range;
	size_t nactive;
	float radius;
	bool local_root;
	bool is_leaf() const {
		return children[0].index == -1;
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
	}
};

struct tree_create_return {
	multipole<float> multi;
	array<fixed32, NDIM> pos;
	tree_id id;
	size_t nactive;
	float radius;
	template<class A>
	void serialize(A&& a, unsigned) {
		a & multi;
		a & id;
		a & pos;
		a & nactive;
		a & radius;
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
