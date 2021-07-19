/*
 * tree.hpp
 *
 *  Created on: Jul 18, 2021
 *      Author: dmarce1
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <tigerfmm/fmm_kernels.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/range.hpp>

struct tree_id {
	int proc;
	int index;
	template<class A>
	void serialize(A&& a, unsigned) {
		a & proc;
		a & index;
	}
};

struct tree_node {
	multipole<float> multi;
	array<tree_id, NCHILD> children;
	array<fixed32, NDIM> pos;
	pair<int, int> proc_range;
	pair<int, int> part_range;
	size_t nactive;
	size_t morton_id;
	float radius;
	int index;
	bool local_root;
	tree_node() {
		morton_id = 0xFFFFFFFFFFFFFFFFLL;
	}
};

struct tree_create_return {
	multipole<float> multi;
	tree_id id;
	array<fixed32, NDIM> pos;
	float radius;
	template<class A>
	void serialize(A&& a, unsigned) {
		a & multi;
		a & id;
		a & pos;
		a & radius;
	}
};

tree_create_return tree_create(pair<int, int> proc_range = pair<int>(0, hpx_size()), pair<int, int> part_range = pair<int>(0, 0),
		range<double> box = unit_box<double>(), int depth = 0, bool local_root = (hpx_size() == 1));
void tree_destroy();

#endif /* TREE_HPP_ */
