/*
 * bh.hpp
 *
 *  Created on: Aug 12, 2021
 *      Author: dmarce1
 */

#ifndef BH_HPP_
#define BH_HPP_

#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/range.hpp>

struct bh_tree_node {
	array<float, NDIM> pos;
	array<int, NDIM> children;
	float count;
	float radius;
	bh_tree_node() {
		children[LEFT] = children[RIGHT] = -1;
		count = 0;
	}
};


vector<float> bh_evaluate_potential(const vector<array<fixed32, NDIM>>& x);
vector<float> bh_cuda_tree_evaluate(const vector<bh_tree_node>& nodes, const vector<array<float, NDIM>>& sinks, float theta);

#endif /* BH_HPP_ */
