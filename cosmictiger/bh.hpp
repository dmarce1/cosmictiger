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
	pair<int> parts;
	float mass;
	float radius;
	bh_tree_node() {
		children[LEFT] = children[RIGHT] = -1;
		mass = 0;
	}
};


struct bh_source {
	array<float, NDIM> x;
	float m;
};


vector<float> bh_evaluate_potential(const vector<array<fixed32, NDIM>>& x);

#endif /* BH_HPP_ */
