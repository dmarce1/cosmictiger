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


vector<float> bh_evaluate_potential_fixed(const vector<array<fixed32, NDIM>>& x);
vector<float> bh_evaluate_potential(vector<array<float, NDIM>>& x);
vector<float> bh_evaluate_points(vector<array<float, NDIM>>& y, vector<array<float, NDIM>>& x);



#endif /* BH_HPP_ */
