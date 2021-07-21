/*
 * gravity.hpp
 *
 *  Created on: Jul 19, 2021
 *      Author: dmarce1
 */

#ifndef GRAVITY_HPP_
#define GRAVITY_HPP_

#include <tigerfmm/cuda.hpp>
#include <tigerfmm/fixed.hpp>
#include <tigerfmm/tree.hpp>

CUDA_EXPORT inline float distance(fixed32 a, fixed32 b) {
	return (fixed<int32_t>(a) - fixed<int32_t>(b)).to_float();
}

struct force_vectors {
	vector<float> phi;
	vector<float> gx;
	vector<float> gy;
	vector<float> gz;
	force_vectors() = default;
	force_vectors(int sz) {
		phi.resize(sz);
		gx.resize(sz);
		gy.resize(sz);
		gz.resize(sz);
	}
};

enum gravity_cc_type {
	GRAVITY_CC_DIRECT, GRAVITY_CC_EWALD
};

void gravity_cc(expansion<float>&, const vector<tree_id>&, tree_id, gravity_cc_type, bool do_phi);
void gravity_cp(expansion<float>&, const vector<tree_id>&, tree_id, bool do_phi);
void gravity_pc(force_vectors&, int, tree_id, const vector<tree_id>&);
void gravity_pp(force_vectors&, int, tree_id, const vector<tree_id>&);

#endif /* GRAVITY_HPP_ */
