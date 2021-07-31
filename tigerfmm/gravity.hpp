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
#include <tigerfmm/fixedcapvec.hpp>
#include <tigerfmm/tree.hpp>
#include <tigerfmm/kick.hpp>

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

int cpu_gravity_cc(expansion<float>&, const vector<tree_id>&, tree_id, gravity_cc_type, bool do_phi);
int cpu_gravity_cp(expansion<float>&, const vector<tree_id>&, tree_id, bool do_phi);
int cpu_gravity_pc(force_vectors&, int, tree_id, const vector<tree_id>&);
int cpu_gravity_pp(force_vectors&, int, tree_id, const vector<tree_id>&, float h);

#ifdef __CUDACC__
__device__
int cuda_gravity_cc(const cuda_kick_data&, expansion<float>&, const tree_node&, const fixedcapvec<int, MULTLIST_SIZE>&, gravity_cc_type, bool do_phi);
__device__
int cuda_gravity_cp(const cuda_kick_data&, expansion<float>&, const tree_node&, const fixedcapvec<int, PARTLIST_SIZE>&, bool do_phi);
__device__
int cuda_gravity_pc(const cuda_kick_data& data, const tree_node&, const fixedcapvec<int, MULTLIST_SIZE>&, int, bool);
__device__
int cuda_gravity_pp(const cuda_kick_data& data, const tree_node&, const fixedcapvec<int, PARTLIST_SIZE>&, int, float h, bool);
#endif
#endif /* GRAVITY_HPP_ */
