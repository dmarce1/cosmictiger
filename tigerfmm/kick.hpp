/*
 * kick.hpp
 *
 *  Created on: Jul 19, 2021
 *      Author: dmarce1
 */

#ifndef KICK_HPP_
#define KICK_HPP_

#include <tigerfmm/tree.hpp>

struct kick_return {
	char max_rung;
	double flops;
	double pot;
	double fx;
	double fy;
	double fz;
	double fnorm;
	int nactive;
	CUDA_EXPORT
	kick_return() {
		max_rung = 0;
		flops = 0.0;
		pot = 0.0;
		fx = 0.0;
		fy = 0.0;
		fz = 0.0;
		fnorm = 0.0;
		nactive = 0;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & max_rung;
		arc & flops;
		arc & pot;
		arc & fx;
		arc & fy;
		arc & fz;
		arc & fnorm;
		arc & nactive;
	}
};

struct kick_params {
	int min_rung;
	float a;
	float t0;
	float theta;
	float h;
	float eta;
	float GM;
	bool save_force;
	bool first_call;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & min_rung;
		arc & a;
		arc & t0;
		arc & theta;
		arc & first_call;
		arc & h;
		arc & eta;
		arc & GM;
		arc & save_force;
	}
};

struct kick_workitem {
	expansion<float> L;
	array<fixed32, NDIM> pos;
	tree_id self;
	vector<tree_id> dchecklist;
	vector<tree_id> echecklist;
};

#ifndef __CUDACC__
hpx::future<kick_return> kick(kick_params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist);
#endif
void kick_show_timings();
vector<kick_return, pinned_allocator<kick_return>> cuda_execute_kicks(kick_params params, fixed32*, fixed32*, fixed32*, tree_node*, vector<kick_workitem> workitems, cudaStream_t stream);



#endif /* KICK_HPP_ */
