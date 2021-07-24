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
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & max_rung;
		arc & flops;
	}
};

struct kick_params {
	int min_rung;
	double a;
	double t0;
	double theta;
	bool first_call;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & min_rung;
		arc & a;
		arc & t0;
		arc & theta;
		arc & first_call;
	}
};


kick_return kick(kick_params, expansion<float> L, array<fixed32,NDIM> pos, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist );
void kick_show_timings();

#endif /* KICK_HPP_ */
