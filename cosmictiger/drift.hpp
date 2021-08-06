/*
 * drift.hpp
 *
 *  Created on: Jul 12, 2021
 *      Author: dmarce1
 */

#ifndef DRIFT_HPP_
#define DRIFT_HPP_

#include <cosmictiger/defs.hpp>


struct drift_return {
	double kin;
	double momx;
	double momy;
	double momz;
	int nmapped;
	template<class Arc>
	void serialize(Arc&& a, unsigned) {
		a & kin;
		a & momx;
		a & momy;
		a & momz;
		a & nmapped;
	}
};

drift_return drift(double scale, double t, double dt);


#endif /* DRIFT_HPP_ */