/*
 * analytic.hpp
 *
 *  Created on: Jul 22, 2021
 *      Author: dmarce1
 */

#ifndef ANALYTIC_HPP_
#define ANALYTIC_HPP_


#include <tigerfmm/containers.hpp>
#include <tigerfmm/fixed.hpp>

std::pair<vector<double>, array<vector<double>, NDIM>> gravity_analytic_call_kernel(const vector<fixed32>& sinkx,
		const vector<fixed32>& sinky, const vector<fixed32>& sinkz);
void analytic_compare(int Nsamples);



#endif /* ANALYTIC_HPP_ */
