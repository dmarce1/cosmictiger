/*
 * sedov.hpp
 *
 *  Created on: Mar 15, 2022
 *      Author: dmarce1
 */

#ifndef SEDOV_HPP_
#define SEDOV_HPP_

namespace sedov {

void solution(double time, double r, double rmax, double& d, double& velocity, double& p, int ndim,
    double gamma = 5.0 / 3.0);
}

#endif /* SEDOV_HPP_ */
