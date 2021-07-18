/*
 * math.hpp
 *
 *  Created on: Jul 18, 2021
 *      Author: dmarce1
 */

#ifndef MATH_HPP_
#define MATH_HPP_

#include <tigerfmm/cuda.hpp>

template<class T>
CUDA_EXPORT inline T sqr(const T& a ) {
	return a * a;
}

template<class T>
CUDA_EXPORT inline T sqr(const T& a, const T& b, const T& c ) {
	return a * a + b * b + c * c;
}

#endif /* MATH_HPP_ */
