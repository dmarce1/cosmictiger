/*
 * float2double.hpp
 *
 *  Created on: Aug 19, 2022
 *      Author: dmarce1
 */

#ifndef FLOAT23DOUBLE_HPP_
#define FLOAT23DOUBLE_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/simd.hpp>

template<class T>
struct todouble {
};

template<>
struct todouble<float> {
	using type = double;
};

#ifndef __CUDACC__

template<>
struct todouble<simd_float> {
	using type = simd_double8;
};


#endif



#endif /* FLOAT2DOUBLE_HPP_ */
