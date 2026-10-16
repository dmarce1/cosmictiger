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

template <class T>
struct Todouble {};

template <>
struct Todouble<float> {
    using type = double;
};

#ifndef __CUDACC__

template <>
struct Todouble<SimdFloat> {
    using type = SimdDouble8;
};

#endif

#endif /* FLOAT2DOUBLE_HPP_ */
