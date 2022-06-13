/*
 * kahansum.hpp
 *
 *  Created on: Jun 12, 2022
 *      Author: dmarce1
 */

#ifndef KAHANSUM_HPP_
#define KAHANSUM_HPP_

#include <cosmictiger/cuda.hpp>

template<class T>
class kahan_sum {
	T sum;
	T cor;
public:
	operator T() {
		return sum + cor;
	}
	kahan_sum& operator=(const T& other) {
		sum = other;
		cor = T(0);
		return *this;
	}
	CUDA_EXPORT
	kahan_sum& operator+=(const T& other) {
		const T y = other - cor;
		const volatile T t = sum + y;
		const volatile T z = t - sum;
		cor = z - y;
		return *this;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & sum;
		arc & cor;
	}
};

#endif /* KAHANSUM_HPP_ */
