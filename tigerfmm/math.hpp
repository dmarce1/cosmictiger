/*
 * math.hpp
 *
 *  Created on: Jul 18, 2021
 *      Author: dmarce1
 */

#ifndef MATH_HPP_
#define MATH_HPP_

#include <tigerfmm/cuda.hpp>
#include <tigerfmm/simd.hpp>

template<class T>
CUDA_EXPORT inline T sqr(const T& a) {
	return a * a;
}

template<class T>
CUDA_EXPORT inline T sqr(const T& a, const T& b, const T& c) {
	return a * a + b * b + c * c;
}

template<class T>
CUDA_EXPORT void constrain_range(T& x) {
	while (x >= T(1)) {
		x -= T(1);
	}
	while (x < T(0)) {
		x += T(1);
	}
}

__device__ inline void erfcexp(float x, float* ec, float *ex) {				// 18 + FLOP_DIV + FLOP_EXP
	const float p(0.3275911f);
	const float a1(0.254829592f);
	const float a2(-0.284496736f);
	const float a3(1.421413741f);
	const float a4(-1.453152027f);
	const float a5(1.061405429f);
	const float t1 = 1.f / fma(p, x, 1.f);			            // FLOP_DIV + 2
	const float t2 = t1 * t1;											// 1
	const float t3 = t2 * t1;											// 1
	const float t4 = t2 * t2;											// 1
	const float t5 = t2 * t3;											// 1
	*ex = expf(-x * x);												  // 2 + FLOP_EXP
	*ec = fmaf(a1, t1, fmaf(a2, t2, fmaf(a3, t3, fmaf(a4, t4, a5 * t5)))) * *ex; 			// 10
}

template<class T>
CUDA_EXPORT
inline T round_up(T a, T b) {
	if( a > 0 ) {
		return (((a - 1) / b) + 1) * b;
	} else {
		return 0;
	}
}

template<class T>
CUDA_EXPORT
inline T round_down(T a, T b) {
	return (a / b) * b;
}



CUDA_EXPORT inline float anytrue(float x) {
	return x;
}

#ifndef __CUDACC__

inline float anytrue(const simd_float& x) {
	return x.sum();
}

#endif

#endif /* MATH_HPP_ */
