#pragma once

#include <cosmictiger/defs.hpp>
#include <math.h>
#include <cosmictiger/cuda.hpp>

#include <cosmictiger/fixed.hpp>


template<class T>
CUDA_EXPORT inline T sqr(T a, T b, T c) {
	return fmaf(a, a, fmaf(b, b, sqr(c)));
}
__global__
void generate_random_vectors(fixed32* x, fixed32* y, fixed32* z, size_t N, int seed);

CUDA_DEVICE inline float erfcexp(float x, float *e) {				// 18 + FLOP_DIV + FLOP_EXP
	const float p(0.3275911f);
	const float a1(0.254829592f);
	const float a2(-0.284496736f);
	const float a3(1.421413741f);
	const float a4(-1.453152027f);
	const float a5(1.061405429f);
	const float t1 = 1.f / FMA(p, x, 1.f);			            // FLOP_DIV + 2
	const float t2 = t1 * t1;											// 1
	const float t3 = t2 * t1;											// 1
	const float t4 = t2 * t2;											// 1
	const float t5 = t2 * t3;											// 1
	*e = EXP(-x * x);												  // 2 + FLOP_EXP
	return FMA(a1, t1, FMA(a2, t2, FMA(a3, t3, FMA(a4, t4, a5 * t5)))) * *e; 			// 10
}

/*
 * math.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_MATH_HPP_
#define GPUTIGER_MATH_HPP_

#include <nvfunctional>
#include <cstdio>
#include <cstdint>
#include <cosmictiger/vector.hpp>

/*#define POW(a,b) powf(a,b)
 #define LOG(a) logf(a)
 #define EXP(a) expf(a)
 #define SQRT(a) sqrtf(a)*/
#define COS(a) cosf(a)
#define SIN(a) sinf(a)
//#define SINCOS(a,b,c) sincosf(a,b,c)

#ifdef __CUDACC__

template<class FUNC, class REAL>
__global__
void integrate(FUNC *fptr, REAL a, REAL b, REAL* result, REAL toler) {
	const int block_size = blockDim.x;
	int thread = threadIdx.x;
	const auto& f = *fptr;
	__syncthreads();
	__shared__ REAL* sums1;
	__shared__ REAL* sums2;
	__shared__ REAL* error_ptr;
	if (thread == 0) {
		sums1 = new REAL[512];
		sums2 = new REAL[512];
		error_ptr = new REAL;
	}
	__syncthreads();
	REAL& err = *error_ptr;
	int N = 6 * ((block_size - 1) / 6) + 1;
	REAL sum1, sum2;
	do {
		sum1 = REAL(0);
		sum2 = REAL(0);
		REAL dx = (b - a) / REAL(N - 1);
		sum1 = sum2 = 0.0;
		constexpr REAL wt1[3] = {6.0 / 8.0, 9.0 / 8.0, 9.0 / 8.0};
		constexpr REAL wt2[6] = {82.0 / 140.0, 216.0 / 140.0, 27.0 / 140.0, 272.0 / 140.0, 27.0 / 140.0, 216.0 / 140.0};
		for (int i = thread; i < N; i += block_size) {
			REAL x = a + REAL(i) * dx;
			REAL this_f = f(x);
			sum2 += this_f * dx * wt2[i % 6] * (i == 0 || i == N - 1 ? REAL(0.5) : REAL(1));
			sum1 += this_f * dx * wt1[i % 3] * (i == 0 || i == N - 1 ? REAL(0.5) : REAL(1));
		}
		sums1[thread] = sum1;
		sums2[thread] = sum2;
		__syncthreads();
		for (int M = block_size / 2; M >= 1; M /= 2) {
			if (thread < M) {
				sums1[thread] += sums1[thread + M];
				sums2[thread] += sums2[thread + M];
			}
			__syncthreads();
		}
		if (thread == 0) {
			sum1 = sums1[0];
			sum2 = sums2[0];
			if (sum2 != REAL(0)) {
				err = abs((sum2 - sum1) / sum2);
			} else if (sum1 != REAL(0.0)) {
				err = abs((sum1 - sum2) / sum1);
			} else {
				err = REAL(0.0);
			}
			*result = sum2;
		}
		N = 2 * (N - 1) + 1;
		__syncthreads();
	}while (err > toler);
	__syncthreads();
	if (thread == 0) {
		delete[] sums1;
		delete[] sums2;
		delete error_ptr;
	}
	__syncthreads();

}

CUDA_EXPORT inline float pow2(float r) {
	return r * r;
}

#endif

template<class T>
CUDA_EXPORT inline T round_up(T num, T mod) {
	return ((num - 1) / mod + 1) * mod;
}

__global__
void generate_random_vectors(fixed32* x, fixed32* y, fixed32* z, size_t N, int seed);


#ifndef __CUDA_ARCH__
inline double erfcexp(double x, double* exp) {
	double y = std::erfc(x);
	*exp = std::exp(-x * x);
	return y;
}

inline void sincos(double x, double* s, double *c ) {
	*s = sin(x);
	*c = cos(x);
}
#else
CUDA_EXPORT
double erfcexp(double x, double* exp);
#endif

#endif /* GPUTIGER_MATH_HPP_ */

