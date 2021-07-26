/*
 * cuda_reduce.hpp
 *
 *  Created on: Jul 26, 2021
 *      Author: dmarce1
 */

#ifndef CUDA_REDUCE_HPP_
#define CUDA_REDUCE_HPP_

#include <tigerfmm/defs.hpp>

__device__ inline void compute_indices(int& index, int& total) {
	const int& tid = threadIdx.x;
	for (int P = 1; P < WARP_SIZE; P *= 2) {
		auto tmp = __shfl_up_sync(0xFFFFFFFF, index, P);
		if (tid >= P) {
			index += tmp;
		}
	}
	total = __shfl_sync(0xFFFFFFFF, index, WARP_SIZE - 1);
	auto tmp = __shfl_up_sync(0xFFFFFFFF, index, 1);
	if (tid >= 1) {
		index = tmp;
	} else {
		index = 0;
	}
}

template<class T>
__device__ inline void shared_reduce_add(T& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number += __shfl_xor_sync(0xffffffff, number, P);
	}
}

__device__ inline void shared_reduce_min(float& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number = fminf(number, __shfl_xor_sync(0xffffffff, number, P));
	}
}

__device__ inline void shared_reduce_max(int& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number = max(number, __shfl_xor_sync(0xffffffff, number, P));
	}
}


#endif /* CUDA_REDUCE_HPP_ */
