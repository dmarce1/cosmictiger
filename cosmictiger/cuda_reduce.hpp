/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef CUDA_REDUCE_HPP_
#define CUDA_REDUCE_HPP_

#include <cosmictiger/defs.hpp>

template<int NTHREADS>
__device__ inline void compute_indices_shmem(int& index, int& total) {
	__shared__ int indices[NTHREADS];
	const int& tid = threadIdx.x;
	indices[tid] = index;
	__syncthreads();
	for (int P = 1; P < NTHREADS; P *= 2) {
		int tmp = indices[tid];
		if (tid >= P) {
			tmp += indices[tid - P];
		}
		__syncthreads();
		indices[tid] = tmp;
	}
	__syncthreads();
	total = indices[NTHREADS - 1];
	int tmp;
	if (tid > 0) {
		tmp = indices[tid - 1];
	} else {
		tmp = 0;
	}
	__syncthreads();
	indices[tid] = tmp;
}

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

template<class T, int BLOCK_SIZE>
__device__ inline void shared_reduce_add(T& number) {
	const int tid = threadIdx.x;
	__shared__ T sum[BLOCK_SIZE];
	sum[tid] = number;
	__syncthreads();
	for (int bit = BLOCK_SIZE / 2; bit > 0; bit /= 2) {
		const int inbr = (tid + bit) % BLOCK_SIZE;
		const T t = sum[tid] + sum[inbr];
		__syncthreads();
		sum[tid] = t;
		__syncthreads();
	}
	number = sum[tid];
}

#include <cosmictiger/safe_io.hpp>

template<int BLOCK_SIZE>
__device__ inline void compute_indices(int& index, int& total) {
	__shared__ int sum[BLOCK_SIZE];
	const int& tid = threadIdx.x;
	sum[tid] = index;
	__syncthreads();
	for (int P = 1; P < BLOCK_SIZE; P *= 2) {
		int tmp = 0;
		if (tid >= P) {
			tmp = sum[tid - P];
		}
		__syncthreads();
		sum[tid] += tmp;
		__syncthreads();
	}
	total = sum[BLOCK_SIZE - 1];
	index = tid == 0 ? 0 : sum[tid - 1];
}

__device__ inline void shared_reduce_min(int& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number = min(number, __shfl_xor_sync(0xffffffff, number, P));
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
