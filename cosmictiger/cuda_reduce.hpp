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
#include <cosmictiger/containers.hpp>

#ifdef __CUDACC__

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

template<int N>
__device__ inline void compute_indices_array(array<int,N>& indices, array<int,N>& total) {
	const int& tid = threadIdx.x;
	for( int n = 0; n < N; n++) {
		for (int P = 1; P < WARP_SIZE; P *= 2) {
			auto tmp = __shfl_up_sync(0xFFFFFFFF, indices[n], P);
			if (tid >= P) {
				indices[n] += tmp;
			}
		}
		total[n] = __shfl_sync(0xFFFFFFFF, indices[n], WARP_SIZE - 1);
		auto tmp = __shfl_up_sync(0xFFFFFFFF, indices[n], 1);
		if (tid >= 1) {
			indices[n] = tmp;
		} else {
			indices[n] = 0;
		}
	}
}

template<class T>
__device__ inline void shared_reduce_add(T& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number += __shfl_xor_sync(0xffffffff, number, P);
	}
}

template<class T, int N>
__device__ inline void shared_reduce_add_array(T* number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		for( int n = 0; n < N; n++ ) {
			number[n] += __shfl_xor_sync(0xffffffff, number[n], P);
		}
	}
}

template<class T, int BLOCK_SIZE>
__device__ inline void shared_reduce_add(T& number) {
	int P = WARP_SIZE;
	while (P < BLOCK_SIZE) {
		P *= 2;
	}
	const int tid = threadIdx.x;
	__shared__ T sum[BLOCK_SIZE];
	sum[tid] = number;
	__syncthreads();
	for (int bit = P / 2; bit > 0; bit /= 2) {
		const int inbr = (tid + bit) % P;
		T t;
		if (inbr < BLOCK_SIZE) {
			t = sum[tid] + sum[inbr];
		} else {
			t = sum[tid];
		}
		__syncthreads();
		sum[tid] = t;
		__syncthreads();
	}
	__syncthreads();
	number = sum[0];
	__syncthreads();
}

template<int BLOCK_SIZE>
__device__ inline void shared_reduce_max(float& number) {
	int P = WARP_SIZE;
	while (P < BLOCK_SIZE) {
		P *= 2;
	}
	const int tid = threadIdx.x;
	__shared__ float mx[BLOCK_SIZE];
	mx[tid] = number;
	__syncthreads();
	for (int bit = P / 2; bit > 0; bit /= 2) {
		const int inbr = (tid + bit) % P;
		float t;
		if( inbr < BLOCK_SIZE ) {
			t = fmaxf(mx[tid], mx[inbr]);
		} else {
			t = mx[tid];
		}
		__syncthreads();
		mx[tid] = t;
		__syncthreads();
	}
	__syncthreads();
	number = mx[0];
	__syncthreads();
}

template<int BLOCK_SIZE>
__device__ inline void shared_reduce_min(unsigned long long & number) {
	int P = WARP_SIZE;
	while (P < BLOCK_SIZE) {
		P *= 2;
	}
	const int tid = threadIdx.x;
	__shared__ unsigned long long mx[BLOCK_SIZE];
	mx[tid] = number;
	__syncthreads();
	for (int bit = P / 2; bit > 0; bit /= 2) {
		const int inbr = (tid + bit) % P;
		unsigned long long t;
		if( inbr < BLOCK_SIZE ) {
			t = min(mx[tid], mx[inbr]);
		} else {
			t = mx[tid];
		}
		__syncthreads();
		mx[tid] = t;
		__syncthreads();
	}
	__syncthreads();
	number = mx[0];
	__syncthreads();
}

template<int BLOCK_SIZE>
__device__ inline void shared_reduce_min(int & number) {
	int P = WARP_SIZE;
	while (P < BLOCK_SIZE) {
		P *= 2;
	}
	const int tid = threadIdx.x;
	__shared__ int mx[BLOCK_SIZE];
	mx[tid] = number;
	__syncthreads();
	for (int bit = P / 2; bit > 0; bit /= 2) {
		const int inbr = (tid + bit) % P;
		int t;
		if( inbr < BLOCK_SIZE ) {
			t = min(mx[tid], mx[inbr]);
		} else {
			t = mx[tid];
		}
		__syncthreads();
		mx[tid] = t;
		__syncthreads();
	}
	__syncthreads();
	number = mx[0];
	__syncthreads();
}

template<int BLOCK_SIZE>
__device__ inline void shared_reduce_max(int& number) {
	int P = WARP_SIZE;
	while (P < BLOCK_SIZE) {
		P *= 2;
	}
	const int tid = threadIdx.x;
	__shared__ int mx[BLOCK_SIZE];
	mx[tid] = number;
	__syncthreads();
	for (int bit = P / 2; bit > 0; bit /= 2) {
		const int inbr = (tid + bit) % P;
		int t;
		if( inbr < BLOCK_SIZE) {
			t = max(mx[tid], mx[inbr]);
		} else {
			t = mx[tid];
		}
		__syncthreads();
		mx[tid] = t;
		__syncthreads();
	}
	__syncthreads();
	number = mx[0];
	__syncthreads();
}

#include <cosmictiger/safe_io.hpp>

template<int BLOCK_SIZE>
__device__ inline void compute_indices(int& index, int& total) {
	int M = WARP_SIZE;
	while (M < BLOCK_SIZE) {
		M *= 2;
	}
	__shared__ int ind[BLOCK_SIZE];
	const int& tid = threadIdx.x;
	ind[tid] = index;
	__syncthreads();
	for (int P = 1; P < M; P *= 2) {
		int tmp = 0;
		if (tid >= P) {
			tmp = ind[tid - P];
		}
		__syncthreads();
		ind[tid] += tmp;
		__syncthreads();
	}
	total = ind[BLOCK_SIZE - 1];
	index = tid == 0 ? 0 : ind[tid - 1];
	__syncthreads();
}

__device__ inline void shared_reduce_min(int& number) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		number = min(number, __shfl_xor_sync(0xffffffff, number, P));
	}
}

__device__ inline void shared_reduce_min(unsigned long long& number) {
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

#endif
#endif /* CUDA_REDUCE_HPP_ */
