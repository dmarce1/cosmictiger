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

#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>

#ifdef __CUDACC__

__device__ inline void computeIndices(int& index, int& total) {
    int const& tid = threadIdx.x;
    for (int p = 1; p < WARP_SIZE; p *= 2) {
        auto tmp = __shfl_up_sync(0xFFFFFFFF, index, p);
        if (tid >= p) {
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

template <int nValue>
__device__ inline void computeIndicesArray(Array<int, nValue>& indices, Array<int, nValue>& total) {
    int const& tid = threadIdx.x;
    for (int n = 0; n < nValue; n++) {
        for (int p = 1; p < WARP_SIZE; p *= 2) {
            auto tmp = __shfl_up_sync(0xFFFFFFFF, indices[n], p);
            if (tid >= p) {
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

template <class T>
__device__ inline void sharedReduceAdd(T& number) {
    for (int p = warpSize / 2; p >= 1; p /= 2) {
        number += __shfl_xor_sync(0xffffffff, number, p);
    }
}

template <class T, int nValue>
__device__ inline void sharedReduceAddArray(Array<T, nValue>& number) {
    for (int p = warpSize / 2; p >= 1; p /= 2) {
        for (int n = 0; n < nValue; n++) {
            number[n] += __shfl_xor_sync(0xffffffff, number[n], p);
        }
    }
}

template <class T, int blockSize>
__device__ inline void sharedReduceAdd(T& number) {
    int p = WARP_SIZE;
    while (p < blockSize) {
        p *= 2;
    }
    int const tid = threadIdx.x;
    __shared__ T sum[blockSize];
    sum[tid] = number;
    __syncthreads();
    for (int bit = p / 2; bit > 0; bit /= 2) {
        int const inbr = (tid + bit) % p;
        T t;
        if (inbr < blockSize) {
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

template <int blockSize>
__device__ inline void sharedReduceMax(float& number) {
    int p = WARP_SIZE;
    while (p < blockSize) {
        p *= 2;
    }
    int const tid = threadIdx.x;
    __shared__ float mx[blockSize];
    mx[tid] = number;
    __syncthreads();
    for (int bit = p / 2; bit > 0; bit /= 2) {
        int const inbr = (tid + bit) % p;
        float t;
        if (inbr < blockSize) {
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

template <int blockSize>
__device__ inline void sharedReduceMin(unsigned long long& number) {
    int p = WARP_SIZE;
    while (p < blockSize) {
        p *= 2;
    }
    int const tid = threadIdx.x;
    __shared__ unsigned long long mx[blockSize];
    mx[tid] = number;
    __syncthreads();
    for (int bit = p / 2; bit > 0; bit /= 2) {
        int const inbr = (tid + bit) % p;
        unsigned long long t;
        if (inbr < blockSize) {
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

template <int blockSize>
__device__ inline void sharedReduceMin(int& number) {
    int p = WARP_SIZE;
    while (p < blockSize) {
        p *= 2;
    }
    int const tid = threadIdx.x;
    __shared__ int mx[blockSize];
    mx[tid] = number;
    __syncthreads();
    for (int bit = p / 2; bit > 0; bit /= 2) {
        int const inbr = (tid + bit) % p;
        int t;
        if (inbr < blockSize) {
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

template <int blockSize>
__device__ inline void sharedReduceMax(int& number) {
    int p = WARP_SIZE;
    while (p < blockSize) {
        p *= 2;
    }
    int const tid = threadIdx.x;
    __shared__ int mx[blockSize];
    mx[tid] = number;
    __syncthreads();
    for (int bit = p / 2; bit > 0; bit /= 2) {
        int const inbr = (tid + bit) % p;
        int t;
        if (inbr < blockSize) {
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

template <int blockSize>
__device__ inline void computeIndices(int& index, int& total) {
    int m = WARP_SIZE;
    while (m < blockSize) {
        m *= 2;
    }
    __shared__ int ind[blockSize];
    int const& tid = threadIdx.x;
    ind[tid] = index;
    __syncthreads();
    for (int p = 1; p < m; p *= 2) {
        int tmp = 0;
        if (tid >= p) {
            tmp = ind[tid - p];
        }
        __syncthreads();
        ind[tid] += tmp;
        __syncthreads();
    }
    total = ind[blockSize - 1];
    index = tid == 0 ? 0 : ind[tid - 1];
    __syncthreads();
}

__device__ inline void sharedReduceMin(int& number) {
    for (int p = warpSize / 2; p >= 1; p /= 2) {
        number = min(number, __shfl_xor_sync(0xffffffff, number, p));
    }
}

__device__ inline void sharedReduceMin(unsigned long long& number) {
    for (int p = warpSize / 2; p >= 1; p /= 2) {
        number = min(number, __shfl_xor_sync(0xffffffff, number, p));
    }
}

__device__ inline void sharedReduceMin(float& number) {
    for (int p = warpSize / 2; p >= 1; p /= 2) {
        number = fminf(number, __shfl_xor_sync(0xffffffff, number, p));
    }
}

__device__ inline void sharedReduceMax(int& number) {
    for (int p = warpSize / 2; p >= 1; p /= 2) {
        number = max(number, __shfl_xor_sync(0xffffffff, number, p));
    }
}

#endif
#endif /* CUDA_REDUCE_HPP_ */
