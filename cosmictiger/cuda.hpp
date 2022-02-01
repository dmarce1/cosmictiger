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

#ifndef COSMICTIGER_CUDA_HPP_
#define COSMICTIGER_CUDA_HPP_

#include <cosmictiger/assert.hpp>
#include <cosmictiger/defs.hpp>

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdlib.h>
#include <stdio.h>

#define CUDA_MALLOC(a,b) cuda_malloc2(a,b,__FILE__,__LINE__)
#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

/* error checker from https://forums.developer.nvidia.com/t/cufft-error-handling/29231 */
static const char *_cudaGetErrorEnum(cufftResult error) {
	switch (error) {
		case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

		case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

		case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

		case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

		case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

		case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

		case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

		case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

		case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

		case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}

inline void _cuda_fft_check(cufftResult err, const char *file, const int line) {
	if (CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n", file, line, err,
				_cudaGetErrorEnum(err));
		cudaDeviceReset();
		ASSERT(0);
	}
}

#define CUDA_FFT_CHECK(a) _cuda_fft_check(a,__FILE__,__LINE__)

void cuda_set_device();

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
/**** TAKEN FROM THE CUDA DOCUMENTATION *****/
inline __device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
	(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
						__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	}while (assumed != old);

	return __longlong_as_double(old);
}
#endif

#endif

#define CUDA_EXPORT __host__ __device__

#include <thrust/system/cuda/experimental/pinned_allocator.h>

template<class T>
using pinned_allocator = thrust::system::cuda::experimental::pinned_allocator< T >;

void cuda_set_device();
size_t cuda_free_mem();
int cuda_smp_count();
size_t cuda_free_mem();
size_t cuda_total_mem();
int cuda_smp_count();
void cuda_init();
cudaStream_t cuda_get_stream();
void cuda_end_stream(cudaStream_t stream);
int cuda_get_device();
void cuda_stream_synchronize(cudaStream_t stream);

void cuda_malloc(void** ptr, size_t size, const char* file, int line );

template<class T>
void cuda_malloc2(T** ptr, size_t size, const char* file, int line ) {
	cuda_malloc((void**)ptr,size,file,line);
}

/*********************************************************************************/
//From https://localcoder.org/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
#ifdef __CUDACC__
__device__ static float atomicMax(float* address, float val)
{
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	}while (assumed != old);
	return __int_as_float(old);
}
__device__ static float atomicMin(float* address, float val)
{
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fminf(val, __int_as_float(assumed))));
	}while (assumed != old);
	return __int_as_float(old);
}
#endif
/*********************************************************************************/
#else

#include <memory>

template<class T>
using pinned_allocator = std::allocator< T >;

#define CUDA_EXPORT

#endif

#endif /* COSMICTIGER_CUDA_HPP_ */
