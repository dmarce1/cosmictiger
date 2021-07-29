

#ifndef COSMICTIGER_CUDA_HPP_
#define COSMICTIGER_CUDA_HPP_


#include <cuda_runtime.h>
#include <cufft.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


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
		assert(0);
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
    } while (assumed != old);

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


#endif /* COSMICTIGER_CUDA_HPP_ */
