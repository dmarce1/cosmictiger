#include <tigerfmm/cuda.hpp>
#include <tigerfmm/hpx.hpp>

void cuda_set_device() {
	int count;
	CUDA_CHECK(cudaGetDeviceCount(&count));
	const int device_num = hpx_rank() % count;
	CUDA_CHECK(cudaSetDevice(device_num));
}


