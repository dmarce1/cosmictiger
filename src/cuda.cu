#include <tigerfmm/cuda.hpp>
#include <tigerfmm/hpx.hpp>

void cuda_set_device() {
	int count;
	CUDA_CHECK(cudaGetDeviceCount(&count));
	const int device_num = hpx_rank() % count;
	CUDA_CHECK(cudaSetDevice(device_num));
}

int cuda_get_device() {
	int count;
	CUDA_CHECK(cudaGetDeviceCount(&count));
	const int device_num = hpx_rank() % count;
	return device_num;
}

size_t cuda_total_mem() {
	size_t total;
	size_t free;
	CUDA_CHECK(cudaMemGetInfo(&free, &total));
	return total;
}

size_t cuda_free_mem() {
	size_t total;
	size_t free;
	CUDA_CHECK(cudaMemGetInfo(&free, &total));
	return free;
}

int cuda_smp_count() {
	int count;
	CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, cuda_get_device()));
	return count;

}

#define HEAP_SIZE 5
#define STACK_SIZE (16*1024)

cudaStream_t cuda_get_stream() {
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	return stream;
}

void cuda_end_stream(cudaStream_t stream) {
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

void cuda_init() {
	cuda_set_device();
	CUDA_CHECK(cudaDeviceReset());
	size_t heap_size = size_t(cuda_total_mem()) * HEAP_SIZE / 100;
	size_t value = 1;
	while (value < heap_size) {
		value *= 2;
	}
	heap_size = value;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize));
	if (value != heap_size) {
		THROW_ERROR("Unable to set heap to %li\n", heap_size);
	}
	value = STACK_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitStackSize));
	if (value != STACK_SIZE) {
		THROW_ERROR("Unable to set stack size to %li\n", STACK_SIZE);
	}

}
