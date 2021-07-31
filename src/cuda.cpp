#include <tigerfmm/cuda.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/ewald_indices.hpp>


HPX_PLAIN_ACTION(cuda_cycle_devices);

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
#ifndef NDEBUG
	size_t value = STACK_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitStackSize));
	if (value != STACK_SIZE) {
		THROW_ERROR("Unable to set stack size to %li\n", STACK_SIZE);
	}
#endif

}


void cuda_cycle_devices() {
	vector<hpx::future<void>> futs;
	for( auto& c : hpx_children()) {
		futs.push_back(hpx::async<cuda_cycle_devices_action>(c));
	}

	particles_unpin();
	cuda_init();
	ewald_const::init();
	particles_pin();

	hpx::wait_all(futs.begin(), futs.end());
}

