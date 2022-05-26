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

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/ewald_indices.hpp>

#ifdef USE_CUDA

void cuda_malloc(void** ptr, size_t size, const char* file, int line ) {
	static mutex_type mutex;
	std::lock_guard<mutex_type> lock(mutex);
	double free_mem = cuda_free_mem();
	double total_mem = cuda_total_mem();
	if( (free_mem - (double) size) / total_mem < 0.15 ) {
		PRINT( "Attempt to allocate %li bytes on rank %i in %s on line %i leaves less than 15%% memory\n", size, hpx_rank(), file, line);
		abort();
	}
	CUDA_CHECK(cudaMalloc(ptr,size));
}

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
	cuda_set_device();
	CUDA_CHECK(cudaMemGetInfo(&free, &total));
	return total;
}

size_t cuda_free_mem() {
	size_t total;
	size_t free;
	cuda_set_device();
	CUDA_CHECK(cudaMemGetInfo(&free, &total));
	return free;
}

int cuda_smp_count() {
	int count;
	cuda_set_device();
	CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, cuda_get_device()));
	return count;

}

cudaStream_t cuda_get_stream() {
	cudaStream_t stream;
	cuda_set_device();
	CUDA_CHECK(cudaStreamCreate(&stream));
	return stream;
}

void cuda_stream_synchronize(cudaStream_t stream) {
	int device;
	CUDA_CHECK(cudaGetDevice(&device));
	while(cudaStreamQuery(stream) != cudaSuccess) {
		hpx_yield();
	}
	CUDA_CHECK(cudaSetDevice(device));
}

void cuda_end_stream(cudaStream_t stream) {
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

void cuda_init() {
	cuda_set_device();
	CUDA_CHECK(cudaDeviceReset());
	size_t value = STACK_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitStackSize));
	value = 11;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth));

	if (value != STACK_SIZE) {
		THROW_ERROR("Unable to set stack size to %li\n", STACK_SIZE);
	}
	cuda_mem_init(HEAP_SIZE);
}

#endif
