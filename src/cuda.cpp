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
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/ewald_indices.hpp>

#ifdef USE_CUDA


HPX_PLAIN_ACTION(cuda_init);

static vector<int> mydevices;

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

void cuda_set_device(int i) {
	CUDA_CHECK(cudaSetDevice(mydevices[i]));
}

int cuda_device_count() {
	return mydevices.size();
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
	CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, mydevices[0]));
	return count;

}

cudaStream_t cuda_get_stream(int dvc) {
	cudaStream_t stream;
	cuda_set_device(dvc);
	CUDA_CHECK(cudaStreamCreate(&stream));
	return stream;
}

void cuda_end_stream(cudaStream_t stream) {
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

void cuda_stream_synchronize(cudaStream_t stream) {
	bool done;
	int device;
	CUDA_CHECK(cudaGetDevice(&device));
	do {
		if( cudaStreamQuery(stream)==cudaSuccess) {
			done = true;
		} else {
			done = false;
			hpx_yield();
			CUDA_CHECK(cudaSetDevice(device));
		}
	}while(!done);
}

const vector<int>& cuda_get_devices() {
	return mydevices;
}

int cuda_get_device_id(int i) {
	return mydevices[i];
}


void cuda_init(int procs_per_node) {
	vector<hpx::future<void>> futs;
	for( auto& c : hpx_children()) {
		futs.push_back(hpx::async<cuda_init_action>(c,procs_per_node));
	}
	int device_count;
	CUDA_CHECK(cudaGetDeviceCount ( &device_count) );
	const int local_num = hpx_rank() % procs_per_node;
	const int device_begin = local_num * device_count / procs_per_node;
	const int device_end = (local_num + 1) * device_count / procs_per_node;
	for( int i = device_begin; i < device_end; i++) {
		mydevices.push_back(i);
	}
	if( hpx_rank() == 0 ) {
		PRINT( "Running with %i processes per node and %i GPUs per process\n", procs_per_node, mydevices.size());
	}
	for( int dvc = 0; dvc < cuda_device_count(); dvc++) {
		cuda_set_device(dvc);
		CUDA_CHECK(cudaDeviceReset());
		size_t value = STACK_SIZE;
		CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, value));
		CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitStackSize));
		if (value != STACK_SIZE) {
			THROW_ERROR("Unable to set stack size to %li on device %i and rank %i\n", STACK_SIZE, dvc, hpx_rank());
		}
	}
	kick_workspace::initialize();
	hpx::wait_all(futs.begin(), futs.end());
}

#endif
