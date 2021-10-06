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

std::set<std::string> get_hostnames();

HPX_PLAIN_ACTION (get_hostnames);

static vector<int> mydevices;

std::set<std::string> get_hostnames() {
	std::set < std::string > hostnames;
	std::vector < hpx::future<std::set<std::string>>>futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<get_hostnames_action>(c));
	}
	char hostname[256];
	gethostname(hostname, 255);
	hostnames.insert(std::string(hostname));
	for (auto& f : futs) {
		const auto tmp = f.get();
		for (const auto& hname : tmp) {
			hostnames.insert(hname);
		}
	}
	return hostnames;
}

static int compute_procs_per_node() {
	return hpx_size() / get_hostnames().size();
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

const vector<int>& cuda_get_devices() {
	return mydevices;
}

int cuda_get_device_id(int i) {
	return mydevices[i];
}

void cuda_init() {
	const int procs_per_node = compute_procs_per_node();
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
}

#endif
