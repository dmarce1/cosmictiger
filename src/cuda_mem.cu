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
#define CUDA_MEM_CU
#include <cosmictiger/cuda_mem.hpp>


__device__
void cuda_mem::push(int bin, char* ptr) {
	using itype = unsigned long long int;
	auto& this_q = q[bin];
	auto in = qin[bin];
	const auto& out = qout[bin];
	if (in - out >= CUDA_MEM_STACK_SIZE) {
		PRINT("Q full! %li %li\n", out, in);
		__trap();
	}
	while (atomicCAS((itype*) &this_q[in % CUDA_MEM_STACK_SIZE], (itype) 0, (itype) ptr) != 0) {
		in++;
		if (in - out >= CUDA_MEM_STACK_SIZE) {
			PRINT("cuda mem Q full! %li %li\n", out, in);
			__trap();
		}
	}
	in++;
	atomicMax((itype*) &qin[bin], (itype) in);
}

__device__
char* cuda_mem::pop(int bin) {
	using itype = unsigned long long int;
	auto& this_q = q[bin];
	const auto& in = qin[bin];
	auto out = qout[bin];
	if (out >= in) {
		return nullptr;
	}
	char* ptr;
	while ((ptr = (char*) atomicExch((itype*) &this_q[out % CUDA_MEM_STACK_SIZE], (itype) 0)) == nullptr) {
		if (out >= in) {
			return nullptr;
		}
		out++;
	}
	out++;
	atomicMax((itype*) &qout[bin], (itype) out);
	return ptr;

}

__device__
bool cuda_mem::create_new_allocation(int bin) {
	size_t size = (1 << bin) + sizeof(size_t);
	auto* ptr = (char*) atomicAdd((unsigned long long*)&next, (unsigned long long) size);
	if (next >= heap_end) {
		return false;
	} else {
		push(bin, ptr + sizeof(size_t));
		*((size_t*) ptr) = bin;
		__threadfence();
		return true;
	}
}

__device__
void* cuda_mem::allocate(size_t sz) {
	int alloc_size = 8;
	int bin = 3;
	while (alloc_size < sz) {
		alloc_size *= 2;
		bin++;
	}
	if (bin >= CUDA_MEM_NBIN) {
		printf("Allocation request for %li too large\n", sz);
		__trap();
	}
	char* ptr;
	while ((ptr = pop(bin)) == nullptr) {
		if (!create_new_allocation(bin)) {
			return nullptr;
		}
	}
	return ptr;
}

__device__ void cuda_mem::free(void* ptr) {
	size_t* binptr = (size_t*) ((char*) ptr - sizeof(size_t));
	if (*binptr >= CUDA_MEM_NBIN) {
		printf("Corrupt free! %li\n", *binptr);
		__trap();
	}
	push(*binptr, (char*) ptr);
}

cuda_mem::cuda_mem(size_t heap_size) {
	CUDA_CHECK(cudaMalloc(&heap_begin, heap_size));
	heap_end = heap_begin + heap_size;
	reset();
}

void cuda_mem::reset() {
	next = heap_begin;
	for (int i = 0; i < CUDA_MEM_NBIN; i++) {
		for (int j = 0; j < CUDA_MEM_STACK_SIZE; j++) {
			q[i][j] = nullptr;
		}
		qin[i] = 0;
		qout[i] = 0;
	}
}

cuda_mem::~cuda_mem() {
	CUDA_CHECK(cudaFree(heap_begin));
}

__managed__ cuda_mem* memory;

__device__ cuda_mem* get_cuda_heap() {
	return memory;
}

void cuda_mem_init(size_t heap_size) {
	CUDA_CHECK(cudaMallocManaged(&memory, sizeof(cuda_mem)));
	new (memory) cuda_mem(heap_size);
}

