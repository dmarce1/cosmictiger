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

#include <cosmictiger/cuda_mem.hpp>

__device__
char* cuda_mem::allocate_blocks(int nblocks) {
	size_t base = atomicAdd(&next_block, nblocks);
//	PRINT( "%lli %i\n", base, heap_max);
	if (base + nblocks >= heap_max) {
		return nullptr;
	} else {
		char* ptr = heap + base * CUDA_MEM_BLOCK_SIZE;
		return ptr;
	}
}

__device__
void cuda_mem::push(int bin, char* ptr) {
	using itype = unsigned long long int;
	auto& this_q = q[bin];
	auto in = qin[bin];
	const auto& out = qout[bin];
///	PRINT("%lli\n", CUDA_MEM_STACK_SIZE - (in -out));
	if (in - out >= CUDA_MEM_STACK_SIZE) {
		PRINT("Q full! %li %li\n", out, in);
		__trap();
	}
	while (atomicCAS((itype*) &this_q[in % CUDA_MEM_STACK_SIZE], (itype) 0, (itype) ptr) != 0) {
	//	PRINT( "push %i %li %li\n", bin, in, out);
		in++;
		if (in - out >= CUDA_MEM_STACK_SIZE) {
			PRINT("cuda mem Q full! %li %li\n", out, in);
			__trap();
		}
	}
	in++;
///	PRINT( "push\n");
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
		//	PRINT( "pop %li %li\n", in, out);
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
bool cuda_mem::create_new_allocations(int bin) {
	size_t alloc_size = (1 << bin) + sizeof(size_t);
	int nallocs, nblocks;
	nblocks = 1;
	do {
		nallocs = nblocks * CUDA_MEM_BLOCK_SIZE / alloc_size;
	} while (nallocs == 0);
	char* base = allocate_blocks(nblocks);
	if (base == nullptr) {
		return false;
	} else {
		for (int i = 0; i < nallocs; i++) {
			char* ptr = base + i * alloc_size;
			push(bin, ptr + sizeof(size_t));
			*((size_t*) ptr) = bin;
		}
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
		if (!create_new_allocations(bin)) {
			return nullptr;
		}
	}
	return ptr;
}

__device__ void cuda_mem::free(void* ptr) {
	size_t* binptr = (size_t*) ((char*) ptr - sizeof(size_t));
	if (*binptr >= CUDA_MEM_NBIN) {
		printf("Corrupt free!\n");
		__trap();
	}
	push(*binptr, (char*) ptr);
}

cuda_mem::cuda_mem(size_t heap_size) {
	CUDA_CHECK(cudaMallocManaged(&lock, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&heap, heap_size));
	heap_max = heap_size / CUDA_MEM_BLOCK_SIZE;
//	PRINT( "HEAP MAX = %i\n", heap_max);
	lock = 0;
	next_block = 0;
	for (int i = 0; i < CUDA_MEM_NBIN; i++) {
		for (int j = 0; j < CUDA_STACK_SIZE; j++) {
			q[i][j] = nullptr;
		}
		qin[i] = 0;
		qout[i] = 0;
	}
}

cuda_mem::~cuda_mem() {
	CUDA_CHECK(cudaFree(lock));
	CUDA_CHECK(cudaFree(heap));
}

