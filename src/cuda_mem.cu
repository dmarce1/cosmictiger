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
void cuda_mem::acquire_lock() {
	while (atomicAdd(lock, 1) != 0) {
		atomicAdd(lock, -1);
	}
}

__device__
void cuda_mem::release_lock() {
	atomicAdd(lock, -1);
}

__device__
char* cuda_mem::allocate_blocks(int nblocks) {
	if (next_block + nblocks >= heap_max) {
		return nullptr;
	} else {
		char* ptr = heap + next_block * CUDA_MEM_BLOCK_SIZE;
		next_block += nblocks;
		return ptr;
	}
}

__device__
void cuda_mem::push(int bin, char* ptr) {
	if (stack_pointers[bin] >= CUDA_MEM_STACK_SIZE) {
		printf("internal stack size exceeded for cuda mem\n");
		__trap();
	}
	free_stacks[bin][stack_pointers[bin]++] = ptr;
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
	acquire_lock();
	int alloc_size = 1;
	int bin = 0;
	while (alloc_size < sz) {
		alloc_size *= 2;
		bin++;
	}
	if (bin >= CUDA_MEM_NBIN) {
		printf("Allocation request for %li too large\n", sz);
		__trap();
	}
	if (stack_pointers[bin] == 0) {
		if (!create_new_allocations(bin)) {
			return nullptr;
		}
	}
	char* ptr = free_stacks[bin][stack_pointers[bin]--];
	release_lock();
	return ptr;
}

__device__ void cuda_mem::free(void* ptr) {
	acquire_lock();
	size_t* binptr = (size_t*) ((char*) ptr - sizeof(size_t));
	if (*binptr >= CUDA_MEM_NBIN) {
		printf("Corrupt free!\n");
		__trap();
	}
	push(*binptr, (char*) ptr);
	release_lock();
}

cuda_mem::cuda_mem(size_t heap_size) {
	CUDA_CHECK(cudaMallocManaged(&lock, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&heap, heap_size));
	heap_max = heap_size / CUDA_MEM_BLOCK_SIZE;
	lock = 0;
	next_block = 0;
	for (int i = 0; i < CUDA_MEM_NBIN; i++) {
		stack_pointers[i] = 0;
	}
}

cuda_mem::~cuda_mem() {
	CUDA_CHECK(cudaFree(lock));
	CUDA_CHECK(cudaFree(heap));
}

