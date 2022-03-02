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
	if (base + nblocks >= heap_max) {
		return nullptr;
	} else {
		char* ptr = heap + base * CUDA_MEM_BLOCK_SIZE;
		return ptr;
	}
}

__device__
void cuda_mem::push(int bin, char* ptr) {
	constexpr auto NBITS = sizeof(cuda_mem_int) * CHAR_BIT;
	bool full = false;
	while (!full) {
		full = true;
		for (int i = 0; i < CUDA_MEM_STACK_SIZE / sizeof(cuda_mem_int) + 1; i++) {
			if (~free_bits[bin][i] != 0ULL) {
				full = false;
				for (int k = 0; k < NBITS; k++) {
					const auto mask = ((cuda_mem_int) 1) << k;
					if (~free_bits[bin][i] & mask) {
						if (atomicCAS(((unsigned long long int*) &free_ptrs[bin][NBITS * i + k]), (unsigned long long int) 0, (unsigned long long int) ptr) == 0) {
							atomicOr((unsigned long long int*) &free_bits[bin][i], (unsigned long long int) mask);
				//			__threadfence();
							return;
						}
					}
				}
			}
		}
		PRINT( "Searching push\n");
	}
	if( full) {
		PRINT( "STACK FULL\n");
		__trap();
	}
}

__device__
char* cuda_mem::pop(int bin) {
	constexpr auto NBITS = sizeof(cuda_mem_int) * CHAR_BIT;
	char* ptr = nullptr;
	bool empty = false;
	while (!empty) {
		empty = true;
		for (int i = 0; i < CUDA_MEM_STACK_SIZE / sizeof(cuda_mem_int) + 1; i++) {
			if (free_bits[bin][i] != 0) {
				empty = false;
				for (int k = 0; k < NBITS; k++) {
					const auto mask = ((cuda_mem_int) 1) << k;
					if (free_bits[bin][i] & mask) {
						const auto old = free_bits[bin][i];
						if (atomicAnd((unsigned long long int*) &free_bits[bin][i], (unsigned long long int) ~mask) == old) {
							ptr = free_ptrs[bin][NBITS * i + k];
							free_ptrs[bin][NBITS * i + k] = nullptr;
							PRINT( "pop Success  %i  %i\n", i, k);
							__threadfence();
							return ptr;
						}
					}
				}
			}
		}
		PRINT( "Searching pop\n");
	}
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
	lock = 0;
	next_block = 0;
	for (int i = 0; i < CUDA_MEM_NBIN; i++) {
		for (int j = 0; j < CUDA_STACK_SIZE; j++) {
			free_ptrs[i][j] = 0;
		}
		for (int j = 0; j < CUDA_STACK_SIZE / sizeof(cuda_mem_int) + 1; j++) {
			free_bits[i][j] = 0;
		}
	}
}

cuda_mem::~cuda_mem() {
	CUDA_CHECK(cudaFree(lock));
	CUDA_CHECK(cudaFree(heap));
}

