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

CUDA_EXPORT
void cuda_mem::push(int bin, char* ptr) {
	if (!q[bin].push(ptr)) {
		PRINT("cuda mem Q full!\n");
		ALWAYS_ASSERT(false);
	}
}

CUDA_EXPORT
char* cuda_mem::pop(int bin) {
	return q[bin].pop();
}


#ifndef __CUDA_ARCH__
__device__
bool cuda_mem::create_new_allocation(int bin) {
	size_t size = (1 << bin) + sizeof(size_t);
	auto* ptr = (char*) __sync_fetch_and_add((unsigned long long*) &next, (unsigned long long) size);
	if (next >= heap_end) {
		return false;
	} else {
		push(bin, ptr + sizeof(size_t));
		*((size_t*) ptr) = bin;
#ifdef __CUDA_ARCH__
		__threadfence();
#endif
		return true;
	}
}
#else
bool cuda_mem::create_new_allocation(int bin) {
	size_t size = (1 << bin) + sizeof(size_t);
	auto* ptr = (char*) atomicAdd((unsigned long long*) &next, (unsigned long long) size);
	if (next >= heap_end) {
		return false;
	} else {
		push(bin, ptr + sizeof(size_t));
		*((size_t*) ptr) = bin;
		__threadfence();
		return true;
	}
}
#endif

CUDA_EXPORT
void* cuda_mem::allocate(size_t sz) {
	int alloc_size = 8;
	int bin = 3;
	while (alloc_size < sz) {
		alloc_size *= 2;
		bin++;
	}
	if (bin >= CUDA_MEM_NBIN) {
		printf("Allocation request for %li too large\n", sz);
		ALWAYS_ASSERT(false);
	}
	char* ptr;
	while ((ptr = pop(bin)) == nullptr) {
		if (!create_new_allocation(bin)) {
			return nullptr;
		}
	}
	return ptr;
}

CUDA_EXPORT void cuda_mem::free(void* ptr) {
	size_t* binptr = (size_t*) ((char*) ptr - sizeof(size_t));
	if (*binptr >= CUDA_MEM_NBIN) {
		printf("Corrupt free! %li\n", *binptr);
		ALWAYS_ASSERT(false);
	}
	push(*binptr, (char*) ptr);
}

cuda_mem::cuda_mem(size_t heap_size) {
	CUDA_CHECK(cudaMallocManaged(&heap_begin, heap_size));
	heap_end = heap_begin + heap_size;
	reset();
}

void cuda_mem::reset() {
	next = heap_begin;
}

cuda_mem::~cuda_mem() {
	CUDA_CHECK(cudaFree(heap_begin));
}

__managed__ cuda_mem* memory;

CUDA_EXPORT cuda_mem* get_cuda_heap() {
	return memory;
}

CUDA_EXPORT void* cuda_malloc(size_t sz) {
	return memory->allocate(sz);
}

CUDA_EXPORT void cuda_free(void* ptr) {
	memory->free(ptr);
}

void cuda_mem_init(size_t heap_size) {
	CUDA_CHECK(cudaMallocManaged(&memory, sizeof(cuda_mem)));
	new (memory) cuda_mem(heap_size);
}

