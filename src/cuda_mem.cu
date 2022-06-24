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

static __managed__ array<lockfree_queue<CUDA_MEM_STACK_SIZE>, CUDA_MEM_NBIN>* qptr;
static __managed__ char* heap_begin;
static __managed__ char* next;
static __managed__ char* heap_end;

struct init_type {
	init_type() {
		PRINT("IN\n");
		CUDA_CHECK(cudaMallocManaged(&qptr, sizeof(array<lockfree_queue<CUDA_MEM_STACK_SIZE>, CUDA_MEM_NBIN> )));
		CUDA_CHECK(cudaMallocManaged(&heap_begin, HEAP_SIZE));
		heap_end = heap_begin + HEAP_SIZE;
		next = heap_begin;
		PRINT("OUT\n");
	}
};

static init_type& init_func() {
	static init_type a;
	return a;
}

void cuda_mem_init() {
	init_func();
}

CUDA_EXPORT
static void cuda_mem_push(int bin, char* ptr) {
	if (!(*qptr)[bin].push(ptr)) {
		PRINT("cuda mem Q full!\n");ALWAYS_ASSERT(false);
	}
}

CUDA_EXPORT
static char* cuda_mem_pop(int bin) {
	return (*qptr)[bin].pop();
}

CUDA_EXPORT char* atomic_add(unsigned long long* ptr, unsigned long long size) {
#ifdef __CUDA_ARCH__
	return (char*) atomicAdd(ptr, size);
#else
	return (char*) __sync_fetch_and_add(ptr, size);
#endif
}

CUDA_EXPORT void fence() {
#ifdef __CUDA_ARCH__
	__threadfence();
#endif
}

CUDA_EXPORT static bool create_new_allocation(int bin) {
	size_t size = (1 << bin) + sizeof(size_t);
	auto* ptr = atomic_add((unsigned long long*) &next, (unsigned long long) size);
	if (next >= heap_end) {
		return false;
	} else {
		cuda_mem_push(bin, ptr + sizeof(size_t));
		*((size_t*) ptr) = bin;
		fence();
		return true;
	}
}

CUDA_EXPORT static void cuda_mem_free(void* ptr) {
	size_t* binptr = (size_t*) ((char*) ptr - sizeof(size_t));
	if (*binptr >= CUDA_MEM_NBIN) {
		printf("Corrupt free! %li\n", *binptr);
		ALWAYS_ASSERT(false);
	}
	cuda_mem_push(*binptr, (char*) ptr);
}

static CUDA_EXPORT void* cuda_mem_allocate(size_t sz) {

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
	while ((ptr = cuda_mem_pop(bin)) == nullptr) {
		if (!create_new_allocation(bin)) {
			return nullptr;
		}
	}
	return ptr;
}

CUDA_EXPORT
void* cuda_malloc(size_t sz) {
	return cuda_mem_allocate(sz);
}

CUDA_EXPORT void cuda_free(void* ptr) {
	cuda_mem_free(ptr);
}

