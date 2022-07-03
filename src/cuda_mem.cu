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
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cosmictiger/atomic.hpp>

using barrier_type = cuda::barrier<cuda::thread_scope::thread_scope_block>;

#define HEADER_SIZE (16)

CUDA_EXPORT
bool cuda_mem::create_new_allocation(int bin) {
	size_t size = (1 << bin) + HEADER_SIZE;
	auto* ptr = (char*) atomic_add((unsigned long long*) &next, (unsigned long long) size);
	if (next >= heap_end) {
		return false;
	} else {
		q[bin].push(ptr + HEADER_SIZE);
		*((size_t*) (ptr)) = bin;
		fence();
		return true;
	}
}

CUDA_EXPORT
void* cuda_mem::allocate(size_t sz) {
	int alloc_size = 16;
	int bin = 4;
	while (alloc_size < sz) {
		alloc_size *= 2;
		bin++;
	}
	if (bin >= CUDA_MEM_NBIN) {
		printf("Allocation request for %li too large\n", sz);
		ALWAYS_ASSERT(false);
	}
	char* ptr;
	while ((ptr = q[bin].pop()) == nullptr) {
		if (!create_new_allocation(bin)) {
			return nullptr;
		}
	}
	return ptr;
}

CUDA_EXPORT void cuda_mem::free(void* ptr) {
	size_t* binptr = (size_t*) ((char*) ptr - HEADER_SIZE);
	if (*binptr >= CUDA_MEM_NBIN) {
		printf("Corrupt free! %li\n", *binptr);
		ALWAYS_ASSERT(false);
	}
	q[*binptr].push((char*) ptr);
}

cuda_mem::cuda_mem(size_t heap_size) {
	CUDA_CHECK(cudaMallocManaged(&heap_begin0, heap_size));
	heap_begin = (char*) round_up((uintptr_t) heap_begin0, (uintptr_t) HEADER_SIZE);
	heap_end = heap_begin + heap_size;
	next = heap_begin;
}

cuda_mem::~cuda_mem() {
	CUDA_CHECK(cudaFree(heap_begin0));
}

__managed__ cuda_mem* memory;

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

void cuda_memcpy(void* dest, void* src, int size) {
#ifdef __CUDA_ARCH__
	__shared__ barrier_type barrier;
	auto group = cooperative_groups::this_thread_block();
	if (group.thread_rank() == 0) {
		init(&barrier, group.size());
	}
	syncthreads();
	cuda::memcpy_async(group, dest, src, size, barrier);
	barrier.arrive_and_wait();
#else
	memcpy(dest, src, size);
#endif

}
