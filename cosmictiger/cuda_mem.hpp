#pragma once

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
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/lockfree_queue.hpp>


CUDA_EXPORT
inline void syncthreads() {
#ifdef __CUDA_ARCH__
	__syncthreads();
#endif
}

#define CUDA_MEM_STACK_SIZE (1024*1024)
#define CUDA_MEM_NBIN 32

using cuda_mem_int = uint64_t;

class cuda_mem {
	array<lockfree_queue<char, CUDA_MEM_STACK_SIZE>, CUDA_MEM_NBIN> q;
	char* heap_begin0;
	char* heap_begin;
	char* next;
	char* heap_end;CUDA_EXPORT
	void push(int bin, char* ptr);CUDA_EXPORT
	char* pop(int bin);CUDA_EXPORT
	bool create_new_allocation(int bin);
public:
	CUDA_EXPORT
	void* allocate(size_t sz);CUDA_EXPORT
	void free(void* ptr);
	cuda_mem(size_t heap_size);
	~cuda_mem();
};

void cuda_mem_init(size_t heap_size);
CUDA_EXPORT void* cuda_malloc(size_t sz);
CUDA_EXPORT void cuda_free(void* ptr);
CUDA_EXPORT void cuda_memcpy(void* d, void* s, int sz);

