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

#define CUDA_MEM_STACK_SIZE (128*1024)
#define CUDA_MEM_NBIN 32
#define CUDA_MEM_BLOCK_SIZE (4*1024)

using cuda_mem_int = uint64_t;

class cuda_mem {
	array<array<char*, CUDA_MEM_STACK_SIZE>, CUDA_MEM_NBIN> q;
	array<long long, CUDA_MEM_NBIN> qin;
	array<long long, CUDA_MEM_NBIN> qout;
	char* heap;
	int next_block;
	int heap_max;
	__device__
	char* allocate_blocks(int nblocks);
	__device__
	void push(int bin, char* ptr);
	__device__
	char* pop(int bin);
	__device__
	bool create_new_allocations(int bin);
public:

	__device__
	void* allocate(size_t sz);
	__device__
	void free(void* ptr);
	cuda_mem(size_t heap_size);
	~cuda_mem();
	void reset();
};


void cuda_mem_init(size_t heap_size);
__device__
void* cuda_mem_allocate(size_t sz);
__device__
void cuda_mem_free(void* ptr);

