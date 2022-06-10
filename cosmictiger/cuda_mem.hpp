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

#define CUDA_MEM_STACK_SIZE (128*1024)
#define CUDA_MEM_NBIN 32

using cuda_mem_int = uint64_t;

class cuda_mem {
	array<array<char*, CUDA_MEM_STACK_SIZE>, CUDA_MEM_NBIN> q;
	array<long long, CUDA_MEM_NBIN> qin;
	array<long long, CUDA_MEM_NBIN> qout;
	char* heap_begin;
	char* next;
	char* heap_end;
	__device__
	void push(int bin, char* ptr);
	__device__
	char* pop(int bin);
	__device__
	bool create_new_allocation(int bin);
	__device__
	static inline void discard_memory_one(volatile void* __ptr, std::size_t nbytes) noexcept {
#if __CUDA_ARCH__ >= 800
		char* p = reinterpret_cast<char*>(const_cast<void*>(__ptr));
		static constexpr std::size_t line_size = 128;
		std::size_t start = round_up(reinterpret_cast<std::uintptr_t>(p), line_size);
		std::size_t end = round_down(reinterpret_cast<std::uintptr_t>(p) + nbytes, line_size);
		for (std::size_t i = start; i < end; i += line_size) {
			asm volatile ("discard.global.L2 [%0], 128;" ::"l"(p + (i * line_size)) :);
		}
#endif
	}
public:
	__device__
	static inline void discard_memory(volatile void* __ptr, std::size_t nbytes) noexcept {
#if __CUDA_ARCH__ >= 800
		const int& tid = threadIdx.x;
		const int& block_size = blockDim.x;
		char* p = reinterpret_cast<char*>(const_cast<void*>(__ptr));
		static constexpr std::size_t line_size = 128;
		std::size_t start = round_up(reinterpret_cast<std::uintptr_t>(p), line_size);
		std::size_t end = round_down(reinterpret_cast<std::uintptr_t>(p) + nbytes, line_size);
		for (std::size_t i = start + tid * line_size; i < end; i += block_size * line_size) {
			asm volatile ("discard.global.L2 [%0], 128;" ::"l"(p + (i * line_size)) :);
		}
#endif
	}
	__device__
	static inline void prefetch(volatile void* ptr, std::size_t size) {
#if __CUDA_ARCH__
		constexpr std::uintptr_t line_size = 128;
		const auto start = round_down((std::uintptr_t)(ptr), line_size);
		const auto stop = round_up((std::uintptr_t)(ptr + size), line_size);
		for (std::uintptr_t i = start; i < stop; i += line_size) {
			char* p = (char*) i;
			asm volatile ("prefetch.global.L2::evict_normal [%0];" : "=l"(p));
		}
#endif
	}
	__device__
	void* allocate(size_t sz);
	__device__
	void free(void* ptr);
	cuda_mem(size_t heap_size);
	~cuda_mem();
	void reset();
};

void cuda_mem_init(size_t heap_size);
__device__ cuda_mem* get_cuda_heap();
__device__ void* cuda_malloc(size_t sz);
__device__ void cuda_free(void* ptr);

