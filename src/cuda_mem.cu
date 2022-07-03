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

template<class T>
CUDA_EXPORT T atomic_cas(T* this_, T expected, T value) {
#ifdef __CUDA_ARCH__
	return atomicCAS_system(this_, expected, value);
#else
	return __sync_val_compare_and_swap(this_, expected, value);
#endif
}


template<class T>
CUDA_EXPORT T atomic_add(T* this_, T value) {
#ifdef __CUDA_ARCH__
	return atomicAdd_system(this_, value);
#else
	T expected, sum, rc;
	do {
		expected = *this_;
		sum = expected + value;
		rc = atomic_cas(this_, expected, sum);
	} while (rc != expected);
	return expected;
#endif
}

CUDA_EXPORT void fence() {
#ifdef __CUDA_ARCH__
	__threadfence_system();
#else
	__sync_synchronize();
#endif
}


template<class T>
CUDA_EXPORT T atomic_max(T* this_, T value) {
#ifdef __CUDA_ARCH__
	return atomicMax_system(this_, value);
#else
	T expected, maxvalue, rc;
	do {
		expected = *this_;
		maxvalue = std::max(expected, value);
		rc = atomic_cas(this_, expected, maxvalue);
	} while (rc != expected);
	return maxvalue;
#endif
}

template<class T>
CUDA_EXPORT T atomic_exch(T* this_, T value) {
#ifdef __CUDA_ARCH__
	return atomicExch_system(this_, value);
#else
	T expected, rc;
	do {
		expected = *this_;
		rc = atomic_cas(this_, expected, value);
	} while (rc != expected);
	return expected;
#endif
}



CUDA_EXPORT
void cuda_mem::push(int bin, char* ptr) {
	using itype = unsigned long long int;
	auto& this_q = q[bin];
	auto in = qin[bin];
	const auto& out = qout[bin];
	if (in - out >= CUDA_MEM_STACK_SIZE) {
		PRINT("Q full! %li %li\n", out, in);
		ALWAYS_ASSERT(false);
	}
	while (atomic_cas((itype*) &this_q[in % CUDA_MEM_STACK_SIZE], (itype) 0, (itype) ptr) != 0) {
		in++;
		if (in - out >= CUDA_MEM_STACK_SIZE) {
			PRINT("cuda mem Q full! %li %li\n", out, in);
			ALWAYS_ASSERT(false);
		}
	}
	in++;
	atomic_max((itype*) &qin[bin], (itype) in);
}

CUDA_EXPORT
char* cuda_mem::pop(int bin) {
	using itype = unsigned long long int;
	auto& this_q = q[bin];
	const auto& in = qin[bin];
	auto out = qout[bin];
	if (out >= in) {
		return nullptr;
	}
	char* ptr;
	while ((ptr = (char*) atomic_exch((itype*) &this_q[out % CUDA_MEM_STACK_SIZE], (itype) 0)) == nullptr) {
		if (out >= in) {
			return nullptr;
		}
		out++;
	}
	out++;
	atomic_max((itype*) &qout[bin], (itype) out);
	return ptr;

}

CUDA_EXPORT
bool cuda_mem::create_new_allocation(int bin) {
	size_t size = (1 << bin) + sizeof(size_t);
	auto* ptr = (char*) atomic_add((unsigned long long*) &next, (unsigned long long) size);
	if (next >= heap_end) {
		return false;
	} else {
		push(bin, ptr + sizeof(size_t));
		*((size_t*) ptr) = bin;
		fence();
		return true;
	}
}

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

