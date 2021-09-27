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


#ifndef FIXEDCAPVEC_HPP_
#define FIXEDCAPVEC_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/safe_io.hpp>

#ifdef __CUDACC__
template<class T, int N>
class fixedcapvec {
	array<T, N> data_;
	int sz;
public:
	__device__
	constexpr fixedcapvec() {
		const int tid = threadIdx.x;
		if (tid == 0) {
			sz = 0;
		}
		__syncwarp();
	}
	__device__
	void initialize() {
		const int tid = threadIdx.x;
		if (tid == 0) {
			sz = 0;
		}
		__syncwarp();
	}
	fixedcapvec(const fixedcapvec&) = default;
	fixedcapvec& operator=(const fixedcapvec&) = default;
	__device__
	inline int size() const {
		return sz;
	}
	__device__
	inline void resize(int new_sz) {
		const int tid = threadIdx.x;
		__syncwarp();
		if (tid == 0) {
			sz = new_sz;
			if (sz >= N) {
				PRINT("fixedcapvec exceeded size of %i in %s on line %i\n", N, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
				__trap();
#else
				abort();
#endif
			}
		}
		__syncwarp();
	}
	__device__
	inline void push_back(const T& item) {
		const int tid = threadIdx.x;
		if (tid == 0) {
			data_[sz] = item;
			sz++;
			if (sz >= N) {
				PRINT("fixedcapvec exceeded size of %i in %s on line %i\n", N, __FILE__, __LINE__);
#ifdef __CUDA_ARCH__
				__trap();
#else
				abort();
#endif
			}
		}
		__syncwarp();
	}
	__device__
	inline void pop_back() {
		const int tid = threadIdx.x;
		if( tid == 0 ) {
			sz--;
		}
		__syncwarp();
	}
	__device__
	inline T& back() {
		ASSERT(sz>=1);
		return data_[sz - 1];
	}
	__device__
	inline T back() const {
		return data_[sz - 1];
	}
	__device__
	T* begin() {
		return data_;
	}
	__device__
	T* end() {
		return data_ + sz;
	}
	__device__
	T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if( i < 0 || i >= sz) {
			PRINT( "index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, sz);
#ifdef __CUDA_ARCH__
			__trap();
#else
			abort();
#endif
		}
#endif
		return data_[i];
	}
	__device__
	T operator[](int i) const {
#ifdef CHECK_BOUNDS
		if( i < 0 || i >= sz) {
			PRINT( "index out of bounds for fixedcapvec %i should be between 0 and %i.\n", i, sz);
#ifdef __CUDA_ARCH__
			__trap();
#else
			abort();
#endif
		}
#endif
		return data_[i];
	}
	__device__
	T* data() {
		return data_;
	}

};
#endif
#endif /* FIXEDCAPVEC_HPP_ */
