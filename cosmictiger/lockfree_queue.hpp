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

#ifdef __CUDACC__

template<class T, int N>
class lockfree_queue {
	array<T*,N> Q;
	long long IN;
	long long OUT;
	lockfree_queue() {
		IN = OUT = 0;
		for( int i = 0; i < N; i++) {
			Q[i] = nullptr;
		}
	}
public:
	__device__
	void push(T* ptr) {
		const int& tid = threadIdx.x;
		__syncthreads();
		if( tid == 0 ) {
			using itype = unsigned long long int;
			auto in = IN;
			const auto& out = OUT;
			if (in - out >= N) {
				PRINT("Q full! %li %li\n", out, in);
				__trap();
			}
			while (atomicCAS((itype*) &Q[in % N], (itype) 0, (itype) ptr) != 0) {
				in++;
				if (in - out >= N) {
					PRINT("cuda mem Q full! %li %li\n", out, in);
					__trap();
				}
			}
			in++;
			atomicMax((itype*) &IN, (itype) in);
		}
		__syncthreads();
	}
	__device__
	T* pop() {
		const int& tid = threadIdx.x;
		__shared__ T* ptr;
		__syncthreads();
		if( tid == 0) {
			using itype = unsigned long long int;
			const auto& in = IN;
			auto out = OUT;
			if (out >= in) {
				return nullptr;
			}
			while ((ptr = (T*) atomicExch((itype*) &Q[out % N], (itype) 0)) == nullptr) {
				if (out >= in) {
					return nullptr;
				}
				out++;
			}
			out++;
			atomicMax((itype*) &OUT, (itype) out);
		}
		__syncthreads();
		return ptr;

	}
};

#endif
