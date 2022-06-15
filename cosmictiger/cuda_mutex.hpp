/*
 * cuda_mutex.hpp
 *
 *  Created on: Jun 13, 2022
 *      Author: dmarce1
 */

#ifndef CUDA_MUTEX_HPP_
#define CUDA_MUTEX_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/hpx.hpp>

class cuda_mutex {
	int lck;
#ifdef __CUDA_ARCH__
	__device__
	inline int atomic_exch(int* ptr, int num) {
		return atomicExch_system(ptr,num);
	}
	__device__
	inline void yield() {
		__nanosleep(10);
	}
#else
	inline int atomic_exch(int* ptr, int num) {
		int rc;
		__atomic_exchange(ptr, &num, &rc, __ATOMIC_RELAXED);
		return rc;
	}
	inline void yield() {
		hpx_yield();
	}
#endif
public:
	CUDA_EXPORT
	inline cuda_mutex() {
		lck = 0;
	}
	CUDA_EXPORT
	inline void lock() {
		while (atomic_exch(&lck, 1) != 0) {
	//		yield();
		}
	}
	CUDA_EXPORT
	inline void unlock() {
		atomic_exch(&lck, 0);
	}
};

#endif /* CUDA_MUTEX_HPP_ */
