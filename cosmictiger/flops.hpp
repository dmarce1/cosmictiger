/*
 * flops.hpp
 *
 *  Created on: Jun 1, 2022
 *      Author: dmarce1
 */

#ifndef FLOPS_HPP_
#define FLOPS_HPP_

#include <cosmictiger/cuda.hpp>

template<class T>
class flop_counter {
	T cnt;
public:
	CUDA_EXPORT
	inline flop_counter& operator=(T i) {
#ifdef COUNT_FLOPS
		cnt = i;
#endif
	}
	CUDA_EXPORT
	inline flop_counter(T i) {
#ifdef COUNT_FLOPS
		cnt = i;
#endif
	}
	CUDA_EXPORT
	inline flop_counter& operator+=(T i) {
#ifdef COUNT_FLOPS
		cnt += i;
#endif
		return *this;
	}
	CUDA_EXPORT
	inline flop_counter& operator-=(T i) {
#ifdef COUNT_FLOPS
		cnt -= i;
#endif
		return *this;
	}
	CUDA_EXPORT
	inline operator T() const {
		return cnt;
	}
};

void reset_flops();
void reset_gpu_flops();
double flops_per_second();
double get_gpu_flops();
void add_cpu_flops(int count);
#ifdef __CUDACC__
__device__
void add_gpu_flops(int count);
#endif

#endif /* FLOPS_HPP_ */
