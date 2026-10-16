/*
 * flops.hpp
 *
 *  Created on: Jun 1, 2022
 *      Author: dmarce1
 */

#ifndef FLOPS_HPP_
#define FLOPS_HPP_

#include <cosmictiger/cuda.hpp>

template <class T>
class FlopCounter {
    T cnt_;

public:
    CUDA_EXPORT
    inline FlopCounter& operator=(T i) {
#ifdef COUNT_FLOPS
        cnt_ = i;
#endif
    }
    CUDA_EXPORT
    inline FlopCounter(T i) {
#ifdef COUNT_FLOPS
        cnt_ = i;
#endif
    }
    CUDA_EXPORT
    inline FlopCounter& operator+=(T i) {
#ifdef COUNT_FLOPS
        cnt_ += i;
#endif
        return *this;
    }
    CUDA_EXPORT
    inline FlopCounter& operator-=(T i) {
#ifdef COUNT_FLOPS
        cnt_ -= i;
#endif
        return *this;
    }
    CUDA_EXPORT
    inline operator T() const {
        return cnt_;
    }
};

void resetFlops();
void resetGpuFlops();
double flopsPerSecond();
double getGpuFlops();
void addCpuFlops(int count);
#ifdef __CUDACC__
__device__ void addGpuFlops(int count);
#endif

#endif /* FLOPS_HPP_ */
