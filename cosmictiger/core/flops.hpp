/*
 * flops.hpp
 *
 *  Created on: Jun 1, 2022
 *      Author: dmarce1
 */

#ifndef FLOPS_HPP_
#define FLOPS_HPP_

#include <cosmictiger/cuda/cuda.hpp>

// Source-operation estimates: FMA counts twice, other arithmetic/math results
// once; integer operations, comparisons and unary signs are excluded.
using FlopCount = unsigned long long;
static_assert(sizeof(FlopCount) == 8);

#ifdef COUNT_FLOPS
constexpr bool kCountFlops = true;
#else
constexpr bool kCountFlops = false;
#endif

template <class Count = FlopCount, bool enabled = kCountFlops>
class FlopCounter {
    Count count_ = 0;

public:
    CUDA_EXPORT
    inline FlopCounter& operator=(Count count) {
        if constexpr (enabled) {
            count_ = count;
        }
        return *this;
    }
    CUDA_EXPORT
    inline FlopCounter(Count count = 0) {
        *this = count;
    }
    CUDA_EXPORT
    inline FlopCounter& operator+=(Count count) {
        if constexpr (enabled) {
            count_ += count;
        }
        return *this;
    }
    CUDA_EXPORT
    inline FlopCounter& operator-=(Count count) {
        if constexpr (enabled) {
            count_ -= count;
        }
        return *this;
    }
    CUDA_EXPORT
    inline operator Count() const {
        return count_;
    }
};

// Call reset/read only at a completed work boundary, after CPU tasks and CUDA
// stream futures finish. Read aggregates every locality; reset starts a new
// wall-clock interval after the distributed reset completes.
void resetFlops();
FlopCount readFlops();
void resetGpuFlops();
double flopsPerSecond();
FlopCount getGpuFlops();
void addCpuFlops(FlopCount count);
#ifdef __CUDACC__
__device__ void addGpuFlops(FlopCount count);
#endif

#endif /* FLOPS_HPP_ */
