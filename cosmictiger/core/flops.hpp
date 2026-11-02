/*
 * flops.hpp
 *
 *  Created on: Jun 1, 2022
 *      Author: dmarce1
 */

#ifndef FLOPS_HPP_
#define FLOPS_HPP_

#include <type_traits>

#include <cosmictiger/cuda/cuda.hpp>

// Source-operation estimates: FMA counts twice, other arithmetic/math results
// once; integer operations, comparisons and unary signs are excluded.
// Converting between floating-point precisions counts once at the higher
// precision, including narrowing conversions. Instrumentation bookkeeping is
// excluded. Same-precision copies and integer conversions do not add FLOPs.
using FlopCount = unsigned long long;
static_assert(sizeof(FlopCount) == 8);

enum class FlopPrecision { Single, Double, Unclassified };

struct FlopCounts {
    FlopCount singlePrecision = 0;
    FlopCount doublePrecision = 0;
    FlopCount unclassified = 0;

    CUDA_EXPORT
    constexpr FlopCount total() const {
        return singlePrecision + doublePrecision + unclassified;
    }

    CUDA_EXPORT
    constexpr FlopCounts& operator+=(FlopCounts const& other) {
        singlePrecision += other.singlePrecision;
        doublePrecision += other.doublePrecision;
        unclassified += other.unclassified;
        return *this;
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & singlePrecision & doublePrecision & unclassified;
    }
};

// These are the two arithmetic precisions tracked explicitly by the current
// application. Keep this helper independent of the actual physics conversion.
template <class Source, class Target>
CUDA_EXPORT constexpr FlopCounts floatingPointConversionFlops(FlopCount conversions = 1) {
    using SourceType = std::remove_cvref_t<Source>;
    using TargetType = std::remove_cvref_t<Target>;
    static_assert(std::is_same_v<SourceType, float> || std::is_same_v<SourceType, double>);
    static_assert(std::is_same_v<TargetType, float> || std::is_same_v<TargetType, double>);
    if constexpr (std::is_same_v<SourceType, TargetType>) {
        return {};
    } else {
        return {0, conversions, 0};
    }
}

struct FlopRates {
    double singlePrecision = 0.0;
    double doublePrecision = 0.0;
    double unclassified = 0.0;

    double total() const {
        return singlePrecision + doublePrecision + unclassified;
    }
};

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
FlopCounts readFlopCounts();
FlopCount readFlops();
void resetGpuFlops();
FlopRates flopRatesPerSecond();
double flopsPerSecond();
FlopCounts getGpuFlopCounts();
FlopCount getGpuFlops();
// Legacy, one-argument submissions retain their totals without claiming a
// precision: some count mixed float/double expressions. New instrumentation
// must submit audited single/double batches explicitly.
void addCpuFlops(FlopCount count);
void addCpuFlops(FlopCount count, FlopPrecision precision);
#ifdef __CUDACC__
__device__ void addGpuFlops(FlopCount count);
__device__ void addGpuFlops(FlopCount count, FlopPrecision precision);
#endif

#endif /* FLOPS_HPP_ */
