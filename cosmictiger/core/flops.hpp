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
// excluded. Same-precision copies and float/integer conversions do not add FLOPs;
// conversions between double and integer count as double.
using FlopCount = unsigned long long;
static_assert(sizeof(FlopCount) == 8);

enum class FlopPrecision { Single, Double, Unclassified };

struct FlopCounts {
    FlopCount singlePrecision = 0;
    FlopCount doublePrecision = 0;
    FlopCount unclassified = 0;

    /// Sum single-precision, double-precision, and unclassified operation counts.
    CUDA_EXPORT
    constexpr FlopCount total() const {
        return singlePrecision + doublePrecision + unclassified;
    }

    /// Accumulate operation counts, honoring the counter's compile-time enablement when
    /// applicable.
    CUDA_EXPORT
    constexpr FlopCounts& operator+=(FlopCounts const& other) {
        singlePrecision += other.singlePrecision;
        doublePrecision += other.doublePrecision;
        unclassified += other.unclassified;
        return *this;
    }

    /// Archive this record in the field order required by distributed transport.
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
    static_assert(std::is_same_v<SourceType, float> || std::is_same_v<SourceType, double> ||
        std::is_integral_v<SourceType>);
    static_assert(std::is_same_v<TargetType, float> || std::is_same_v<TargetType, double> ||
        std::is_integral_v<TargetType>);
    if constexpr (!std::is_same_v<SourceType, TargetType> &&
        (std::is_same_v<SourceType, double> || std::is_same_v<TargetType, double>)) {
        return {0, conversions, 0};
    } else {
        return {};
    }
}

struct FlopRates {
    double singlePrecision = 0.0;
    double doublePrecision = 0.0;
    double unclassified = 0.0;

    /// Sum single-precision, double-precision, and unclassified operation counts.
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
    Count count_{};

public:
    /// Replace the stored count when this counter is enabled at compile time.
    CUDA_EXPORT
    inline FlopCounter& operator=(Count count) {
        if constexpr (enabled) {
            count_ = count;
        }
        return *this;
    }
    /// Initialize an operation counter from the supplied count.
    CUDA_EXPORT
    inline FlopCounter(Count count = {}) {
        *this = count;
    }
    /// Accumulate operation counts, honoring the counter's compile-time enablement when
    /// applicable.
    CUDA_EXPORT
    inline FlopCounter& operator+=(Count count) {
        if constexpr (enabled) {
            count_ += count;
        }
        return *this;
    }
    /// Subtract operations when this counter is enabled at compile time.
    CUDA_EXPORT
    inline FlopCounter& operator-=(Count count) {
        if constexpr (enabled) {
            count_ -= count;
        }
        return *this;
    }
    /// Return the counter's currently stored value.
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
// Scalar submissions must identify the arithmetic precision. Mixed routines
// submit separate audited batches; unclassified is reserved for diagnostics.
void addCpuFlops(FlopCount count, FlopPrecision precision);
/// Record CPU operations in their single, double, and unclassified precision categories.
inline void addCpuFlops(FlopCounts const& counts) {
    addCpuFlops(counts.singlePrecision, FlopPrecision::Single);
    addCpuFlops(counts.doublePrecision, FlopPrecision::Double);
    if (counts.unclassified != 0) {
        addCpuFlops(counts.unclassified, FlopPrecision::Unclassified);
    }
}
#ifdef __CUDACC__
__device__ void addGpuFlops(FlopCount count, FlopPrecision precision);
/// Record GPU operations in their single, double, and unclassified precision categories.
__device__ inline void addGpuFlops(FlopCounts const& counts) {
    addGpuFlops(counts.singlePrecision, FlopPrecision::Single);
    addGpuFlops(counts.doublePrecision, FlopPrecision::Double);
    if (counts.unclassified != 0) {
        addGpuFlops(counts.unclassified, FlopPrecision::Unclassified);
    }
}
#endif

#endif /* FLOPS_HPP_ */
