/* CosmicTiger - SPH source-operation counts; BSL-1.0. */
#pragma once

#include <type_traits>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

namespace sph_flops {

// SPH arithmetic defaults to float; geometry, table preparation and mixed
// expressions enter the double bucket explicitly. This keeps the precision
// alongside each contribution instead of classifying a whole launch afterward.
struct Operations {
    FlopCount singlePrecision = 0;
    FlopCount doublePrecision = 0;
};

CUDA_EXPORT inline Operations operator+(Operations left, Operations right) {
    Operations sum;
    sum.singlePrecision = left.singlePrecision + right.singlePrecision;
    sum.doublePrecision = left.doublePrecision + right.doublePrecision;
    return sum;
}
CUDA_EXPORT inline Operations operator+(FlopCount left, Operations right) {
    return Operations{left, 0} + right;
}
CUDA_EXPORT inline Operations operator+(Operations left, FlopCount right) {
    return left + Operations{right, 0};
}

struct Counter {
    FlopCounter<> singlePrecision;
    FlopCounter<> doublePrecision;

    CUDA_EXPORT Counter(FlopCount single = 0)
      : singlePrecision(single) {}

    CUDA_EXPORT Counter& operator+=(FlopCount count) {
        singlePrecision += count;
        return *this;
    }
    CUDA_EXPORT Counter& operator+=(Operations count) {
        singlePrecision += count.singlePrecision;
        doublePrecision += count.doublePrecision;
        return *this;
    }
    CUDA_EXPORT operator FlopCount() const {
        return FlopCount(singlePrecision) + FlopCount(doublePrecision);
    }
    CUDA_EXPORT FlopCounts counts() const {
        return {FlopCount(singlePrecision), FlopCount(doublePrecision), 0};
    }
};

CUDA_EXPORT constexpr FlopCount positionConstructionFlops() {
#ifdef __CUDA_ARCH__
    return 1;
#else
    // Fixed64::floatingBits separates the two 32-bit limbs on the host.
    return kPositionBits == 64 ? 4 : 1;
#endif
}

// Spatial-vector costs over D = kSpatialDimensions active axes. A separation, sum, difference
// or scalar product costs one operation per axis; a·b takes D products and D − 1 sums, and
// |a| one more root. Padded Vector3 arithmetic always executes all three components.
inline constexpr FlopCount kVectorOperation = kSpatialDimensions;
inline constexpr FlopCount kDotProduct = 2 * kSpatialDimensions - 1;
inline constexpr FlopCount kNorm = kDotProduct + 1;

// Products of sph_geometry::inverseVolume and inverseGradientScale.
inline constexpr FlopCount kInverseVolume = kSpatialDimensions - 1;
inline constexpr FlopCount kInverseGradientScale = kSpatialDimensions == 1 ? 1 : 2;

// Costs of the expressions in sph/viscosity/monaghan.hpp. Selection and unary signs
// are free; both expressions passed to math::select are evaluated. The pair length
// h̄_M𞁑𞁓 (3 operations) is formed and counted once per pair by the caller.
constexpr FlopCount kViscosityMu = 5;
constexpr FlopCount kArtificialViscosity = 10;

// Two gamma-1 subtractions, two powers, multiplication, and two divisions.
// A non-monatomic gas additionally forms 5-3*gamma.
CUDA_EXPORT constexpr FlopCount entropyConversionFlops(float adiabaticIndex) {
    return adiabaticIndex == 5.0f / 3.0f ? 7 : 9;
}

CUDA_EXPORT inline FlopCount bernsteinFlops(int degree) {
    return FlopCount(3) * degree * (degree + 1) / 2;
}

// Density and its derivative use two explicitly formed Bernstein weights.
// Scalar evaluation visits only the selected span (none outside support),
// whereas SIMD eagerly evaluates every span, including padded lanes. Each
// span also forms its scalar table-precision width once, outside the SIMD arithmetic.
template <class Real, class Kernel>
CUDA_EXPORT inline FlopCount densityKernelFlops(
    Kernel const& kernel, Real const& q, bool derivative = false) {
    auto const polynomial = bernsteinFlops(kernel.degree - int(derivative));
    if constexpr (std::is_arithmetic_v<Real>) {
        return q < Real(kernel.changePoints[kernel.pieceCount]) ? 5 + polynomial : 0;
    } else {
        return FlopCount(kernel.pieceCount) * (1 + Real::size() * (4 + polynomial));
    }
}

// The optional central force uses the same selected gravity radial table.
template <class Real, class TableReal>
CUDA_EXPORT inline FlopCount forceKernelFlops(
    radial_kernel::KernelTable<TableReal> const& kernel, Real const& q) {
    if constexpr (std::is_arithmetic_v<Real>) {
        for (int piece = 0; piece < kernel.pieceCount; ++piece) {
            if (q < Real(kernel.changePoints[piece + 1])) {
                return piece == 0 ? 4 + bernsteinFlops(kernel.degree) :
                                    8 + bernsteinFlops(kernel.degree + 3);
            }
        }
        return 3;
    } else {
        return kernel.pieceCount +
            Real::size() *
            (3 + bernsteinFlops(kernel.degree) +
                FlopCount(kernel.pieceCount - 1) * (7 + bernsteinFlops(kernel.degree + 3)) + 3);
    }
}

// Arithmetic-only helpers above support the CountingReal references. Production
// counts additionally classify runtime conversions when table and lane precisions differ.
// math::broadcast narrows once before broadcasting, so these are per span, not
// per SIMD lane. Polynomial arithmetic remains at the scalar/lane precision.
template <class Real, class Kernel>
CUDA_EXPORT inline Operations densityKernelOperations(
    Kernel const& kernel, Real const& q, bool derivative = false) {
    using TableReal = std::remove_cvref_t<decltype(kernel.coefficients[0][0])>;
    int const degree = kernel.degree - int(derivative);
    FlopCount spans = 0;
    FlopCount comparisons = 0;
    if constexpr (std::is_arithmetic_v<Real>) {
        for (int piece = 0; piece < kernel.pieceCount; ++piece) {
            ++comparisons;
            if (q < Real(kernel.changePoints[piece + 1])) {
                spans = 1;
                break;
            }
        }
    } else {
        spans = kernel.pieceCount;
        comparisons = 2 * spans - 1;
    }
    FlopCount const arithmetic = densityKernelFlops(kernel, q, derivative);
    Operations result;
    if constexpr (std::is_same_v<SimdScalar<Real>, double>) {
        result.doublePrecision = arithmetic - spans;
    } else {
        result.singlePrecision = arithmetic - spans;
    }
    if constexpr (std::is_same_v<TableReal, double>) {
        result.doublePrecision += spans;
    } else {
        result.singlePrecision += spans;
    }
    if constexpr (!std::is_same_v<SimdScalar<Real>, TableReal>) {
        result.doublePrecision += spans * (degree + 6) + comparisons;
    }
    return result;
}

template <class Real, class TableReal>
CUDA_EXPORT inline Operations forceKernelOperations(
    radial_kernel::KernelTable<TableReal> const& kernel, Real const& q) {
    FlopCount spans = 0;
    FlopCount conversions = 0;
    if constexpr (std::is_arithmetic_v<Real>) {
        for (int piece = 0; piece < kernel.pieceCount; ++piece) {
            ++conversions;
            if (q < Real(kernel.changePoints[piece + 1])) {
                spans = 1;
                conversions += kernel.degree + (piece == 0 ? 5 : 10);
                break;
            }
        }
    } else {
        spans = kernel.pieceCount;
        conversions = 2 * spans - 1 + kernel.degree + 5 + (spans - 1) * (kernel.degree + 10);
    }
    FlopCount const arithmetic = forceKernelFlops(kernel, q);
    Operations result;
    if constexpr (std::is_same_v<SimdScalar<Real>, double>) {
        result.doublePrecision = arithmetic - spans;
    } else {
        result.singlePrecision = arithmetic - spans;
    }
    if constexpr (std::is_same_v<TableReal, double>) {
        result.doublePrecision += spans;
    } else {
        result.singlePrecision += spans;
    }
    if constexpr (!std::is_same_v<SimdScalar<Real>, TableReal>) {
        result.doublePrecision += conversions;
    }
    return result;
}

}    // namespace sph_flops
