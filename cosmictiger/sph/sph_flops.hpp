/* CosmicTiger - SPH source-operation counts; BSL-1.0. */
#pragma once

#include <type_traits>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

namespace sph_flops {

// Costs of the expressions in monaghan/{equations,viscosity}.hpp. Selection and unary signs
// are free; both expressions passed to simdSelect are evaluated.
constexpr FlopCount kViscosityMu = 9;
constexpr FlopCount kArtificialViscosity = 10;
constexpr FlopCount kPressureCoefficient = 5;
constexpr FlopCount kPairEnergyRate = 4;

CUDA_EXPORT inline FlopCount bernsteinFlops(int degree) {
    return FlopCount(3) * degree * (degree + 1) / 2;
}

// Density and its derivative use two explicitly formed Bernstein weights.
// Scalar evaluation visits only the selected span (none outside support),
// whereas SIMD eagerly evaluates every span, including padded lanes. Each
// span also forms its scalar double width once, outside the SIMD arithmetic.
template <class Real>
CUDA_EXPORT inline FlopCount densityKernelFlops(
    radial_kernel::KernelTable const& kernel, Real const& q, bool derivative = false) {
    auto const polynomial = bernsteinFlops(kernel.degree - int(derivative));
    if constexpr (std::is_arithmetic_v<Real>) {
        return q < Real(kernel.changePoints[kernel.pieceCount]) ? 5 + polynomial : 0;
    } else {
        return FlopCount(kernel.pieceCount) * (1 + Real::size() * (4 + polynomial));
    }
}

// The optional central force uses the same selected gravity radial table.
template <class Real>
CUDA_EXPORT inline FlopCount forceKernelFlops(
    radial_kernel::KernelTable const& kernel, Real const& q) {
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

}    // namespace sph_flops
