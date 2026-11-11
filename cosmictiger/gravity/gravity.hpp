/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef GRAVITY_HPP_
#define GRAVITY_HPP_

#include <cmath>
#include <type_traits>

#include <cosmictiger/core/fixed_capacity_vector.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/gravity/fmm_geometry.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/source_access.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/kernel.hpp>

struct ForceVectors {
    Vector<float> phi;
    Vector<float> gx;
    Vector<float> gy;
    Vector<float> gz;
    ForceVectors() = default;
    /// Resize potential and all three acceleration output arrays for the requested sink
    /// count.
    ForceVectors(int sz) {
        phi.resize(sz);
        gx.resize(sz);
        gy.resize(sz);
        gz.resize(sz);
    }
};

// Order-templated CPU operators; definitions live in gravity_cpu.hpp and are
// instantiated once per FMM order by the per-order instance translation units.
template <int order>
void cpuGravityCc(cosmictiger::gravity::cartesian::Expansion<float>&, Vector<TreeId> const&, TreeId,
    bool doPhi, FmmGeometry const& geometry = {});
template <int order>
void cpuGravityCp(cosmictiger::gravity::cartesian::Expansion<float>&, Vector<TreeId> const&, TreeId,
    bool doPhi, FmmGeometry const& geometry = {});
template <int order>
void cpuGravityPc(
    ForceVectors&, int, TreeId, Vector<TreeId> const&, FmmGeometry const& geometry = {});
void cpuGravityPp(
    ForceVectors&, int, TreeId, Vector<TreeId> const&, float h, FmmGeometry const& geometry = {});

// CUDA operators are order-templated device functions in cosmictiger/gravity_cuda.hpp.

// Distributed CUDA PP interaction totals, normalized by CUDA-processed sinks.
// CPU walk work is not included. Invoke from the root between completed kicks.
void resetGravityCounters();
void getGravityCounters(double& close, double& direct);
void setGravityCounterUse(bool code);

namespace detail {

/// Count the arithmetic in Bernstein-polynomial evaluation at the supplied degree.
CUDA_EXPORT inline FlopCount bernsteinFlops(int degree) {
    return FlopCount(3) * degree * (degree + 1) / 2;
}

// Cost of the radial table evaluation, including q and the final support
// scaling. Scalar evaluation stops at its selected span; SIMD evaluates every
// span and the outside expression. Width subtractions are scalar in both cases.
template <class T>
CUDA_EXPORT inline FlopCount gravitySofteningFlops(
    radial_kernel::KernelTable<double> const& kernel, T const& q, bool doPhi) {
    FlopCount const central = bernsteinFlops(kernel.degree);
    FlopCount const mass = bernsteinFlops(kernel.degree + 3);
    FlopCount const shell = bernsteinFlops(kernel.degree + 2);
    if constexpr (std::is_arithmetic_v<T>) {
        for (int piece = 0; piece < kernel.pieceCount; ++piece) {
            if (q < T(kernel.changePoints[piece + 1])) {
                return piece == 0 ? 7 + central + (doPhi ? 8 + central : 0) :
                                    11 + mass + (doPhi ? 9 + mass + shell : 0);
            }
        }
        return 6 + (doPhi ? 2 : 0);
    } else {
        FlopCount const width = T::size();
        FlopCount const outerPieces = kernel.pieceCount - 1;
        FlopCount count = width * (6 + 3 + central + outerPieces * (7 + mass)) + kernel.pieceCount;
        if (doPhi) {
            count +=
                width * (2 + 6 + central + outerPieces * (7 + mass + shell)) + kernel.pieceCount;
        }
        return count;
    }
}

// The arithmetic oracle above includes each double span-width subtraction but
// excludes table casts. Float kernels narrow each scalar table value once per
// broadcast, not once per SIMD lane; polynomial evaluation remains FP32.
template <class T>
CUDA_EXPORT inline FlopCounts gravitySofteningOperations(
    radial_kernel::KernelTable<double> const& kernel, T const& q, bool doPhi) {
    FlopCount const arithmetic = gravitySofteningFlops(kernel, q, doPhi);
    if constexpr (std::is_same_v<SimdScalar<T>, double>) {
        return {0, arithmetic, 0};
    } else if constexpr (std::is_arithmetic_v<T>) {
        FlopCount visited = 0;
        for (int piece = 0; piece < kernel.pieceCount; ++piece) {
            ++visited;
            if (q < T(kernel.changePoints[piece + 1])) {
                FlopCount const forceDouble = piece == 0 ? kernel.degree + 6 : kernel.degree + 11;
                FlopCount const potentialDouble =
                    piece == 0 ? kernel.degree + 9 : 2 * kernel.degree + 14;
                FlopCounts counts;
                counts.singlePrecision = arithmetic - 1 - int(doPhi);
                counts.doublePrecision =
                    visited * (1 + int(doPhi)) + forceDouble + (doPhi ? potentialDouble : 0);
                counts.unclassified = 0;
                return counts;
            }
        }
        return {arithmetic, visited * (1 + int(doPhi)), 0};
    } else {
        FlopCount const outer = kernel.pieceCount - 1;
        FlopCount const masks = 2 * kernel.pieceCount - 1;
        FlopCount doublePrecision = masks + kernel.degree + 6 + outer * (kernel.degree + 11);
        if (doPhi) {
            doublePrecision += masks + kernel.degree + 9 + outer * (2 * kernel.degree + 14);
        }
        return {arithmetic - kernel.pieceCount * (1 + int(doPhi)), doublePrecision, 0};
    }
}

}    // namespace detail

/// Evaluate the selected softened force factor and optional potential, returning and
/// optionally recording operation counts.
template <class T>
CUDA_EXPORT inline FlopCount gravitySofteningKernel(T& force, T& phi, T radiusSquared,
    T inverseSupport, T inverseSupportSquared, T inverseSupportCubed, bool doPhi,
    FlopCounts* operations = nullptr) {
    T const q = math::sqrt(radiusSquared * inverseSupportSquared);
    auto const& kernel = selectedGravityKernel();
    force = radial_kernel::forceFactor(kernel, q) * inverseSupportCubed;
    if (doPhi) {
        phi = radial_kernel::potential(kernel, q) * inverseSupport;
    } else {
        phi = math::broadcast<T>(0);
    }
    if constexpr (kCountFlops) {
        if (operations != nullptr) {
            *operations += detail::gravitySofteningOperations(kernel, q, doPhi);
        }
    }
    return detail::gravitySofteningFlops(kernel, q, doPhi);
}

/// Return the selected gravity kernel's dimensionless central potential magnitude.
CUDA_EXPORT inline float selfPhi() {
    return selectedGravityKernel().centralPotential;
}

#endif /* GRAVITY_HPP_ */
