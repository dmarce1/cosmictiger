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
void cpuGravityCc(cosmictiger::gravity::cartesian::Expansion<float, order>&, Vector<TreeId> const&,
    TreeId, bool doPhi, FmmGeometry const& geometry = {});
template <int order>
void cpuGravityCp(cosmictiger::gravity::cartesian::Expansion<float, order>&, Vector<TreeId> const&,
    TreeId, bool doPhi, FmmGeometry const& geometry = {});
template <int order>
void cpuGravityPc(
    ForceVectors&, int, TreeId, Vector<TreeId> const&, FmmGeometry const& geometry = {});
void cpuGravityPp(
    ForceVectors&, int, TreeId, Vector<TreeId> const&, float h, FmmGeometry const& geometry = {});

void showTimings();

// CUDA operators are order-templated device functions in cosmictiger/gravity_cuda.hpp.

void resetGravityCounters();
void getGravityCounters(double& close, double& direct);
void setGravityCounterUse(bool code);

namespace detail {

CUDA_EXPORT inline FlopCount bernsteinFlops(int degree) {
    return FlopCount(3) * degree * (degree + 1) / 2;
}

// Cost of the radial table evaluation, including q and the final support
// scaling. Scalar evaluation stops at its selected span; SIMD evaluates every
// span and the outside expression. Width subtractions are scalar in both cases.
template <class T>
CUDA_EXPORT inline FlopCount gravitySofteningFlops(
    radial_kernel::KernelTable const& kernel, T const& q, bool doPhi) {
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

}    // namespace detail

template <class T>
CUDA_EXPORT inline FlopCount gravitySofteningKernel(T& force, T& phi, T radiusSquared,
    T inverseSupport, T inverseSupportSquared, T inverseSupportCubed, bool doPhi) {
    T const q = simdSqrt(radiusSquared * inverseSupportSquared);
    auto const& kernel = selectedGravityKernel();
    force = radial_kernel::forceFactor(kernel, q) * inverseSupportCubed;
    if (doPhi) {
        phi = radial_kernel::potential(kernel, q) * inverseSupport;
    } else {
        phi = simdBroadcast<T>(0);
    }
    return detail::gravitySofteningFlops(kernel, q, doPhi);
}

CUDA_EXPORT inline float selfPhi() {
    return selectedGravityKernel().centralPotential;
}

#endif /* GRAVITY_HPP_ */
