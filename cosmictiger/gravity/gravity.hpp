/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef GRAVITY_HPP_
#define GRAVITY_HPP_

#include <cmath>

#include <cosmictiger/core/fixed_capacity_vector.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/gravity/kick.hpp>
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

using GravityCcType = int;
constexpr GravityCcType kGravityDirect = 0;
constexpr GravityCcType kGravityEwald = 1;

// Order-templated CPU operators; definitions live in gravity_cpu.hpp and are
// instantiated once per FMM order by the per-order instance translation units.
template <int order>
void cpuGravityCc(
    GravityCcType, FmmExpansion<float, order>&, Vector<TreeId> const&, TreeId, bool doPhi);
template <int order>
void cpuGravityCp(FmmExpansion<float, order>&, Vector<TreeId> const&, TreeId, bool doPhi);
template <int order>
void cpuGravityPc(ForceVectors&, int, TreeId, Vector<TreeId> const&);
void cpuGravityPp(ForceVectors&, int, TreeId, Vector<TreeId> const&, float h);

void showTimings();

// CUDA operators are order-templated device functions in cosmictiger/gravity_cuda.hpp.

void resetGravityCounters();
void getGravityCounters(double& close, double& direct);
void setGravityCounterUse(bool code);

template <class T>
CUDA_EXPORT inline void gravitySofteningKernel(T& force, T& phi, T radiusSquared, T inverseSupport,
    T inverseSupportSquared, T inverseSupportCubed, bool doPhi) {
    T const q = simdSqrt(radiusSquared * inverseSupportSquared);
    auto const& kernel = selectedGravityKernel();
    force = radial_kernel::forceFactor(kernel, q) * inverseSupportCubed;
    if (doPhi) {
        phi = radial_kernel::potential(kernel, q) * inverseSupport;
    } else {
        phi = simdBroadcast<T>(0);
    }
}

CUDA_EXPORT inline float selfPhi() {
    return selectedGravityKernel().centralPotential;
}

#endif /* GRAVITY_HPP_ */
