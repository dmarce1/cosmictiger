/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once
#include <cosmictiger/gravity/kernels/spherical/ewald_common.hpp>
#include <cosmictiger/gravity/kernels/spherical_types.hpp>

namespace cosmictiger::gravity::spherical {
template <>
inline constexpr int kEwaldGreenFlops<6> = 577958;
template <>
inline constexpr int kEwaldGreenConventionalFlops<6> = 627304;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
#if defined(COSMICTIGER_FMM_COUNTING_ONLY) || defined(COSMICTIGER_FMM_INSTRUMENTATION)
template <typename Real>

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(Real x, Real y, Real z, Real radiusSquared, Real* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(float x, float y, float z, float radiusSquared, float* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(double x, double y, double z, double radiusSquared, double* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(SimdVector<float, 8> x, SimdVector<float, 8> y, SimdVector<float, 8> z,
        SimdVector<float, 8> radiusSquared, SimdVector<float, 8>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(SimdVector<double, 2> x, SimdVector<double, 2> y, SimdVector<double, 2> z,
        SimdVector<double, 2> radiusSquared, SimdVector<double, 2>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(SimdVector<float, 4> x, SimdVector<float, 4> y, SimdVector<float, 4> z,
        SimdVector<float, 4> radiusSquared, SimdVector<float, 4>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(SimdVector<double, 4> x, SimdVector<double, 4> y, SimdVector<double, 4> z,
        SimdVector<double, 4> radiusSquared, SimdVector<double, 4>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void ewaldRegular(SimdVector<double, 8> x, SimdVector<double, 8> y, SimdVector<double, 8> z,
        SimdVector<double, 8> radiusSquared, SimdVector<double, 8>* output);
#endif
#endif

/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
#if defined(COSMICTIGER_FMM_COUNTING_ONLY) || defined(COSMICTIGER_FMM_INSTRUMENTATION)
template <typename Real>

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(Real& greenTrace, Real x, Real y, Real z, Real* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(float& greenTrace, float x, float y, float z, float* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(double& greenTrace, double x, double y, double z, double* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(SimdVector<float, 8>& greenTrace, SimdVector<float, 8> x,
        SimdVector<float, 8> y, SimdVector<float, 8> z, SimdVector<float, 8>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(SimdVector<double, 2>& greenTrace, SimdVector<double, 2> x,
        SimdVector<double, 2> y, SimdVector<double, 2> z, SimdVector<double, 2>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(SimdVector<float, 4>& greenTrace, SimdVector<float, 4> x,
        SimdVector<float, 4> y, SimdVector<float, 4> z, SimdVector<float, 4>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(SimdVector<double, 4>& greenTrace, SimdVector<double, 4> x,
        SimdVector<double, 4> y, SimdVector<double, 4> z, SimdVector<double, 4>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(SimdVector<double, 8>& greenTrace, SimdVector<double, 8> x,
        SimdVector<double, 8> y, SimdVector<double, 8> z, SimdVector<double, 8>* output);
#endif
#endif
template <typename Real, typename Output>
#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldWorking(Real& greenTrace, Real x, Real y, Real z, Output& output) {
    using namespace math;
    evaluateEwaldWorking(greenTrace, x, y, z, &output[0]);
}

/// Evaluate the periodic correction with promoted working precision, then convert the result.
#if defined(COSMICTIGER_FMM_COUNTING_ONLY) || defined(COSMICTIGER_FMM_INSTRUMENTATION)
template <typename Real>

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(Real& greenTrace, Real x, Real y, Real z, Real* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(float& greenTrace, float x, float y, float z, float* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(double& greenTrace, double x, double y, double z, double* output);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(SimdVector<float, 8>& greenTrace, SimdVector<float, 8> x,
        SimdVector<float, 8> y, SimdVector<float, 8> z, SimdVector<float, 8>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(SimdVector<double, 2>& greenTrace, SimdVector<double, 2> x,
        SimdVector<double, 2> y, SimdVector<double, 2> z, SimdVector<double, 2>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(SimdVector<float, 4>& greenTrace, SimdVector<float, 4> x,
        SimdVector<float, 4> y, SimdVector<float, 4> z, SimdVector<float, 4>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(SimdVector<double, 4>& greenTrace, SimdVector<double, 4> x,
        SimdVector<double, 4> y, SimdVector<double, 4> z, SimdVector<double, 4>* output);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__

#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(SimdVector<double, 8>& greenTrace, SimdVector<double, 8> x,
        SimdVector<double, 8> y, SimdVector<double, 8> z, SimdVector<double, 8>* output);
#endif
#endif
template <typename Real, typename Output>
#ifdef __CUDA_ARCH__
__device__
#endif
    void evaluateEwaldGreen(Real& greenTrace, Real x, Real y, Real z, Output& output) {
    using namespace math;
    evaluateEwaldGreen(greenTrace, x, y, z, &output[0]);
}

}    // namespace cosmictiger::gravity::spherical
