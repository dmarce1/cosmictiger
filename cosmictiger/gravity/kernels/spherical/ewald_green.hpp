/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once
#include <cosmictiger/gravity/kernels/spherical/ewald_common.hpp>

namespace cosmictiger::gravity::spherical {
template <>
inline constexpr int kEwaldGreenFlops<6> = 577958;
template <>
inline constexpr int kEwaldGreenConventionalFlops<6> = 627304;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 6)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared);
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
template <>
CUDA_EXPORT void ewaldRegular<6, float>(
    float* output, float x, float y, float z, float radiusSquared);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
template <>
CUDA_EXPORT void ewaldRegular<6, double>(
    double* output, double x, double y, double z, double radiusSquared);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void ewaldRegular<6, SimdVector<float, 8>>(SimdVector<float, 8>* output,
    SimdVector<float, 8> x, SimdVector<float, 8> y, SimdVector<float, 8> z,
    SimdVector<float, 8> radiusSquared);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void ewaldRegular<6, SimdVector<double, 2>>(SimdVector<double, 2>* output,
    SimdVector<double, 2> x, SimdVector<double, 2> y, SimdVector<double, 2> z,
    SimdVector<double, 2> radiusSquared);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void ewaldRegular<6, SimdVector<float, 4>>(SimdVector<float, 4>* output,
    SimdVector<float, 4> x, SimdVector<float, 4> y, SimdVector<float, 4> z,
    SimdVector<float, 4> radiusSquared);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void ewaldRegular<6, SimdVector<double, 4>>(SimdVector<double, 4>* output,
    SimdVector<double, 4> x, SimdVector<double, 4> y, SimdVector<double, 4> z,
    SimdVector<double, 4> radiusSquared);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void ewaldRegular<6, SimdVector<double, 8>>(SimdVector<double, 8>* output,
    SimdVector<double, 8> x, SimdVector<double, 8> y, SimdVector<double, 8> z,
    SimdVector<double, 8> radiusSquared);
#endif
#endif

/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real>
    requires(degree == 6)
CUDA_EXPORT void evaluateEwaldWorking(Real* output, Real& greenTrace, Real x, Real y, Real z);
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
template <>
CUDA_EXPORT void evaluateEwaldWorking<6, float>(
    float* output, float& greenTrace, float x, float y, float z);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
template <>
CUDA_EXPORT void evaluateEwaldWorking<6, double>(
    double* output, double& greenTrace, double x, double y, double z);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldWorking<6, SimdVector<float, 8>>(SimdVector<float, 8>* output,
    SimdVector<float, 8>& greenTrace, SimdVector<float, 8> x, SimdVector<float, 8> y,
    SimdVector<float, 8> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldWorking<6, SimdVector<double, 2>>(SimdVector<double, 2>* output,
    SimdVector<double, 2>& greenTrace, SimdVector<double, 2> x, SimdVector<double, 2> y,
    SimdVector<double, 2> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldWorking<6, SimdVector<float, 4>>(SimdVector<float, 4>* output,
    SimdVector<float, 4>& greenTrace, SimdVector<float, 4> x, SimdVector<float, 4> y,
    SimdVector<float, 4> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldWorking<6, SimdVector<double, 4>>(SimdVector<double, 4>* output,
    SimdVector<double, 4>& greenTrace, SimdVector<double, 4> x, SimdVector<double, 4> y,
    SimdVector<double, 4> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldWorking<6, SimdVector<double, 8>>(SimdVector<double, 8>* output,
    SimdVector<double, 8>& greenTrace, SimdVector<double, 8> x, SimdVector<double, 8> y,
    SimdVector<double, 8> z);
#endif
#endif
template <int degree, typename Real, typename Output>
    requires(degree == 6)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    evaluateEwaldWorking<degree, Real>(&output[0], greenTrace, x, y, z);
}

/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real>
    requires(degree == 6)
CUDA_EXPORT void evaluateEwaldGreen(Real* output, Real& greenTrace, Real x, Real y, Real z);
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
template <>
CUDA_EXPORT void evaluateEwaldGreen<6, float>(
    float* output, float& greenTrace, float x, float y, float z);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
template <>
CUDA_EXPORT void evaluateEwaldGreen<6, double>(
    double* output, double& greenTrace, double x, double y, double z);
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldGreen<6, SimdVector<float, 8>>(SimdVector<float, 8>* output,
    SimdVector<float, 8>& greenTrace, SimdVector<float, 8> x, SimdVector<float, 8> y,
    SimdVector<float, 8> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldGreen<6, SimdVector<double, 2>>(SimdVector<double, 2>* output,
    SimdVector<double, 2>& greenTrace, SimdVector<double, 2> x, SimdVector<double, 2> y,
    SimdVector<double, 2> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldGreen<6, SimdVector<float, 4>>(SimdVector<float, 4>* output,
    SimdVector<float, 4>& greenTrace, SimdVector<float, 4> x, SimdVector<float, 4> y,
    SimdVector<float, 4> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldGreen<6, SimdVector<double, 4>>(SimdVector<double, 4>* output,
    SimdVector<double, 4>& greenTrace, SimdVector<double, 4> x, SimdVector<double, 4> y,
    SimdVector<double, 4> z);
#endif
#endif
#ifndef COSMICTIGER_FMM_COUNTING_ONLY
#ifndef __CUDACC__
template <>
CUDA_EXPORT void evaluateEwaldGreen<6, SimdVector<double, 8>>(SimdVector<double, 8>* output,
    SimdVector<double, 8>& greenTrace, SimdVector<double, 8> x, SimdVector<double, 8> y,
    SimdVector<double, 8> z);
#endif
#endif
template <int degree, typename Real, typename Output>
    requires(degree == 6)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    evaluateEwaldGreen<degree, Real>(&output[0], greenTrace, x, y, z);
}

}    // namespace cosmictiger::gravity::spherical
