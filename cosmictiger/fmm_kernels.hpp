/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */
#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/ewald_kernels.hpp>
#include <cosmictiger/fmm_order.hpp>
#include <cosmictiger/fmm_scale.hpp>
#include <cosmictiger/fmm_types.hpp>
#include <cosmictiger/gravity/stf/generated/generated_config.hpp>
#include <cosmictiger/gravity/stf/generated/generated_m2l.hpp>
#include <cosmictiger/gravity/stf/generated/generated_operators.hpp>
#include <cosmictiger/simd_math.hpp>

// Order-templated FMM operators over the generated STF kernels. Every FMM order
// in [kFmmMinOrder, kFmmMaxOrder] has its own generated specialization; the
// wrappers deduce the degrees from their tensor arguments, so an instantiation
// for one order never touches another order's code.
static_assert(cosmictiger::gravity::stf::kMinLocalDegree <= fmmLocalDegree(kFmmMinOrder) &&
        cosmictiger::gravity::stf::kMaxLocalDegree >= fmmLocalDegree(kFmmMaxOrder),
    "Generated STF kernels do not cover the supported FMM orders; regenerate them");

namespace detail {
namespace stf = cosmictiger::gravity::stf::detail;

template <class Real>
CUDA_EXPORT Real fmmInverseRadius(Array<Real, NDIM> const& position) {
    Real const radius = simdSqrt(simdFma(
        position[0], position[0], simdFma(position[1], position[1], position[2] * position[2])));
    return simdSelect(
               radius > simdBroadcast<Real>(0), simdBroadcast<Real>(1), simdBroadcast<Real>(0)) /
        simdMax(radius, simdBroadcast<Real>(1.0e-20));
}
}    // namespace detail

template <int order, class Real>
CUDA_EXPORT FmmMultipole<Real, order> p2M(Array<Real, NDIM> const& position) {
    FmmMultipole<Real, order> result{};
    detail::stf::P2mKernel<fmmMultipoleDegree(order)>::add(result, simdBroadcast<Real>(1),
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR));
    return result;
}

template <class Real, int degree>
CUDA_EXPORT StfTensor<Real, degree> m2M(
    StfTensor<Real, degree> const& source, Array<Real, NDIM> position) {
    StfTensor<Real, degree> result{};
    detail::stf::M2mKernel<degree>::add(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), simdBroadcast<Real>(1));
    return result;
}

template <class Real, int degree>
CUDA_EXPORT int greensFunction(
    StfTensor<Real, degree, true>& local, Array<Real, NDIM> position, bool scale = true) {
    if (scale) {
        for (int dim = 0; dim < NDIM; dim++) {
            position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
        }
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    local = simdBroadcast<Real>(0);
    detail::stf::P2lKernel<degree>::add(local, simdBroadcast<Real>(1), position[0] * inverseRadius,
        position[1] * inverseRadius, position[2] * inverseRadius, inverseRadius, inverseRadius);
    return detail::stf::P2lKernel<degree>::kFlops_ + 14;
}

// Direct M2L uses the imported partial-contraction kernel, avoiding the
// intermediate Green tensor. centered=true is valid only for zero source dipole.
template <class Real, int localDegree, int sourceDegree, bool centered = false>
CUDA_EXPORT int m2LDirect(StfTensor<Real, localDegree, true>& local,
    StfTensor<Real, sourceDegree> const& source, Array<Real, NDIM> position, bool doPhi) {
    static_assert(localDegree == sourceDegree + 1, "Local degree is the source degree plus one");
    for (int dim = 0; dim < NDIM; dim++) {
        position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    Real const oldPotential = local[0];
    detail::stf::M2lKernel<localDegree, centered>::template add<Real,
        StfTensor<Real, localDegree, true>, StfTensor<Real, sourceDegree>, sourceDegree>(local,
        source, position[0] * inverseRadius, position[1] * inverseRadius,
        position[2] * inverseRadius, inverseRadius, inverseRadius, inverseRadius);
    if (!doPhi) {
        local[0] = oldPotential;
    }
    return detail::stf::M2lKernel<localDegree, centered>::kFlops_ + 14;
}

template <class Real, int sourceDegree, bool centered = false>
CUDA_EXPORT int m2P(Expansion2<Real>& local, StfTensor<Real, sourceDegree> const& source,
    Array<Real, NDIM> position, bool doPhi) {
    for (int dim = 0; dim < NDIM; dim++) {
        position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    Array<Real, NDIM + 1> field{};
    detail::stf::M2pKernel<sourceDegree, centered>::add(field, source,
        position[0] * inverseRadius, position[1] * inverseRadius, position[2] * inverseRadius,
        inverseRadius, inverseRadius);
    if (doPhi) {
        local[0] += field[0];
    }
    local(1, 0, 0) -= field[1];
    local(0, 1, 0) -= field[2];
    local(0, 0, 1) -= field[3];
    return detail::stf::M2pKernel<sourceDegree, centered>::kFlops_ + 18;
}

// Ewald is evaluated on the nearest-image branch. Harmonicize it using the
// unwrapped separation, so the removed quadratic sums to one root polynomial.
template <class Real, int degree>
CUDA_EXPORT int ewaldGreensFunction(StfTensor<Real, degree, true>& local,
    Array<Real, NDIM> const& wrappedPosition, Array<Real, NDIM> const& unwrappedPosition) {
    constexpr int kTensorOrder = degree + 1;
    SymmetricTracelessTensor<Real, kTensorOrder> derivatives;
    int const flops = detail::ewaldCorrectionDerivatives(derivatives, wrappedPosition);
    for (int rank = 0; rank <= degree; rank++) {
        for (int nz = 0; nz <= 1 && nz <= rank; nz++) {
            for (int nx = 0; nx <= rank - nz; nx++) {
                int const ny = rank - nx - nz;
                local(nx, ny, nz) = derivatives(nx, ny, nz);
            }
        }
    }
    Real const coefficient = simdBroadcast<Real>(4.0 * M_PI / 3.0);
    Real radiusSquared = simdBroadcast<Real>(0);
    for (int dim = 0; dim < NDIM; dim++) {
        radiusSquared = simdFma(unwrappedPosition[dim], unwrappedPosition[dim], radiusSquared);
        int const nx = dim == XDIM;
        int const ny = dim == YDIM;
        int const nz = dim == ZDIM;
        local(nx, ny, nz) +=
            coefficient * simdBroadcast<Real>(SCALE_FACTOR_INV2) * unwrappedPosition[dim];
    }
    local[0] += simdBroadcast<Real>(0.5 * SCALE_FACTOR_INV1) * coefficient * radiusSquared;
    local(2, 0, 0) += coefficient * simdBroadcast<Real>(SCALE_FACTOR_INV3);
    local(0, 2, 0) += coefficient * simdBroadcast<Real>(SCALE_FACTOR_INV3);
    local[StfTensor<Real, degree, true>::kCount_] = simdBroadcast<Real>(0);
    return flops + 25;
}

template <class Real, int degree>
CUDA_EXPORT int ewaldGreensFunction(
    StfTensor<Real, degree, true>& local, Array<Real, NDIM> const& position) {
    return ewaldGreensFunction(local, position, position);
}

template <class Real, int localDegree, int sourceDegree, bool centered = false>
CUDA_EXPORT int m2L(StfTensor<Real, localDegree, true>& local,
    StfTensor<Real, sourceDegree> const& source,
    StfTensor<Real, localDegree, true> const& derivatives, bool doPhi) {
    static_assert(localDegree == sourceDegree + 1, "Local degree is the source degree plus one");
    Real const oldPotential = local[0];
    detail::stf::EwaldM2lKernel<localDegree, centered>::template add<Real>(
        local, source, derivatives);
    if (!doPhi) {
        local[0] = oldPotential;
    }
    return detail::stf::EwaldM2lKernel<localDegree, centered>::kFlops_;
}

template <class Real, int degree>
CUDA_EXPORT StfTensor<Real, degree, true> l2L(
    StfTensor<Real, degree, true> const& source, Array<Real, NDIM> position, bool doPhi) {
    StfTensor<Real, degree, true> result{};
    detail::stf::L2lKernel<degree, true>::add(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), simdBroadcast<Real>(1));
    if (!doPhi) {
        result[0] = source[0];
    }
    return result;
}

template <class Real, int degree>
CUDA_EXPORT Expansion2<Real> l2P(
    StfTensor<Real, degree, true> const& source, Array<Real, NDIM> position, bool doPhi) {
    Array<Real, NDIM + 1> field{};
    detail::stf::L2pKernel<degree, true>::add(field, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), simdBroadcast<Real>(1));
    Expansion2<Real> result{};
    result[0] = doPhi ? field[0] : simdBroadcast<Real>(0);
    result(1, 0, 0) = -field[1];
    result(0, 1, 0) = -field[2];
    result(0, 0, 1) = -field[3];
    return result;
}

template <int order>
inline void initializeBackgroundExpansion(FmmExpansion<float, order>& local,
    FmmMultipole<float, order> const& rootMultipole, double secondMomentTrace) {
    double const coefficient = 4.0 * M_PI / 3.0;
    local = 0.0f;
    local[0] = -0.5 * coefficient * secondMomentTrace * SCALE_FACTOR_INV1;
    local(1, 0, 0) = coefficient * rootMultipole(1, 0, 0) * SCALE_FACTOR_INV3;
    local(0, 1, 0) = coefficient * rootMultipole(0, 1, 0) * SCALE_FACTOR_INV3;
    local(0, 0, 1) = coefficient * rootMultipole(0, 0, 1) * SCALE_FACTOR_INV3;
    local(2, 0, 0) = local(0, 2, 0) = -coefficient * rootMultipole[0] * SCALE_FACTOR_INV3;
    local[FmmOrder<order>::kLocalTraceIndex_] = -4.0 * M_PI * rootMultipole[0] * SCALE_FACTOR_INV3;
}

#ifdef __CUDACC__
template <class Real, int degree>
__device__ StfTensor<Real, degree, true> l2LCuda(
    StfTensor<Real, degree, true> const& source, Array<Real, NDIM> position, bool doPhi) {
    StfTensor<Real, degree, true> result{};
    detail::stf::L2lWarpKernel<degree, true>::translate(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), doPhi);
    return result;
}

// Cooperative translation: group leaders write their coefficients straight into
// `target`, which must not alias `source`. Complete on return for every lane.
template <class Real, int degree>
__device__ void l2LCudaInto(Real* target, StfTensor<Real, degree, true> const& source,
    Array<Real, NDIM> position, bool doPhi) {
    detail::stf::L2lWarpKernel<degree, true>::translateInto(target, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), doPhi);
}
#endif
