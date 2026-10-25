/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */
#pragma once

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/generated_cartesian_m2l.hpp>
#include <cosmictiger/gravity/generated_cartesian_operators.hpp>
#include <cosmictiger/gravity/generated_config.hpp>
#include <cosmictiger/math/simd_math.hpp>

// Order-templated FMM operators over the generated Cartesian kernels. Every FMM
// order in [kFmmMinOrder, kFmmMaxOrder] has its own generated specialization; the wrappers deduce
// the degrees from their tensor arguments, so an instantiation for one order never touches another
// order's code.
static_assert(cosmictiger::gravity::cartesian::kMinLocalDegree <= fmmLocalDegree(kFmmMinOrder) &&
        cosmictiger::gravity::cartesian::kMaxLocalDegree >= fmmLocalDegree(kFmmMaxOrder),
    "Generated Cartesian kernels do not cover the supported FMM orders; "
    "regenerate them");

namespace detail {
namespace cartesian = cosmictiger::gravity::cartesian::detail;

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
    detail::cartesian::P2mKernel<fmmMultipoleDegree(order)>::add(result, simdBroadcast<Real>(1),
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR));
    return result;
}

template <class Real, int degree>
CUDA_EXPORT Cartesian<Real, degree> m2M(
    Cartesian<Real, degree> const& source, Array<Real, NDIM> position) {
    Cartesian<Real, degree> result{};
    detail::cartesian::M2mKernel<degree>::add(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), simdBroadcast<Real>(1));
    return result;
}

template <class Real, int degree>
CUDA_EXPORT int greensFunction(
    Cartesian<Real, degree, true>& local, Array<Real, NDIM> position, bool scale = true) {
    if (scale) {
        for (int dim = 0; dim < NDIM; dim++) {
            position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
        }
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    local = simdBroadcast<Real>(0);
    detail::cartesian::P2lKernel<degree>::add(local, simdBroadcast<Real>(1),
        position[0] * inverseRadius, position[1] * inverseRadius, position[2] * inverseRadius,
        inverseRadius, inverseRadius);
    return detail::cartesian::P2lKernel<degree>::kFlops_ + 10 + 3 * int(scale);
}

// Direct M2L uses the imported partial-contraction kernel, avoiding the
// intermediate Green tensor. centered=true is valid only for zero source dipole.
template <class Real, int localDegree, int sourceDegree, bool centered = false>
CUDA_EXPORT int m2LDirect(Cartesian<Real, localDegree, true>& local,
    Cartesian<Real, sourceDegree> const& source, Array<Real, NDIM> position, bool doPhi) {
    static_assert(localDegree == sourceDegree + 1, "Local degree is the source degree plus one");
    for (int dim = 0; dim < NDIM; dim++) {
        position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    Real const oldPotential = local[0];
    detail::cartesian::M2lKernel<localDegree, centered>::template add<Real,
        Cartesian<Real, localDegree, true>, Cartesian<Real, sourceDegree>, sourceDegree>(local,
        source, position[0] * inverseRadius, position[1] * inverseRadius,
        position[2] * inverseRadius, inverseRadius, inverseRadius, inverseRadius);
    if (!doPhi) {
        local[0] = oldPotential;
    }
    return detail::cartesian::M2lKernel<localDegree, centered>::kFlops_ + 13;
}

template <class Real, int sourceDegree, bool centered = false>
CUDA_EXPORT int m2P(Expansion2<Real>& local, Cartesian<Real, sourceDegree> const& source,
    Array<Real, NDIM> position, bool doPhi) {
    for (int dim = 0; dim < NDIM; dim++) {
        position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    Array<Real, NDIM + 1> field{};
    detail::cartesian::M2pKernel<sourceDegree, centered>::add(field, source,
        position[0] * inverseRadius, position[1] * inverseRadius, position[2] * inverseRadius,
        inverseRadius, inverseRadius);
    if (doPhi) {
        local[0] += field[0];
    }
    local(1, 0, 0) -= field[1];
    local(0, 1, 0) -= field[2];
    local(0, 0, 1) -= field[3];
    return detail::cartesian::M2pKernel<sourceDegree, centered>::kFlops_ + 16 + int(doPhi);
}

template <class Real, int degree>
CUDA_EXPORT Cartesian<Real, degree, true> l2L(
    Cartesian<Real, degree, true> const& source, Array<Real, NDIM> position, bool doPhi) {
    Cartesian<Real, degree, true> result{};
    detail::cartesian::L2lKernel<degree, true>::add(result, source,
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
    Cartesian<Real, degree, true> const& source, Array<Real, NDIM> position, bool doPhi) {
    Array<Real, NDIM + 1> field{};
    detail::cartesian::L2pKernel<degree, true>::add(field, source,
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

#ifdef __CUDACC__
template <class Real, int degree>
__device__ Cartesian<Real, degree, true> l2LCuda(
    Cartesian<Real, degree, true> const& source, Array<Real, NDIM> position, bool doPhi) {
    Cartesian<Real, degree, true> result{};
    detail::cartesian::L2lWarpKernel<degree, true>::translate(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), doPhi);
    return result;
}

// Cooperative translation: group leaders write their coefficients straight into
// `target`, which must not alias `source`. Complete on return for every lane.
template <class Real, int degree>
__device__ void l2LCudaInto(Real* target, Cartesian<Real, degree, true> const& source,
    Array<Real, NDIM> position, bool doPhi) {
    detail::cartesian::L2lWarpKernel<degree, true>::translateInto(target, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), doPhi);
}
#endif

// Flop costs of the interactions that can replace direct summation for a leaf
// pair, from the generated kernel counts plus the per-particle overhead of the
// CPU operators. Both backends use these constants, so the choice is the same
// on either side of a CPU/CUDA handoff.
template <int order>
struct FmmHybridCost {
    static constexpr int kMultipoleDegree_ = FmmOrder<order>::kMultipoleDegree_;
    static constexpr int kLocalDegree_ = FmmOrder<order>::kLocalDegree_;
    // M2P per sink particle: the kernel, the displacement and the accumulation.
    static constexpr float kParticleCellPerParticle_ =
        float(detail::cartesian::M2pKernel<kMultipoleDegree_, false>::kFlops_) + 18.0f + 7.0f;
    // P2L per source particle: the Green function, the displacement and the
    // accumulation into the local expansion.
    static constexpr float kCellParticlePerParticle_ =
        float(detail::cartesian::P2lKernel<kLocalDegree_>::kFlops_) + 14.0f + 3.0f +
        float(FmmOrder<order>::kLocalCoefficients_);
    // Both walks evaluate a pairing: every explicit particle costs one M2P on
    // its own side and one P2L on the other side, so charge the mean per side.
    static constexpr float kHybridPerParticle_ =
        0.5f * (kParticleCellPerParticle_ + kCellParticlePerParticle_);
    // Newtonian pair in the direct kernel. Hybrids are considered only for
    // support-separated pairs, which never take the softened branch.
    static constexpr float kParticlePair_ = 38.0f;
};
