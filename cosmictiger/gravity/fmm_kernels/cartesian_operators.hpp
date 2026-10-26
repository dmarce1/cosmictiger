/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_config.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_kernels.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_m2l.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/math/simd_math.hpp>

static_assert(cosmictiger::gravity::cartesian::kMinLocalDegree <= fmmLocalDegree(kFmmMinOrder) &&
        cosmictiger::gravity::cartesian::kMaxLocalDegree >= fmmLocalDegree(kFmmMaxOrder),
    "Generated Cartesian kernels do not cover the supported FMM orders; regenerate them");

// Application coordinates are scaled at the boundary. The typed FMM API below
// uses the same coordinates as its moments and expansions.
template <class Real>
CUDA_EXPORT Array<Real, kDimensions> fmmScaleDisplacement(Array<Real, kDimensions> value) {
    for (int dim = 0; dim < kDimensions; dim++) {
        value[dim] *= simdBroadcast<Real>(kScaleFactor);
    }
    return value;
}

namespace cosmictiger::gravity::cartesian {
namespace detail {
    template <class Real>
    CUDA_EXPORT Real inverseRadius(Array<Real, kDimensions> const& position) {
        Real const radius = simdSqrt(simdFma(position[0], position[0],
            simdFma(position[1], position[1], position[2] * position[2])));
        return simdSelect(radius > simdBroadcast<Real>(0), simdBroadcast<Real>(1),
                   simdBroadcast<Real>(0)) /
            simdMax(radius, simdBroadcast<Real>(1.0e-20));
    }
}    // namespace detail

template <class Real, int order>
CUDA_EXPORT int P2M(
    Multipole<Real, order>& target, Real mass, Array<Real, kDimensions> const& particleOffset) {
    detail::P2M<order - 2>::add(
        target, mass, particleOffset[0], particleOffset[1], particleOffset[2]);
    return detail::P2M<order - 2>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int P2L(Expansion<Real, order>& target, Real mass,
    Array<Real, kDimensions> const& displacement, bool doPotential = true) {
    Real const inverse = detail::inverseRadius(displacement);
    Real const oldPotential = target[0];
    detail::P2L<order - 1>::add(target, mass, displacement[0] * inverse, displacement[1] * inverse,
        displacement[2] * inverse, inverse, inverse);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::P2L<order - 1>::kFlops_ + 10;
}

template <class Real, int order>
CUDA_EXPORT int M2M(Multipole<Real, order>& target, Multipole<Real, order> const& source,
    Array<Real, kDimensions> const& displacement) {
    ASSERT(&target != &source);
    detail::M2M<order - 2>::add(
        target, source, displacement[0], displacement[1], displacement[2], simdBroadcast<Real>(1));
    return detail::M2M<order - 2>::kFlops_;
}

template <bool centered = false, class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Array<Real, kDimensions> const& displacement, bool doPotential = true) {
    Real const inverse = detail::inverseRadius(displacement);
    Real const oldPotential = target[0];
    detail::M2L<order - 1, centered>::template add<Real, Expansion<Real, order>,
        Multipole<Real, order>, order - 2>(target, source, displacement[0] * inverse,
        displacement[1] * inverse, displacement[2] * inverse, inverse, inverse, inverse);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2L<order - 1, centered>::kFlops_ + 10;
}

template <class Real, int order>
CUDA_EXPORT void evaluateGreen(
    Green<Real, order>& target, Array<Real, kDimensions> const& displacement) {
    Real const inverse = detail::inverseRadius(displacement);
    target = simdBroadcast<Real>(0);
    detail::P2L<order - 1>::add(target, simdBroadcast<Real>(1), displacement[0] * inverse,
        displacement[1] * inverse, displacement[2] * inverse, inverse, inverse);
}

template <bool centered = false, class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Green<Real, order> const& green, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::M2LDerivatives<order - 1, centered>::template add<Real>(target, source, green);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2LDerivatives<order - 1, centered>::kFlops_;
}

template <bool centered = false, class Real, int order>
CUDA_EXPORT int M2P(gravity::Field<Real>& target, Multipole<Real, order> const& source,
    Array<Real, kDimensions> const& displacement, bool doPotential = true) {
    Real const inverse = detail::inverseRadius(displacement);
    Real const oldPotential = target[0];
    detail::M2P<order - 2, centered>::add(target, source, displacement[0] * inverse,
        displacement[1] * inverse, displacement[2] * inverse, inverse, inverse);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2P<order - 2, centered>::kFlops_ + 10;
}

template <class Real, int order>
CUDA_EXPORT int L2L(Expansion<Real, order>& target, Expansion<Real, order> const& source,
    Array<Real, kDimensions> const& displacement, bool doPotential = true) {
    ASSERT(&target != &source);
    Real const oldPotential = target[0];
    detail::L2L<order - 1, true>::add(
        target, source, displacement[0], displacement[1], displacement[2], simdBroadcast<Real>(1));
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::L2L<order - 1, true>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int L2P(gravity::Field<Real>& target, Expansion<Real, order> const& source,
    Array<Real, kDimensions> const& displacement, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::L2P<order - 1, true>::add(
        target, source, displacement[0], displacement[1], displacement[2], simdBroadcast<Real>(1));
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::L2P<order - 1, true>::kFlops_;
}

#ifdef __CUDACC__
// Every warp lane participates. The raw target is caller-owned shared storage.
template <class Real, int order>
__device__ void L2LWarpInto(Real* target, Expansion<Real, order> const& source,
    Array<Real, kDimensions> const& displacement, bool doPotential) {
    detail::detail::L2LWarp<order - 1, true>::translateInto(
        target, source, displacement[0], displacement[1], displacement[2], doPotential);
}
#endif

}    // namespace cosmictiger::gravity::cartesian

// Weighted walk-choice costs use conventional FLOP counts, not a representation
// switch or measured runtime.
template <int order>
struct FmmHybridCost {
    static constexpr int kMultipoleDegree_ = FmmOrder<order>::kMultipoleDegree_;
    static constexpr int kLocalDegree_ = FmmOrder<order>::kLocalDegree_;
    static constexpr float kParticleCellPerParticle_ =
        float(cosmictiger::gravity::cartesian::detail::M2P<kMultipoleDegree_, false>::kFlops_) +
        18.0f + 7.0f;
    static constexpr float kCellParticlePerParticle_ =
        float(cosmictiger::gravity::cartesian::detail::P2L<kLocalDegree_>::kFlops_) + 14.0f + 3.0f +
        float(FmmOrder<order>::kLocalCoefficients_);
    static constexpr float kHybridPerParticle_ =
        0.5f * (kParticleCellPerParticle_ + kCellParticlePerParticle_);
    static constexpr float kParticlePair_ = 38.0f;
};
