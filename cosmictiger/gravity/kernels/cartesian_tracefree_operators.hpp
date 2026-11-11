/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

// Symbol directory (typed Cartesian FMM operators):
// R: source-target separation radius; iR: 1/R.
// M: source or destination multipole; L: source or destination local expansion.
// u: separation divided by R; G: derivatives of the negative Newtonian Green function.
// Particle and center displacements use the same coordinates as M and L.

#pragma once

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/l2l.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/l2l_warp.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/l2p.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/m2l.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/m2m.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/m2p.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/p2l.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree/p2m.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_config.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>

static_assert(
    cosmictiger::gravity::cartesian::kMinLocalDegree <= fmmLocalDegree(kFmmOrder) &&
        cosmictiger::gravity::cartesian::kMaxLocalDegree >= fmmLocalDegree(kFmmOrder),
    "Generated Cartesian kernels do not cover the supported FMM orders; regenerate them");

// Application coordinates are scaled at the boundary. The typed FMM API below
// uses the same coordinates as its moments and expansions.
template <class Real>
CUDA_EXPORT Vector3<Real> fmmScaleDisplacement(Vector3<Real> value) {
    using namespace math;
    for (int dim = 0; dim < kDimensions; dim++) {
        value[dim] *= broadcast<Real>(kScaleFactor);
    }
    return value;
}

namespace cosmictiger::gravity::cartesian {
/// Accumulate a point mass into the multipole and return the generated operation count.
template <class Real, int order>
CUDA_EXPORT int P2M(
    Multipole<Real, order>& target, Real mass, Vector3<Real> const& particleOffset) {
    using namespace math;
    detail::P2M(target, mass, particleOffset);
    return detail::kP2MFlops<order - 2>;
}

/// Accumulate a point-mass local expansion, optionally preserving the potential
/// coefficient.
template <class Real, int order>
CUDA_EXPORT int P2L(Expansion<Real, order>& target, Real mass, Vector3<Real> const& displacement,
    bool doPotential = true) {
    using namespace math;
    Real const oldPotential = target[0];
    detail::P2L(target, mass, displacement);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::kP2LFlops<order - 1> + 7;
}

/// Accumulate translated moments at unit scale ratio; source and target must be distinct.
template <class Real, int order>
CUDA_EXPORT int M2M(Multipole<Real, order>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement) {
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(1.000000000000000000E+00);
    using namespace math;
    ASSERT(&target != &source);
    detail::M2M(target, source, displacement, broadcast<Real>(k0));
    return detail::kM2MFlops<order - 2>;
}

/// Accumulate the source multipole's local field, optionally preserving the potential
/// coefficient.
template <class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    using namespace math;
    Real const oldPotential = target[0];
    detail::M2L(target, source, displacement);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::kM2LFlops<order - 1> + 7;
}

/// Replace the coefficients with the unit-mass Newtonian Green expansion.
template <class Real, int order>
CUDA_EXPORT void evaluateGreen(Green<Real, order>& target, Vector3<Real> const& displacement) {
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(1.000000000000000000E+00);
    using namespace math;
    target = broadcast<Real>(k0);
    detail::P2L(target, broadcast<Real>(k1), displacement);
}

/// Accumulate multipole potential and acceleration, optionally preserving potential.
template <class Real, int order>
CUDA_EXPORT int M2P(gravity::Field<Real>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    using namespace math;
    Real const oldPotential = target[0];
    detail::M2P(target, source, displacement);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::kM2PFlops<order - 2> + 7;
}

/// Accumulate a translated local expansion; source and target must be distinct.
template <class Real, int order>
CUDA_EXPORT int L2L(Expansion<Real, order>& target, Expansion<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(1.000000000000000000E+00);
    using namespace math;
    ASSERT(&target != &source);
    Real const oldPotential = target[0];
    detail::L2L(target, source, displacement, broadcast<Real>(k0));
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::kL2LFlops<order - 1>;
}

/// Accumulate local potential and acceleration at unit scale ratio, optionally preserving
/// potential.
template <class Real, int order>
CUDA_EXPORT int L2P(gravity::Field<Real>& target, Expansion<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(1.000000000000000000E+00);
    using namespace math;
    Real const oldPotential = target[0];
    detail::L2P(target, source, displacement, broadcast<Real>(k0));
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::kL2PFlops<order - 1>;
}

#ifdef __CUDACC__
// Every warp lane participates. The raw target is caller-owned shared storage.
template <class Real, int order>
__device__ void L2LWarpInto(Real* target, Expansion<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential) {
    detail::L2LWarpInto(target, source, displacement, doPotential);
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
        float(cosmictiger::gravity::cartesian::detail::kM2PFlops<kMultipoleDegree_>) +
        15.0f + 7.0f;
    static constexpr float kCellParticlePerParticle_ =
        float(cosmictiger::gravity::cartesian::detail::kP2LFlops<kLocalDegree_>) +
        11.0f + 3.0f + float(FmmOrder<order>::kLocalCoefficients_);
    static constexpr float kHybridPerParticle_ =
        0.5f * (kParticleCellPerParticle_ + kCellParticlePerParticle_);
    static constexpr float kParticlePair_ = 38.0f;
    // M2L arithmetic and displacement scaling, excluding SIMD padding and
    // reductions shared by a list. This is a work estimate, not a timing model.
    static constexpr float kCellCell_ =
        float(cosmictiger::gravity::cartesian::detail::kM2LFlops<kLocalDegree_>) + 10.0f;
};
