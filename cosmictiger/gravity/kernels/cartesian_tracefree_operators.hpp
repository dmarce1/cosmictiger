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

static_assert(cosmictiger::gravity::cartesian::kMinLocalDegree <= fmmLocalDegree(kFmmOrder) &&
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
/// Accumulate a point-mass local expansion, optionally preserving the potential
/// coefficient.
template <class Real>
CUDA_EXPORT int P2L(
    Real mass, Vector3<Real> const& displacement, bool doPotential, Expansion<Real>& target) {
    using namespace math;
    Real const oldPotential = target[0];
    P2L(mass, displacement, target);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return kP2LFlops<kFmmOrder - 1> + 7;
}

/// Accumulate translated moments at unit scale ratio; source and target must be distinct.
template <class Real>
CUDA_EXPORT int M2M(
    Multipole<Real> const& source, Vector3<Real> const& displacement, Multipole<Real>& target) {
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(1.000000000000000000E+00);
    using namespace math;
    ASSERT(&target != &source);
    M2M(source, displacement, broadcast<Real>(k0), target);
    return kM2MFlops<kFmmOrder - 2>;
}

/// Accumulate the source multipole's local field, optionally preserving the potential
/// coefficient.
template <class Real>
CUDA_EXPORT int M2L(Multipole<Real> const& source, Vector3<Real> const& displacement,
    bool doPotential, Expansion<Real>& target) {
    using namespace math;
    Real const oldPotential = target[0];
    M2L(source, displacement, target);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return kM2LFlops<kFmmOrder - 1> + 7;
}

/// Replace the coefficients with the unit-mass Newtonian Green expansion.
template <class Real>
CUDA_EXPORT void evaluateGreen(Vector3<Real> const& displacement, Green<Real>& target) {
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(1.000000000000000000E+00);
    using namespace math;
    target = broadcast<Real>(k0);
    P2L(broadcast<Real>(k1), displacement, target);
}

/// Accumulate multipole potential and acceleration, optionally preserving potential.
template <class Real>
CUDA_EXPORT int M2P(Multipole<Real> const& source, Vector3<Real> const& displacement,
    bool doPotential, gravity::Field<Real>& target) {
    using namespace math;
    Real const oldPotential = target[0];
    M2P(source, displacement, target);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return kM2PFlops<kFmmOrder - 2> + 7;
}

/// Accumulate a translated local expansion; source and target must be distinct.
template <class Real>
CUDA_EXPORT int L2L(Expansion<Real> const& source, Vector3<Real> const& displacement,
    bool doPotential, Expansion<Real>& target) {
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(1.000000000000000000E+00);
    using namespace math;
    ASSERT(&target != &source);
    Real const oldPotential = target[0];
    L2L(source, displacement, broadcast<Real>(k0), target);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return kL2LFlops<kFmmOrder - 1>;
}

/// Accumulate local potential and negative gradient in the displacement coordinates,
/// optionally preserving potential.
template <class Real>
CUDA_EXPORT int L2P(Expansion<Real> const& source, Vector3<Real> const& displacement,
    bool doPotential, gravity::Field<Real>& target) {
    using namespace math;
    Real const oldPotential = target[0];
    L2P(source, displacement, target);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return kL2PFlops<kFmmOrder - 1>;
}

/// Accumulate all components, including the potential.
template <class Real>
CUDA_EXPORT int L2L(
    Expansion<Real> const& source, Vector3<Real> const& displacement, Expansion<Real>& target) {
    return L2L(source, displacement, true, target);
}

}    // namespace cosmictiger::gravity::cartesian

// Weighted walk-choice costs use conventional FLOP counts, not a representation
// switch or measured runtime.
template <int order>
struct FmmHybridCost {
    static constexpr int kMultipoleDegree_ = FmmOrder<order>::kMultipoleDegree_;
    static constexpr int kLocalDegree_ = FmmOrder<order>::kLocalDegree_;
    static constexpr float kParticleCellPerParticle_ =
        float(cosmictiger::gravity::cartesian::kM2PFlops<kMultipoleDegree_>) + 15.0f + 7.0f;
    static constexpr float kCellParticlePerParticle_ =
        float(cosmictiger::gravity::cartesian::kP2LFlops<kLocalDegree_>) + 11.0f + 3.0f +
        float(FmmOrder<order>::kLocalCoefficients_);
    static constexpr float kHybridPerParticle_ =
        0.5f * (kParticleCellPerParticle_ + kCellParticlePerParticle_);
    static constexpr float kParticlePair_ = 38.0f;
    // M2L arithmetic and displacement scaling, excluding SIMD padding and
    // reductions shared by a list. This is a work estimate, not a timing model.
    static constexpr float kCellCell_ =
        float(cosmictiger::gravity::cartesian::kM2LFlops<kLocalDegree_>) + 10.0f;
};
