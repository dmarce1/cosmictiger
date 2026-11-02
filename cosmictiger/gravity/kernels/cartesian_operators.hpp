/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kernels/cartesian_kernels.hpp>
#include <cosmictiger/gravity/kernels/cartesian_types.hpp>
#include <cosmictiger/gravity/kernels/gaussian_screening.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace cosmictiger::gravity::cartesian {

// Coordinates use the same physical units as the moments and expansions.
// These wrappers do not apply the application's normalized-coordinate scaling.
// All seven operators add to their destination; translations require distinct
// source and destination objects. Displacement means source center minus target
// center for M2M, target center minus source center for M2L/P2L/M2P, and new
// center or evaluation position minus old center for L2L/L2P.

template <class Real, int order>
CUDA_EXPORT int P2M(
    Multipole<Real, order>& target, Real mass, Vector3<Real> const& particleOffset) {
    detail::P2M<order - 2>::template add<Real>(
        target, mass, particleOffset[0], particleOffset[1], particleOffset[2]);
    return detail::P2M<order - 2>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int M2M(Multipole<Real, order>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement) {
    ASSERT(&target != &source);
    detail::M2M<order - 2>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    return detail::M2M<order - 2>::kFlops_;
}

// Evaluate derivatives of -(r^2 + epsilonSquared)^(-1/2), including traces.
// epsilonSquared must be nonnegative; r^2 + epsilonSquared must be positive.
// This overwrites the Green tensor. A positive epsilonSquared permits r = 0.
template <class Real, int order>
CUDA_EXPORT void evaluateGreen(
    Green<Real, order>& target, Vector3<Real> const& displacement, Real epsilonSquared) {
    detail::PlummerDerivatives<order - 1>::template evaluate<Real>(
        target, displacement[0], displacement[1], displacement[2], epsilonSquared);
}

template <class Real, int order>
CUDA_EXPORT void evaluateGreen(Green<Real, order>& target, Vector3<Real> const& displacement) {
    evaluateGreen(target, displacement, simdBroadcast<Real>(0));
}

// Derivatives of -erfc(alpha*r)/r (Short) or -erf(alpha*r)/r (Long).
// alpha has inverse coordinate units. Short requires r>0; Long includes r=0.
// All traces are retained. No force softening or periodic corrections are added.
template <GaussianComponent component = GaussianComponent::Short, class Real, int order>
CUDA_EXPORT void evaluateGreen(Green<Real, order>& target, Vector3<Real> const& displacement,
    GaussianScreening<SimdScalar<Real>> const& screening) {
    Real const radiusSquared = simdFma(displacement[0], displacement[0],
        simdFma(displacement[1], displacement[1], displacement[2] * displacement[2]));
    auto const radial = evaluateGaussianRadial<component, order - 1>(radiusSquared, screening);
    Real const inverseRadius = radial.inverseRadius();
    detail::RadialDerivatives<order - 1>::template evaluate<Real>(target,
        displacement[0] * inverseRadius, displacement[1] * inverseRadius,
        displacement[2] * inverseRadius, radial);
}

template <class Real, int order>
CUDA_EXPORT int P2L(Expansion<Real, order>& target, Real mass, Green<Real, order> const& green,
    bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::P2L<order - 1>::template add<Real>(target, mass, green);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::P2L<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int P2L(Expansion<Real, order>& target, Real mass, Vector3<Real> const& displacement,
    Real epsilonSquared, bool doPotential = true) {
    Green<Real, order> green;
    evaluateGreen(green, displacement, epsilonSquared);
    return P2L(target, mass, green, doPotential) + detail::PlummerDerivatives<order - 1>::kFlops_;
}

// Require a bool here so an epsilonSquared of the wrong scalar type cannot
// silently select the Newtonian overload by conversion to bool.
template <class Real, int order, class Potential = bool>
    requires std::is_same_v<Potential, bool>
CUDA_EXPORT int P2L(Expansion<Real, order>& target, Real mass, Vector3<Real> const& displacement,
    Potential doPotential = true) {
    return P2L(target, mass, displacement, simdBroadcast<Real>(0), doPotential);
}

// centered=true is an explicit caller promise of zero dipoles. All other source
// moments, including odd ranks three and higher, remain in the contraction.
// M2L retains terms with total source plus local degree at most order - 1.
template <bool centered = false, class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Green<Real, order> const& green, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::M2L<order - 1, centered>::template add<Real>(target, source, green);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2L<order - 1, centered>::kFlops_;
}

template <bool centered = false, class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, Real epsilonSquared, bool doPotential = true) {
    Green<Real, order> green;
    evaluateGreen(green, displacement, epsilonSquared);
    return M2L<centered>(target, source, green, doPotential) +
        detail::PlummerDerivatives<order - 1>::kFlops_;
}

template <bool centered = false, class Real, int order, class Potential = bool>
    requires std::is_same_v<Potential, bool>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, Potential doPotential = true) {
    return M2L<centered>(target, source, displacement, simdBroadcast<Real>(0), doPotential);
}

template <bool centered = false, class Real, int order>
CUDA_EXPORT int M2P(gravity::Field<Real>& target, Multipole<Real, order> const& source,
    Green<Real, order> const& green, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::M2P<order - 2, centered>::template add<Real>(target, source, green);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2P<order - 2, centered>::kFlops_;
}

template <bool centered = false, class Real, int order>
CUDA_EXPORT int M2P(gravity::Field<Real>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, Real epsilonSquared, bool doPotential = true) {
    Green<Real, order> green;
    evaluateGreen(green, displacement, epsilonSquared);
    return M2P<centered>(target, source, green, doPotential) +
        detail::PlummerDerivatives<order - 1>::kFlops_;
}

template <bool centered = false, class Real, int order, class Potential = bool>
    requires std::is_same_v<Potential, bool>
CUDA_EXPORT int M2P(gravity::Field<Real>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, Potential doPotential = true) {
    return M2P<centered>(target, source, displacement, simdBroadcast<Real>(0), doPotential);
}

template <class Real, int order>
CUDA_EXPORT int L2L(Expansion<Real, order>& target, Expansion<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    ASSERT(&target != &source);
    Real const oldPotential = target[0];
    detail::L2L<order - 1>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::L2L<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int L2P(gravity::Field<Real>& target, Expansion<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::L2P<order - 1>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::L2P<order - 1>::kFlops_;
}

}    // namespace cosmictiger::gravity::cartesian
