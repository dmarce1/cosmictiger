/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

// Symbol directory (typed real-spherical FMM operators):
// M: source or destination multipole; L: source or destination local expansion.
// G: free-space Green harmonics; R: source-target separation radius.
// The radial source moment is independent of the harmonic coefficients.
// A completed periodic local may use an implicit known background curvature.

#pragma once

#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/kernels/spherical_kernels.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace cosmictiger::gravity::spherical {

template <class Real, int order>
CUDA_EXPORT int P2M(
    Multipole<Real, order>& target, Real mass, Vector3<Real> const& particleOffset) {
    using namespace math;
    detail::P2M<order - 2>::template add<Real>(
        target, mass, particleOffset[0], particleOffset[1], particleOffset[2]);
    return detail::P2M<order - 2>::kFlops_;
}

// The radial second moment is independent of the harmonic coefficients.
template <class Real, int order>
CUDA_EXPORT int P2M(Multipole<Real, order>& target, Real& targetTrace, Real mass,
    Vector3<Real> const& particleOffset) {
    using namespace math;
    Fma const fma{};
    int const flops = P2M(target, mass, particleOffset);
    Real const radiusSquared = fma(particleOffset[0], particleOffset[0],
        fma(particleOffset[1], particleOffset[1], particleOffset[2] * particleOffset[2]));
    targetTrace = fma(mass, radiusSquared, targetTrace);
    return flops + 7;
}

template <class Real, int order>
CUDA_EXPORT int P2L(Expansion<Real, order>& target, Real mass, Vector3<Real> const& displacement,
    bool doPotential = true) {
    using namespace math;
    Real const oldPotential = target[0];
    detail::P2L<order - 1>::template add<Real>(
        target, mass, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::P2L<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int M2M(Multipole<Real, order>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement) {
    using namespace math;
    ASSERT(&target != &source);
    detail::M2M<order - 2>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    return detail::M2M<order - 2>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int M2M(Multipole<Real, order>& target, Real& targetTrace,
    Multipole<Real, order> const& source, Real sourceTrace, Vector3<Real> const& displacement) {
    using namespace math;
    Fma const fma{};
    Real const radiusSquared = fma(displacement[0], displacement[0],
        fma(displacement[1], displacement[1], displacement[2] * displacement[2]));
    // M_1^1 = -(sum mass*x + i*sum mass*y)/2 and M_1^0 = -sum mass*z.
    Real const firstMoment = fma(-broadcast<Real>(4) * displacement[0], source[3],
        fma(-broadcast<Real>(4) * displacement[1], source[1],
            -broadcast<Real>(2) * displacement[2] * source[2]));
    targetTrace += sourceTrace + firstMoment + source[0] * radiusSquared;
    return M2M(target, source, displacement) + 20;
}

template <class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    using namespace math;
    Real const oldPotential = target[0];
    detail::M2L<order - 1>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2L<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT void evaluateGreen(Green<Real, order>& target, Vector3<Real> const& displacement) {
    using namespace math;
    detail::Green<order - 1>::template evaluate<Real>(
        target, displacement[0], displacement[1], displacement[2]);
}

template <class Real, int order>
CUDA_EXPORT int M2P(gravity::Field<Real>& target, Multipole<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    using namespace math;
    Real const oldPotential = target[0];
    detail::M2P<order - 1>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2P<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int L2L(Expansion<Real, order>& target, Expansion<Real, order> const& source,
    Vector3<Real> const& displacement, bool doPotential = true) {
    using namespace math;
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
    using namespace math;
    Real const oldPotential = target[0];
    detail::L2P<order - 1>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::L2P<order - 1>::kFlops_;
}

// Derived for the negative Newtonian periodic potential:
// ∇²Φ = -3β, Φ(x) = H′(x) - 3β z² / 2, β = 4π M / (3V).
// The harmonic proxy H′ is initialized once at the completed far local.
// Symbol directory: β is the known uniform-background coefficient;
// H′ is the compact harmonic proxy; z is displacement from the local center.
// Code names: backgroundBeta = β; curvature = -3β.
template <class Real>
struct PeriodicBackground {
    Real backgroundBeta;
};

template <class Real, int order>
CUDA_EXPORT void preparePeriodicLocal(
    Expansion<Real, order>& target, PeriodicBackground<Real> const& background) {
    using namespace math;
    // Derived: U₂⁰ = (2z² - x² - y²)/4; add 2β once to the harmonic local.
    target.real(2, 0) += broadcast<Real>(2) * background.backgroundBeta;
}

template <class Real, int order>
CUDA_EXPORT int translatePeriodicLocal(Expansion<Real, order>& target,
    Expansion<Real, order> const& source, Vector3<Real> const& displacement,
    PeriodicBackground<Real> const& background, bool doPotential = true) {
    using namespace math;
    Real const curvature = -broadcast<Real>(3) * background.backgroundBeta;
    Real const z = displacement[2];
    int const flops = L2L(target, source, displacement, doPotential);
    target.real(1, 0) = math::fma(curvature, z, target.real(1, 0));
    if (doPotential) {
        target.real(0, 0) = math::fma(curvature * broadcast<Real>(0.5) * z, z, target.real(0, 0));
    }
    return flops + (doPotential ? 5 : 2);
}

template <class Real, int order>
CUDA_EXPORT int evaluatePeriodicLocal(gravity::Field<Real>& target,
    Expansion<Real, order> const& source, Vector3<Real> const& displacement,
    PeriodicBackground<Real> const& background, bool doPotential = true) {
    using namespace math;
    Real const curvature = -broadcast<Real>(3) * background.backgroundBeta;
    Real const z = displacement[2];
    int const flops = L2P(target, source, displacement, doPotential);
    target[3] = math::fma(-curvature, z, target[3]);
    if (doPotential) {
        target[0] = math::fma(curvature * broadcast<Real>(0.5) * z, z, target[0]);
    }
    return flops + (doPotential ? 5 : 2);
}

template <class Real, int order>
CUDA_EXPORT int L2L(Expansion<Real, order>& target, Real& targetTrace,
    Expansion<Real, order> const& source, Real sourceTrace, Vector3<Real> const& displacement,
    bool doPotential = true) {
    using namespace math;
    ASSERT(&target != &source);
    Real const oldPotential = target[0];
    EwaldL2L<order - 1>::template add<Real>(target, targetTrace, source, sourceTrace,
        displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return EwaldL2L<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int L2P(gravity::Field<Real>& target, Expansion<Real, order> const& source,
    Real sourceTrace, Vector3<Real> const& displacement, bool doPotential = true) {
    using namespace math;
    Real const oldPotential = target[0];
    EwaldL2P<order - 1>::template add<Real>(
        target, source, sourceTrace, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return EwaldL2P<order - 1>::kFlops_;
}

}    // namespace cosmictiger::gravity::spherical
