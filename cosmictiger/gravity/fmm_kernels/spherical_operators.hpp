/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/gravity/fmm_kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>

namespace cosmictiger::gravity::spherical {

template <class Real, int order>
CUDA_EXPORT int P2M(
    Multipole<Real, order>& target, Real mass, Array<Real, NDIM> const& particleOffset) {
    detail::P2M<order - 2>::template add<Real>(
        target, mass, particleOffset[0], particleOffset[1], particleOffset[2]);
    return detail::P2M<order - 2>::kFlops_;
}

// The radial second moment is independent of the harmonic coefficients.
template <class Real, int order>
CUDA_EXPORT int P2M(Multipole<Real, order>& target, Real& targetTrace, Real mass,
    Array<Real, NDIM> const& particleOffset) {
    int const flops = P2M(target, mass, particleOffset);
    Real const radiusSquared = simdFma(particleOffset[0], particleOffset[0],
        simdFma(particleOffset[1], particleOffset[1], particleOffset[2] * particleOffset[2]));
    targetTrace = simdFma(mass, radiusSquared, targetTrace);
    return flops + 7;
}

template <class Real, int order>
CUDA_EXPORT int P2L(Expansion<Real, order>& target, Real mass,
    Array<Real, NDIM> const& displacement, bool doPotential = true) {
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
    Array<Real, NDIM> const& displacement) {
    ASSERT(&target != &source);
    detail::M2M<order - 2>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    return detail::M2M<order - 2>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int M2M(Multipole<Real, order>& target, Real& targetTrace,
    Multipole<Real, order> const& source, Real sourceTrace, Array<Real, NDIM> const& displacement) {
    Real const radiusSquared = simdFma(displacement[0], displacement[0],
        simdFma(displacement[1], displacement[1], displacement[2] * displacement[2]));
    // M_1^1 = -(sum mass*x + i*sum mass*y)/2 and M_1^0 = -sum mass*z.
    Real const firstMoment = simdFma(-simdBroadcast<Real>(4) * displacement[0], source[3],
        simdFma(-simdBroadcast<Real>(4) * displacement[1], source[1],
            -simdBroadcast<Real>(2) * displacement[2] * source[2]));
    targetTrace += sourceTrace + firstMoment + source[0] * radiusSquared;
    return M2M(target, source, displacement) + 20;
}

template <class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Array<Real, NDIM> const& displacement, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::M2L<order - 1>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2L<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT void evaluateGreen(Green<Real, order>& target, Array<Real, NDIM> const& displacement) {
    detail::Green<order - 1>::template evaluate<Real>(
        target, displacement[0], displacement[1], displacement[2]);
}

template <class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Multipole<Real, order> const& source,
    Green<Real, order> const& green, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::M2LDerivatives<order - 1>::template add<Real>(target, source, green);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::M2LDerivatives<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int M2P(gravity::Field<Real>& target, Multipole<Real, order> const& source,
    Array<Real, NDIM> const& displacement, bool doPotential = true) {
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
    Array<Real, NDIM> const& displacement, bool doPotential = true) {
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
    Array<Real, NDIM> const& displacement, bool doPotential = true) {
    Real const oldPotential = target[0];
    detail::L2P<order - 1>::template add<Real>(
        target, source, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return detail::L2P<order - 1>::kFlops_;
}

// The periodic Laplacian is independent of the harmonic coefficients.
template <class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order>& target, Real& localTrace,
    Multipole<Real, order> const& source, Real sourceTrace, Green<Real, order> const& green,
    Real greenTrace, bool doPotential = true) {
    Real const oldPotential = target[0];
    EwaldM2L<order - 1>::template add<Real>(
        target, localTrace, source, sourceTrace, green, greenTrace);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return EwaldM2L<order - 1>::kFlops_;
}

template <class Real, int order>
CUDA_EXPORT int L2L(Expansion<Real, order>& target, Real& targetTrace,
    Expansion<Real, order> const& source, Real sourceTrace, Array<Real, NDIM> const& displacement,
    bool doPotential = true) {
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
    Real sourceTrace, Array<Real, NDIM> const& displacement, bool doPotential = true) {
    Real const oldPotential = target[0];
    EwaldL2P<order - 1>::template add<Real>(
        target, source, sourceTrace, displacement[0], displacement[1], displacement[2]);
    if (!doPotential) {
        target[0] = oldPotential;
    }
    return EwaldL2P<order - 1>::kFlops_;
}

}    // namespace cosmictiger::gravity::spherical
