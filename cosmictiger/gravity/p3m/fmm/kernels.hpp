/* CosmicTiger - Gaussian-screened FMM kernel adapter; BSL-1.0. */

#pragma once

#include <cosmictiger/gravity/kernels/cartesian_operators.hpp>

namespace cosmictiger::gravity::p3m::fmm {

// Fill the full symmetric Cartesian derivatives of -erfc(alpha*r)/r.
// Requires r>0. The PM solve supplies the complementary long-range field.
// The supplied-Green Cartesian operators perform the subsequent contractions.
template <class Real, int order>
CUDA_EXPORT void evaluateGreen(cartesian::Green<Real, order>& target,
    Vector3<Real> const& displacement, GaussianScreening<SimdScalar<Real>> const& screening) {
    cartesian::evaluateGreen<GaussianComponent::Short>(target, displacement, screening);
}

}    // namespace cosmictiger::gravity::p3m::fmm
