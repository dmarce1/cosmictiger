/* CosmicTiger - Gaussian-screened particle kernel; BSL-1.0. */

#pragma once

#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kernels/gaussian_screening.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace cosmictiger::gravity::p3m::pp {

// Unit-source-mass short field for target minus source displacement. The
// displacement and screening parameter use the same physical coordinates.
// Requires r>0; physical force softening and periodic images are separate.
template <class Real>
CUDA_EXPORT Field<Real> evaluateField(
    Vector3<Real> const& displacement, GaussianScreening<SimdScalar<Real>> const& screening) {
    Real const radiusSquared = simdFma(displacement[0], displacement[0],
        simdFma(displacement[1], displacement[1], displacement[2] * displacement[2]));
    auto const radial =
        evaluateGaussianRadial<GaussianComponent::Short, 1>(radiusSquared, screening);
    Real const inwardRadial = -radial.template coefficient<1, 0>();
    Real const inverseRadius = radial.inverseRadius();
    Field<Real> field;
    field.potential() = radial.template coefficient<0, 0>();
    field.accelerationX() = inwardRadial * (displacement[0] * inverseRadius);
    field.accelerationY() = inwardRadial * (displacement[1] * inverseRadius);
    field.accelerationZ() = inwardRadial * (displacement[2] * inverseRadius);
    return field;
}

}    // namespace cosmictiger::gravity::p3m::pp
