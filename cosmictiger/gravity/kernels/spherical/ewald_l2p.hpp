/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once
#include <cosmictiger/gravity/kernels/spherical/l2p.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace cosmictiger::gravity::spherical {
template <int degree>
inline constexpr int kEwaldL2PFlops = detail::kL2PFlops<degree> + 8;
template <int degree>
inline constexpr int kEwaldL2PConventionalFlops = detail::kL2PConventionalFlops<degree> + 14;
/// Accumulate the local field including the isotropic trace term of the periodic correction.
template <int degree, typename Real, typename Field, typename Source>
CUDA_EXPORT void ewaldL2P(
    Field& field, Source const& source, Real sourceTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+5.00000000000000000E-01);
    detail::L2P<degree, Real>(field, source, x, y, z);
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    field[0] = fma(-broadcast<Real>(k0) * sourceTrace, radiusSquared, field[0]);
    field[1] = fma(sourceTrace, x, field[1]);
    field[2] = fma(sourceTrace, y, field[2]);
    field[3] = fma(sourceTrace, z, field[3]);
}
}    // namespace cosmictiger::gravity::spherical
