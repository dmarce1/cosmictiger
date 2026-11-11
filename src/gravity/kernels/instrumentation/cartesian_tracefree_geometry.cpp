/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Generated counting-scalar definitions. Tests only; not a library source.
#include <cosmictiger/gravity/kernels/cartesian_tracefree/geometry.hpp>
namespace cosmictiger::gravity::cartesian::detail {
template <class Real>
Real inverseRadius(Vector3<Real> const& distance) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(1.000000000000000000E-20);
    constexpr auto k2 = static_cast<Scalar>(1.000000000000000000E+00);
    Real const dx = distance[0];
    Real const dy = distance[1];
    Real const dz = distance[2];
    Real const R = sqrt(fma(dx, dx, fma(dy, dy, dz * dz)));
    return select(R > broadcast<Real>(k0), broadcast<Real>(k2), broadcast<Real>(k0)) /
        max(R, broadcast<Real>(k1));
}
}    // namespace cosmictiger::gravity::cartesian::detail
