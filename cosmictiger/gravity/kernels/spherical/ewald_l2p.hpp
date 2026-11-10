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
class EwaldL2P {
public:
    static constexpr int kFlops_ = detail::L2P<degree>::kFlops_ + 8;
    static constexpr int kConventionalFlops_ = detail::L2P<degree>::kConventionalFlops_ + 14;
    template <typename Real, typename Field, typename Source>
    CUDA_EXPORT static void add(
        Field& field, Source const& source, Real sourceTrace, Real x, Real y, Real z) {
        using namespace math;
        Fma const fma{};
        detail::L2P<degree>::template add<Real>(field, source, x, y, z);
        Real const radiusSquared = fma(x, x, fma(y, y, z * z));
        field[0] = fma(-broadcast<Real>(0.5) * sourceTrace, radiusSquared, field[0]);
        field[1] = fma(sourceTrace, x, field[1]);
        field[2] = fma(sourceTrace, y, field[2]);
        field[3] = fma(sourceTrace, z, field[3]);
    }
};
}    // namespace cosmictiger::gravity::spherical
