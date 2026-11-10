/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once
#include <cosmictiger/gravity/kernels/spherical/l2l.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace cosmictiger::gravity::spherical {
template <int degree>
class EwaldL2L {
public:
    static constexpr int kFlops_ = detail::L2L<degree>::kFlops_ + 9;
    static constexpr int kConventionalFlops_ = detail::L2L<degree>::kConventionalFlops_ + 15;
    template <typename Real, typename Local, typename Source>
    CUDA_EXPORT static void add(Local& output, Real& outputTrace, Source const& source,
        Real sourceTrace, Real x, Real y, Real z) {
        using namespace math;
        Fma const fma{};
        detail::L2L<degree>::template add<Real>(output, source, x, y, z);
        Real const radiusSquared = fma(x, x, fma(y, y, z * z));
        output[0] = fma(-broadcast<Real>(0.5) * sourceTrace, radiusSquared, output[0]);
        output[3] = fma(-sourceTrace, x, output[3]);
        output[1] = fma(-sourceTrace, y, output[1]);
        output[2] = fma(-sourceTrace, z, output[2]);
        outputTrace += sourceTrace;
    }
};
}    // namespace cosmictiger::gravity::spherical
