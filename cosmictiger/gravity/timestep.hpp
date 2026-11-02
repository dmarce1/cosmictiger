/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cmath>
#include <limits>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace gravity {

// pairCurvature = sum_j m_j F(r_ij/H)/H^3 over direct, nonself pairs, without G.
// For every supported nonnegative, decreasing spherical density kernel the
// pair's tidal eigenvalues are -F and 2F - 4*pi*rho, hence bounded by 2F, and
// the diagonal plus off-diagonal blocks give a row bound of 4*G*pairCurvature.
// Unlike the acceleration criterion this stays finite at a softened encounter's
// zero-force crossing. The scale factor converts canonical momentum to drift.
// Pairs with no curvature, or a repulsive (glass) gm, impose no bound.
template <class Real>
CUDA_EXPORT inline Real pairTimestep(Real scaleFactor, Real gm, Real eta, Real pairCurvature) {
    if (!(pairCurvature > Real(0) && gm > Real(0))) {
        return std::numeric_limits<Real>::max();
    }
    return eta * std::sqrt(scaleFactor / (Real(4) * gm * pairCurvature));
}

}    // namespace gravity
