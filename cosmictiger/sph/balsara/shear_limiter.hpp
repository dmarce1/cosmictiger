/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/math/simd_math.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_BALSARA_EXPORT __host__ __device__
#else
#define COSMICTIGER_BALSARA_EXPORT
#endif

namespace balsara {

// NOLINTBEGIN(readability-identifier-naming)

// Balsara (1995) shear correction in Springel et al., GADGET-4 (2021),
// MNRAS 506, 2871-2949, doi:10.1093/mnras/stab1855, equation (72):
// https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf
//
//   f_a^AV = |div v_a| / (|div v_a| + |curl v_a|),
//   Π_ab <- ½(f_a^AV + f_b^AV) Π_ab.
//
// The common density normalization cancels from the derivative sums. Set f_a^AV
// to zero when both vanish; a safe denominator prevents SIMD lanes evaluating 0/0.
// Use div_v_a and curl_v_a because ∇ is not a valid C++ identifier character.
template <typename Real>
COSMICTIGER_BALSARA_EXPORT inline Real balsaraFactor(Real divergence, Real curlMagnitude) {
    Real const div_v_a = divergence;
    Real const abs_div_v_a = simdMax(div_v_a, -div_v_a);
    Real const abs_curl_v_a = curlMagnitude;
    Real const D_a = abs_div_v_a + abs_curl_v_a;
    Real const f_a_AV = abs_div_v_a / simdSelect(D_a > Real(0), D_a, Real(1));
    return f_a_AV;
}

// NOLINTEND(readability-identifier-naming)

}    // namespace balsara

#undef COSMICTIGER_BALSARA_EXPORT
