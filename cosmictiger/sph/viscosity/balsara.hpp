/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace balsara {

/*
 * Balsara (1995) shear correction in Springel et al., GADGET-4 (2021),
 * MNRAS 506, 2871-2949, doi:10.1093/mnras/stab1855, equation (72):
 * https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf
 *
 *   f_a^AV = |div v_a| / (|div v_a| + |curl v_a|),
 *   Π_ab <- ½(f_a^AV + f_b^AV) Π_ab.
 *
 * The common density normalization cancels from the derivative sums. Set f_a^AV
 * to zero when both vanish; a safe denominator prevents SIMD lanes evaluating 0/0.
 */
// clang-format off
/*
 * Symbol directory for GADGET-4 equation (72) above:
 *   ∇·v_a → divergence; |∇×v_a| → curlMagnitude
 *   |∇·v_a| → absoluteDivergence; |∇×v_a| → absoluteCurl
 *   |∇·v_a| + |∇×v_a| → limiterDenominator; f_aᴬⱽ → shearLimiter.
 * Density-weighted inputs are also valid because their common factor cancels.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real balsaraFactor(Real divergence, Real curlMagnitude) {
    Real const velocityDivergence = divergence;
    Real const absoluteDivergence = simdMax(velocityDivergence, -velocityDivergence);
    Real const absoluteCurl = curlMagnitude;
    Real const limiterDenominator = absoluteDivergence + absoluteCurl;
    Real const shearLimiter =
        absoluteDivergence / simdSelect(limiterDenominator > Real(0), limiterDenominator, Real(1));
    return shearLimiter;
}

}    // namespace balsara
