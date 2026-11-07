/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <bit>
#include <cstdint>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace balsara {

/*
 * Derived storage quantization of the bounded shear factor:
 *   q = ⌊255 clamp(f, 0, 1) + ½⌋,   f̂ = q/255,
 *   |f̂ − f| ≤ 1/510 for 0 ≤ f ≤ 1, before floating-point reconstruction.
 * Symbol directory: f → factor; q → stored factor; f̂ → decoded factor.
 * Nonfinite factors become zero. Integer IEEE checks preserve this guard under
 * finite-math optimization. The double intermediate preserves which side of a
 * half-code threshold the input float occupies; ties round upward.
 */
CUDA_EXPORT inline uint8_t encodeFactor(float f) {
#ifdef __CUDA_ARCH__
    uint32_t const factorBits = __float_as_uint(f);
#else
    uint32_t const factorBits = std::bit_cast<uint32_t>(f);
#endif
    if ((factorBits & 0x7f800000U) == 0x7f800000U || f <= 0.0f) {
        return 0;
    }
    if (f >= 1.0f) {
        return 255;
    }
    return static_cast<uint8_t>(double(f) * 255.0 + 0.5);
}

CUDA_EXPORT inline float decodeFactor(uint8_t q) {
    // A float reciprocal under fast-math can map 255 slightly above one.
    // Decode in double before rounding to float to preserve both endpoints.
    return float(double(q) / 255.0);
}

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
 *   ∇·v_a → divergence; |∇×v_a| → curl magnitude
 *   |∇·v_a| → absolute divergence
 *   |∇·v_a| + |∇×v_a| → limiter denominator; f_aᴬⱽ → shear limiter.
 *   Code names: div_v_a = ∇·v_a; abs_curl_v_a = |∇×v_a|; abs_div_v_a = |∇·v_a|;
 *     abs_div_v_a_plus_abs_curl_v_a = |∇·v_a| + |∇×v_a|; fᴬⱽ_a = f_aᴬⱽ.
 * Density-weighted inputs are also valid because their common factor cancels.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real balsaraFactor(Real div_v_a, Real abs_curl_v_a) {
    Real const abs_div_v_a = simdMax(div_v_a, -div_v_a);
    Real const abs_div_v_a_plus_abs_curl_v_a = abs_div_v_a + abs_curl_v_a;
    Real const fᴬⱽ_a = abs_div_v_a /
        simdSelect(abs_div_v_a_plus_abs_curl_v_a > Real(0), abs_div_v_a_plus_abs_curl_v_a, Real(1));
    return fᴬⱽ_a;
}

}    // namespace balsara
