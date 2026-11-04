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
 * Symbol directory: f → factor; q → storedFactor; f̂ → decodeFactor result.
 * Nonfinite factors become zero. Integer IEEE checks preserve this guard under
 * finite-math optimization. The double intermediate preserves which side of a
 * half-code threshold the input float occupies; ties round upward.
 */
CUDA_EXPORT inline uint8_t encodeFactor(float factor) {
#ifdef __CUDA_ARCH__
    uint32_t const factorBits = __float_as_uint(factor);
#else
    uint32_t const factorBits = std::bit_cast<uint32_t>(factor);
#endif
    if ((factorBits & 0x7f800000U) == 0x7f800000U || factor <= 0.0f) {
        return 0;
    }
    if (factor >= 1.0f) {
        return 255;
    }
    return static_cast<uint8_t>(double(factor) * 255.0 + 0.5);
}

CUDA_EXPORT inline float decodeFactor(uint8_t storedFactor) {
    // A float reciprocal under fast-math can map 255 slightly above one.
    // Decode in double before rounding to float to preserve both endpoints.
    return float(double(storedFactor) / 255.0);
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

/*
 * Derived evaluation of GADGET-4 equation (72) from the reconstructed peculiar
 * velocity gradient used by Cullen & Dehnen (2010), equations (B2), (B4), (B7):
 * https://arxiv.org/abs/1006.1524
 *   V_ij = ∂w_i/∂x_j,  G = tr(V),
 *   ω = (V_zy − V_yz, V_xz − V_zx, V_yx − V_xy),
 *   f = |G| / (|G| + |ω|).
 * The common conversion from stored to physical peculiar gradients cancels.
 * V → velocityGradient; G → divergence; ω → curlX/curlY/curlZ.
 * Count scalar arithmetic only: trace (2), curl (3), norm (6), factor (2).
 * Comparisons, selection and unary signs do not count as floating operations.
 */
template <typename Real, typename Counter>
CUDA_EXPORT inline Real factorFromVelocityGradient(
    Real const (&velocityGradient)[3][3], Counter& floatingPointOperations) {
    Real const divergence =
        velocityGradient[0][0] + velocityGradient[1][1] + velocityGradient[2][2];
    Real const curlX = velocityGradient[2][1] - velocityGradient[1][2];
    Real const curlY = velocityGradient[0][2] - velocityGradient[2][0];
    Real const curlZ = velocityGradient[1][0] - velocityGradient[0][1];
    Real const curlMagnitude = simdSqrt(curlX * curlX + curlY * curlY + curlZ * curlZ);
    floatingPointOperations += 13;
    return balsaraFactor(divergence, curlMagnitude);
}

}    // namespace balsara
