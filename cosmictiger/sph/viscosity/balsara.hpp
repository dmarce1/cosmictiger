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
 * Balsara (1995) shear correction with the sound-speed term of Springel (2005), GADGET-2,
 * MNRAS 364, 1105-1134, doi:10.1111/j.1365-2966.2005.09655.x, equation (14), and the
 * divergence and curl estimates of Springel et al., GADGET-4 (2021), MNRAS 506, 2871-2949,
 * doi:10.1093/mnras/stab1855, equation (72):
 *
 *   f𞁑^AV = |div v𞁑| / (|div v𞁑| + |curl v𞁑| + ε_B c𞁑/h𞁑),
 *   Π𞁑𞁓 <- ½(f𞁑^AV + f𞁓^AV) Π𞁑𞁓.
 *
 * GADGET-2 fixes ε_B = 10⁻⁴; here it is hydro.balsaraSoundSpeedCoefficient. The sound-speed
 * term keeps noisy, nearly static flow from receiving full viscosity; a larger ε_B lowers f𞁑^AV
 * toward zero, removing viscosity, and never toward one.
 * A common density normalization of all three inputs cancels. Set f𞁑^AV to zero when the
 * denominator vanishes; a safe denominator prevents SIMD lanes evaluating 0/0.
 */
// clang-format off
/*
 * Symbol directory for GADGET-2 equation (14) above:
 *   ∇·v𞁑 → divergence; |∇×v𞁑| → curl magnitude; |∇·v𞁑| → absolute divergence
 *   c𞁑 → sound speed; h𞁑 → smoothing length (GADGET-2's h, the full kernel support)
 *   ε_B → Balsara sound-speed coefficient
 *   |∇·v𞁑| + |∇×v𞁑| + ε_B c𞁑/h𞁑 → limiter denominator; f𞁑ᴬⱽ → shear limiter.
 *   Code names: div_v_a = ∇·v𞁑; abs_curl_v𞁑 = |∇×v𞁑|; abs_div_v𞁑 = |∇·v𞁑|
 *     c𞁑_over_h𞁑 = c𞁑/h𞁑; denominator = |∇·v𞁑| + |∇×v𞁑| + ε_B c𞁑/h𞁑; fᴬⱽ𞁑 = f𞁑ᴬⱽ.
 * Density-weighted inputs are also valid when all three carry the factor.
 */
// clang-format on
template <typename Real>
CUDA_EXPORT inline Real balsaraFactor(Real div_v_a, Real abs_curl_v𞁑, Real c𞁑_over_h𞁑, Real ε_B) {
    Real const abs_div_v𞁑 = simdMax(div_v_a, -div_v_a);
    Real const denominator = abs_div_v𞁑 + abs_curl_v𞁑 + ε_B * c𞁑_over_h𞁑;
    Real const fᴬⱽ𞁑 = abs_div_v𞁑 / simdSelect(denominator > Real(0), denominator, Real(1));
    return fᴬⱽ𞁑;
}

}    // namespace balsara
