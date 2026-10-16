/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef SIMD_HPP_
#define SIMD_HPP_

#ifndef __CUDACC__

#include <cmath>

#include <immintrin.h>

#include <cosmictiger/defs.hpp>

#if defined(__AVX2__)
#define USE_AVX2

#elif defined(__AVX__)
#define USE_AVX

#else
#ifndef __VEC__
#error "No AVX"
#endif
#endif

#include <cosmictiger/simd_float.hpp>
#include <cosmictiger/simd_int.hpp>

#ifdef USE_AVX
using SimdFloat = SimdFloat4;
using SimdInt = SimdInt4;
#define SIMD_FLOAT_SIZE SIMD_FLOAT4_SIZE

#elif defined(USE_AVX2)
using SimdFloat = SimdFloat8;
using SimdInt = SimdInt8;
#define SIMD_FLOAT_SIZE SIMD_FLOAT8_SIZE

#elif defined(USE_AVX512)
using SimdFloat = SimdFloat16;
using SimdInt = SimdInt16;
#define SIMD_FLOAT_SIZE SIMD_FLOAT16_SIZE

#endif

#include <cosmictiger/simd_altivec.hpp>

inline SimdFloat twoPow(SimdFloat const& r) {    // 21
    static SimdFloat const zero = SimdFloat(0.0);
    static SimdFloat const one = SimdFloat(1.0);
    static SimdFloat const c1 = SimdFloat(std::log(2));
    static SimdFloat const c2 = SimdFloat((0.5) * std::pow(std::log(2), 2));
    static SimdFloat const c3 = SimdFloat((1.0 / 6.0) * std::pow(std::log(2), 3));
    static SimdFloat const c4 = SimdFloat((1.0 / 24.0) * std::pow(std::log(2), 4));
    static SimdFloat const c5 = SimdFloat((1.0 / 120.0) * std::pow(std::log(2), 5));
    static SimdFloat const c6 = SimdFloat((1.0 / 720.0) * std::pow(std::log(2), 6));
    static SimdFloat const c7 = SimdFloat((1.0 / 5040.0) * std::pow(std::log(2), 7));
    static SimdFloat const c8 = SimdFloat((1.0 / 40320.0) * std::pow(std::log(2), 8));
    SimdFloat r0;
    SimdInt n;
    r0 = round(r);      // 1
    n = SimdInt(r0);    // 1
    auto x = r - r0;
    auto y = c8;
    y = fmaf(y, x, c7);     // 2
    y = fmaf(y, x, c6);     // 2
    y = fmaf(y, x, c5);     // 2
    y = fmaf(y, x, c4);     // 2
    y = fmaf(y, x, c3);     // 2
    y = fmaf(y, x, c2);     // 2
    y = fmaf(y, x, c1);     // 2
    y = fmaf(y, x, one);    // 2
    SimdInt sevenf(0x7f);
    SimdInt imm00 = n + sevenf;
    imm00 <<= 23;
    r0 = *((SimdFloat*) &imm00);
    auto res = y * r0;    // 1
    return res;
}

inline SimdFloat sin(SimdFloat const& x0) {    // 17
    auto x = x0;
    // From : http://mooooo.ooo/chebyshev-sine-approximation/
    static SimdFloat const piMajor(3.1415927);
    static SimdFloat const piMinor(-0.00000008742278);
    x = x - round(x * (1.0 / (2.0 * M_PI))) * (2.0 * M_PI);    // 4
    SimdFloat const x2 = x * x;                                // 1
    SimdFloat p = SimdFloat(0.00000000013291342);
    p = fmaf(p, x2, SimdFloat(-0.000000023317787));    // 2
    p = fmaf(p, x2, SimdFloat(0.0000025222919));       // 2
    p = fmaf(p, x2, SimdFloat(-0.00017350505));        // 2
    p = fmaf(p, x2, SimdFloat(0.0066208798));          // 2
    p = fmaf(p, x2, SimdFloat(-0.10132118));           // 2
    auto const x1 = (x - piMajor - piMinor);           // 2
    auto const x3 = (x + piMajor + piMinor);           // 2
    auto res = x1 * x3 * p * x;                        // 3
    return res;
}

inline SimdFloat sinf(SimdFloat const& x0) {
    return sin(x0);
}

inline SimdFloat cos(SimdFloat const& x) {    // 18
    return sin(x + SimdFloat(M_PI / 2.0));
}

inline void sincos(SimdFloat const& x, SimdFloat* s, SimdFloat* c) {    // 35
    *s = sin(x);
    *c = cos(x);
}

inline SimdFloat exp(SimdFloat a) {    // 24
    static SimdFloat const c0 = 1.0 / std::log(2);
    static auto const hi = SimdFloat(88);
    static auto const lo = SimdFloat(-88);
    a = min(a, hi);
    a = max(a, lo);
    return twoPow(a * c0);
}

inline SimdFloat expf(SimdFloat a) {
    return exp(a);
}

inline void erfcexp(SimdFloat const& x, SimdFloat* ec, SimdFloat* ex) {    // 76
    SimdFloat const p(0.3275911);
    SimdFloat const a1(0.254829592);
    SimdFloat const a2(-0.284496736);
    SimdFloat const a3(1.421413741);
    SimdFloat const a4(-1.453152027);
    SimdFloat const a5(1.061405429);
    SimdFloat const t1 = SimdFloat(1) / (SimdFloat(1) + p * x);       // 37
    SimdFloat const t2 = t1 * t1;                                     // 1
    SimdFloat const t3 = t2 * t1;                                     // 1
    SimdFloat const t4 = t2 * t2;                                     // 1
    SimdFloat const t5 = t2 * t3;                                     // 1
    *ex = exp(-x * x);                                                // 16
    *ec = (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *ex;    // 11
}

inline SimdFloat erfc(SimdFloat const x) {    // 76
    SimdFloat ec, ex;
    erfcexp(x, &ec, &ex);
    return ec;
}

inline SimdFloat erf(SimdFloat const x) {    // 76
    return SimdFloat(1) - erfc(x);
}

inline SimdFloat erff(SimdFloat const x) {    // 76
    return erf(x);
}

inline SimdFloat operator*(float r, SimdFloat const& y) {
    return y * r;
}

inline SimdFloat8 pow(SimdFloat8 const& a, SimdFloat8 const& b) {
    return exp(log(a) * b);
}

inline SimdDouble8::SimdDouble8(SimdInt8 const& a) {
    __m128i& v0 = *((__m128i*) &a.v_);
    __m128i& v1 = *((__m128i*) (((float*) &a.v_) + 4));
    v_[0] = _mm256_cvtepi32_pd(v0);
    v_[1] = _mm256_cvtepi32_pd(v1);
}
/*
inline simd_double8 two_pow(const simd_double8 &r) {											//
21 static const simd_double8 zero = simd_double8(0.0); static const simd_double8 one =
simd_double8(1.0); static const simd_double8 c1 = simd_double8(std::log(2)); static const
simd_double8 c2 = simd_double8((0.5) * std::pow(std::log(2), 2)); static const simd_double8 c3 =
simd_double8((1.0 / 6.0) * std::pow(std::log(2), 3)); static const simd_double8 c4 =
simd_double8((1.0 / 24.0) * std::pow(std::log(2), 4)); static const simd_double8 c5 =
simd_double8((1.0 / 120.0) * std::pow(std::log(2), 5)); static const simd_double8 c6 =
simd_double8((1.0 / 720.0) * std::pow(std::log(2), 6)); static const simd_double8 c7 =
simd_double8((1.0 / 5040.0) * std::pow(std::log(2), 7)); static const simd_double8 c8 =
simd_double8((1.0 / 40320.0) * std::pow(std::log(2), 8)); static const simd_double8 c9 =
simd_double8((1.0 / 40320.0 / (9.0)) * std::pow(std::log(2), 9)); static const simd_double8 c10 =
simd_double8((1.0 / 40320.0 / (9.0 * 10.0)) * std::pow(std::log(2), 10)); static const simd_double8
c11 = simd_double8((1.0 / 40320.0 / (9.0 * 10.0 * 11.0)) * std::pow(std::log(2), 11)); static const
simd_double8 c12 = simd_double8((1.0 / 40320.0 / (9.0 * 10.0 * 11.0 * 12.0)) * std::pow(std::log(2),
12)); static const simd_double8 c13 = simd_double8((1.0 / 40320.0 / (9.0 * 10.0 * 11.0 * 12.0
* 13.0)) * std::pow(std::log(2), 13)); static const simd_double8 c14 = simd_double8((1.0 / 40320.0 /
(9.0 * 10.0 * 11.0 * 12.0 * 13.0 * 14.0)) * std::pow(std::log(2), 14)); static const simd_double8
c15 = simd_double8((1.0 / 40320.0 / (9.0 * 10.0 * 11.0 * 12.0 * 13.0 * 14.0 * 15.0)) *
std::pow(std::log(2), 15)); simd_double8 r0; simd_int n; r0 = round(r);							// 1
    n = simd_int(r0);														// 1
    auto x = r - r0;
    auto y = c15;
    y = fmaf(y, x, c14);																		// 2
    y = fmaf(y, x, c13);																		// 2
    y = fmaf(y, x, c12);																		// 2
    y = fmaf(y, x, c11);																		// 2
    y = fmaf(y, x, c10);																		// 2
    y = fmaf(y, x, c9);																		// 2
    y = fmaf(y, x, c8);																		// 2
    y = fmaf(y, x, c7);																		// 2
    y = fmaf(y, x, c6);																		// 2
    y = fmaf(y, x, c5);																		// 2
    y = fmaf(y, x, c4);																		// 2
    y = fmaf(y, x, c3);																		// 2
    y = fmaf(y, x, c2);																		// 2
    y = fmaf(y, x, c1);																		// 2
    y = fmaf(y, x, one);																		// 2
    simd_int sevenf(0x1ff);
    simd_int imm00 = n + sevenf;
    imm00 <<= 52;
    r0 = *((simd_double8*) &imm00);
    auto res = y * r0;																			// 1
    return res;
}
*/

inline SimdDouble8 exp(SimdDouble8 a) {    // 24
    SimdDouble8 c;
    for (int i = 0; i < 8; i++) {
        c[i] = exp(a[i]);
    }
    return c;
}

#endif /* SIMD_HPP_ */

#endif
