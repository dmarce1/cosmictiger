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

#pragma once

#include <cmath>

#include <cosmictiger/math/simd_math.hpp>

#ifndef __CUDACC__

// Preserve the existing kernel approximations while library types handle the arithmetic.
inline SimdFloat twoPow(SimdFloat const& r) {
    static SimdFloat const one = simdBroadcast<SimdFloat>(1.0);
    static SimdFloat const c1 = simdBroadcast<SimdFloat>(std::log(2));
    static SimdFloat const c2 = simdBroadcast<SimdFloat>((0.5) * std::pow(std::log(2), 2));
    static SimdFloat const c3 = simdBroadcast<SimdFloat>((1.0 / 6.0) * std::pow(std::log(2), 3));
    static SimdFloat const c4 = simdBroadcast<SimdFloat>((1.0 / 24.0) * std::pow(std::log(2), 4));
    static SimdFloat const c5 = simdBroadcast<SimdFloat>((1.0 / 120.0) * std::pow(std::log(2), 5));
    static SimdFloat const c6 = simdBroadcast<SimdFloat>((1.0 / 720.0) * std::pow(std::log(2), 6));
    static SimdFloat const c7 = simdBroadcast<SimdFloat>((1.0 / 5040.0) * std::pow(std::log(2), 7));
    static SimdFloat const c8 =
        simdBroadcast<SimdFloat>((1.0 / 40320.0) * std::pow(std::log(2), 8));
    SimdFloat r0;
    SimdInt n;
    r0 = simdRound(r);
    n = simdCast<SimdInt>(r0);
    auto x = r - r0;
    auto y = c8;
    y = simdFma(y, x, c7);
    y = simdFma(y, x, c6);
    y = simdFma(y, x, c5);
    y = simdFma(y, x, c4);
    y = simdFma(y, x, c3);
    y = simdFma(y, x, c2);
    y = simdFma(y, x, c1);
    y = simdFma(y, x, one);
    SimdInt sevenf(0x7f);
    SimdInt imm00 = n + sevenf;
    imm00 <<= 23;
    r0 = simdBitCast<SimdFloat>(imm00);
    auto res = y * r0;
    return res;
}

inline SimdFloat sin(SimdFloat const& x0) {
    auto x = x0;
    // From : http://mooooo.ooo/chebyshev-sine-approximation/
    static SimdFloat const piMajor = simdBroadcast<SimdFloat>(3.1415927);
    static SimdFloat const piMinor = simdBroadcast<SimdFloat>(-0.00000008742278);
    x = x -
        simdRound(x * simdBroadcast<SimdFloat>(1.0 / (2.0 * M_PI))) *
            simdBroadcast<SimdFloat>(2.0 * M_PI);
    SimdFloat const x2 = x * x;
    SimdFloat p = simdBroadcast<SimdFloat>(0.00000000013291342);
    p = simdFma(p, x2, simdBroadcast<SimdFloat>(-0.000000023317787));
    p = simdFma(p, x2, simdBroadcast<SimdFloat>(0.0000025222919));
    p = simdFma(p, x2, simdBroadcast<SimdFloat>(-0.00017350505));
    p = simdFma(p, x2, simdBroadcast<SimdFloat>(0.0066208798));
    p = simdFma(p, x2, simdBroadcast<SimdFloat>(-0.10132118));
    auto const x1 = (x - piMajor - piMinor);
    auto const x3 = (x + piMajor + piMinor);
    auto res = x1 * x3 * p * x;
    return res;
}

inline SimdFloat sinf(SimdFloat const& x0) {
    return sin(x0);
}

inline SimdFloat cos(SimdFloat const& x) {
    return sin(x + simdBroadcast<SimdFloat>(M_PI / 2.0));
}

inline void sincos(SimdFloat const& x, SimdFloat* s, SimdFloat* c) {
    *s = sin(x);
    *c = cos(x);
}

inline SimdFloat exp(SimdFloat a) {
    static SimdFloat const c0 = simdBroadcast<SimdFloat>(1.0 / std::log(2));
    static auto const hi = simdBroadcast<SimdFloat>(88);
    static auto const lo = simdBroadcast<SimdFloat>(-88);
    a = simdMin(a, hi);
    a = simdMax(a, lo);
    return twoPow(a * c0);
}

inline SimdFloat expf(SimdFloat a) {
    return exp(a);
}

inline void erfcexp(SimdFloat const& x, SimdFloat* ec, SimdFloat* ex) {
    SimdFloat const p = simdBroadcast<SimdFloat>(0.3275911);
    SimdFloat const a1 = simdBroadcast<SimdFloat>(0.254829592);
    SimdFloat const a2 = simdBroadcast<SimdFloat>(-0.284496736);
    SimdFloat const a3 = simdBroadcast<SimdFloat>(1.421413741);
    SimdFloat const a4 = simdBroadcast<SimdFloat>(-1.453152027);
    SimdFloat const a5 = simdBroadcast<SimdFloat>(1.061405429);
    SimdFloat const t1 = simdBroadcast<SimdFloat>(1) / (simdBroadcast<SimdFloat>(1) + p * x);
    SimdFloat const t2 = t1 * t1;
    SimdFloat const t3 = t2 * t1;
    SimdFloat const t4 = t2 * t2;
    SimdFloat const t5 = t2 * t3;
    *ex = exp(-x * x);
    *ec = (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *ex;
}

inline SimdFloat erfc(SimdFloat const x) {
    SimdFloat ec, ex;
    erfcexp(x, &ec, &ex);
    return ec;
}

inline SimdFloat erf(SimdFloat const x) {
    return simdBroadcast<SimdFloat>(1) - erfc(x);
}

inline SimdFloat erff(SimdFloat const x) {
    return erf(x);
}

inline SimdFloat log2(SimdFloat const& value) {
    using UnsignedPack = SimdVector<uint32_t, SIMD_FLOAT_SIZE>;
    auto const exponent = simdBitCast<UnsignedPack>(value) >> 23;
    auto const exponentScale = simdBitCast<SimdFloat>(exponent << 23);
    auto const exponentValue = simdCast<SimdFloat>(simdCast<SimdInt>(exponent) - SimdInt(127));
    auto x = simdSqrt(simdSqrt(value / exponentScale)) - SimdFloat(1.0f);
    float const scale = 1.0f / std::log(2.0f);
    auto polynomial = SimdFloat(scale / 9.0f);
    for (int order = 8; order >= 1; --order) {
        auto const coefficient = SimdFloat((order % 2 ? scale : -scale) / float(order));
        polynomial = simdFma(polynomial, x, coefficient);
    }
    return exponentValue + SimdFloat(4.0f) * polynomial * x;
}

inline SimdFloat log(SimdFloat const& value) {
    return log2(value) * simdBroadcast<SimdFloat>(std::log(2.0));
}

inline SimdFloat pow(SimdFloat const& base, SimdFloat const& exponent) {
    return exp(log(base) * exponent);
}

inline SimdDouble8 exp(SimdDouble8 const& value) {
    return SimdDouble8([&](auto lane) { return std::exp(value[lane]); });
}

#endif
