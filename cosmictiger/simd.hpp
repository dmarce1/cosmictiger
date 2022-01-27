
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

#include <cosmictiger/defs.hpp>

#include <immintrin.h>

#include <cmath>

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
using simd_float = simd_float4;
using simd_int = simd_int4;
#define SIMD_FLOAT_SIZE SIMD_FLOAT4_SIZE

#elif defined(USE_AVX2)
using simd_float = simd_float8;
using simd_int = simd_int8;
#define SIMD_FLOAT_SIZE SIMD_FLOAT8_SIZE

#elif defined(USE_AVX512)
using simd_float = simd_float16;
using simd_int = simd_int16;
#define SIMD_FLOAT_SIZE SIMD_FLOAT16_SIZE

#endif

#include <cosmictiger/simd_altivec.hpp>

inline simd_float two_pow(const simd_float &r) {											// 21
	static const simd_float zero = simd_float(0.0);
	static const simd_float one = simd_float(1.0);
	static const simd_float c1 = simd_float(std::log(2));
	static const simd_float c2 = simd_float((0.5) * std::pow(std::log(2), 2));
	static const simd_float c3 = simd_float((1.0 / 6.0) * std::pow(std::log(2), 3));
	static const simd_float c4 = simd_float((1.0 / 24.0) * std::pow(std::log(2), 4));
	static const simd_float c5 = simd_float((1.0 / 120.0) * std::pow(std::log(2), 5));
	static const simd_float c6 = simd_float((1.0 / 720.0) * std::pow(std::log(2), 6));
	static const simd_float c7 = simd_float((1.0 / 5040.0) * std::pow(std::log(2), 7));
	static const simd_float c8 = simd_float((1.0 / 40320.0) * std::pow(std::log(2), 8));
	simd_float r0;
	simd_int n;
	r0 = round(r);							// 1
	n = simd_int(r0);														// 1
	auto x = r - r0;
	auto y = c8;
	y = fmaf(y, x, c7);																		// 2
	y = fmaf(y, x, c6);																		// 2
	y = fmaf(y, x, c5);																		// 2
	y = fmaf(y, x, c4);																		// 2
	y = fmaf(y, x, c3);																		// 2
	y = fmaf(y, x, c2);																		// 2
	y = fmaf(y, x, c1);																		// 2
	y = fmaf(y, x, one);																		// 2
	simd_int sevenf(0x7f);
	simd_int imm00 = n + sevenf;
	imm00 <<= 23;
	r0 = *((simd_float*) &imm00);
	auto res = y * r0;																			// 1
	return res;
}

inline simd_float sin(const simd_float &x0) {						// 17
	auto x = x0;
	// From : http://mooooo.ooo/chebyshev-sine-approximation/
	static const simd_float pi_major(3.1415927);
	static const simd_float pi_minor(-0.00000008742278);
	x = x - round(x * (1.0 / (2.0 * M_PI))) * (2.0 * M_PI);			// 4
	const simd_float x2 = x * x;									// 1
	simd_float p = simd_float(0.00000000013291342);
	p = fmaf(p, x2, simd_float(-0.000000023317787));				// 2
	p = fmaf(p, x2, simd_float(0.0000025222919));					// 2
	p = fmaf(p, x2, simd_float(-0.00017350505));					// 2
	p = fmaf(p, x2, simd_float(0.0066208798));						// 2
	p = fmaf(p, x2, simd_float(-0.10132118));						// 2
	const auto x1 = (x - pi_major - pi_minor);						// 2
	const auto x3 = (x + pi_major + pi_minor);						// 2
	auto res = x1 * x3 * p * x;										// 3
	return res;
}

inline simd_float cos(const simd_float &x) {		// 18
	return sin(x + simd_float(M_PI / 2.0));
}

inline void sincos(const simd_float &x, simd_float *s, simd_float *c) {		// 35
	*s = sin(x);
	*c = cos(x);
}

inline simd_float exp(simd_float a) { 	// 24
	static const simd_float c0 = 1.0 / std::log(2);
	static const auto hi = simd_float(88);
	static const auto lo = simd_float(-88);
	a = min(a, hi);
	a = max(a, lo);
	return two_pow(a * c0);
}

inline void erfcexp(const simd_float &x, simd_float *ec, simd_float* ex) {				// 76
	const simd_float p(0.3275911);
	const simd_float a1(0.254829592);
	const simd_float a2(-0.284496736);
	const simd_float a3(1.421413741);
	const simd_float a4(-1.453152027);
	const simd_float a5(1.061405429);
	const simd_float t1 = simd_float(1) / (simd_float(1) + p * x);			//37
	const simd_float t2 = t1 * t1;											// 1
	const simd_float t3 = t2 * t1;											// 1
	const simd_float t4 = t2 * t2;											// 1
	const simd_float t5 = t2 * t3;											// 1
	*ex = exp(-x * x);														// 16
	*ec = (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *ex; 			// 11
}

inline simd_float operator*(float r, const simd_float& y) {
	return y * r;
}

inline simd_float8 pow(const simd_float8& a, const simd_float8& b) {
	return exp(log(a)*b);
}

#endif /* SIMD_HPP_ */

#endif
