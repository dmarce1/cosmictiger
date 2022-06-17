/*
 * float40.hpp
 *
 *  Created on: Jun 12, 2022
 *      Author: dmarce1
 */

#ifndef FLOAT40_HPP_
#define FLOAT40_HPP_

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

class float40 {
	constexpr static double dfactor = (1ULL << 32ULL);
	constexpr static float sfactor = (1ULL << 32ULL);
	unsigned m;
	char e;
	char s;
public:
	inline float40() {
	}
	inline float40(double a) {
		*this = a;
	}
	inline float40(float a) {
		*this = a;
	}
	inline operator double() const {
		return (double) s * (1.0 + m / dfactor) * pow(2.0, e);
	}
	inline operator float() const {
		return (float) s * (1.0 + m / sfactor) * pow(2.0, e);
	}
	inline float40(const float40&) = default;
	inline float40& operator=(double a) {
		if (a == 0.0) {
			s = 0;
		} else {
			e = floor(log2(abs(a)));
			const double b = abs(a) * pow(2.0, -e);
			m = (b - 1.0) * dfactor;
			s = copysign(1.0, a);
		}
		return *this;
	}
	inline float40& operator=(float a) {
		if (a == 0.f) {
			s = 0;
		} else {
			e = floorf(log2f(fabs(a)));
			const float b = fabs(a) * powf(2.0f, -e);
			m = (b - 1.0f) * sfactor;
			s = copysignf(1.0f, a);
		}
		return *this;
	}
	inline float40& operator=(const float40&) = default;
	inline bool abslt(const float40& B) const {
		const float40& A = *this;
		if (A.e < B.e) {
			return true;
		} else if (A.e > B.e) {
			return false;
		} else {
			return A.m < B.m;
		}
	}
	inline float40 operator-() const {
		float40 A = *this;
		A.s = -A.s;
		return A;
	}
	inline float40 operator-(const float40& other) {
		return (*this) + -other;
	}
	inline float40 operator+(const float40& other) {
		float40 A = *this;
		float40 B = other;
		float40 C;
		if (A.abslt(B)) {
			C = A;
			A = B;
			B = C;
		}
		if (A.s == 0) {
			C = B;
		} else if (B.s == 0) {
			C = A;
		} else {
			const signed dif = A.e - B.e;
			const unsigned sha = (1 + std::max(-dif, 0));
			const unsigned shb = (1 + std::max(+dif, 0));
			unsigned ma = (A.m >> sha) | ((unsigned) 1 << (32 - sha));
			unsigned mb = (B.m >> shb) | ((unsigned) 1 << (32 - shb));
			unsigned maxe = std::max(A.e, B.e);
			bool of;
			bool chngsgn = false;
			const signed sgn = (2 * signed(A.s * B.s > 0) - 1);
			if (__builtin_add_overflow(ma, sgn * mb, &C.m)) {
				C.e = maxe + 1;
			} else {
				C.m <<= 1;
				C.e = maxe;
			}
			C.s = A.s;
			if (sgn == -1) {
				if (C.m != 0) {
					const unsigned shf = 1 + __builtin_clz(C.m);
					C.m <<= shf;
					C.e -= shf;
				} else {
					C.s = 0;
				}
			}
		}
		return C;
	}
	inline float40 operator*(const float40& other) {
		constexpr auto half = ((unsigned) 1 << (unsigned) 31);
		float40 A = *this;
		float40 B = other;
		float40 C;
		if (A.s == 0) {
			C = A;
		} else if (B.s == 0) {
			C = B;
		} else {
			C.e = A.e + B.e;
			A.m >>= 1;
			B.m >>= 1;
			const unsigned s1 = A.m + B.m;
			const unsigned s2 = half + (((size_t) A.m * (size_t) B.m) >> 31);
			if (__builtin_add_overflow(s1, s2, &C.m)) {
				C.e++;
			} else {
				C.m <<= 1;
			}
			C.s = A.s * B.s;
		}
		return C;
	}
	inline float40 operator/(const float40& other) const {
		float40 A = *this;
		float40 B = other;
		float40 C;
		float40 invB = B;
		float40 two;
		two.m = 0;
		two.s = 1;
		two.e = 1;
		invB = 1.f / (float) B;
		invB = invB * (two - invB * B);
		C = A * invB;
		return C;
	}
	friend inline float40 sqrt(const float40& A) {
		float40 B;
		float40 half;
		half.m = 0;
		half.s = 1;
		half.e = -1;
		B = sqrtf((float) A);
		B = half * (B + A / B);
		return B;
	}
}
;

#endif /* FLOAT40_HPP_ */
