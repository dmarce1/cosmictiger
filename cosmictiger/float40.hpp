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
			e = log2(a);
			const double b = a * pow(2.0, -e);
			m = (b - 1.0) * dfactor;
			s = copysign(1.0, a);
		}
		return *this;
	}
	inline float40& operator=(float a) {
		if (a == 0.f) {
			s = 0;
		} else {
			e = log2f(a);
			const float b = a * powf(2.0f, -e);
			m = (b - 1.0f) * sfactor;
			s = copysignf(1.0f, a);
		}
		return *this;
	}
	inline float40& operator=(const float40&) = default;
	inline bool operator<(const float40& B) const {
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
		float40 A = *this;
		float40 B = other;
		float40 C;
		if (A.s == 0) {
			C = -B;
		} else if (B.s == 0) {
			C = A;
		} else {
			if (A.s * B.s > 0) {
				if (A.s > 0) {
					if (A < B) {
						C = -(B - A);
					} else {
						const signed dif = A.e - B.e;
						C.m = (A.m - (B.m >> dif));
						const int lshft = 32 - __builtin_clz(C.m);
						C.m <<= lshft;
						C.e = A.m - lshft;
					}
				} else {
					C = -(-A - -B);
				}
			} else {
				if (A.s > 0) {
					C = A + -B;
				} else {
					C = B + -A;
				}
			}
		}
		return C;
	}
	inline float40 operator+(const float40& other) {
		float40 A = *this;
		float40 B = other;
		float40 C;
		if (A.s == 0) {
			C = B;
		} else if (B.s == 0) {
			C = A;
		} else {
			if (A.s * B.s > 0) {
				const signed dif = A.e - B.e;
				const unsigned sha = (1 + std::max(-dif, 0));
				const unsigned shb = (1 + std::max(+dif, 0));
				unsigned ma = (A.m >> sha) | ((unsigned) 1 << (32 - sha));
				unsigned mb = (B.m >> shb) | ((unsigned) 1 << (32 - shb));
				unsigned maxe = std::max(A.e, B.e);
				if (__builtin_add_overflow(ma, mb, &C.m)) {
					C.e = maxe + 1;
				} else {
					C.m <<= 1;
					C.e = maxe;
				}
				C.s = A.s;
			} else {
				B.s = -B.s;
				C = A - B;
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
				PRINT( "!\n");
				C.e++;
			} else {
				PRINT( "?\n");
				C.m <<= 1;
			}
			C.s = A.s * B.s;
		}
		return C;
	}
}
;

#endif /* FLOAT40_HPP_ */
