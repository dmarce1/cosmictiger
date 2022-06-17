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
	signed short e;
	signed short s;

	CUDA_EXPORT
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
#ifdef __CUDA_ARCH__
	__device__
	inline static signed imax( signed a, signed b) {
		return max(a,b);
	}
	__device__
	inline static bool add_overflow(unsigned a, unsigned b, unsigned* c ) {
		*c = a + b;
		return __uhadd(a, b) & unsigned(0x80000000);
	}
	__device__
	inline static unsigned leading_zeros(unsigned a) {
		return __clz(a);
	}
	__device__
	inline static unsigned mulhi( unsigned a, unsigned b ) {
		return __umulhi(a,b);
	}
#else
	__host__
	inline static signed imax(signed a, signed b) {
		return std::max(a, b);
	}
	__host__
	inline static bool add_overflow(unsigned a, unsigned b, unsigned* c) {
		return __builtin_add_overflow(a, b, c);
	}
	__host__
	inline static unsigned leading_zeros(unsigned a) {
		return __builtin_clz(a);
	}
	__host__
	inline static unsigned mulhi(unsigned a, unsigned b) {
		return (((size_t) a * (size_t) b) >> 32);
	}
#endif
public:
	CUDA_EXPORT
	inline float40() {
	}
	CUDA_EXPORT
	inline float40(double a) {
		*this = a;
	}
	CUDA_EXPORT
	inline float40(float a) {
		*this = a;
	}
	CUDA_EXPORT
	inline float to_float() const {
		if (e >= 0) {
			return (float) s * (1.0f + m / sfactor) * float(1 << e);
		} else {
			return (float) s * (1.0f + m / sfactor) / float(1 << -e);
		}
	}

	CUDA_EXPORT
	inline operator double() const {
		if (e >= 0) {
			return (double) s * (1.0f + m / dfactor) * double(1 << e);
		} else {
			return (double) s * (1.0f + m / dfactor) / double(1 << -e);
		}
	}
	inline float40(const float40&) = default;
	CUDA_EXPORT
	inline float40& operator=(double a) {
		if (a != 0.0) {
			const auto absa = fabs(a);
			e = floor(log2(absa));
			const double b = absa * ((e > 0) ? 1.0 / double(1 << e) : double(1 << -e));
			m = (b - 1.0) * dfactor;
			s = copysign(1.0, a);
		} else {
			s = 0;
		}
		return *this;
	}
	CUDA_EXPORT
	inline float40& operator=(float a) {
		if (a != 0.f) {
			const auto absa = fabsf(a);
			e = floorf(log2f(absa));
			const float b = absa * ((e > 0) ? 1.0 / float(1 << e) : float(1 << -e));
			m = (b - 1.0f) * sfactor;
			s = copysignf(1.0f, a);
		} else {
			s = 0;
		}
		return *this;
	}
	inline float40& operator=(const float40&) = default;
	CUDA_EXPORT
	inline bool operator<(float40 B) const {
		float40 A = *this;
		float40 C;
		if (A.s < B.s) {
			return true;
		} else if (A.s > B.s) {
			return false;
		} else {
			if (A.s == -1) {
				C = A;
				A = B;
				B = C;
			}
			return unsigned(A.s != 0) * A.abslt(B);
		}
	}
	CUDA_EXPORT
	inline bool operator==(const float40& B) const {
		const float40& A = *this;
		if (A.s == 0 && B.s == 0) {
			return true;
		} else {
			return A.s == B.s && A.e == B.e && A.m == B.m;
		}
	}
	CUDA_EXPORT
	inline bool operator!=(const float40& B) const {
		return !(operator==(B));
	}
	CUDA_EXPORT
	inline bool operator<=(const float40& B) const {
		return (*this < B) || (*this == B);
	}
	CUDA_EXPORT
	inline bool operator>=(const float40& B) const {
		return !(operator<(B));
	}
	CUDA_EXPORT
	inline bool operator>(const float40& B) const {
		return !(operator<=(B));
	}
	CUDA_EXPORT
	inline float40 operator-() const {
		float40 A = *this;
		A.s = -A.s;
		return A;
	}
	CUDA_EXPORT
	inline float40 operator-(const float40& other) const {
		return (*this) + -other;
	}
	CUDA_EXPORT
	inline float40 operator+(const float40& other) const {
		float40 A = *this;
		float40 B = other;
		float40 C;
		if (A.abslt(B)) {
			C = A;
			A = B;
			B = C;
		}
		if (A.s * B.s != 0) {
			const signed dif = A.e - B.e;
			unsigned ma, mb;
			unsigned shf = 1 + abs(dif);
			ma = (dif < 0) ? (A.m >> shf) | ((unsigned) 1 << (32 - shf)) :  ((A.m >> 1) | ((unsigned) 0x80000000));
			mb = (dif > 0) ? (B.m >> shf) | ((unsigned) 1 << (32 - shf)) :  ((B.m >> 1) | ((unsigned) 0x80000000));
			unsigned maxe = imax(A.e, B.e);
			const signed sgn = (2 * signed(A.s * B.s > 0) - 1);
			if (add_overflow(ma, sgn * mb, &C.m)) {
				C.e = maxe + 1;
			} else {
				C.m <<= 1;
				C.e = maxe;
			}
			C.s = A.s;
			if (sgn == -1) {
				if (C.m != 0) {
					const unsigned shf = 1 + leading_zeros(C.m);
					C.m <<= shf;
					C.e -= shf;
				} else {
					C.s = 0;
				}
			}
		} else {
			C = A.s == 0 ? B : A;
		}
		return C;
	}
	CUDA_EXPORT
	inline float40 operator*(const float40& other) const {
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
			const unsigned s2 = half + (mulhi(A.m, B.m) << 1);
			if (add_overflow(s1, s2, &C.m)) {
				C.e++;
			} else {
				C.m <<= 1;
			}
			C.s = A.s * B.s;
		}
		return C;
	}
	CUDA_EXPORT
	inline float40 operator/(const float40& other) const {
		float40 A = *this;
		float40 B = other;
		float40 C;
		float40 invB = B;
		float40 two;
		two.m = 0;
		two.s = 1;
		two.e = 1;
		invB = 1.f / B.to_float();
		invB = invB * (two - invB * B);
		C = A * invB;
		return C;
	}
	CUDA_EXPORT
	inline float40& operator+=(const float40& other) {
		*this = *this + other;
		return *this;
	}
	CUDA_EXPORT
	inline float40& operator-=(const float40& other) {
		*this = *this + -other;
		return *this;
	}
	CUDA_EXPORT
	inline float40& operator*=(const float40& other) {
		*this = *this * other;
		return *this;
	}
	CUDA_EXPORT
	inline float40& operator/=(const float40& other) {
		*this = *this / other;
		return *this;
	}
	CUDA_EXPORT
	friend inline float40 sqrt(const float40& A) {
		float40 B;
		float40 half;
		half.m = 0;
		half.s = 1;
		half.e = -1;
		B = sqrtf(A.to_float());
		B = half * (B + A / B);
		return B;
	}
}
;

#endif /* FLOAT40_HPP_ */
