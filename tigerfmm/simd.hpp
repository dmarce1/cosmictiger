/*
 * simd.hpp
 *
 *  Created on: Jul 19, 2021
 *      Author: dmarce1
 */

#ifndef SIMD_HPP_
#define SIMD_HPP_

class simd_float;

class simd_int {
	typedef int type __attribute__ ((vector_size (SIMD_INT_SIZE*8)));
	type a;
public:
	inline int& operator[](int i) {
		return a[i];
	}
	inline int operator[](int i) const {
		return a[i];
	}
	inline simd_int() = default;
	inline simd_int(int b) {
		for (int i = 0; i < SIMD_INT_SIZE; i++) {
			a[i] = b;
		}
	}
	inline simd_int operator-() const {
		simd_int b;
		b.a = -a;
		return b;
	}
	inline simd_int operator+(const simd_int& other) const {
		simd_int b;
		b.a = a + other.a;
		return b;
	}
	inline simd_int operator-(const simd_int& other) const {
		simd_int b;
		b.a = a - other.a;
		return b;
	}
	inline simd_int operator*(const simd_int& other) const {
		simd_int b;
		b.a = a * other.a;
		return b;
	}
	inline simd_int operator/(const simd_int& other) const {
		simd_int b;
		b.a = a / other.a;
		return b;
	}
	inline simd_int& operator+=(const simd_int& other) {
		a += other.a;
		return *this;
	}
	inline simd_int& operator-=(const simd_int& other) {
		a -= other.a;
		return *this;
	}
	inline simd_int& operator*=(const simd_int& other) {
		a *= other.a;
		return *this;
	}
	inline simd_int& operator/=(const simd_int& other) {
		a /= other.a;
		return *this;
	}
	inline simd_int operator*(int other) const {
		simd_int b;
		b.a = a * other;
		return b;
	}
	inline simd_int operator/(int other) const {
		simd_int b;
		b.a = a / other;
		return b;
	}
	inline simd_int& operator*=(int other) {
		a *= other;
		return *this;
	}
	inline simd_int& operator/=(int other) {
		a /= other;
		return *this;
	}
	inline simd_int operator>(const simd_int& other) const {
		simd_int i;
		i.a = -(a > other.a);
		return i;
	}
	inline simd_int operator<(const simd_int& other) const {
		simd_int i;
		i.a = -(a < other.a);
		return i;
	}
	inline simd_int operator>=(const simd_int& other) const {
		simd_int i;
		i.a = -(a >= other.a);
		return i;
	}
	inline simd_int operator<=(const simd_int& other) const {
		simd_int i;
		i.a = -(a <= other.a);
		return i;
	}
	inline simd_int operator==(const simd_int& other) const {
		simd_int i;
		i.a = -(a == other.a);
		return i;
	}
	inline simd_int operator!=(const simd_int& other) const {
		simd_int i;
		i.a = -(a != other.a);
		return i;
	}
	inline simd_int operator<<=(int shift) {
		a <<= shift;
		return *this;
	}
	inline simd_int operator>>=(int shift) {
		a >>= shift;
		return *this;
	}
	inline simd_int operator<<(int shift) {
		simd_int i;
		i.a = a >> shift;
		return i;
	}
	inline simd_int operator>>(int shift) {
		simd_int i;
		i.a = a >> shift;
		return i;
	}
	friend class simd_float;
	inline operator simd_float();
};

class simd_float {
	typedef float type __attribute__ ((vector_size (SIMD_INT_SIZE*8)));
	type a;
public:
	inline float& operator[](int i) {
		return a[i];
	}
	inline float operator[](int i) const {
		return a[i];
	}
	inline simd_float() = default;
	inline simd_float(float b) {
		for (int i = 0; i < SIMD_FLOAT_SIZE; i++) {
			a[i] = b;
		}
	}
	inline simd_float operator-() const {
		simd_float b;
		b.a = -a;
		return b;
	}
	inline simd_float operator+(const simd_float& other) const {
		simd_float b;
		b.a = a + other.a;
		return b;
	}
	inline simd_float operator-(const simd_float& other) const {
		simd_float b;
		b.a = a - other.a;
		return b;
	}
	inline simd_float operator*(const simd_float& other) const {
		simd_float b;
		b.a = a * other.a;
		return b;
	}
	inline simd_float operator/(const simd_float& other) const {
		simd_float b;
		b.a = a / other.a;
		return b;
	}
	inline simd_float& operator+=(const simd_float& other) {
		a += other.a;
		return *this;
	}
	inline simd_float& operator-=(const simd_float& other) {
		a -= other.a;
		return *this;
	}
	inline simd_float& operator*=(const simd_float& other) {
		a *= other.a;
		return *this;
	}
	inline simd_float& operator/=(const simd_float& other) {
		a /= other.a;
		return *this;
	}
	inline simd_float operator*(float other) const {
		simd_float b;
		b.a = a * other;
		return b;
	}
	inline simd_float operator/(float other) const {
		simd_float b;
		b.a = a / other;
		return b;
	}
	inline simd_float& operator*=(float other) {
		a *= other;
		return *this;
	}
	inline simd_float& operator/=(float other) {
		a /= other;
		return *this;
	}
	inline simd_float operator>(const simd_float& other) const {
		simd_float b;
		b.a = -(a > other.a);
		return b;
	}
	inline simd_float operator<(const simd_float& other) const {
		simd_float b;
		b.a = -(a < other.a);
		return b;
	}
	inline simd_float operator>=(const simd_float& other) const {
		simd_float b;
		b.a = -(a >= other.a);
		return b;
	}
	inline simd_float operator<=(const simd_float& other) const {
		simd_float b;
		b.a = -(a <= other.a);
		return b;
	}
	inline simd_float operator==(const simd_float& other) const {
		simd_float b;
		b.a = -(a == other.a);
		return b;
	}
	inline simd_float operator!=(const simd_float& other) const {
		simd_float b;
		b.a = -(a != other.a);
		return b;
	}
	friend class simd_int;
	inline operator simd_int();
};

inline simd_int::operator simd_float() {
	simd_int i;
	i.a = __builtin_convertvector(a, simd_int::type);
	return i;
}

inline simd_float::operator simd_int() {
	simd_float b;
	b.a = __builtin_convertvector(a, simd_float::type);
	return b;
}

inline simd_float fmaf(const simd_float& a, const simd_float& b, const simd_float& c) {
	return a * b + c;
}

inline simd_float round(const simd_float& a) {
	return simd_float(simd_int(a + simd_float(0.5f)));
}

inline simd_float expf(simd_float r) {
	static const simd_float one = simd_float(1.0);
	static const simd_float c0 = 1.0 / std::log(2);
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
	r *= c0;
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
	r0 = *((simd_float*) (&imm00));
	auto res = y * r0;																			// 1
	return res;

}

inline simd_float sinf(const simd_float &x0) {						// 17
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

inline simd_float cosf(const simd_float &x) {		// 18
	return sinf(x + simd_float(M_PI / 2.0));
}

inline void erfcexpf(const simd_float &x, simd_float* ec, simd_float *ex) {				// 76
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
	*ex = expf(-x * x);														// 16
	*ec = (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *ex; 			// 11
}

inline simd_float rsqrt(simd_float number) {
	/* Quake III Arena */
	static const simd_int magic(0x5f3759df);
	static const simd_float pt5 = simd_float(0.5f);
	simd_int i;
	simd_float x2, y;
	static const simd_float threehalfs(1.5f);
	x2 = number * pt5;
	y = number;
	i = *(simd_int *) &y;
	i = magic - (i >> 1);
	y = *(simd_float *) &i;
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));
	return y;
}

inline simd_float sqrt(simd_float n) {
	/* https://www.gamedev.net/forums/topic/704525-3-quick-ways-to-calculate-the-square-root-in-c/ */
	static const simd_float pt5 = simd_float(0.5f);
	static const simd_int magic(0x1FB5AD00);
	union {
		simd_int i;
		simd_float f;
	} u;
	u.i = magic + (*(simd_int*) &n >> 1);
	u.f = pt5 * (u.f + n / u.f);
	u.f = pt5 * (u.f + n / u.f);
	u.f = pt5 * (u.f + n / u.f);
	return u.f;
}

#endif /* SIMD_HPP_ */
