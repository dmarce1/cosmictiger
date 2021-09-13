/*
 * simd_altivec.hpp
 *
 *  Created on: Sep 10, 2021
 *      Author: dmarce1
 */

#ifndef SIMD_ALTIVEC_HPP_
#define SIMD_ALTIVEC_HPP_

#ifdef __VEC__

#define SIMD_DOUBLE_SIZE 2
#define SIMD_FLOAT4_SIZE 4
#define SIMD_FLOAT8_SIZE 8
#define SIMD_INT4_SIZE 4
#define SIMD_INT8_SIZE 8

#include <altivec.h>

class simd_int4;
class simd_int8;

class alignas(sizeof(vector float)) simd_float4 {
	vector float v;
public:
	simd_float4() = default;
	simd_float4(float a1, float a2, float a3, float a4) {
		v[0] = a1;
		v[1] = a2;
		v[2] = a3;
		v[3] = a4;
	}
	simd_float4(simd_int4 a);
	simd_float4(float a) {
		v = vec_load_splats(0, &a);
	}
	float& operator[](int i) {
		return v[i];
	}
	float operator[](int i) const {
		return v[i];
	}
	inline simd_float4 operator+(const simd_float4& other) const {
		simd_float4 a;
		a.v = vec_add(v, other.v);
		return a;
	}
	inline simd_float4 operator-(const simd_float4& other) const {
		simd_float4 a;
		a.v = vec_sub(v, other.v);
		return a;
	}
	inline simd_float4 operator+() const {
		return *this;
	}
	inline simd_float4 operator-() const {
		simd_float4 a;
		a.v = vec_neg(v);
		return a;
	}
	inline simd_float4 operator*(const simd_float4& other) const {
		simd_float4 a;
		a.v = vec_mul(v, other.v);
		return a;
	}
	inline simd_float4 operator/(const simd_float4& other) const {
		simd_float4 a;
		a.v = vec_div(v, other.v);
		return a;
	}
	inline simd_float4 operator+=(const simd_float4& other) {
		v = vec_add(v, other.v);
		return *this;
	}
	inline simd_float4 operator-=(const simd_float4& other) {
		v = vec_sub(v, other.v);
		return *this;
	}
	inline simd_float4 operator*=(const simd_float4& other) {
		v = vec_mul(v, other.v);
		return *this;
	}
	inline simd_float4 operator/=(const simd_float4& other) {
		v = vec_div(v, other.v);
		return *this;
	}
	inline simd_float4 operator*(float d) const {
		const simd_float4 other = d;
		return other * *this;
	}
	inline simd_float4 operator/(float d) const {
		const simd_float4 other = 1.0 / d;
		return *this * other;
	}
	float sum() const {
		float s = (*this)[0];
		for( int i = 1; i < SIMD_FLOAT4_SIZE; i++) {
			s += (*this)[i];
		}
		return s;
	}
	simd_float4 operator<=(const simd_float4& other) const {
		simd_float4 rc;
		rc.v = vec_ctf(vec_cmple(v, other.v), 0);
		return -rc;
	}
	simd_float4 operator<(const simd_float4& other) const {
		simd_float4 rc;
		rc.v = vec_ctf(vec_cmplt(v, other.v), 0);
		return -rc;
	}
	simd_float4 operator>(const simd_float4& other) const {
		simd_float4 rc;
		rc.v = vec_ctf(vec_cmpgt(v, other.v), 0);
		return -rc;
	}
	simd_float4 operator>=(const simd_float4& other) const {
		simd_float4 rc;
		rc.v = vec_ctf(vec_cmpge(v, other.v), 0);
		return -rc;
	}
	friend inline simd_float4 sqrt(const simd_float4&);
	friend inline simd_float4 rsqrt(const simd_float4&);
	friend inline simd_float4 max(const simd_float4&, const simd_float4&);
	friend inline simd_float4 min(const simd_float4&, const simd_float4&);
	friend inline simd_float4 round(const simd_float4&);
	friend inline simd_float4 fmaf(const simd_float4&, const simd_float4&, const simd_float4&);
	friend class simd_int4;
};

inline simd_float4 sqrt(const simd_float4& a) {
	simd_float4 b;
	b.v = vec_sqrt(a.v);
	return b;
}

inline simd_float4 rsqrt(const simd_float4& a) {
	simd_float4 b;
	b.v = vec_rsqrt(a.v);
	return b;
}

inline simd_float4 max(const simd_float4& a, const simd_float4& b) {
	simd_float4 c;
	c.v = vec_max(a.v, b.v);
	return c;
}

inline simd_float4 min(const simd_float4& a, const simd_float4& b) {
	simd_float4 c;
	c.v = vec_min(a.v, b.v);
	return c;
}

inline simd_float4 round(const simd_float4& a) {
	simd_float4 b;
	b.v = vec_round(a.v);
	return b;
}

inline simd_float4 fmaf(const simd_float4& a, const simd_float4& b, const simd_float4& c) {
	simd_float4 d;
	d.v = vec_madd(a.v, b.v, c.v);
	return d;
}

class alignas(sizeof(vector double)) simd_double {
	vector double v;
public:
	simd_double() = default;
	simd_double(double a1, double a2) {
		v[0] = a1;
		v[1] = a2;
	}
	simd_double(simd_int4 a);
	simd_double(double a) {
		v = vec_load_splats(0, &a);
	}
	double& operator[](int i) {
		return v[i];
	}
	double operator[](int i) const {
		return v[i];
	}
	inline simd_double operator+(const simd_double& other) const {
		simd_double a;
		a.v = vec_add(v, other.v);
		return a;
	}
	inline simd_double operator-(const simd_double& other) const {
		simd_double a;
		a.v = vec_sub(v, other.v);
		return a;
	}
	inline simd_double operator+() const {
		return *this;
	}
	inline simd_double operator-() const {
		simd_double a;
		a.v = vec_neg(v);
		return a;
	}
	inline simd_double operator*(const simd_double& other) const {
		simd_double a;
		a.v = vec_mul(v, other.v);
		return a;
	}
	inline simd_double operator/(const simd_double& other) const {
		simd_double a;
		a.v = vec_div(v, other.v);
		return a;
	}
	inline simd_double operator+=(const simd_double& other) {
		v = vec_add(v, other.v);
		return *this;
	}
	inline simd_double operator-=(const simd_double& other) {
		v = vec_sub(v, other.v);
		return *this;
	}
	inline simd_double operator*=(const simd_double& other) {
		v = vec_mul(v, other.v);
		return *this;
	}
	inline simd_double operator/=(const simd_double& other) {
		v = vec_div(v, other.v);
		return *this;
	}
	inline simd_double operator*(float d) const {
		const simd_double other = d;
		return other * *this;
	}
	inline simd_double operator/(float d) const {
		const simd_double other = 1.0 / d;
		return *this * other;
	}
	float sum() const {
		float s = (*this)[0];
		for( int i = 1; i < SIMD_DOUBLE_SIZE; i++) {
			s += (*this)[i];
		}
		return s;
	}
	simd_double operator<=(const simd_double& other) const {
		simd_double rc;
		rc.v = vec_ctf(vec_cmple(v, other.v), 0);
		return -rc;
	}
	simd_double operator<(const simd_double& other) const {
		simd_double rc;
		rc.v = vec_ctf(vec_cmplt(v, other.v), 0);
		return -rc;
	}
	simd_double operator>(const simd_double& other) const {
		simd_double rc;
		rc.v = vec_ctf(vec_cmpgt(v, other.v), 0);
		return -rc;
	}
	simd_double operator>=(const simd_double& other) const {
		simd_double rc;
		rc.v = vec_ctf(vec_cmpge(v, other.v), 0);
		return -rc;
	}
	friend inline simd_double sqrt(const simd_double&);
	friend inline simd_double rsqrt(const simd_double&);
	friend inline simd_double max(const simd_double&, const simd_double&);
	friend inline simd_double min(const simd_double&, const simd_double&);
	friend inline simd_double round(const simd_double&);
	friend inline simd_double fmaf(const simd_double&, const simd_double&, const simd_double&);
	friend class simd_int4;
};

inline simd_double sqrt(const simd_double& a) {
	simd_double b;
	b.v = vec_sqrt(a.v);
	return b;
}

inline simd_double rsqrt(const simd_double& a) {
	simd_double b;
	b.v = vec_rsqrt(a.v);
	return b;
}

inline simd_double max(const simd_double& a, const simd_double& b) {
	simd_double c;
	c.v = vec_max(a.v, b.v);
	return c;
}

inline simd_double min(const simd_double& a, const simd_double& b) {
	simd_double c;
	c.v = vec_min(a.v, b.v);
	return c;
}

inline simd_double round(const simd_double& a) {
	simd_double b;
	b.v = vec_round(a.v);
	return b;
}

inline simd_double fmaf(const simd_double& a, const simd_double& b, const simd_double& c) {
	simd_double d;
	d.v = vec_madd(a.v, b.v, c.v);
	return d;
}

class alignas(sizeof(vector float)) simd_float8 {
	vector float v[2];
public:
	simd_float8() = default;
	simd_float8(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
		v[0][0] = a1;
		v[0][1] = a2;
		v[0][2] = a3;
		v[0][3] = a4;
		v[1][0] = a5;
		v[1][1] = a6;
		v[1][2] = a7;
		v[1][3] = a8;
	}
	simd_float8(simd_int4 a);
	simd_float8(float a) {
		v[0] = vec_load_splats(0, &a);
		v[1] = vec_load_splats(0, &a);
	}
	float& operator[](int i) {
		return v[i/4][i%4];
	}
	float operator[](int i) const {
		return v[i/4][i%4];
	}
	inline simd_float8 operator+(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = vec_add(v[0], other.v[0]);
		a.v[1] = vec_add(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator-(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = vec_sub(v[0], other.v[0]);
		a.v[1] = vec_sub(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator+() const {
		return *this;
	}
	inline simd_float8 operator-() const {
		simd_float8 a;
		a.v[0] = vec_neg(v[0]);
		a.v[1] = vec_neg(v[1]);
		return a;
	}
	inline simd_float8 operator*(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = vec_mul(v[0], other.v[0]);
		a.v[1] = vec_mul(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator/(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = vec_div(v[0], other.v[0]);
		a.v[1] = vec_div(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator+=(const simd_float8& other) {
		v[0] = vec_add(v[0], other.v[0]);
		v[1] = vec_add(v[1], other.v[1]);
		return *this;
	}
	inline simd_float8 operator-=(const simd_float8& other) {
		v[0] = vec_sub(v[0], other.v[0]);
		v[1] = vec_sub(v[1], other.v[1]);
		return *this;
	}
	inline simd_float8 operator*=(const simd_float8& other) {
		v[0] = vec_mul(v[0], other.v[0]);
		v[1] = vec_mul(v[1], other.v[1]);
		return *this;
	}
	inline simd_float8 operator/=(const simd_float8& other) {
		v[0] = vec_div(v[0], other.v[0]);
		v[1] = vec_div(v[1], other.v[1]);
		return *this;
	}
	inline simd_float8 operator*(float d) const {
		const simd_float8 other = d;
		return other * *this;
	}
	inline simd_float8 operator/(float d) const {
		const simd_float8 other = 1.0 / d;
		return *this * other;
	}
	float sum() const {
		float s = (*this)[0];
		for( int i = 1; i < SIMD_FLOAT8_SIZE; i++) {
			s += (*this)[i];
		}
		return s;
	}
	simd_float8 operator<=(const simd_float8& other) const {
		simd_float8 rc;
		rc.v[0] = vec_ctf(vec_cmple(v[0], other.v[0]), 0);
		rc.v[1] = vec_ctf(vec_cmple(v[1], other.v[1]), 0);
		return -rc;
	}
	simd_float8 operator<(const simd_float8& other) const {
		simd_float8 rc;
		rc.v[0] = vec_ctf(vec_cmplt(v[0], other.v[0]), 0);
		rc.v[1] = vec_ctf(vec_cmplt(v[1], other.v[1]), 0);
		return -rc;
	}
	simd_float8 operator>(const simd_float8& other) const {
		simd_float8 rc;
		rc.v[0] = vec_ctf(vec_cmpgt(v[0], other.v[0]), 0);
		rc.v[1] = vec_ctf(vec_cmpgt(v[1], other.v[1]), 0);
		return -rc;
	}
	simd_float8 operator>=(const simd_float8& other) const {
		simd_float8 rc;
		rc.v[0] = vec_ctf(vec_cmpge(v[0], other.v[0]), 0);
		rc.v[1] = vec_ctf(vec_cmpge(v[1], other.v[1]), 0);
		return -rc;
	}
	friend inline simd_float8 sqrt(const simd_float8&);
	friend inline simd_float8 rsqrt(const simd_float8&);
	friend inline simd_float8 max(const simd_float8&, const simd_float8&);
	friend inline simd_float8 min(const simd_float8&, const simd_float8&);
	friend inline simd_float8 round(const simd_float8&);
	friend inline simd_float8 fmaf(const simd_float8&, const simd_float8&, const simd_float8&);
	friend class simd_int4;
};

inline simd_float8 sqrt(const simd_float8& a) {
	simd_float8 b;
	b.v[0] = vec_sqrt(a.v[0]);
	b.v[1] = vec_sqrt(a.v[1]);
	return b;
}

inline simd_float8 rsqrt(const simd_float8& a) {
	simd_float8 b;
	b.v[0] = vec_rsqrt(a.v[0]);
	b.v[1] = vec_rsqrt(a.v[1]);
	return b;
}

inline simd_float8 max(const simd_float8& a, const simd_float8& b) {
	simd_float8 c;
	c.v[0] = vec_max(a.v[0], b.v[0]);
	c.v[1] = vec_max(a.v[1], b.v[1]);
	return c;
}

inline simd_float8 min(const simd_float8& a, const simd_float8& b) {
	simd_float8 c;
	c.v[0] = vec_min(a.v[0], b.v[0]);
	c.v[1] = vec_min(a.v[1], b.v[1]);
	return c;
}

inline simd_float8 round(const simd_float8& a) {
	simd_float8 b;
	b.v[0] = vec_round(a.v[0]);
	b.v[1] = vec_round(a.v[1]);
	return b;
}

inline simd_float8 fmaf(const simd_float8& a, const simd_float8& b, const simd_float8& c) {
	simd_float8 d;
	d.v[0] = vec_madd(a.v[0], b.v[0], c.v[0]);
	d.v[1] = vec_madd(a.v[1], b.v[1], c.v[1]);
	return d;
}

class alignas(sizeof(vector int)) simd_int4 {
private:
	vector int v;
public:
	simd_int4() = default;
	simd_int4(const simd_int4&) = default;
	simd_int4(int i) {
		v = vec_splats(0, &i);
	}
	simd_int4(simd_float4 r) {
		v = vec_cts(vec_floor(r.v));
	}
	inline simd_int4& operator=(const simd_int4 &other) = default;
	simd_int4& operator=(simd_int4 &&other) = default;
	inline simd_int4 operator+(const simd_int4 &other) const {
		simd_int4 r;
		r.v = vec_add(v, other.v);
		return r;
	}
	inline simd_int4 operator-(const simd_int4 &other) const {
		simd_int4 r;
		r.v = vec_sub(v, other.v);
		return r;
	}
	inline simd_int4 operator*(const simd_int4 &other) const {
		simd_int4 r;
		r.v = vec_mul(v, other.v);
		return r;
	}
	inline simd_int4 operator+() const {
		return *this;
	}
	inline simd_int4 operator-() const {
		simd_int4 a;
		a.v = vec_neg(v);
		return a;
	}
	inline int& operator[](std::size_t i) {
		return v[i];
	}
	inline int operator[](std::size_t i) const {
		return v[i];
	}
	inline simd_int4 operator<<=(int shft) {
		simd_int4 s(shft);
		v = vec_sl(v, s);
		return *this;
	}
	friend class simd_float4;
};

inline simd_float4::simd_float4(simd_int4 a) {
	v = vec_ctf(a.v);
}

class alignas(sizeof(vector int)) simd_int8 {
private:
	vector int v[2];
public:
	simd_int8() = default;
	simd_int8(const simd_int8&) = default;
	simd_int8(int i) {
		v[0] = vec_splats(0, &i);
		v[1] = vec_splats(0, &i);
	}
	simd_int8(simd_float4 r) {
		v[0] = vec_cts(vec_floor(r.v[0]));
		v[1] = vec_cts(vec_floor(r.v[1]));
	}
	inline simd_int8& operator=(const simd_int8 &other) = default;
	simd_int8& operator=(simd_int8 &&other) = default;
	inline simd_int8 operator+(const simd_int8 &other) const {
		simd_int8 r;
		r.v[0] = vec_add(v[0], other.v[0]);
		r.v[1] = vec_add(v[1], other.v[1]);
		return r;
	}
	inline simd_int8 operator-(const simd_int8 &other) const {
		simd_int8 r;
		r.v[0] = vec_sub(v[0], other.v[0]);
		r.v[1] = vec_sub(v[1], other.v[1]);
		return r;
	}
	inline simd_int8 operator*(const simd_int8 &other) const {
		simd_int8 r;
		r.v[0] = vec_mul(v[0], other.v[0]);
		r.v[1] = vec_mul(v[1], other.v[1]);
		return r;
	}
	inline simd_int8 operator+() const {
		return *this;
	}
	inline simd_int8 operator-() const {
		simd_int8 a;
		a.v[0] = vec_neg(v[0]);
		a.v[1] = vec_neg(v[1]);
		return a;
	}
	inline int& operator[](std::size_t i) {
		return v[i/4][i%4];
	}
	inline int operator[](std::size_t i) const {
		return v[i/4][i%4];
	}
	inline simd_int8 operator<<=(int shft) {
		simd_int8 s(shft);
		v[0] = vec_sl(v[0], s[0]);
		v[1] = vec_sl(v[1], s[1]);
		return *this;
	}
	friend class simd_float8;
};

inline simd_float8::simd_float8(simd_int8 a) {
	v[0] = vec_ctf(a.v[0]);
	v[1] = vec_ctf(a.v[1]);
}

using simd_int = simd_int4;
using simd_float = simd_float4;

#define SIMD_FLOAT_SIZE SIMD_FLOAT4_SIZE
#define SIMD_INT_SIZE SIMD_INT4_SIZE

#endif

#endif /* SIMD_ALTIVEC_HPP_ */
