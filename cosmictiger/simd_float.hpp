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

#ifndef SIMD_FLOAT_HPP_
#define SIMD_FLOAT_HPP_

#define SIMD_FLOAT4_SIZE 4

class simd_int4;

class alignas(sizeof(__m128)) simd_float4 {
	__m128 v;
public:
	simd_float4() = default;
	simd_float4(float a1, float a2, float a3, float a4) {
		v = _mm_set_ps(a4, a3, a2, a1);
	}
	simd_float4(simd_int4 a);
	simd_float4(float a) {
		v = _mm_broadcast_ss(&a);
	}
	float& operator[](int i) {
		return v[i];
	}
	float operator[](int i) const {
		return v[i];
	}
	inline simd_float4 operator+(const simd_float4& other) const {
		simd_float4 a;
		a.v = _mm_add_ps(v, other.v);
		return a;
	}
	inline simd_float4 operator-(const simd_float4& other) const {
		simd_float4 a;
		a.v = _mm_sub_ps(v, other.v);
		return a;
	}
	inline simd_float4 operator+() const {
		return *this;
	}
	inline simd_float4 operator-() const {
		return simd_float4(0.0f) - *this;
	}
	inline simd_float4 operator*(const simd_float4& other) const {
		simd_float4 a;
		a.v = _mm_mul_ps(v, other.v);
		return a;
	}
	inline simd_float4 operator/(const simd_float4& other) const {
		simd_float4 a;
		a.v = _mm_div_ps(v, other.v);
		return a;
	}
	inline simd_float4 operator+=(const simd_float4& other) {
		v = _mm_add_ps(v, other.v);
		return *this;
	}
	inline simd_float4 operator-=(const simd_float4& other) {
		v = _mm_sub_ps(v, other.v);
		return *this;
	}
	inline simd_float4 operator*=(const simd_float4& other) {
		v = _mm_mul_ps(v, other.v);
		return *this;
	}
	inline simd_float4 operator/=(const simd_float4& other) {
		v = _mm_div_ps(v, other.v);
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
		static const simd_float4 one(1);
		auto mask0 = _mm_cmp_ps(v, other.v, _CMP_LE_OQ);
		rc.v = _mm_and_ps(mask0, one.v);
		return rc;
	}
	simd_float4 operator<(const simd_float4& other) const {
		simd_float4 rc;
		static const simd_float4 one(1);
		auto mask0 = _mm_cmp_ps(v, other.v, _CMP_LT_OQ);
		rc.v = _mm_and_ps(mask0, one.v);
		return rc;
	}
	simd_float4 operator>(const simd_float4& other) const {
		simd_float4 rc;
		static const simd_float4 one(1);
		auto mask0 = _mm_cmp_ps(v, other.v, _CMP_GT_OQ);
		rc.v = _mm_and_ps(mask0, one.v);
		return rc;
	}
	simd_float4 operator>=(const simd_float4& other) const {
		simd_float4 rc;
		static const simd_float4 one(1);
		auto mask0 = _mm_cmp_ps(v, other.v, _CMP_GE_OQ);
		rc.v = _mm_and_ps(mask0, one.v);
		return rc;
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
	b.v = _mm_sqrt_ps(a.v);
	return b;
}

inline simd_float4 rsqrt(const simd_float4& a) {
	simd_float4 b;
	b.v = _mm_rsqrt_ps(a.v);
	return b;
}

inline simd_float4 max(const simd_float4& a, const simd_float4& b) {
	simd_float4 c;
	c.v = _mm_max_ps(a.v, b.v);
	return c;
}

inline simd_float4 min(const simd_float4& a, const simd_float4& b) {
	simd_float4 c;
	c.v = _mm_min_ps(a.v, b.v);
	return c;
}

inline simd_float4 round(const simd_float4& a) {
	simd_float4 b;
	b.v = _mm_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT);
	return b;
}

inline simd_float4 fmaf(const simd_float4& a, const simd_float4& b, const simd_float4& c) {
	simd_float4 d;
#ifdef USE_AVX
	return a * b + c;
#else
	d.v = _mm_fmadd_ps(a.v, b.v, c.v);
#endif
	return d;
}

class alignas(sizeof(__m128d)) simd_double {
	__m128d v;
public:
	simd_double() = default;
	inline double& operator[](int i) {
		return v[i];
	}
	inline double operator[](int i) const {
		return v[i];
	}
	inline simd_double(double a) {
		v = _mm_set_pd(a, a);
	}
	inline simd_double operator+(const simd_double& other) const {
		simd_double c;
		c.v = _mm_add_pd(v, other.v);
		return c;
	}
	inline simd_double operator-(const simd_double& other) const {
		simd_double c;
		c.v = _mm_sub_pd(v, other.v);
		return c;
	}
	inline simd_double operator*(const simd_double& other) const {
		simd_double c;
		c.v = _mm_mul_pd(v, other.v);
		return c;
	}
	inline simd_double operator/(const simd_double& other) const {
		simd_double c;
		c.v = _mm_div_pd(v, other.v);
		return c;
	}
	inline simd_double operator-() const {
		simd_double c(0.0);
		c.v = _mm_sub_pd(c.v, v);
		return c;
	}
	inline simd_double operator+=(const simd_double& other) {
		v = _mm_add_pd(v, other.v);
		return *this;
	}
	inline simd_double operator-=(const simd_double& other) {
		v = _mm_sub_pd(v, other.v);
		return *this;
	}
	inline simd_double operator*=(const simd_double& other) {
		v = _mm_mul_pd(v, other.v);
		return *this;
	}
	inline simd_double operator/=(const simd_double& other) {
		v = _mm_div_pd(v, other.v);
		return *this;
	}
	friend simd_double fmaf(const simd_double& a, const simd_double& b, const simd_double& c);

};

inline simd_double fmaf(const simd_double& a, const simd_double& b, const simd_double& c) {
	simd_double d;
#if defined(USE_AVX2) || defined(USE_AVX512)
	d.v = _mm_fmadd_pd(a.v, b.v, c.v);
#else
	d = a * b + c;
#endif
	return d;
}

#if defined(USE_AVX)

#define SIMD_FLOAT8_SIZE 8

class simd_int8;

class alignas(2*sizeof(__m128)) simd_float8 {
	__m128 v[2];
public:
	simd_float8() = default;
	simd_float8(simd_int8 a);
	simd_float8(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
		v[0] = _mm_set_ps(a4, a3, a2, a1);
		v[1] = _mm_set_ps(a8, a7, a6, a5);
	}
	inline simd_float8(float a) {
		v[0] = _mm_broadcast_ss(&a);
		v[1] = _mm_broadcast_ss(&a);
	}
	float& operator[](int i) {
		return v[i/4][i%4];
	}
	float operator[](int i) const {
		return v[i/4][i%4];
	}
	inline simd_float8 operator+(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = _mm_add_ps(v[0], other.v[0]);
		a.v[1] = _mm_add_ps(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator-(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = _mm_sub_ps(v[0], other.v[0]);
		a.v[1] = _mm_sub_ps(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator+() const {
		return *this;
	}
	inline simd_float8 operator-() const {
		return simd_float8(0.0f) - *this;
	}
	inline simd_float8 operator*(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = _mm_mul_ps(v[0], other.v[0]);
		a.v[1] = _mm_mul_ps(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator/(const simd_float8& other) const {
		simd_float8 a;
		a.v[0] = _mm_div_ps(v[0], other.v[0]);
		a.v[1] = _mm_div_ps(v[1], other.v[1]);
		return a;
	}
	inline simd_float8 operator+=(const simd_float8& other) {
		v[0] = _mm_add_ps(v[0], other.v[0]);
		v[1] = _mm_add_ps(v[1], other.v[1]);
		return *this;
	}
	inline simd_float8 operator-=(const simd_float8& other) {
		v[0] = _mm_sub_ps(v[0], other.v[0]);
		v[1] = _mm_sub_ps(v[1], other.v[1]);
		return *this;
	}
	inline simd_float8 operator*=(const simd_float8& other) {
		v[0] = _mm_mul_ps(v[0], other.v[0]);
		v[1] = _mm_mul_ps(v[1], other.v[1]);
		return *this;
	}
	inline simd_float8 operator/=(const simd_float8& other) {
		v[0] = _mm_div_ps(v[0], other.v[0]);
		v[1] = _mm_div_ps(v[1], other.v[1]);
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
		const auto a = _mm_add_ps(v[0], v[1]);
		float s = a[0];
		for( int i = 1; i < 4; i++) {
			s += a[i];
		}
		return s;
	}
	simd_float8 operator<=(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm_cmp_ps(v[0], other.v[0], _CMP_LE_OQ);
		auto mask1 = _mm_cmp_ps(v[1], other.v[1], _CMP_LE_OQ);
		rc.v[0] = _mm_and_ps(mask0, one.v[0]);
		rc.v[1] = _mm_and_ps(mask1, one.v[1]);
		return rc;
	}
	simd_float8 operator<(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm_cmp_ps(v[0], other.v[0], _CMP_LT_OQ);
		auto mask1 = _mm_cmp_ps(v[1], other.v[1], _CMP_LT_OQ);
		rc.v[0] = _mm_and_ps(mask0, one.v[0]);
		rc.v[1] = _mm_and_ps(mask1, one.v[1]);
		return rc;
	}
	simd_float8 operator>(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm_cmp_ps(v[0], other.v[0], _CMP_GT_OQ);
		auto mask1 = _mm_cmp_ps(v[1], other.v[1], _CMP_GT_OQ);
		rc.v[0] = _mm_and_ps(mask0, one.v[0]);
		rc.v[1] = _mm_and_ps(mask1, one.v[1]);
		return rc;
	}
	simd_float8 operator>=(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm_cmp_ps(v[0], other.v[0], _CMP_GE_OQ);
		auto mask1 = _mm_cmp_ps(v[1], other.v[1], _CMP_GE_OQ);
		rc.v[0] = _mm_and_ps(mask0, one.v[0]);
		rc.v[1] = _mm_and_ps(mask1, one.v[1]);
		return rc;
	}

	friend inline simd_float8 sqrt(const simd_float8&);
	friend inline simd_float8 rsqrt(const simd_float8&);
	friend inline simd_float8 max(const simd_float8&, const simd_float8&);
	friend inline simd_float8 min(const simd_float8&, const simd_float8&);
	friend inline simd_float8 round(const simd_float8&);
	friend inline simd_float8 fmaf(const simd_float8&, const simd_float8&, const simd_float8&);
	friend class simd_int8;
};

inline simd_float8 sqrt(const simd_float8& a) {
	simd_float8 b;
	b.v[0] = _mm_sqrt_ps(a.v[0]);
	b.v[1] = _mm_sqrt_ps(a.v[1]);
	return b;
}

inline simd_float8 rsqrt(const simd_float8& a) {
	simd_float8 b;
	b.v[0] = _mm_rsqrt_ps(a.v[0]);
	b.v[1] = _mm_rsqrt_ps(a.v[1]);
	return b;
}

inline simd_float8 max(const simd_float8& a, const simd_float8& b) {
	simd_float8 c;
	c.v[0] = _mm_max_ps(a.v[0], b.v[0]);
	c.v[1] = _mm_max_ps(a.v[1], b.v[1]);
	return c;
}

inline simd_float8 fmaxf(const simd_float8& a, const simd_float8& b) {
	return max(a,b);
}

inline simd_float8 min(const simd_float8& a, const simd_float8& b) {
	simd_float8 c;
	c.v[0] = _mm_min_ps(a.v[0], b.v[0]);
	c.v[1] = _mm_min_ps(a.v[1], b.v[1]);
	return c;
}

inline simd_float8 round(const simd_float8& a) {
	simd_float8 b;
	b.v[0] = _mm_round_ps(a.v[0], _MM_FROUND_TO_NEAREST_INT);
	b.v[1] = _mm_round_ps(a.v[1], _MM_FROUND_TO_NEAREST_INT);
	return b;
}

inline simd_float8 fmaf(const simd_float8& a, const simd_float8& b, const simd_float8& c) {
	simd_float8 d;
	d.v[0] = _mm_fmadd_ps(a.v[0], b.v[0], c.v[0]);
	d.v[1] = _mm_fmadd_ps(a.v[1], b.v[1], c.v[1]);
	return d;
}

#endif

class simd_int8;

#if defined(USE_AVX2) || defined(USE_AVX512)

class alignas(sizeof(__m256d)) simd_double8 {
	__m256d v[2];
public:
	friend class simd_int8;
	inline simd_double8(const simd_int8&);
	simd_double8() = default;
	inline double& operator[](int i) {
		return v[i/4][i%4];
	}
	inline double operator[](int i) const {
		return v[i/4][i%4];
	}
	inline simd_double8(double a) {
		v[0] = _mm256_set_pd(a, a, a, a);
		v[1] = _mm256_set_pd(a, a, a, a);
	}
	inline simd_double8 operator+(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = _mm256_add_pd(v[0], other.v[0]);
		c.v[1] = _mm256_add_pd(v[1], other.v[1]);
		return c;
	}
	inline simd_double8 operator-(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = _mm256_sub_pd(v[0], other.v[0]);
		c.v[1] = _mm256_sub_pd(v[1], other.v[1]);
		return c;
	}
	inline simd_double8 operator*(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = _mm256_mul_pd(v[0], other.v[0]);
		c.v[1] = _mm256_mul_pd(v[1], other.v[1]);
		return c;
	}
	inline simd_double8 operator/(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = _mm256_div_pd(v[0], other.v[0]);
		c.v[1] = _mm256_div_pd(v[1], other.v[1]);
		return c;
	}
	inline simd_double8 operator-() const {
		simd_double8 c(0.0);
		c.v[0] = _mm256_sub_pd(c.v[0], v[0]);
		c.v[1] = _mm256_sub_pd(c.v[1], v[1]);
		return c;
	}
	inline simd_double8 operator+=(const simd_double8& other) {
		v[0] = _mm256_add_pd(v[0], other.v[0]);
		v[1] = _mm256_add_pd(v[1], other.v[1]);
		return *this;
	}
	inline simd_double8 operator-=(const simd_double8& other) {
		v[0] = _mm256_sub_pd(v[0], other.v[0]);
		v[1] = _mm256_sub_pd(v[1], other.v[1]);
		return *this;
	}
	inline simd_double8 operator*=(const simd_double8& other) {
		v[0] = _mm256_mul_pd(v[0], other.v[0]);
		v[1] = _mm256_mul_pd(v[1], other.v[1]);
		return *this;
	}
	inline simd_double8 operator/=(const simd_double8& other) {
		v[0] = _mm256_div_pd(v[0], other.v[0]);
		v[1] = _mm256_div_pd(v[1], other.v[1]);
		return *this;
	}
	friend inline simd_double8 max(const simd_double8& a, const simd_double8& b) {
		simd_double8 c;
		c.v[0] = _mm256_max_pd(a.v[0], b.v[0]);
		c.v[1] = _mm256_max_pd(a.v[1], b.v[1]);
		return c;
	}

	friend inline simd_double8 min(const simd_double8& a, const simd_double8& b) {
		simd_double8 c;
		c.v[0] = _mm256_min_pd(a.v[0], b.v[0]);
		c.v[1] = _mm256_min_pd(a.v[1], b.v[1]);
		return c;
	}

	inline simd_double8 operator<(const simd_double8& other) const {
		simd_double8 rc;
		static const simd_double8 one(1.0);
		auto mask0 = _mm256_cmp_pd(v[0], other.v[0], _CMP_LT_OQ);
		auto mask1 = _mm256_cmp_pd(v[1], other.v[1], _CMP_LT_OQ);
		rc.v[0] = _mm256_and_pd(mask0, one.v[0]);
		rc.v[1] = _mm256_and_pd(mask1, one.v[1]);
		return rc;
	}

	inline simd_double8 operator>(const simd_double8& other) const {
		simd_double8 rc;
		static const simd_double8 one(1.0);
		auto mask0 = _mm256_cmp_pd(v[0], other.v[0], _CMP_GT_OQ);
		auto mask1 = _mm256_cmp_pd(v[1], other.v[1], _CMP_GT_OQ);
		rc.v[0] = _mm256_and_pd(mask0, one.v[0]);
		rc.v[1] = _mm256_and_pd(mask1, one.v[1]);
		return rc;
	}
	friend inline simd_double8 fmaf(const simd_double8&, const simd_double8&, const simd_double8&);

	friend class simd_float8;


	friend inline simd_double8 round(const simd_double8& a) {
		simd_double8 b;
		b.v[0] = _mm256_round_pd(a.v[0], _MM_FROUND_TO_NEAREST_INT);
		b.v[1] = _mm256_round_pd(a.v[1], _MM_FROUND_TO_NEAREST_INT);
		return b;
	}
	friend inline simd_double8 sqrt(const simd_double8& a) {
		simd_double8 b;
		b.v[0] = _mm256_sqrt_pd(a.v[0]);
		b.v[1] = _mm256_sqrt_pd(a.v[1]);
		return b;
	}

};

inline simd_double8 fmaf(const simd_double8& a, const simd_double8& b, const simd_double8& c) {
	simd_double8 d;
	d.v[0] = _mm256_fmadd_pd(a.v[0], b.v[0], c.v[0]);
	d.v[1] = _mm256_fmadd_pd(a.v[1], b.v[1], c.v[1]);
	return d;
}


#define SIMD_FLOAT8_SIZE 8

class simd_int8;

class alignas(sizeof(__m256)) simd_float8 {
	__m256 v;
public:
	simd_float8() = default;
	simd_float8(simd_int8 a);
	inline simd_float8(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
		v = _mm256_set_ps(a8, a7, a6, a5, a4, a3, a2, a1);
	}
	inline simd_float8(float a) {
		v = _mm256_broadcast_ss(&a);
	}
	inline simd_float8(simd_double8 a) {
		__m128& v0 = (__m128&) v;
		__m128& v1 = *((__m128*)((float*)(&v)+4));
		v0 = _mm256_cvtpd_ps(a.v[0]);
		v1 = _mm256_cvtpd_ps(a.v[1]);
	}
	inline float& operator[](int i) {
		return v[i];
	}
	inline float operator[](int i) const {
		return v[i];
	}
	inline simd_float8 operator+(const simd_float8& other) const {
		simd_float8 a;
		a.v = _mm256_add_ps(v, other.v);
		return a;
	}
	inline simd_float8 operator-(const simd_float8& other) const {
		simd_float8 a;
		a.v = _mm256_sub_ps(v, other.v);
		return a;
	}
	inline simd_float8 operator+() const {
		return *this;
	}
	inline simd_float8 operator-() const {
		return simd_float8(0.0f) - *this;
	}
	inline simd_float8 operator*(const simd_float8& other) const {
		simd_float8 a;
		a.v = _mm256_mul_ps(v, other.v);
		return a;
	}
	inline simd_float8 operator/(const simd_float8& other) const {
		simd_float8 a;
		a.v = _mm256_div_ps(v, other.v);
		return a;
	}
	inline simd_float8 operator+=(const simd_float8& other) {
		v = _mm256_add_ps(v, other.v);
		return *this;
	}
	inline simd_float8 operator-=(const simd_float8& other) {
		v = _mm256_sub_ps(v, other.v);
		return *this;
	}
	inline simd_float8 operator*=(const simd_float8& other) {
		v = _mm256_mul_ps(v, other.v);
		return *this;
	}
	inline simd_float8 operator/=(const simd_float8& other) {
		v = _mm256_div_ps(v, other.v);
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
	inline float sum() const {
		__m128 vsum;
		__m128& a = *(((__m128*) &v) + 0);
		__m128& b = *(((__m128*) &v) + 1);
		vsum = _mm_add_ps(a, b);
		return vsum[0] + vsum[1] + vsum[2] + vsum[3];
	}
	inline simd_float8 operator<=(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_LE_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
		return rc;
	}
	inline simd_float8 operator<(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_LT_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
		return rc;
	}
	inline simd_float8 operator>(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_GT_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
		return rc;
	}
	inline simd_float8 operator>=(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_GE_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
		return rc;
	}
	inline float max() const {
		float mx = v[0];
		for( int i = 1; i < SIMD_FLOAT8_SIZE; i++) {
			mx = std::max(mx, v[i]);
		}
		return mx;
	}
	friend inline simd_float8 sqrt(const simd_float8&);
	friend inline simd_float8 rsqrt(const simd_float8&);
	friend inline simd_float8 max(const simd_float8&, const simd_float8&);
	friend inline simd_float8 min(const simd_float8&, const simd_float8&);
	friend inline simd_float8 round(const simd_float8&);
	friend inline simd_float8 fmaf(const simd_float8&, const simd_float8&, const simd_float8&);
		friend class simd_int8;
	friend simd_float8 log2(const simd_float8& a);

};

inline simd_float8 abs(const simd_float8& a) {
	return max(a,-a);
}

inline simd_float8 log2(const simd_float8& a) {
	__m256i i = _mm256_castps_si256(a.v);
	__m256i j;
	i = _mm256_srli_epi32(i, 23);
	j = _mm256_slli_epi32(i, 23);
	i = _mm256_sub_epi32(i, _mm256_set1_epi32(0x7f));
	simd_float8 ret, x;
	x.v = _mm256_castsi256_ps(j);
	ret.v = _mm256_cvtepi32_ps(i);
	x = a / x;
	x = sqrt(sqrt(x));
	static const float c0 = 1.0 / log(2.0);
	static const simd_float8 c1(c0);
	static const simd_float8 c2(-c0/2);
	static const simd_float8 c3(c0/3);
	static const simd_float8 c4(-c0/4);
	static const simd_float8 c5(c0/5);
	static const simd_float8 c6(-c0/6);
	static const simd_float8 c7(c0/7);
	static const simd_float8 c8(-c0/8);
	static const simd_float8 c9(c0/9);
	x = x - simd_float8(1.0f);
	simd_float8 y = c9;
	y = fmaf(y, x, c8);
	y = fmaf(y, x, c7);
	y = fmaf(y, x, c6);
	y = fmaf(y, x, c5);
	y = fmaf(y, x, c4);
	y = fmaf(y, x, c3);
	y = fmaf(y, x, c2);
	y = fmaf(y, x, c1);
	y = y * x;
	ret += simd_float8(4)*y;
	return ret;
}

inline simd_float8 log(const simd_float8& a) {
	static const simd_float8 b = 1.0/log2(exp(1));
	return log2(a)*b;
}

inline simd_float8 sqrt(const simd_float8& a) {
	simd_float8 b;
	b.v = _mm256_sqrt_ps(a.v);
	return b;
}

inline simd_float8 rsqrt(const simd_float8& a) {
	simd_float8 b;
	b.v = _mm256_rsqrt_ps(a.v);
	return b;
}

inline simd_float8 max(const simd_float8& a, const simd_float8& b) {
	simd_float8 c;
	c.v = _mm256_max_ps(a.v, b.v);
	return c;
}

inline simd_float8 min(const simd_float8& a, const simd_float8& b) {
	simd_float8 c;
	c.v = _mm256_min_ps(a.v, b.v);
	return c;
}

inline simd_float8 round(const simd_float8& a) {
	simd_float8 b;
	b.v = _mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT);
	return b;
}

inline simd_float8 fmaf(const simd_float8& a, const simd_float8& b, const simd_float8& c) {
	simd_float8 d;
	d.v = _mm256_fmadd_ps(a.v, b.v, c.v);
	return d;
}

#endif

#ifdef USE_AVX512

#define SIMD_FLOAT16_SIZE 16

class simd_int16;

class alignas(sizeof(__m512)) simd_float16 {
	__m512 v;
public:
	simd_float16() = default;
	simd_float16(simd_int16 a);
	simd_float16(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15, float a16) {
		v = _mm512_set_ps(a16, a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1);
	}
	simd_float16(float a) {
		v = _mm512_set_ps(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a);
	}
	float& operator[](int i) {
		return v[i];
	}
	float operator[](int i) const {
		return v[i];
	}
	inline simd_float16 operator+(const simd_float16& other) const {
		simd_float16 a;
		a.v = _mm512_add_ps(v, other.v);
		return a;
	}
	inline simd_float16 operator-(const simd_float16& other) const {
		simd_float16 a;
		a.v = _mm512_sub_ps(v, other.v);
		return a;
	}
	inline simd_float16 operator+() const {
		return *this;
	}
	inline simd_float16 operator-() const {
		return simd_float16(0.0f) - *this;
	}
	inline simd_float16 operator*(const simd_float16& other) const {
		simd_float16 a;
		a.v = _mm512_mul_ps(v, other.v);
		return a;
	}
	inline simd_float16 operator/(const simd_float16& other) const {
		simd_float16 a;
		a.v = _mm512_div_ps(v, other.v);
		return a;
	}
	inline simd_float16 operator+=(const simd_float16& other) {
		v = _mm512_add_ps(v, other.v);
		return *this;
	}
	inline simd_float16 operator-=(const simd_float16& other) {
		v = _mm512_sub_ps(v, other.v);
		return *this;
	}
	inline simd_float16 operator*=(const simd_float16& other) {
		v = _mm512_mul_ps(v, other.v);
		return *this;
	}
	inline simd_float16 operator/=(const simd_float16& other) {
		v = _mm512_div_ps(v, other.v);
		return *this;
	}
	inline simd_float16 operator*(float d) const {
		const simd_float16 other = d;
		return other * *this;
	}
	inline simd_float16 operator/(float d) const {
		const simd_float16 other = 1.0 / d;
		return *this * other;
	}
	float sum() const {
		__m256 vsum1;
		__m128 vsum2;
		__m256& a1 = *(((__m256*) &v) + 0);
		__m256& b1 = *(((__m256*) &v) + 1);
		vsum1 = _mm256_add_ps(a1, b1);
		__m128& a2 = *(((__m128*) &vsum1) + 0);
		__m128& b2 = *(((__m128*) &vsum1) + 1);
		vsum2 = _mm_add_ps(a2, b2);
		return vsum2[0] + vsum2[1] + vsum2[2] + vsum2[3];
	}
	simd_float16 operator<=(const simd_float16& other) const {
		simd_float16 rc;
		static const simd_float16 one(1);
		static const simd_float16 zero(0);
		auto mask0 = _mm512_cmp_ps_mask(v, other.v, _CMP_LE_OQ);
		rc.v = _mm512_mask_mov_ps(zero.v, mask0, one.v);
		return rc;
	}
	simd_float16 operator<(const simd_float16& other) const {
		simd_float16 rc;
		static const simd_float16 one(1);
		static const simd_float16 zero(0);
		auto mask0 = _mm512_cmp_ps_mask(v, other.v, _CMP_LT_OQ);
		rc.v = _mm512_mask_mov_ps(zero.v, mask0, one.v);
		return rc;
	}
	simd_float16 operator>(const simd_float16& other) const {
		simd_float16 rc;
		static const simd_float16 one(1);
		static const simd_float16 zero(0);
		auto mask0 = _mm512_cmp_ps_mask(v, other.v, _CMP_GT_OQ);
		rc.v = _mm512_mask_mov_ps(zero.v, mask0, one.v);
		return rc;
	}
	simd_float16 operator>=(const simd_float16& other) const {
		simd_float16 rc;
		static const simd_float16 one(1);
		static const simd_float16 zero(0);
		auto mask0 = _mm512_cmp_ps_mask(v, other.v, _CMP_GE_OQ);
		rc.v = _mm512_mask_mov_ps(zero.v, mask0, one.v);
		return rc;
	}
	friend inline simd_float16 sqrt(const simd_float16&);
	friend inline simd_float16 rsqrt(const simd_float16&);
	friend inline simd_float16 max(const simd_float16&, const simd_float16&);
	friend inline simd_float16 min(const simd_float16&, const simd_float16&);
	friend inline simd_float16 round(const simd_float16&);
	friend inline simd_float16 fmaf(const simd_float16&, const simd_float16&, const simd_float16&);
	friend class simd_int16;

};

inline simd_float16 sqrt(const simd_float16& a) {
	simd_float16 b;
	b.v = _mm512_sqrt_ps(a.v);
	return b;
}

inline simd_float16 rsqrt(const simd_float16& a) {
	simd_float16 b;
	b.v = _mm512_rsqrt14_ps(a.v);
	return b;
}

inline simd_float16 max(const simd_float16& a, const simd_float16& b) {
	simd_float16 c;
	c.v = _mm512_max_ps(a.v, b.v);
	return c;
}

inline simd_float16 min(const simd_float16& a, const simd_float16& b) {
	simd_float16 c;
	c.v = _mm512_min_ps(a.v, b.v);
	return c;
}

inline simd_float16 round(const simd_float16& a) {
	simd_float16 b;
	b.v = _mm512_roundscale_ps(a.v, _MM_FROUND_TO_NEAREST_INT);
	return b;
}

inline simd_float16 fmaf(const simd_float16& a, const simd_float16& b, const simd_float16& c) {
	simd_float16 d;
	d.v = _mm512_fmadd_ps(a.v, b.v, c.v);
	return d;
}

#endif

inline simd_float8 fmaxf(simd_float8 a, simd_float8 b) {
	return max(a, b);
}
#endif /* SIMD_FLOAT_HPP_ */
