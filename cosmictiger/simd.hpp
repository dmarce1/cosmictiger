/*
 * simd.hpp
 *
 *  Created on: Aug 29, 2021
 *      Author: dmarce1
 */

#ifndef SIMD_HPP_
#define SIMD_HPP_

#ifndef __CUDACC__

#include <cosmictiger/defs.hpp>

#include <immintrin.h>

#include <cmath>

#ifdef __AVX2__
#define USE_AVX2
#elif defined(__AVX__)
#define USE_AVX
#else
#error "No AVX"
#endif

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
		v = _mm_set_ps(a, a, a, a);
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
	d.v = _mm_fmadd_ps(a.v, b.v, c.v);
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
#ifdef USE_AVX2
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
	simd_float8(float a) {
		v[0] = _mm_set_ps(a, a, a, a);
		v[1] = _mm_set_ps(a, a, a, a);
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
		a.v[0] = _mm_sub_ps(v[0], other.v[0]);
		a.v[1] = _mm_sub_ps(v[1], other.v[1]);
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

#elif defined(USE_AVX2)

#define SIMD_FLOAT8_SIZE 8

class simd_int8;

class alignas(sizeof(__m256)) simd_float8 {
	__m256 v;
public:
	simd_float8() = default;
	simd_float8(simd_int8 a);
	simd_float8(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
		v = _mm256_set_ps(a8, a7, a6, a5, a4, a3, a2, a1);
	}
	simd_float8(float a) {
		v = _mm256_set_ps(a, a, a, a, a, a, a, a);
	}
	float& operator[](int i) {
		return v[i];
	}
	float operator[](int i) const {
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
	float sum() const {
		__m128 vsum;
		__m128& a = *(((__m128*) &v) + 0);
		__m128& b = *(((__m128*) &v) + 1);
		vsum = _mm_add_ps(a, b);
		return vsum[0] + vsum[1] + vsum[2] + vsum[3];
	}
	simd_float8 operator<=(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_LE_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
		return rc;
	}
	simd_float8 operator<(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_LT_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
		return rc;
	}
	simd_float8 operator>(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_GT_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
		return rc;
	}
	simd_float8 operator>=(const simd_float8& other) const {
		simd_float8 rc;
		static const simd_float8 one(1);
		auto mask0 = _mm256_cmp_ps(v, other.v, _CMP_GE_OQ);
		rc.v = _mm256_and_ps(mask0, one.v);
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

#else

#error "CosmicTiger requires AVX or AVX2"

#endif

class alignas(sizeof(__m128i)) simd_int4 {
private:
	union {
		__m128i v;
		int ints[4];
	};
public:
	simd_int4() = default;
	simd_int4(const simd_int4&) = default;
	simd_int4(int i) {
		v = _mm_set_epi32(i, i, i, i);
	}
	simd_int4(simd_float4 r) {
		v = _mm_cvtps_epi32(_mm_floor_ps(r.v));
	}
	inline simd_int4& operator=(const simd_int4 &other) = default;
	simd_int4& operator=(simd_int4 &&other) = default;
	inline simd_int4 operator+(const simd_int4 &other) const {
		simd_int4 r;
		r.v = _mm_add_epi32(v, other.v);
		return r;
	}
	inline simd_int4 operator-(const simd_int4 &other) const {
		simd_int4 r;
		r.v = _mm_sub_epi32(v, other.v);
		return r;
	}
	inline simd_int4 operator*(const simd_int4 &other) const {
		simd_int4 r;
		r.v = _mm_mul_epi32(v, other.v);
		return r;
	}
	inline simd_int4 operator+() const {
		return *this;
	}
	inline simd_int4 operator-() const {
		return simd_int4(0.0) - *this;
	}
	inline int& operator[](std::size_t i) {
		return ints[i];
	}
	inline int operator[](std::size_t i) const {
		return v[i];
	}
	inline simd_int4 operator<<=(int shft) {
		v = _mm_slli_epi32(v, shft);
		return *this;
	}
	friend class simd_float4;
};


inline simd_float4::simd_float4(simd_int4 a) {
	v = _mm_cvtepi32_ps(a.v);
}


#if defined(USE_AVX2)

class alignas(sizeof(__m256i)) simd_int8 {
private:
	union {
		__m256i v;
		int ints[8];
	};
public:
	simd_int8() = default;
	simd_int8(const simd_int8&) = default;
	simd_int8(simd_float8 r) {
		v = _mm256_cvtps_epi32(r.v);
	}
	simd_int8(int i) {
		v = _mm256_set_epi32(i, i, i, i, i, i, i, i);
	}
	inline simd_int8& operator=(const simd_int8 &other) = default;
	simd_int8& operator=(simd_int8 &&other) = default;
	inline simd_int8 operator+(const simd_int8 &other) const {
		simd_int8 r;
		r.v = _mm256_add_epi32(v, other.v);
		return r;
	}
	inline simd_int8 operator-(const simd_int8 &other) const {
		simd_int8 r;
		r.v = _mm256_sub_epi32(v, other.v);
		return r;
	}
	inline simd_int8 operator*(const simd_int8 &other) const {
		simd_int8 r;
		r.v = _mm256_mul_epi32(v, other.v);
		return r;
	}
	inline simd_int8 operator+() const {
		return *this;
	}
	inline simd_int8 operator-() const {
		return simd_int8(0.0) - *this;
	}
	inline int& operator[](std::size_t i) {
		return ints[i];
	}
	inline int operator[](std::size_t i) const {
		return v[i];
	}
	inline simd_int8 operator<<=(int shft) {
		v = _mm256_slli_epi32(v, shft);
		return *this;
	}
	friend class simd_float8;
};

inline simd_float8::simd_float8(simd_int8 a) {
	v = _mm256_cvtepi32_ps(a.v);
}


#elif defined(USE_AVX)

class alignas(2*sizeof(__m128i)) simd_int8 {
	__m128i v[2];

public:
	simd_int8(int i) {
		v[0] = _mm_set_epi32(i, i, i, i);
		v[1] = _mm_set_epi32(i, i, i, i);
	}
	simd_int8() = default;
	simd_int8(const simd_int8&) = default;
	inline simd_int8(float d) {
		v[0] = _mm_set_epi32(d, d, d, d);
		v[1] = _mm_set_epi32(d, d, d, d);
	}
	inline simd_int8& operator=(const simd_int8 &other) = default;
	simd_int8& operator=(simd_int8 &&other) = default;
	inline simd_int8 operator+(const simd_int8 &other) const {
		simd_int8 r;
		r.v[0] = _mm_add_epi32(v[0], other.v[0]);
		r.v[1] = _mm_add_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int8 operator-(const simd_int8 &other) const {
		simd_int8 r;
		r.v[0] = _mm_sub_epi32(v[0], other.v[0]);
		r.v[1] = _mm_sub_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int8 operator*(const simd_int8 &other) const {
		simd_int8 r;
		r.v[0] = _mm_mul_epi32(v[0], other.v[0]);
		r.v[1] = _mm_mul_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int8 operator+() const {
		return *this;
	}
	inline simd_int8 operator-() const {
		return simd_int8(0) - *this;
	}
	inline simd_int8& operator+=(const simd_int8 &other) {
		v[0] = _mm_add_epi32(v[0], other.v[0]);
		v[1] = _mm_add_epi32(v[1], other.v[1]);
		return *this;
	}
	inline simd_int8& operator-=(const simd_int8 &other) {
		v[0] = _mm_sub_epi32(v[0], other.v[0]);
		v[1] = _mm_sub_epi32(v[1], other.v[1]);
		return *this;
	}
	inline simd_int8& operator*=(const simd_int8 &other) {
		v[0] = _mm_mul_epi32(v[0], other.v[0]);
		v[1] = _mm_mul_epi32(v[1], other.v[1]);
		return *this;
	}
	inline simd_int8 operator*(float d) const {
		const simd_int8 other = d;
		return other * *this;
	}
	inline simd_int8 operator*=(float d) {
		*this = *this * d;
		return *this;
	}
	inline int& operator[](std::size_t i) {
		return *(((int*) &v) + i);
	}
	inline int operator[](std::size_t i) const {
		return v[i / 2][i % 4];
	}
	simd_int8(const simd_float8& r) {
		v[0] = _mm_cvtps_epi32(_mm_floor_ps(r.v[0]));
		v[1] = _mm_cvtps_epi32(_mm_floor_ps(r.v[1]));
	}
	friend class simd_float8;
};

inline simd_float8::simd_float8(simd_int8 a) {
	v[0] = _mm_cvtepi32_ps(a.v[0]);
	v[1] = _mm_cvtepi32_ps(a.v[1]);
}


#else

#error "CosmicTiger requires AVX or AVX2"

#endif

#ifdef USE_AVX2
using simd_float = simd_float8;
using simd_int = simd_int8;
#define SIMD_FLOAT_SIZE SIMD_FLOAT8_SIZE
#elif defined(USE_AVX)
using simd_float = simd_float4;
using simd_int = simd_int4;
#define SIMD_FLOAT_SIZE SIMD_FLOAT4_SIZE
#else
#error "CosmicTiger requires AVX or AVX2"
#endif

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

#endif /* SIMD_HPP_ */

#endif
