/*
 * simd.hpp
 *
 *  Created on: Aug 8, 2023
 *      Author: dmarce1
 */

#ifndef SIMD1_HPP_
#define SIMD1_HPP_

#include <simd.hpp>
#include <cosmictiger/sfmm.hpp>

static constexpr int SIMD_FLOAT_SIZE = simd::simd_f32::size();
#ifndef __CUDACC__

using simd_float = simd::simd_f32;
using simd_double = simd::simd_f64;
using simd_int = simd::simd_i32;

class simd_double8 {
	simd::simd_f64 v[2];
public:
	inline simd_double8(const simd_fixed& a) {
		__m128i& v0 = *((__m128i*) &a);
		__m128i& v1 = *((__m128i*) (((float*) &a) + 4));
		sfmm::simd_f64 factor(1.0 / (std::numeric_limits<std::uint32_t>::max() + 1.0));
		const auto A0 = _mm256_mul_pd(_mm256_cvtepi32_pd(v0), (__m256d&) factor);
		const auto A1 = _mm256_mul_pd(_mm256_cvtepi32_pd(v1), (__m256d&) factor);
		v[0] = *((simd::simd_f64*) (&A0));
		v[1] = *((simd::simd_f64*) (&A1));
	}
	simd_double8() = default;
	inline double& operator[](int i) {
		return v[i/4][i%4];
	}
	inline double operator[](int i) const {
		return v[i/4][i%4];
	}
	inline simd_double8(double a) {
		v[0] = a;
		v[1] = a;
	}
	inline simd_double8 operator+(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = v[0] + other.v[0];
		c.v[1] = v[1] + other.v[1];
		return c;
	}
	inline simd_double8 operator-(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = v[0] - other.v[0];
		c.v[1] = v[1] - other.v[1];
		return c;
	}
	inline simd_double8 operator*(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = v[0] * other.v[0];
		c.v[1] = v[1] * other.v[1];
		return c;
	}
	inline simd_double8 operator/(const simd_double8& other) const {
		simd_double8 c;
		c.v[0] = v[0] / other.v[0];
		c.v[1] = v[1] / other.v[1];
		return c;
	}
	inline simd_double8 operator-() const {
		simd_double8 c(0.0);
		c.v[0] = c.v[0] - v[0];
		c.v[1] = c.v[1] - v[1];
		return c;
	}
	inline simd_double8 operator+=(const simd_double8& other) {
		v[0] = v[0] + other.v[0];
		v[1] = v[1] + other.v[1];
		return *this;
	}
	inline simd_double8 operator-=(const simd_double8& other) {
		v[0] = v[0] - other.v[0];
		v[1] = v[1] - other.v[1];
		return *this;
	}
	inline simd_double8 operator*=(const simd_double8& other) {
		v[0] = v[0] * other.v[0];
		v[1] = v[1] * other.v[1];
		return *this;
	}
	inline simd_double8 operator/=(const simd_double8& other) {
		v[0] = v[0] / other.v[0];
		v[1] = v[1] / other.v[1];
		return *this;
	}
	friend inline simd_double8 max(const simd_double8& a, const simd_double8& b) {
		simd_double8 c;
		c.v[0] = fmax(a.v[0], b.v[0]);
		c.v[1] = fmax(a.v[1], b.v[1]);
		return c;
	}

	friend inline simd_double8 min(const simd_double8& a, const simd_double8& b) {
		simd_double8 c;
		c.v[0] = fmin(a.v[0], b.v[0]);
		c.v[1] = fmin(a.v[1], b.v[1]);
		return c;
	}

	inline simd_double8 operator<(const simd_double8& other) const {
		simd_double8 rc;
		rc.v[0] = v[0] < other.v[0];
		rc.v[1] = v[1] < other.v[1];
		return rc;
	}

	inline simd_double8 operator>(const simd_double8& other) const {
		simd_double8 rc;
		rc.v[0] = v[0] > other.v[0];
		rc.v[1] = v[1] > other.v[1];
		return rc;
	}
	friend inline simd_double8 fmaf(const simd_double8&, const simd_double8&, const simd_double8&);

	friend class simd_float8;

	friend inline simd_double8 round(const simd_double8& a) {
		simd_double8 b;
		b.v[0] = round(a.v[0]);
		b.v[1] = round(a.v[1]);
		return b;
	}
	friend inline simd_double8 sqrt(const simd_double8& a) {
		simd_double8 b;
		b.v[0] = sqrt(a.v[0]);
		b.v[1] = sqrt(a.v[1]);
		return b;
	}
	double sum() const {
		double s = v[0][0] + v[0][1] + v[0][2] + v[0][3];
		s += v[1][0] + v[1][1] + v[1][2] + v[1][3];
		return s;
	}

};

inline simd_float simd_double8_2_simd_float(const simd_double8& v) {
	simd_float a;
	__m128& a0 = *((__m128*) &a);
	__m128& a1 = *((__m128*) (((float*) &a) + 4));
	__m256d& v0 = *((__m256d*) &v);
	__m256d& v1 = *((__m256d*) (((double*) &v) + 4));
	a0 = _mm256_cvtpd_ps(v0);
	a1 = _mm256_cvtpd_ps(v1);
	return a;
}
#endif

#endif /* SIMD_HPP_ */
