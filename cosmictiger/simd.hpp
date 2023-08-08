/*
 * simd.hpp
 *
 *  Created on: Aug 8, 2023
 *      Author: dmarce1
 */

#ifndef SIMD_HPP_
#define SIMD_HPP_


#include <simd.hpp>

static constexpr int SIMD_FLOAT_SIZE = simd::simd_f32::size();

using simd_float = simd::simd_f32;
using simd_int = simd::simd_i32;



class simd_double8 {
	simd::simd_f64 v[2];
public:
	inline simd_double8(const simd_i32&) {

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


#endif /* SIMD_HPP_ */
