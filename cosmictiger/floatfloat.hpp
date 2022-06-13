/*
 * floatfloat.hpp
 *
 *  Created on: Jun 13, 2022
 *      Author: dmarce1
 */

#ifndef FLOATFLOAT_HPP_
#define FLOATFLOAT_HPP_

#include <cosmictiger/cuda.hpp>

struct two_float {
	float x;
	float y;

	CUDA_EXPORT
	inline two_float(float a, float b) {
		x = a;
		y = b;
	}
	CUDA_EXPORT
	inline two_float() {

	}
};

class floatfloat {
	two_float A;

	const double SPLITTER = (1 << 29) + 1;

	CUDA_EXPORT
	inline void split(double a, float* a_hi, float* a_lo) {
		const volatile double t = a * SPLITTER;
		const volatile double t_hi1 = t - a;
		const volatile double t_hi = t - t_hi1;
		const double t_lo = a - t_hi;
		*a_hi = (float) t_hi;
		*a_lo = (float) t_lo;
	}

	CUDA_EXPORT
	inline two_float quickTwoSum(float a, float b) {  // 3
		const volatile float s = a + b;
		const float e0 = s - a;
		const volatile float e = b - e0;
		return two_float(s, e);
	}

	CUDA_EXPORT
	inline two_float twoSum(float a, float b) { // 6
		const volatile float s = a + b;
		const float v = s - a;
		const volatile float e0 = s - v;
		const volatile float e2 = b - v;
		return two_float(s, (a - e0) + e2);
	}

	CUDA_EXPORT
	inline two_float df64_add(two_float a, two_float b) { // 19
		two_float s = twoSum(a.x, b.x);
		two_float t = twoSum(a.y, b.y);
		s.y += t.x;
		s = quickTwoSum(s.x, s.y);
		s.y += t.y;
		s = quickTwoSum(s.x, s.y);
		return s;
	}

	CUDA_EXPORT
	inline two_float df64_diff(two_float a, two_float b) {
		b.x = -b.x;
		b.y = -b.y;
		return df64_add(a, b);
	}
	CUDA_EXPORT
	inline two_float split(float a) {
		const float split = 4097;
		const float t = a * split;
		const float a0 = t - a;
		const float a_hi = t - a0;
		const float a_lo = a - a_hi;
		return two_float(a_hi, a_lo);
	}

	CUDA_EXPORT
	inline two_float twoProd(float a, float b) {
		const volatile float p = a * b;
		const two_float aS = split(a);
		const volatile two_float bS = split(b);
		const float err1 = (aS.x * bS.x - p);
		const volatile float err = (err1 + aS.x * bS.y + aS.y * bS.x + aS.y * bS.y);
		return two_float(p, err);
	}

	CUDA_EXPORT
	inline two_float df64_mult(two_float a, two_float b) {
		two_float p;
		p = twoProd(a.x, b.x);
		p.y += a.x * b.y;
		p.y += a.y * b.x;
		p = quickTwoSum(p.x, p.y);
		return p;
	}

public:
	inline floatfloat() = default;
	inline floatfloat(const floatfloat&) = default;
	inline floatfloat& operator=(floatfloat&&) = default;
	inline floatfloat& operator=(const floatfloat&) = default;
	CUDA_EXPORT
	operator double() {
		return (double) A.x + (double) A.y;
	}
	CUDA_EXPORT
	inline floatfloat(float r) {
		*this = r;
	}
	CUDA_EXPORT
	inline floatfloat(double r) {
		*this = r;
	}
	CUDA_EXPORT
	inline floatfloat& operator=(float r) {
		A.x = r;
		A.y = 0;
		return *this;
	}
	CUDA_EXPORT
	inline floatfloat& operator=(double r) {
		split(r, &A.x, &A.y);
		return *this;
	}
	CUDA_EXPORT
	inline floatfloat& operator+=(const floatfloat& other) {
		A = df64_add(A, other.A);
		return *this;
	}
	CUDA_EXPORT
	inline floatfloat& operator-=(const floatfloat& other) {
		A = df64_diff(A, other.A);
		return *this;
	}
	CUDA_EXPORT
	inline floatfloat& operator*=(const floatfloat& other) {
		A = df64_mult(A, other.A);
		return *this;
	}
	CUDA_EXPORT
	inline floatfloat operator+(const floatfloat& other) const {
		auto B = *this;
		B += other;
		return B;
	}
	CUDA_EXPORT
	inline floatfloat operator-(const floatfloat& other) const {
		auto B = *this;
		B -= other;
		return B;
	}
	CUDA_EXPORT
	inline floatfloat operator*(const floatfloat& other) const {
		auto B = *this;
		B *= other;
		return B;
	}
	CUDA_EXPORT
	inline floatfloat operator-() const {
		auto B = *this;
		B.A.x = -B.A.x;
		B.A.y = -B.A.y;
		return B;
	}

};
#endif /* FLOATFLOAT_HPP_ */

