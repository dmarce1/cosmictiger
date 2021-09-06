/*
 * simd_int.hpp
 *
 *  Created on: Aug 29, 2021
 *      Author: dmarce1
 */

#ifndef SIMD_INT_HPP_
#define SIMD_INT_HPP_


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

#if defined(USE_AVX)

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
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)

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
		v = _mm256_cvtps_epi32(_mm256_floor_ps(r.v));
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

#endif

#if defined(USE_AVX512)

class alignas(sizeof(__m512i)) simd_int16 {
private:
	union {
		__m512i v;
		int ints[16];
	};
public:
	simd_int16() = default;
	simd_int16(const simd_int16&) = default;
	simd_int16(simd_float16 r) {
		v = _mm512_cvtps_epi32(r.v);
	}
	simd_int16(int i) {
		v = _mm512_set_epi32(i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i);
	}
	inline simd_int16& operator=(const simd_int16 &other) = default;
	simd_int16& operator=(simd_int16 &&other) = default;
	inline simd_int16 operator+(const simd_int16 &other) const {
		simd_int16 r;
		r.v = _mm512_add_epi32(v, other.v);
		return r;
	}
	inline simd_int16 operator-(const simd_int16 &other) const {
		simd_int16 r;
		r.v = _mm512_sub_epi32(v, other.v);
		return r;
	}
	inline simd_int16 operator*(const simd_int16 &other) const {
		simd_int16 r;
		r.v = _mm512_mul_epi32(v, other.v);
		return r;
	}
	inline simd_int16 operator+() const {
		return *this;
	}
	inline simd_int16 operator-() const {
		return simd_int16(0.0) - *this;
	}
	inline int& operator[](std::size_t i) {
		return ints[i];
	}
	inline int operator[](std::size_t i) const {
		return v[i];
	}
	inline simd_int16 operator<<=(int shft) {
		v = _mm512_slli_epi32(v, shft);
		return *this;
	}
	friend class simd_float16;
};

inline simd_float16::simd_float16(simd_int16 a) {
	v = _mm512_cvtepi32_ps(a.v);
}

#endif


#endif /* SIMD_INT_HPP_ */
