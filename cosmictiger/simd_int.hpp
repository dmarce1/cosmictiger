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

#ifndef SIMD_INT_HPP_
#define SIMD_INT_HPP_

#include <stddef.h>

class alignas(sizeof(__m128i)) SimdInt4 {
private:
    union {
        __m128i v_;
        int ints_[4];
    };

public:
    SimdInt4() = default;
    SimdInt4(SimdInt4 const&) = default;
    SimdInt4(int i) {
        v_ = _mm_set_epi32(i, i, i, i);
    }
    SimdInt4(SimdFloat4 r) {
        v_ = _mm_cvtps_epi32(_mm_floor_ps(r.v_));
    }
    inline SimdInt4& operator=(SimdInt4 const& other) = default;
    SimdInt4& operator=(SimdInt4&& other) = default;
    inline SimdInt4 operator+(SimdInt4 const& other) const {
        SimdInt4 r;
        r.v_ = _mm_add_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt4 operator-(SimdInt4 const& other) const {
        SimdInt4 r;
        r.v_ = _mm_sub_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt4 operator*(SimdInt4 const& other) const {
        SimdInt4 r;
        r.v_ = _mm_mul_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt4 operator+() const {
        return *this;
    }
    inline SimdInt4 operator-() const {
        return SimdInt4(0.0) - *this;
    }
    inline int& operator[](size_t i) {
        return ints_[i];
    }
    inline int operator[](size_t i) const {
        return v_[i];
    }
    inline SimdInt4 operator<<=(int shft) {
        v_ = _mm_slli_epi32(v_, shft);
        return *this;
    }
    friend class SimdFloat4;
};

inline SimdFloat4::SimdFloat4(SimdInt4 a) {
    v_ = _mm_cvtepi32_ps(a.v_);
}

#if defined(USE_AVX)

class alignas(2 * sizeof(__m128i)) SimdInt8 {
    __m128i v_[2];

public:
    SimdInt8(int i) {
        v_[0] = _mm_set_epi32(i, i, i, i);
        v_[1] = _mm_set_epi32(i, i, i, i);
    }
    SimdInt8() = default;
    SimdInt8(SimdInt8 const&) = default;
    inline SimdInt8(float d) {
        v_[0] = _mm_set_epi32(d, d, d, d);
        v_[1] = _mm_set_epi32(d, d, d, d);
    }
    inline SimdInt8& operator=(SimdInt8 const& other) = default;
    SimdInt8& operator=(SimdInt8&& other) = default;
    inline SimdInt8 operator+(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_[0] = _mm_add_epi32(v_[0], other.v_[0]);
        r.v_[1] = _mm_add_epi32(v_[1], other.v_[1]);
        return r;
    }
    inline SimdInt8 operator-(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_[0] = _mm_sub_epi32(v_[0], other.v_[0]);
        r.v_[1] = _mm_sub_epi32(v_[1], other.v_[1]);
        return r;
    }
    inline SimdInt8 operator*(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_[0] = _mm_mul_epi32(v_[0], other.v_[0]);
        r.v_[1] = _mm_mul_epi32(v_[1], other.v_[1]);
        return r;
    }
    inline SimdInt8 operator+() const {
        return *this;
    }
    inline SimdInt8 operator-() const {
        return SimdInt8(0) - *this;
    }
    inline SimdInt8& operator+=(SimdInt8 const& other) {
        v_[0] = _mm_add_epi32(v_[0], other.v_[0]);
        v_[1] = _mm_add_epi32(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdInt8& operator-=(SimdInt8 const& other) {
        v_[0] = _mm_sub_epi32(v_[0], other.v_[0]);
        v_[1] = _mm_sub_epi32(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdInt8& operator*=(SimdInt8 const& other) {
        v_[0] = _mm_mul_epi32(v_[0], other.v_[0]);
        v_[1] = _mm_mul_epi32(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdInt8 operator<<(SimdInt8 const& other) {}
    inline SimdInt8 operator*(float d) const {
        SimdInt8 const other = d;
        return other * *this;
    }
    inline SimdInt8 operator*=(float d) {
        *this = *this * d;
        return *this;
    }
    inline int& operator[](size_t i) {
        return *(((int*) &v_) + i);
    }
    inline int operator[](size_t i) const {
        return v_[i / 2][i % 4];
    }
    SimdInt8(SimdFloat8 const& r) {
        v_[0] = _mm_cvtps_epi32(_mm_floor_ps(r.v_[0]));
        v_[1] = _mm_cvtps_epi32(_mm_floor_ps(r.v_[1]));
    }
    friend class SimdFloat8;
};

inline SimdFloat8::SimdFloat8(SimdInt8 a) {
    v_[0] = _mm_cvtepi32_ps(a.v_[0]);
    v_[1] = _mm_cvtepi32_ps(a.v_[1]);
}
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)

class alignas(sizeof(__m256i)) SimdInt8 {
private:
    union {
        __m256i v_;
        int ints_[8];
    };

public:
    SimdInt8() = default;
    SimdInt8(SimdInt8 const&) = default;
    SimdInt8(SimdFloat8 r) {
        v_ = _mm256_cvtps_epi32(_mm256_floor_ps(r.v_));
    }
    SimdInt8(int i) {
        v_ = _mm256_set_epi32(i, i, i, i, i, i, i, i);
    }
    inline SimdInt8& operator=(SimdInt8 const& other) = default;
    SimdInt8& operator=(SimdInt8&& other) = default;
    inline SimdInt8 operator+(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_ = _mm256_add_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt8 operator-(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_ = _mm256_sub_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt8 operator*(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_ = _mm256_mul_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt8 operator+() const {
        return *this;
    }
    inline SimdInt8 operator-() const {
        return SimdInt8(0.0) - *this;
    }
    inline int& operator[](size_t i) {
        return ints_[i];
    }
    inline int operator[](size_t i) const {
        return v_[i];
    }
    inline SimdInt8 operator<<=(int shft) {
        v_ = _mm256_slli_epi32(v_, shft);
        return *this;
    }
    SimdInt8 operator==(SimdInt8 const& other) const {
        SimdInt8 rc;
        rc.v_ = _mm256_cmpeq_epi32(v_, other.v_);
        rc.v_ = -rc.v_;
        return rc;
    }
    friend class SimdDouble8;

    friend class SimdFloat8;
};

inline SimdFloat8::SimdFloat8(SimdInt8 a) {
    v_ = _mm256_cvtepi32_ps(a.v_);
}

#endif

#if defined(USE_AVX512)

class alignas(sizeof(__m512i)) SimdInt16 {
private:
    union {
        __m512i v_;
        int ints_[16];
    };

public:
    SimdInt16() = default;
    SimdInt16(SimdInt16 const&) = default;
    SimdInt16(SimdFloat16 r) {
        v_ = _mm512_cvtps_epi32(r.v_);
    }
    SimdInt16(int i) {
        v_ = _mm512_set_epi32(i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i);
    }
    inline SimdInt16& operator=(SimdInt16 const& other) = default;
    SimdInt16& operator=(SimdInt16&& other) = default;
    inline SimdInt16 operator+(SimdInt16 const& other) const {
        SimdInt16 r;
        r.v_ = _mm512_add_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt16 operator-(SimdInt16 const& other) const {
        SimdInt16 r;
        r.v_ = _mm512_sub_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt16 operator*(SimdInt16 const& other) const {
        SimdInt16 r;
        r.v_ = _mm512_mul_epi32(v_, other.v_);
        return r;
    }
    inline SimdInt16 operator+() const {
        return *this;
    }
    inline SimdInt16 operator-() const {
        return SimdInt16(0.0) - *this;
    }
    inline int& operator[](size_t i) {
        return ints_[i];
    }
    inline int operator[](size_t i) const {
        return v_[i];
    }
    inline SimdInt16 operator<<=(int shft) {
        v_ = _mm512_slli_epi32(v_, shft);
        return *this;
    }
    friend class SimdFloat16;
};

inline SimdFloat16::SimdFloat16(SimdInt16 a) {
    v_ = _mm512_cvtepi32_ps(a.v_);
}

#endif

#endif /* SIMD_INT_HPP_ */
