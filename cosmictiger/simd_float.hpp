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

class SimdInt4;

class alignas(sizeof(__m128)) SimdFloat4 {
    __m128 v_;

public:
    SimdFloat4() = default;
    SimdFloat4(float a1, float a2, float a3, float a4) {
        v_ = _mm_set_ps(a4, a3, a2, a1);
    }
    SimdFloat4(SimdInt4 a);
    SimdFloat4(float a) {
        v_ = _mm_broadcast_ss(&a);
    }
    float& operator[](int i) {
        return v_[i];
    }
    float operator[](int i) const {
        return v_[i];
    }
    inline SimdFloat4 operator+(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = _mm_add_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator-(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = _mm_sub_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator+() const {
        return *this;
    }
    inline SimdFloat4 operator-() const {
        return SimdFloat4(0.0f) - *this;
    }
    inline SimdFloat4 operator*(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = _mm_mul_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator/(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = _mm_div_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator+=(SimdFloat4 const& other) {
        v_ = _mm_add_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat4 operator-=(SimdFloat4 const& other) {
        v_ = _mm_sub_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat4 operator*=(SimdFloat4 const& other) {
        v_ = _mm_mul_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat4 operator/=(SimdFloat4 const& other) {
        v_ = _mm_div_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat4 operator*(float d) const {
        SimdFloat4 const other = d;
        return other * *this;
    }
    inline SimdFloat4 operator/(float d) const {
        SimdFloat4 const other = 1.0 / d;
        return *this * other;
    }
    float sum() const {
        float s = (*this)[0];
        for (int i = 1; i < SIMD_FLOAT4_SIZE; i++) {
            s += (*this)[i];
        }
        return s;
    }
    SimdFloat4 operator<=(SimdFloat4 const& other) const {
        SimdFloat4 rc;
        static SimdFloat4 const one(1);
        auto mask0 = _mm_cmp_ps(v_, other.v_, _CMP_LE_OQ);
        rc.v_ = _mm_and_ps(mask0, one.v_);
        return rc;
    }
    SimdFloat4 operator<(SimdFloat4 const& other) const {
        SimdFloat4 rc;
        static SimdFloat4 const one(1);
        auto mask0 = _mm_cmp_ps(v_, other.v_, _CMP_LT_OQ);
        rc.v_ = _mm_and_ps(mask0, one.v_);
        return rc;
    }
    SimdFloat4 operator>(SimdFloat4 const& other) const {
        SimdFloat4 rc;
        static SimdFloat4 const one(1);
        auto mask0 = _mm_cmp_ps(v_, other.v_, _CMP_GT_OQ);
        rc.v_ = _mm_and_ps(mask0, one.v_);
        return rc;
    }
    SimdFloat4 operator>=(SimdFloat4 const& other) const {
        SimdFloat4 rc;
        static SimdFloat4 const one(1);
        auto mask0 = _mm_cmp_ps(v_, other.v_, _CMP_GE_OQ);
        rc.v_ = _mm_and_ps(mask0, one.v_);
        return rc;
    }
    friend inline SimdFloat4 sqrt(SimdFloat4 const&);
    friend inline SimdFloat4 rsqrt(SimdFloat4 const&);
    friend inline SimdFloat4 max(SimdFloat4 const&, SimdFloat4 const&);
    friend inline SimdFloat4 min(SimdFloat4 const&, SimdFloat4 const&);
    friend inline SimdFloat4 round(SimdFloat4 const&);
    friend inline SimdFloat4 fmaf(SimdFloat4 const&, SimdFloat4 const&, SimdFloat4 const&);
    friend class SimdInt4;
};

inline SimdFloat4 sqrt(SimdFloat4 const& a) {
    SimdFloat4 b;
    b.v_ = _mm_sqrt_ps(a.v_);
    return b;
}

inline SimdFloat4 rsqrt(SimdFloat4 const& a) {
    SimdFloat4 b;
    b.v_ = _mm_rsqrt_ps(a.v_);
    return b;
}

inline SimdFloat4 max(SimdFloat4 const& a, SimdFloat4 const& b) {
    SimdFloat4 c;
    c.v_ = _mm_max_ps(a.v_, b.v_);
    return c;
}

inline SimdFloat4 min(SimdFloat4 const& a, SimdFloat4 const& b) {
    SimdFloat4 c;
    c.v_ = _mm_min_ps(a.v_, b.v_);
    return c;
}

inline SimdFloat4 round(SimdFloat4 const& a) {
    SimdFloat4 b;
    b.v_ = _mm_round_ps(a.v_, _MM_FROUND_TO_NEAREST_INT);
    return b;
}

inline SimdFloat4 fmaf(SimdFloat4 const& a, SimdFloat4 const& b, SimdFloat4 const& c) {
    SimdFloat4 d;
#ifdef USE_AVX
    return a * b + c;
#else
    d.v_ = _mm_fmadd_ps(a.v_, b.v_, c.v_);
#endif
    return d;
}

class alignas(sizeof(__m128d)) SimdDouble {
    __m128d v_;

public:
    SimdDouble() = default;
    inline double& operator[](int i) {
        return v_[i];
    }
    inline double operator[](int i) const {
        return v_[i];
    }
    inline SimdDouble(double a) {
        v_ = _mm_set_pd(a, a);
    }
    inline SimdDouble operator+(SimdDouble const& other) const {
        SimdDouble c;
        c.v_ = _mm_add_pd(v_, other.v_);
        return c;
    }
    inline SimdDouble operator-(SimdDouble const& other) const {
        SimdDouble c;
        c.v_ = _mm_sub_pd(v_, other.v_);
        return c;
    }
    inline SimdDouble operator*(SimdDouble const& other) const {
        SimdDouble c;
        c.v_ = _mm_mul_pd(v_, other.v_);
        return c;
    }
    inline SimdDouble operator/(SimdDouble const& other) const {
        SimdDouble c;
        c.v_ = _mm_div_pd(v_, other.v_);
        return c;
    }
    inline SimdDouble operator-() const {
        SimdDouble c(0.0);
        c.v_ = _mm_sub_pd(c.v_, v_);
        return c;
    }
    inline SimdDouble operator+=(SimdDouble const& other) {
        v_ = _mm_add_pd(v_, other.v_);
        return *this;
    }
    inline SimdDouble operator-=(SimdDouble const& other) {
        v_ = _mm_sub_pd(v_, other.v_);
        return *this;
    }
    inline SimdDouble operator*=(SimdDouble const& other) {
        v_ = _mm_mul_pd(v_, other.v_);
        return *this;
    }
    inline SimdDouble operator/=(SimdDouble const& other) {
        v_ = _mm_div_pd(v_, other.v_);
        return *this;
    }
    friend SimdDouble fmaf(SimdDouble const& a, SimdDouble const& b, SimdDouble const& c);
};

inline SimdDouble fmaf(SimdDouble const& a, SimdDouble const& b, SimdDouble const& c) {
    SimdDouble d;
#if defined(USE_AVX2) || defined(USE_AVX512)
    d.v_ = _mm_fmadd_pd(a.v_, b.v_, c.v_);
#else
    d = a * b + c;
#endif
    return d;
}

#if defined(USE_AVX)

#define SIMD_FLOAT8_SIZE 8

class SimdInt8;

class alignas(2 * sizeof(__m128)) SimdFloat8 {
    __m128 v_[2];

public:
    SimdFloat8() = default;
    SimdFloat8(SimdInt8 a);
    SimdFloat8(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
        v_[0] = _mm_set_ps(a4, a3, a2, a1);
        v_[1] = _mm_set_ps(a8, a7, a6, a5);
    }
    inline SimdFloat8(float a) {
        v_[0] = _mm_broadcast_ss(&a);
        v_[1] = _mm_broadcast_ss(&a);
    }
    float& operator[](int i) {
        return v_[i / 4][i % 4];
    }
    float operator[](int i) const {
        return v_[i / 4][i % 4];
    }
    inline SimdFloat8 operator+(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = _mm_add_ps(v_[0], other.v_[0]);
        a.v_[1] = _mm_add_ps(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator-(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = _mm_sub_ps(v_[0], other.v_[0]);
        a.v_[1] = _mm_sub_ps(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator+() const {
        return *this;
    }
    inline SimdFloat8 operator-() const {
        return SimdFloat8(0.0f) - *this;
    }
    inline SimdFloat8 operator*(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = _mm_mul_ps(v_[0], other.v_[0]);
        a.v_[1] = _mm_mul_ps(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator/(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = _mm_div_ps(v_[0], other.v_[0]);
        a.v_[1] = _mm_div_ps(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator+=(SimdFloat8 const& other) {
        v_[0] = _mm_add_ps(v_[0], other.v_[0]);
        v_[1] = _mm_add_ps(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdFloat8 operator-=(SimdFloat8 const& other) {
        v_[0] = _mm_sub_ps(v_[0], other.v_[0]);
        v_[1] = _mm_sub_ps(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdFloat8 operator*=(SimdFloat8 const& other) {
        v_[0] = _mm_mul_ps(v_[0], other.v_[0]);
        v_[1] = _mm_mul_ps(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdFloat8 operator/=(SimdFloat8 const& other) {
        v_[0] = _mm_div_ps(v_[0], other.v_[0]);
        v_[1] = _mm_div_ps(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdFloat8 operator*(float d) const {
        SimdFloat8 const other = d;
        return other * *this;
    }
    inline SimdFloat8 operator/(float d) const {
        SimdFloat8 const other = 1.0 / d;
        return *this * other;
    }
    float sum() const {
        auto const a = _mm_add_ps(v_[0], v_[1]);
        float s = a[0];
        for (int i = 1; i < 4; i++) {
            s += a[i];
        }
        return s;
    }
    SimdFloat8 operator<=(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm_cmp_ps(v_[0], other.v_[0], _CMP_LE_OQ);
        auto mask1 = _mm_cmp_ps(v_[1], other.v_[1], _CMP_LE_OQ);
        rc.v_[0] = _mm_and_ps(mask0, one.v_[0]);
        rc.v_[1] = _mm_and_ps(mask1, one.v_[1]);
        return rc;
    }
    SimdFloat8 operator<(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm_cmp_ps(v_[0], other.v_[0], _CMP_LT_OQ);
        auto mask1 = _mm_cmp_ps(v_[1], other.v_[1], _CMP_LT_OQ);
        rc.v_[0] = _mm_and_ps(mask0, one.v_[0]);
        rc.v_[1] = _mm_and_ps(mask1, one.v_[1]);
        return rc;
    }
    SimdFloat8 operator>(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm_cmp_ps(v_[0], other.v_[0], _CMP_GT_OQ);
        auto mask1 = _mm_cmp_ps(v_[1], other.v_[1], _CMP_GT_OQ);
        rc.v_[0] = _mm_and_ps(mask0, one.v_[0]);
        rc.v_[1] = _mm_and_ps(mask1, one.v_[1]);
        return rc;
    }
    SimdFloat8 operator>=(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm_cmp_ps(v_[0], other.v_[0], _CMP_GE_OQ);
        auto mask1 = _mm_cmp_ps(v_[1], other.v_[1], _CMP_GE_OQ);
        rc.v_[0] = _mm_and_ps(mask0, one.v_[0]);
        rc.v_[1] = _mm_and_ps(mask1, one.v_[1]);
        return rc;
    }

    friend inline SimdFloat8 sqrt(SimdFloat8 const&);
    friend inline SimdFloat8 rsqrt(SimdFloat8 const&);
    friend inline SimdFloat8 max(SimdFloat8 const&, SimdFloat8 const&);
    friend inline SimdFloat8 min(SimdFloat8 const&, SimdFloat8 const&);
    friend inline SimdFloat8 round(SimdFloat8 const&);
    friend inline SimdFloat8 fmaf(SimdFloat8 const&, SimdFloat8 const&, SimdFloat8 const&);
    friend class SimdInt8;
};

inline SimdFloat8 sqrt(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_[0] = _mm_sqrt_ps(a.v_[0]);
    b.v_[1] = _mm_sqrt_ps(a.v_[1]);
    return b;
}

inline SimdFloat8 rsqrt(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_[0] = _mm_rsqrt_ps(a.v_[0]);
    b.v_[1] = _mm_rsqrt_ps(a.v_[1]);
    return b;
}

inline SimdFloat8 max(SimdFloat8 const& a, SimdFloat8 const& b) {
    SimdFloat8 c;
    c.v_[0] = _mm_max_ps(a.v_[0], b.v_[0]);
    c.v_[1] = _mm_max_ps(a.v_[1], b.v_[1]);
    return c;
}

inline SimdFloat8 fmaxf(SimdFloat8 const& a, SimdFloat8 const& b) {
    return max(a, b);
}

inline SimdFloat8 min(SimdFloat8 const& a, SimdFloat8 const& b) {
    SimdFloat8 c;
    c.v_[0] = _mm_min_ps(a.v_[0], b.v_[0]);
    c.v_[1] = _mm_min_ps(a.v_[1], b.v_[1]);
    return c;
}

inline SimdFloat8 round(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_[0] = _mm_round_ps(a.v_[0], _MM_FROUND_TO_NEAREST_INT);
    b.v_[1] = _mm_round_ps(a.v_[1], _MM_FROUND_TO_NEAREST_INT);
    return b;
}

inline SimdFloat8 fmaf(SimdFloat8 const& a, SimdFloat8 const& b, SimdFloat8 const& c) {
    SimdFloat8 d;
    d.v_[0] = _mm_fmadd_ps(a.v_[0], b.v_[0], c.v_[0]);
    d.v_[1] = _mm_fmadd_ps(a.v_[1], b.v_[1], c.v_[1]);
    return d;
}

#endif

class SimdInt8;

#if defined(USE_AVX2) || defined(USE_AVX512)

class alignas(sizeof(__m256d)) SimdDouble8 {
    __m256d v_[2];

public:
    friend class SimdInt8;
    inline SimdDouble8(SimdInt8 const&);
    SimdDouble8() = default;
    inline double& operator[](int i) {
        return v_[i / 4][i % 4];
    }
    inline double operator[](int i) const {
        return v_[i / 4][i % 4];
    }
    inline SimdDouble8(double a) {
        v_[0] = _mm256_set_pd(a, a, a, a);
        v_[1] = _mm256_set_pd(a, a, a, a);
    }
    inline SimdDouble8 operator+(SimdDouble8 const& other) const {
        SimdDouble8 c;
        c.v_[0] = _mm256_add_pd(v_[0], other.v_[0]);
        c.v_[1] = _mm256_add_pd(v_[1], other.v_[1]);
        return c;
    }
    inline SimdDouble8 operator-(SimdDouble8 const& other) const {
        SimdDouble8 c;
        c.v_[0] = _mm256_sub_pd(v_[0], other.v_[0]);
        c.v_[1] = _mm256_sub_pd(v_[1], other.v_[1]);
        return c;
    }
    inline SimdDouble8 operator*(SimdDouble8 const& other) const {
        SimdDouble8 c;
        c.v_[0] = _mm256_mul_pd(v_[0], other.v_[0]);
        c.v_[1] = _mm256_mul_pd(v_[1], other.v_[1]);
        return c;
    }
    inline SimdDouble8 operator/(SimdDouble8 const& other) const {
        SimdDouble8 c;
        c.v_[0] = _mm256_div_pd(v_[0], other.v_[0]);
        c.v_[1] = _mm256_div_pd(v_[1], other.v_[1]);
        return c;
    }
    inline SimdDouble8 operator-() const {
        SimdDouble8 c(0.0);
        c.v_[0] = _mm256_sub_pd(c.v_[0], v_[0]);
        c.v_[1] = _mm256_sub_pd(c.v_[1], v_[1]);
        return c;
    }
    inline SimdDouble8 operator+=(SimdDouble8 const& other) {
        v_[0] = _mm256_add_pd(v_[0], other.v_[0]);
        v_[1] = _mm256_add_pd(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdDouble8 operator-=(SimdDouble8 const& other) {
        v_[0] = _mm256_sub_pd(v_[0], other.v_[0]);
        v_[1] = _mm256_sub_pd(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdDouble8 operator*=(SimdDouble8 const& other) {
        v_[0] = _mm256_mul_pd(v_[0], other.v_[0]);
        v_[1] = _mm256_mul_pd(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdDouble8 operator/=(SimdDouble8 const& other) {
        v_[0] = _mm256_div_pd(v_[0], other.v_[0]);
        v_[1] = _mm256_div_pd(v_[1], other.v_[1]);
        return *this;
    }
    friend inline SimdDouble8 max(SimdDouble8 const& a, SimdDouble8 const& b) {
        SimdDouble8 c;
        c.v_[0] = _mm256_max_pd(a.v_[0], b.v_[0]);
        c.v_[1] = _mm256_max_pd(a.v_[1], b.v_[1]);
        return c;
    }

    friend inline SimdDouble8 min(SimdDouble8 const& a, SimdDouble8 const& b) {
        SimdDouble8 c;
        c.v_[0] = _mm256_min_pd(a.v_[0], b.v_[0]);
        c.v_[1] = _mm256_min_pd(a.v_[1], b.v_[1]);
        return c;
    }

    inline SimdDouble8 operator<(SimdDouble8 const& other) const {
        SimdDouble8 rc;
        static SimdDouble8 const one(1.0);
        auto mask0 = _mm256_cmp_pd(v_[0], other.v_[0], _CMP_LT_OQ);
        auto mask1 = _mm256_cmp_pd(v_[1], other.v_[1], _CMP_LT_OQ);
        rc.v_[0] = _mm256_and_pd(mask0, one.v_[0]);
        rc.v_[1] = _mm256_and_pd(mask1, one.v_[1]);
        return rc;
    }

    inline SimdDouble8 operator>(SimdDouble8 const& other) const {
        SimdDouble8 rc;
        static SimdDouble8 const one(1.0);
        auto mask0 = _mm256_cmp_pd(v_[0], other.v_[0], _CMP_GT_OQ);
        auto mask1 = _mm256_cmp_pd(v_[1], other.v_[1], _CMP_GT_OQ);
        rc.v_[0] = _mm256_and_pd(mask0, one.v_[0]);
        rc.v_[1] = _mm256_and_pd(mask1, one.v_[1]);
        return rc;
    }
    friend inline SimdDouble8 fmaf(SimdDouble8 const&, SimdDouble8 const&, SimdDouble8 const&);

    friend class SimdFloat8;

    friend inline SimdDouble8 round(SimdDouble8 const& a) {
        SimdDouble8 b;
        b.v_[0] = _mm256_round_pd(a.v_[0], _MM_FROUND_TO_NEAREST_INT);
        b.v_[1] = _mm256_round_pd(a.v_[1], _MM_FROUND_TO_NEAREST_INT);
        return b;
    }
    friend inline SimdDouble8 sqrt(SimdDouble8 const& a) {
        SimdDouble8 b;
        b.v_[0] = _mm256_sqrt_pd(a.v_[0]);
        b.v_[1] = _mm256_sqrt_pd(a.v_[1]);
        return b;
    }
};

inline SimdDouble8 fmaf(SimdDouble8 const& a, SimdDouble8 const& b, SimdDouble8 const& c) {
    SimdDouble8 d;
    d.v_[0] = _mm256_fmadd_pd(a.v_[0], b.v_[0], c.v_[0]);
    d.v_[1] = _mm256_fmadd_pd(a.v_[1], b.v_[1], c.v_[1]);
    return d;
}

#define SIMD_FLOAT8_SIZE 8

class SimdInt8;

class alignas(sizeof(__m256)) SimdFloat8 {
    __m256 v_;

public:
    SimdFloat8() = default;
    SimdFloat8(SimdInt8 a);
    inline SimdFloat8(
        float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
        v_ = _mm256_set_ps(a8, a7, a6, a5, a4, a3, a2, a1);
    }
    inline SimdFloat8(float a) {
        v_ = _mm256_broadcast_ss(&a);
    }
    inline SimdFloat8(SimdDouble8 a) {
        __m128& v0 = (__m128&) v_;
        __m128& v1 = *((__m128*) ((float*) (&v_) + 4));
        v0 = _mm256_cvtpd_ps(a.v_[0]);
        v1 = _mm256_cvtpd_ps(a.v_[1]);
    }
    inline float& operator[](int i) {
        return v_[i];
    }
    inline float operator[](int i) const {
        return v_[i];
    }
    inline SimdFloat8 operator+(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_ = _mm256_add_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat8 operator-(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_ = _mm256_sub_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat8 operator+() const {
        return *this;
    }
    inline SimdFloat8 operator-() const {
        return SimdFloat8(0.0f) - *this;
    }
    inline SimdFloat8 operator*(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_ = _mm256_mul_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat8 operator/(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_ = _mm256_div_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat8 operator+=(SimdFloat8 const& other) {
        v_ = _mm256_add_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat8 operator-=(SimdFloat8 const& other) {
        v_ = _mm256_sub_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat8 operator*=(SimdFloat8 const& other) {
        v_ = _mm256_mul_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat8 operator/=(SimdFloat8 const& other) {
        v_ = _mm256_div_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat8 operator*(float d) const {
        SimdFloat8 const other = d;
        return other * *this;
    }
    inline SimdFloat8 operator/(float d) const {
        SimdFloat8 const other = 1.0 / d;
        return *this * other;
    }
    inline float sum() const {
        __m128 vsum;
        __m128& a = *(((__m128*) &v_) + 0);
        __m128& b = *(((__m128*) &v_) + 1);
        vsum = _mm_add_ps(a, b);
        return vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }
    inline SimdFloat8 operator<=(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm256_cmp_ps(v_, other.v_, _CMP_LE_OQ);
        rc.v_ = _mm256_and_ps(mask0, one.v_);
        return rc;
    }
    inline SimdFloat8 operator<(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm256_cmp_ps(v_, other.v_, _CMP_LT_OQ);
        rc.v_ = _mm256_and_ps(mask0, one.v_);
        return rc;
    }
    inline SimdFloat8 operator>(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm256_cmp_ps(v_, other.v_, _CMP_GT_OQ);
        rc.v_ = _mm256_and_ps(mask0, one.v_);
        return rc;
    }
    inline SimdFloat8 operator>=(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        static SimdFloat8 const one(1);
        auto mask0 = _mm256_cmp_ps(v_, other.v_, _CMP_GE_OQ);
        rc.v_ = _mm256_and_ps(mask0, one.v_);
        return rc;
    }
    inline float max() const {
        float mx = v_[0];
        for (int i = 1; i < SIMD_FLOAT8_SIZE; i++) {
            mx = std::max(mx, v_[i]);
        }
        return mx;
    }
    friend inline SimdFloat8 sqrt(SimdFloat8 const&);
    friend inline SimdFloat8 rsqrt(SimdFloat8 const&);
    friend inline SimdFloat8 max(SimdFloat8 const&, SimdFloat8 const&);
    friend inline SimdFloat8 min(SimdFloat8 const&, SimdFloat8 const&);
    friend inline SimdFloat8 round(SimdFloat8 const&);
    friend inline SimdFloat8 fmaf(SimdFloat8 const&, SimdFloat8 const&, SimdFloat8 const&);
    friend class SimdInt8;
    friend SimdFloat8 log2(SimdFloat8 const& a);
};

inline SimdFloat8 abs(SimdFloat8 const& a) {
    return max(a, -a);
}

inline SimdFloat8 log2(SimdFloat8 const& a) {
    __m256i i = _mm256_castps_si256(a.v_);
    __m256i j;
    i = _mm256_srli_epi32(i, 23);
    j = _mm256_slli_epi32(i, 23);
    i = _mm256_sub_epi32(i, _mm256_set1_epi32(0x7f));
    SimdFloat8 ret, x;
    x.v_ = _mm256_castsi256_ps(j);
    ret.v_ = _mm256_cvtepi32_ps(i);
    x = a / x;
    x = sqrt(sqrt(x));
    static float const c0 = 1.0 / log(2.0);
    static SimdFloat8 const c1(c0);
    static SimdFloat8 const c2(-c0 / 2);
    static SimdFloat8 const c3(c0 / 3);
    static SimdFloat8 const c4(-c0 / 4);
    static SimdFloat8 const c5(c0 / 5);
    static SimdFloat8 const c6(-c0 / 6);
    static SimdFloat8 const c7(c0 / 7);
    static SimdFloat8 const c8(-c0 / 8);
    static SimdFloat8 const c9(c0 / 9);
    x = x - SimdFloat8(1.0f);
    SimdFloat8 y = c9;
    y = fmaf(y, x, c8);
    y = fmaf(y, x, c7);
    y = fmaf(y, x, c6);
    y = fmaf(y, x, c5);
    y = fmaf(y, x, c4);
    y = fmaf(y, x, c3);
    y = fmaf(y, x, c2);
    y = fmaf(y, x, c1);
    y = y * x;
    ret += SimdFloat8(4) * y;
    return ret;
}

inline SimdFloat8 log(SimdFloat8 const& a) {
    static SimdFloat8 const b = 1.0 / log2(exp(1));
    return log2(a) * b;
}

inline SimdFloat8 sqrt(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_ = _mm256_sqrt_ps(a.v_);
    return b;
}

inline SimdFloat8 rsqrt(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_ = _mm256_rsqrt_ps(a.v_);
    return b;
}

inline SimdFloat8 max(SimdFloat8 const& a, SimdFloat8 const& b) {
    SimdFloat8 c;
    c.v_ = _mm256_max_ps(a.v_, b.v_);
    return c;
}

inline SimdFloat8 min(SimdFloat8 const& a, SimdFloat8 const& b) {
    SimdFloat8 c;
    c.v_ = _mm256_min_ps(a.v_, b.v_);
    return c;
}

inline SimdFloat8 round(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_ = _mm256_round_ps(a.v_, _MM_FROUND_TO_NEAREST_INT);
    return b;
}

inline SimdFloat8 fmaf(SimdFloat8 const& a, SimdFloat8 const& b, SimdFloat8 const& c) {
    SimdFloat8 d;
    d.v_ = _mm256_fmadd_ps(a.v_, b.v_, c.v_);
    return d;
}

#endif

#ifdef USE_AVX512

#define SIMD_FLOAT16_SIZE 16

class SimdInt16;

class alignas(sizeof(__m512)) SimdFloat16 {
    __m512 v_;

public:
    SimdFloat16() = default;
    SimdFloat16(SimdInt16 a);
    SimdFloat16(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8,
        float a9, float a10, float a11, float a12, float a13, float a14, float a15, float a16) {
        v_ = _mm512_set_ps(a16, a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1);
    }
    SimdFloat16(float a) {
        v_ = _mm512_set_ps(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a);
    }
    float& operator[](int i) {
        return v_[i];
    }
    float operator[](int i) const {
        return v_[i];
    }
    inline SimdFloat16 operator+(SimdFloat16 const& other) const {
        SimdFloat16 a;
        a.v_ = _mm512_add_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat16 operator-(SimdFloat16 const& other) const {
        SimdFloat16 a;
        a.v_ = _mm512_sub_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat16 operator+() const {
        return *this;
    }
    inline SimdFloat16 operator-() const {
        return SimdFloat16(0.0f) - *this;
    }
    inline SimdFloat16 operator*(SimdFloat16 const& other) const {
        SimdFloat16 a;
        a.v_ = _mm512_mul_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat16 operator/(SimdFloat16 const& other) const {
        SimdFloat16 a;
        a.v_ = _mm512_div_ps(v_, other.v_);
        return a;
    }
    inline SimdFloat16 operator+=(SimdFloat16 const& other) {
        v_ = _mm512_add_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat16 operator-=(SimdFloat16 const& other) {
        v_ = _mm512_sub_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat16 operator*=(SimdFloat16 const& other) {
        v_ = _mm512_mul_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat16 operator/=(SimdFloat16 const& other) {
        v_ = _mm512_div_ps(v_, other.v_);
        return *this;
    }
    inline SimdFloat16 operator*(float d) const {
        SimdFloat16 const other = d;
        return other * *this;
    }
    inline SimdFloat16 operator/(float d) const {
        SimdFloat16 const other = 1.0 / d;
        return *this * other;
    }
    float sum() const {
        __m256 vsum1;
        __m128 vsum2;
        __m256& a1 = *(((__m256*) &v_) + 0);
        __m256& b1 = *(((__m256*) &v_) + 1);
        vsum1 = _mm256_add_ps(a1, b1);
        __m128& a2 = *(((__m128*) &vsum1) + 0);
        __m128& b2 = *(((__m128*) &vsum1) + 1);
        vsum2 = _mm_add_ps(a2, b2);
        return vsum2[0] + vsum2[1] + vsum2[2] + vsum2[3];
    }
    SimdFloat16 operator<=(SimdFloat16 const& other) const {
        SimdFloat16 rc;
        static SimdFloat16 const one(1);
        static SimdFloat16 const zero(0);
        auto mask0 = _mm512_cmp_ps_mask(v_, other.v_, _CMP_LE_OQ);
        rc.v_ = _mm512_mask_mov_ps(zero.v_, mask0, one.v_);
        return rc;
    }
    SimdFloat16 operator<(SimdFloat16 const& other) const {
        SimdFloat16 rc;
        static SimdFloat16 const one(1);
        static SimdFloat16 const zero(0);
        auto mask0 = _mm512_cmp_ps_mask(v_, other.v_, _CMP_LT_OQ);
        rc.v_ = _mm512_mask_mov_ps(zero.v_, mask0, one.v_);
        return rc;
    }
    SimdFloat16 operator>(SimdFloat16 const& other) const {
        SimdFloat16 rc;
        static SimdFloat16 const one(1);
        static SimdFloat16 const zero(0);
        auto mask0 = _mm512_cmp_ps_mask(v_, other.v_, _CMP_GT_OQ);
        rc.v_ = _mm512_mask_mov_ps(zero.v_, mask0, one.v_);
        return rc;
    }
    SimdFloat16 operator>=(SimdFloat16 const& other) const {
        SimdFloat16 rc;
        static SimdFloat16 const one(1);
        static SimdFloat16 const zero(0);
        auto mask0 = _mm512_cmp_ps_mask(v_, other.v_, _CMP_GE_OQ);
        rc.v_ = _mm512_mask_mov_ps(zero.v_, mask0, one.v_);
        return rc;
    }
    friend inline SimdFloat16 sqrt(SimdFloat16 const&);
    friend inline SimdFloat16 rsqrt(SimdFloat16 const&);
    friend inline SimdFloat16 max(SimdFloat16 const&, SimdFloat16 const&);
    friend inline SimdFloat16 min(SimdFloat16 const&, SimdFloat16 const&);
    friend inline SimdFloat16 round(SimdFloat16 const&);
    friend inline SimdFloat16 fmaf(SimdFloat16 const&, SimdFloat16 const&, SimdFloat16 const&);
    friend class SimdInt16;
};

inline SimdFloat16 sqrt(SimdFloat16 const& a) {
    SimdFloat16 b;
    b.v_ = _mm512_sqrt_ps(a.v_);
    return b;
}

inline SimdFloat16 rsqrt(SimdFloat16 const& a) {
    SimdFloat16 b;
    b.v_ = _mm512_rsqrt14_ps(a.v_);
    return b;
}

inline SimdFloat16 max(SimdFloat16 const& a, SimdFloat16 const& b) {
    SimdFloat16 c;
    c.v_ = _mm512_max_ps(a.v_, b.v_);
    return c;
}

inline SimdFloat16 min(SimdFloat16 const& a, SimdFloat16 const& b) {
    SimdFloat16 c;
    c.v_ = _mm512_min_ps(a.v_, b.v_);
    return c;
}

inline SimdFloat16 round(SimdFloat16 const& a) {
    SimdFloat16 b;
    b.v_ = _mm512_roundscale_ps(a.v_, _MM_FROUND_TO_NEAREST_INT);
    return b;
}

inline SimdFloat16 fmaf(SimdFloat16 const& a, SimdFloat16 const& b, SimdFloat16 const& c) {
    SimdFloat16 d;
    d.v_ = _mm512_fmadd_ps(a.v_, b.v_, c.v_);
    return d;
}

#endif

inline SimdFloat8 fmaxf(SimdFloat8 a, SimdFloat8 b) {
    return max(a, b);
}
#endif /* SIMD_FLOAT_HPP_ */
