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

#ifndef SIMD_ALTIVEC_HPP_
#define SIMD_ALTIVEC_HPP_

#include <stddef.h>

#ifdef __VEC__

#define SIMD_DOUBLE_SIZE 2
#define SIMD_FLOAT4_SIZE 4
#define SIMD_FLOAT8_SIZE 8
#define SIMD_INT4_SIZE 4
#define SIMD_INT8_SIZE 8

#include <altivec.h>

class SimdInt4;
class SimdInt8;

class alignas(sizeof(vector float)) SimdFloat4 {
    vector float v_;

public:
    SimdFloat4() = default;
    SimdFloat4(float a1, float a2, float a3, float a4) {
        v_[0] = a1;
        v_[1] = a2;
        v_[2] = a3;
        v_[3] = a4;
    }
    SimdFloat4(SimdInt4 a);
    SimdFloat4(float a) {
        v_ = vec_load_splats(0, &a);
    }
    float& operator[](int i) {
        return v_[i];
    }
    float operator[](int i) const {
        return v_[i];
    }
    inline SimdFloat4 operator+(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = vec_add(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator-(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = vec_sub(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator+() const {
        return *this;
    }
    inline SimdFloat4 operator-() const {
        SimdFloat4 a;
        a.v_ = vec_neg(v_);
        return a;
    }
    inline SimdFloat4 operator*(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = vec_mul(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator/(SimdFloat4 const& other) const {
        SimdFloat4 a;
        a.v_ = vec_div(v_, other.v_);
        return a;
    }
    inline SimdFloat4 operator+=(SimdFloat4 const& other) {
        v_ = vec_add(v_, other.v_);
        return *this;
    }
    inline SimdFloat4 operator-=(SimdFloat4 const& other) {
        v_ = vec_sub(v_, other.v_);
        return *this;
    }
    inline SimdFloat4 operator*=(SimdFloat4 const& other) {
        v_ = vec_mul(v_, other.v_);
        return *this;
    }
    inline SimdFloat4 operator/=(SimdFloat4 const& other) {
        v_ = vec_div(v_, other.v_);
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
        rc.v_ = vec_ctf(vec_cmple(v_, other.v_), 0);
        return -rc;
    }
    SimdFloat4 operator<(SimdFloat4 const& other) const {
        SimdFloat4 rc;
        rc.v_ = vec_ctf(vec_cmplt(v_, other.v_), 0);
        return -rc;
    }
    SimdFloat4 operator>(SimdFloat4 const& other) const {
        SimdFloat4 rc;
        rc.v_ = vec_ctf(vec_cmpgt(v_, other.v_), 0);
        return -rc;
    }
    SimdFloat4 operator>=(SimdFloat4 const& other) const {
        SimdFloat4 rc;
        rc.v_ = vec_ctf(vec_cmpge(v_, other.v_), 0);
        return -rc;
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
    b.v_ = vec_sqrt(a.v_);
    return b;
}

inline SimdFloat4 rsqrt(SimdFloat4 const& a) {
    SimdFloat4 b;
    b.v_ = vec_rsqrt(a.v_);
    return b;
}

inline SimdFloat4 max(SimdFloat4 const& a, SimdFloat4 const& b) {
    SimdFloat4 c;
    c.v_ = vec_max(a.v_, b.v_);
    return c;
}

inline SimdFloat4 min(SimdFloat4 const& a, SimdFloat4 const& b) {
    SimdFloat4 c;
    c.v_ = vec_min(a.v_, b.v_);
    return c;
}

inline SimdFloat4 round(SimdFloat4 const& a) {
    SimdFloat4 b;
    b.v_ = vec_round(a.v_);
    return b;
}

inline SimdFloat4 fmaf(SimdFloat4 const& a, SimdFloat4 const& b, SimdFloat4 const& c) {
    SimdFloat4 d;
    d.v_ = vec_madd(a.v_, b.v_, c.v_);
    return d;
}

class alignas(sizeof(vector double)) SimdDouble {
    vector double v_;

public:
    SimdDouble() = default;
    SimdDouble(double a1, double a2) {
        v_[0] = a1;
        v_[1] = a2;
    }
    SimdDouble(SimdInt4 a);
    SimdDouble(double a) {
        v_ = vec_load_splats(0, &a);
    }
    double& operator[](int i) {
        return v_[i];
    }
    double operator[](int i) const {
        return v_[i];
    }
    inline SimdDouble operator+(SimdDouble const& other) const {
        SimdDouble a;
        a.v_ = vec_add(v_, other.v_);
        return a;
    }
    inline SimdDouble operator-(SimdDouble const& other) const {
        SimdDouble a;
        a.v_ = vec_sub(v_, other.v_);
        return a;
    }
    inline SimdDouble operator+() const {
        return *this;
    }
    inline SimdDouble operator-() const {
        SimdDouble a;
        a.v_ = vec_neg(v_);
        return a;
    }
    inline SimdDouble operator*(SimdDouble const& other) const {
        SimdDouble a;
        a.v_ = vec_mul(v_, other.v_);
        return a;
    }
    inline SimdDouble operator/(SimdDouble const& other) const {
        SimdDouble a;
        a.v_ = vec_div(v_, other.v_);
        return a;
    }
    inline SimdDouble operator+=(SimdDouble const& other) {
        v_ = vec_add(v_, other.v_);
        return *this;
    }
    inline SimdDouble operator-=(SimdDouble const& other) {
        v_ = vec_sub(v_, other.v_);
        return *this;
    }
    inline SimdDouble operator*=(SimdDouble const& other) {
        v_ = vec_mul(v_, other.v_);
        return *this;
    }
    inline SimdDouble operator/=(SimdDouble const& other) {
        v_ = vec_div(v_, other.v_);
        return *this;
    }
    inline SimdDouble operator*(float d) const {
        SimdDouble const other = d;
        return other * *this;
    }
    inline SimdDouble operator/(float d) const {
        SimdDouble const other = 1.0 / d;
        return *this * other;
    }
    float sum() const {
        float s = (*this)[0];
        for (int i = 1; i < SIMD_DOUBLE_SIZE; i++) {
            s += (*this)[i];
        }
        return s;
    }
    SimdDouble operator<=(SimdDouble const& other) const {
        SimdDouble rc;
        rc.v_ = vec_ctf(vec_cmple(v_, other.v_), 0);
        return -rc;
    }
    SimdDouble operator<(SimdDouble const& other) const {
        SimdDouble rc;
        rc.v_ = vec_ctf(vec_cmplt(v_, other.v_), 0);
        return -rc;
    }
    SimdDouble operator>(SimdDouble const& other) const {
        SimdDouble rc;
        rc.v_ = vec_ctf(vec_cmpgt(v_, other.v_), 0);
        return -rc;
    }
    SimdDouble operator>=(SimdDouble const& other) const {
        SimdDouble rc;
        rc.v_ = vec_ctf(vec_cmpge(v_, other.v_), 0);
        return -rc;
    }
    friend inline SimdDouble sqrt(SimdDouble const&);
    friend inline SimdDouble rsqrt(SimdDouble const&);
    friend inline SimdDouble max(SimdDouble const&, SimdDouble const&);
    friend inline SimdDouble min(SimdDouble const&, SimdDouble const&);
    friend inline SimdDouble round(SimdDouble const&);
    friend inline SimdDouble fmaf(SimdDouble const&, SimdDouble const&, SimdDouble const&);
    friend class SimdInt4;
};

inline SimdDouble sqrt(SimdDouble const& a) {
    SimdDouble b;
    b.v_ = vec_sqrt(a.v_);
    return b;
}

inline SimdDouble rsqrt(SimdDouble const& a) {
    SimdDouble b;
    b.v_ = vec_rsqrt(a.v_);
    return b;
}

inline SimdDouble max(SimdDouble const& a, SimdDouble const& b) {
    SimdDouble c;
    c.v_ = vec_max(a.v_, b.v_);
    return c;
}

inline SimdDouble min(SimdDouble const& a, SimdDouble const& b) {
    SimdDouble c;
    c.v_ = vec_min(a.v_, b.v_);
    return c;
}

inline SimdDouble round(SimdDouble const& a) {
    SimdDouble b;
    b.v_ = vec_round(a.v_);
    return b;
}

inline SimdDouble fmaf(SimdDouble const& a, SimdDouble const& b, SimdDouble const& c) {
    SimdDouble d;
    d.v_ = vec_madd(a.v_, b.v_, c.v_);
    return d;
}

class alignas(sizeof(vector float)) SimdFloat8 {
    vector float v_[2];

public:
    SimdFloat8() = default;
    SimdFloat8(float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
        v_[0][0] = a1;
        v_[0][1] = a2;
        v_[0][2] = a3;
        v_[0][3] = a4;
        v_[1][0] = a5;
        v_[1][1] = a6;
        v_[1][2] = a7;
        v_[1][3] = a8;
    }
    SimdFloat8(SimdInt4 a);
    SimdFloat8(float a) {
        v_[0] = vec_load_splats(0, &a);
        v_[1] = vec_load_splats(0, &a);
    }
    float& operator[](int i) {
        return v_[i / 4][i % 4];
    }
    float operator[](int i) const {
        return v_[i / 4][i % 4];
    }
    inline SimdFloat8 operator+(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = vec_add(v_[0], other.v_[0]);
        a.v_[1] = vec_add(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator-(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = vec_sub(v_[0], other.v_[0]);
        a.v_[1] = vec_sub(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator+() const {
        return *this;
    }
    inline SimdFloat8 operator-() const {
        SimdFloat8 a;
        a.v_[0] = vec_neg(v_[0]);
        a.v_[1] = vec_neg(v_[1]);
        return a;
    }
    inline SimdFloat8 operator*(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = vec_mul(v_[0], other.v_[0]);
        a.v_[1] = vec_mul(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator/(SimdFloat8 const& other) const {
        SimdFloat8 a;
        a.v_[0] = vec_div(v_[0], other.v_[0]);
        a.v_[1] = vec_div(v_[1], other.v_[1]);
        return a;
    }
    inline SimdFloat8 operator+=(SimdFloat8 const& other) {
        v_[0] = vec_add(v_[0], other.v_[0]);
        v_[1] = vec_add(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdFloat8 operator-=(SimdFloat8 const& other) {
        v_[0] = vec_sub(v_[0], other.v_[0]);
        v_[1] = vec_sub(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdFloat8 operator*=(SimdFloat8 const& other) {
        v_[0] = vec_mul(v_[0], other.v_[0]);
        v_[1] = vec_mul(v_[1], other.v_[1]);
        return *this;
    }
    inline SimdFloat8 operator/=(SimdFloat8 const& other) {
        v_[0] = vec_div(v_[0], other.v_[0]);
        v_[1] = vec_div(v_[1], other.v_[1]);
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
        float s = (*this)[0];
        for (int i = 1; i < SIMD_FLOAT8_SIZE; i++) {
            s += (*this)[i];
        }
        return s;
    }
    SimdFloat8 operator<=(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        rc.v_[0] = vec_ctf(vec_cmple(v_[0], other.v_[0]), 0);
        rc.v_[1] = vec_ctf(vec_cmple(v_[1], other.v_[1]), 0);
        return -rc;
    }
    SimdFloat8 operator<(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        rc.v_[0] = vec_ctf(vec_cmplt(v_[0], other.v_[0]), 0);
        rc.v_[1] = vec_ctf(vec_cmplt(v_[1], other.v_[1]), 0);
        return -rc;
    }
    SimdFloat8 operator>(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        rc.v_[0] = vec_ctf(vec_cmpgt(v_[0], other.v_[0]), 0);
        rc.v_[1] = vec_ctf(vec_cmpgt(v_[1], other.v_[1]), 0);
        return -rc;
    }
    SimdFloat8 operator>=(SimdFloat8 const& other) const {
        SimdFloat8 rc;
        rc.v_[0] = vec_ctf(vec_cmpge(v_[0], other.v_[0]), 0);
        rc.v_[1] = vec_ctf(vec_cmpge(v_[1], other.v_[1]), 0);
        return -rc;
    }
    friend inline SimdFloat8 sqrt(SimdFloat8 const&);
    friend inline SimdFloat8 rsqrt(SimdFloat8 const&);
    friend inline SimdFloat8 max(SimdFloat8 const&, SimdFloat8 const&);
    friend inline SimdFloat8 min(SimdFloat8 const&, SimdFloat8 const&);
    friend inline SimdFloat8 round(SimdFloat8 const&);
    friend inline SimdFloat8 fmaf(SimdFloat8 const&, SimdFloat8 const&, SimdFloat8 const&);
    friend class SimdInt4;
};

inline SimdFloat8 sqrt(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_[0] = vec_sqrt(a.v_[0]);
    b.v_[1] = vec_sqrt(a.v_[1]);
    return b;
}

inline SimdFloat8 rsqrt(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_[0] = vec_rsqrt(a.v_[0]);
    b.v_[1] = vec_rsqrt(a.v_[1]);
    return b;
}

inline SimdFloat8 max(SimdFloat8 const& a, SimdFloat8 const& b) {
    SimdFloat8 c;
    c.v_[0] = vec_max(a.v_[0], b.v_[0]);
    c.v_[1] = vec_max(a.v_[1], b.v_[1]);
    return c;
}

inline SimdFloat8 min(SimdFloat8 const& a, SimdFloat8 const& b) {
    SimdFloat8 c;
    c.v_[0] = vec_min(a.v_[0], b.v_[0]);
    c.v_[1] = vec_min(a.v_[1], b.v_[1]);
    return c;
}

inline SimdFloat8 round(SimdFloat8 const& a) {
    SimdFloat8 b;
    b.v_[0] = vec_round(a.v_[0]);
    b.v_[1] = vec_round(a.v_[1]);
    return b;
}

inline SimdFloat8 fmaf(SimdFloat8 const& a, SimdFloat8 const& b, SimdFloat8 const& c) {
    SimdFloat8 d;
    d.v_[0] = vec_madd(a.v_[0], b.v_[0], c.v_[0]);
    d.v_[1] = vec_madd(a.v_[1], b.v_[1], c.v_[1]);
    return d;
}

class alignas(sizeof(vector int)) SimdInt4 {
private:
    vector int v_;

public:
    SimdInt4() = default;
    SimdInt4(SimdInt4 const&) = default;
    SimdInt4(int i) {
        v_ = vec_splats(0, &i);
    }
    SimdInt4(SimdFloat4 r) {
        v_ = vec_cts(vec_floor(r.v_));
    }
    inline SimdInt4& operator=(SimdInt4 const& other) = default;
    SimdInt4& operator=(SimdInt4&& other) = default;
    inline SimdInt4 operator+(SimdInt4 const& other) const {
        SimdInt4 r;
        r.v_ = vec_add(v_, other.v_);
        return r;
    }
    inline SimdInt4 operator-(SimdInt4 const& other) const {
        SimdInt4 r;
        r.v_ = vec_sub(v_, other.v_);
        return r;
    }
    inline SimdInt4 operator*(SimdInt4 const& other) const {
        SimdInt4 r;
        r.v_ = vec_mul(v_, other.v_);
        return r;
    }
    inline SimdInt4 operator+() const {
        return *this;
    }
    inline SimdInt4 operator-() const {
        SimdInt4 a;
        a.v_ = vec_neg(v_);
        return a;
    }
    inline int& operator[](size_t i) {
        return v_[i];
    }
    inline int operator[](size_t i) const {
        return v_[i];
    }
    inline SimdInt4 operator<<=(int shft) {
        SimdInt4 s(shft);
        v_ = vec_sl(v_, s);
        return *this;
    }
    friend class SimdFloat4;
};

inline SimdFloat4::SimdFloat4(SimdInt4 a) {
    v_ = vec_ctf(a.v_);
}

class alignas(sizeof(vector int)) SimdInt8 {
private:
    vector int v_[2];

public:
    SimdInt8() = default;
    SimdInt8(SimdInt8 const&) = default;
    SimdInt8(int i) {
        v_[0] = vec_splats(0, &i);
        v_[1] = vec_splats(0, &i);
    }
    SimdInt8(SimdFloat4 r) {
        v_[0] = vec_cts(vec_floor(r.v_[0]));
        v_[1] = vec_cts(vec_floor(r.v_[1]));
    }
    inline SimdInt8& operator=(SimdInt8 const& other) = default;
    SimdInt8& operator=(SimdInt8&& other) = default;
    inline SimdInt8 operator+(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_[0] = vec_add(v_[0], other.v_[0]);
        r.v_[1] = vec_add(v_[1], other.v_[1]);
        return r;
    }
    inline SimdInt8 operator-(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_[0] = vec_sub(v_[0], other.v_[0]);
        r.v_[1] = vec_sub(v_[1], other.v_[1]);
        return r;
    }
    inline SimdInt8 operator*(SimdInt8 const& other) const {
        SimdInt8 r;
        r.v_[0] = vec_mul(v_[0], other.v_[0]);
        r.v_[1] = vec_mul(v_[1], other.v_[1]);
        return r;
    }
    inline SimdInt8 operator+() const {
        return *this;
    }
    inline SimdInt8 operator-() const {
        SimdInt8 a;
        a.v_[0] = vec_neg(v_[0]);
        a.v_[1] = vec_neg(v_[1]);
        return a;
    }
    inline int& operator[](size_t i) {
        return v_[i / 4][i % 4];
    }
    inline int operator[](size_t i) const {
        return v_[i / 4][i % 4];
    }
    inline SimdInt8 operator<<=(int shft) {
        SimdInt8 s(shft);
        v_[0] = vec_sl(v_[0], s[0]);
        v_[1] = vec_sl(v_[1], s[1]);
        return *this;
    }
    friend class SimdFloat8;
};

inline SimdFloat8::SimdFloat8(SimdInt8 a) {
    v_[0] = vec_ctf(a.v_[0]);
    v_[1] = vec_ctf(a.v_[1]);
}

using SimdInt = SimdInt4;
using SimdFloat = SimdFloat4;

#define SIMD_FLOAT_SIZE SIMD_FLOAT4_SIZE
#define SIMD_INT_SIZE SIMD_INT4_SIZE

#endif

#endif /* SIMD_ALTIVEC_HPP_ */
