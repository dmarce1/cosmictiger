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

#ifndef COMPLEX_HPP_
#define COMPLEX_HPP_

#include <algorithm>
#include <math.h>

template <class T = float>
class Complex {
    T x_, y_;

public:
    Complex() = default;

    Complex(T a) {
        x_ = a;
        y_ = T(0.0);
    }

    Complex(T a, T b) {
        x_ = a;
        y_ = b;
    }

    Complex& operator+=(Complex other) {
        x_ += other.x_;
        y_ += other.y_;
        return *this;
    }

    Complex& operator-=(Complex other) {
        x_ -= other.x_;
        y_ -= other.y_;
        return *this;
    }

    Complex operator*(Complex other) const {
        Complex a;
        a.x_ = x_ * other.x_ - y_ * other.y_;
        a.y_ = x_ * other.y_ + y_ * other.x_;
        return a;
    }

    Complex operator/(Complex other) const {
        return *this * other.conj() / other.norm();
    }

    Complex operator/(T other) const {
        Complex b;
        b.x_ = x_ / other;
        b.y_ = y_ / other;
        return b;
    }

    Complex operator*(T other) const {
        Complex b;
        b.x_ = x_ * other;
        b.y_ = y_ * other;
        return b;
    }

    Complex& operator*=(T other) {
        x_ *= other;
        y_ *= other;
        return *this;
    }

    Complex operator+(Complex other) const {
        Complex a;
        a.x_ = x_ + other.x_;
        a.y_ = y_ + other.y_;
        return a;
    }

    Complex operator-(Complex other) const {
        Complex a;
        a.x_ = x_ - other.x_;
        a.y_ = y_ - other.y_;
        return a;
    }

    Complex conj() const {
        Complex a;
        a.x_ = x_;
        a.y_ = -y_;
        return a;
    }

    T real() const {
        return x_;
    }

    T imag() const {
        return y_;
    }

    T& real() {
        return x_;
    }

    T& imag() {
        return y_;
    }

    T norm() const {
        return ((*this) * conj()).real();
    }

    T abs() const {
        return sqrtf(norm());
    }

    Complex operator-() const {
        Complex a;
        a.x_ = -x_;
        a.y_ = -y_;
        return a;
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & x_;
        arc & y_;
    }
};

template <class T>
inline void swap(Complex<T>& a, Complex<T>& b) {
    std::swap(a.real(), b.real());
    std::swap(a.imag(), b.imag());
}

using Cmplx = Complex<float>;

inline Cmplx expc(Cmplx z) {
    float x, y;
    float t = std::exp(z.real());
    sincosf(z.imag(), &y, &x);
    x *= t;
    y *= t;
    return Cmplx(x, y);
}
#endif /* COMPLEX_HPP_ */
