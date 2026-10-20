/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
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

using ComplexFloat = Complex<float>;

inline ComplexFloat expc(ComplexFloat z) {
    float x, y;
    float t = std::exp(z.real());
    sincosf(z.imag(), &y, &x);
    x *= t;
    y *= t;
    return ComplexFloat(x, y);
}
#endif /* COMPLEX_HPP_ */
