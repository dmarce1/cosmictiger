/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef COMPLEX_HPP_
#define COMPLEX_HPP_

#include <cmath>
#include <complex>

using ComplexFloat = std::complex<float>;

inline ComplexFloat expc(ComplexFloat z) {
    float x, y;
    float t = std::exp(z.real());
    sincosf(z.imag(), &y, &x);
    x *= t;
    y *= t;
    return ComplexFloat(x, y);
}
#endif /* COMPLEX_HPP_ */
