/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef MATH_HPP_
#define MATH_HPP_

#include <atomic>
#include <cmath>
#include <cstdlib>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>

/// Return the square of the supplied value.
template <class T>
CUDA_EXPORT inline T sqr(T const& a) {
    return a * a;
}

/// Return the sum of squares of three components.
template <class T>
CUDA_EXPORT inline T sqr(T const& a, T const& b, T const& c) {
    return a * a + b * b + c * c;
}

/// Map the C random generator to a uniform deviate strictly between zero and one.
inline double rand1() {
    return (double(rand()) + 0.5) / (double(RAND_MAX) + 1.0);
}

/// Round a positive integer up to a multiple of b; return zero for nonpositive input.
template <class T>
CUDA_EXPORT inline T roundUp(T a, T b) {
    if (a > 0) {
        return (((a - 1) / b) + 1) * b;
    } else {
        return 0;
    }
}

/// Evaluate sin(x)/x with the exact limiting value one at zero.
CUDA_EXPORT
inline double sinc(double x) {
    if (x == 0.0) {
        return 1.0;
    } else {
        return sin(x) / x;
    }
}

/// Add a float to an atomic value with a compare-and-exchange retry loop.
inline void atomicAdd(std::atomic<float>& y, float x) {
    float z = y;
    while (!y.compare_exchange_strong(z, z + x)) {
    }
}

#endif /* MATH_HPP_ */
