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

template <class T>
CUDA_EXPORT inline T sqr(T const& a) {
    return a * a;
}

template <class T>
CUDA_EXPORT inline T sqr(T const& a, T const& b, T const& c) {
    return a * a + b * b + c * c;
}

inline double rand1() {
    return (double(rand()) + 0.5) / (double(RAND_MAX) + 1.0);
}

template <class T>
CUDA_EXPORT inline T roundUp(T a, T b) {
    if (a > 0) {
        return (((a - 1) / b) + 1) * b;
    } else {
        return 0;
    }
}

CUDA_EXPORT
inline double sinc(double x) {
    if (x == 0.0) {
        return 1.0;
    } else {
        return sin(x) / x;
    }
}

inline void atomicAdd(std::atomic<float>& y, float x) {
    float z = y;
    while (!y.compare_exchange_strong(z, z + x)) {
    }
}

#endif /* MATH_HPP_ */
