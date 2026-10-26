/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef MATH_HPP_
#define MATH_HPP_

#include <atomic>
#include <numbers>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/simd.hpp>

template <class T>
CUDA_EXPORT inline T sqr(T const& a) {
    return a * a;
}

template <class T>
CUDA_EXPORT inline T sqr(T const& a, T const& b, T const& c) {
    return a * a + b * b + c * c;
}

inline float rand1() {
    return ((float) rand() + 0.5) / (float) RAND_MAX;
}

inline float randNormal() {
    double const x = rand1();
    double const y = rand1();
    return sqrt(-2.0 * log(x)) * cos(2 * std::numbers::pi * y);
}

template <class T>
CUDA_EXPORT int constrainRange(T& x) {
    int flops = 2;
    if (x > T(1)) {
        x -= T(1);
        flops++;
    }
    if (x < T(0)) {
        x += T(1);
        flops++;
    }
    if (x > T(1)) {
        PRINT("Print particle out of range %e\n", x);
        ALWAYS_ASSERT(false);
    }
    if (x < T(0)) {
        PRINT("Print particle out of range %e\n", x);
        ALWAYS_ASSERT(false);
    }
    return flops;
}

#ifdef USE_CUDA
__device__ inline void erfcexp(float x, float* ec, float* ex) {    // 18 + FLOP_DIV + FLOP_EXP
    float const p(0.3275911f);
    float const a1(0.254829592f);
    float const a2(-0.284496736f);
    float const a3(1.421413741f);
    float const a4(-1.453152027f);
    float const a5(1.061405429f);
    float const t1 = 1.f / fma(p, x, 1.f);                                          // FLOP_DIV + 2
    float const t2 = t1 * t1;                                                       // 1
    float const t3 = t2 * t1;                                                       // 1
    float const t4 = t2 * t2;                                                       // 1
    float const t5 = t2 * t3;                                                       // 1
    *ex = expf(-x * x);                                                             // 2 + FLOP_EXP
    *ec = fmaf(a1, t1, fmaf(a2, t2, fmaf(a3, t3, fmaf(a4, t4, a5 * t5)))) * *ex;    // 10
}
#endif

inline void erfcexp(double x, double* ec, double* ex) {    // 18 + FLOP_DIV + FLOP_EXP
    *ex = exp(-x * x);                                     // 2 + FLOP_EXP
    *ec = erfc(x);
}

namespace math {
inline double max(double a, double b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
}    // namespace math

template <class T>
CUDA_EXPORT inline T roundUp(T a, T b) {
    if (a > 0) {
        return (((a - 1) / b) + 1) * b;
    } else {
        return 0;
    }
}

template <class T>
CUDA_EXPORT inline T roundDown(T a, T b) {
    return (a / b) * b;
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

CUDA_EXPORT
inline float tsc(float x) {
    float const absx = fabsf(x);
    if (absx < 0.5) {
        return 0.75 - sqr(x);
    } else if (absx < 1.5) {
        return 0.5 * sqr(1.5 - absx);
    } else {
        return 0.0;
    }
}

template <class T>
CUDA_EXPORT inline T erfnearzero(T x) {
    double const c[15] = {1.1283791670955126, -0.37612638903183752, 0.11283791670955126,
        -0.026866170645131252, 0.0052239776254421878, -0.00085483270234508528,
        0.00012055332981789664, -0.000014925650358406251, 1.6462114365889247e-6,
        -1.6365844691234924e-7, 1.4807192815879217e-8, -1.2290555301717927e-9,
        9.422759064650411e-11, -6.7113668551641104e-12, 4.4632242632864773e-13};
    T x2 = x * x;
    T w = simdBroadcast<T>(c[14]);
    for (int i = 13; i >= 0; i--) {
        w = simdFma(w, x2, simdBroadcast<T>(c[i]));
    }
    w *= x;
    return w;
}

template <class T>
CUDA_EXPORT T expnearzero(T x) {
    T c0 = simdBroadcast<T>(1.);
    T c1 = simdBroadcast<T>(1.);
    T c2 = simdBroadcast<T>(0.5);
    T c3 = simdBroadcast<T>(0.166666666666667);
    T c4 = simdBroadcast<T>(0.0416666666666667);
    T c5 = simdBroadcast<T>(0.00833333333333333);
    T c6 = simdBroadcast<T>(0.00138888888888889);
    T c7 = simdBroadcast<T>(0.000198412698412698);
    T c8 = simdBroadcast<T>(0.0000248015873015873);
    T c9 = simdBroadcast<T>(2.75573192239859e-6);
    T c10 = simdBroadcast<T>(2.75573192239859e-7);
    T c11 = simdBroadcast<T>(2.50521083854417e-8);
    T c12 = simdBroadcast<T>(2.08767569878681e-9);
    T c13 = simdBroadcast<T>(1.60590438368216e-10);
    T c14 = simdBroadcast<T>(1.1470745597729725e-11);
    T c15 = simdBroadcast<T>(7.647163731819816e-13);
    T c16 = simdBroadcast<T>(4.779477332387385e-14);
    T w = c16;
    w = simdFma(w, x, c15);
    w = simdFma(w, x, c14);
    w = simdFma(w, x, c13);
    w = simdFma(w, x, c12);
    w = simdFma(w, x, c11);
    w = simdFma(w, x, c10);
    w = simdFma(w, x, c9);
    w = simdFma(w, x, c8);
    w = simdFma(w, x, c7);
    w = simdFma(w, x, c6);
    w = simdFma(w, x, c5);
    w = simdFma(w, x, c4);
    w = simdFma(w, x, c3);
    w = simdFma(w, x, c2);
    w = simdFma(w, x, c1);
    w = simdFma(w, x, c0);
    return w;
}

inline double doubleFactorial(int n) {
    if (n < 1) {
        return 1;
    } else {
        return n * doubleFactorial(n - 2);
    }
}

#include <cosmictiger/core/containers.hpp>

Pair<double, Array<double, kDimensions>> findEigenpair(
    Array<Array<double, kDimensions>, kDimensions> const& aValue, double mu);
#endif /* MATH_HPP_ */
