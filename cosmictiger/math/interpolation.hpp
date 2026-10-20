/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef COSMICTIGER_INTERPOLATION_HPP_
#define COSMICTIGER_INTERPOLATION_HPP_
#include <cmath>
#include <stdio.h>
#include <vector>

template <class T>
struct InterpolationFunctor {
    std::vector<T> values;
    T amin;
    T amax;
    T minloga;
    T maxloga;
    int n;
    T dloga;
    T operator()(T a) const {
        T loga = log(a);
        if (loga < minloga || loga > maxloga) {
            printf("Error in interpolation_function out of range %e %e %e\n", a, amin, amax);
        }
        int i1 = int((loga - minloga) / (dloga));
        if (i1 > n - 3) {
            i1 = n - 3;
        } else if (i1 < 1) {
            i1 = 1;
        }
        int i0 = i1 - 1;
        int i2 = i1 + 1;
        int i3 = i2 + 1;
        T const c0 = values[i1];
        T const c1 =
            -values[i0] / (T) 3.0 - (T) 0.5 * values[i1] + values[i2] - values[i3] / (T) 6.0;
        T const c2 = (T) 0.5 * values[i0] - values[i1] + (T) 0.5 * values[i2];
        T const c3 = -values[i0] / (T) 6.0 + (T) 0.5 * values[i1] - (T) 0.5 * values[i2] +
            values[i3] / (T) 6.0;
        T x = (loga - i1 * dloga - minloga) / dloga;
        return c0 + c1 * x + c2 * x * x + c3 * x * x * x;
    }
};

template <class T>
inline void buildInterpolationFunction(
    InterpolationFunctor<T>* f, std::vector<T> const& values, T amin, T amax) {
    T minloga = log(amin);
    T maxloga = log(amax);
    int n = values.size();
    T dloga = (maxloga - minloga) / (n - 1);
    InterpolationFunctor<T> functor;
    functor.values = std::move(values);
    functor.maxloga = maxloga;
    functor.minloga = minloga;
    functor.dloga = dloga;
    functor.amin = amin;
    functor.amax = amax;
    functor.n = n;
    *f = functor;
}

#endif /* COSMICTIGER_INTERPOLATION_HPP_ */
