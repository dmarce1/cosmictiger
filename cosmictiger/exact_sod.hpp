//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXACT_SOD__C
#define EXACT_SOD__C

typedef struct {
    double rhol, rhor, pl, pr, gamma;
} SodInitT;

typedef struct {
    double rho, v, p;
} SodStateT;

void exactSod(SodStateT* out, SodInitT const* in, double x, double t, double dx);

// constexpr sod_init_t sod_init = { 1.0, 0.125, 1.0, 0.1, 7.0 / 5.0 };

#endif
