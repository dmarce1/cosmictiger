/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

void domainsBegin(int rung);
void domainsEnd();
void domainsRebound();
Range<double> domainsFindMyBox();
Range<double> domainsRange(size_t key);
void domainsSave(FILE* fp);
void domainsLoad(FILE* fp);
double domainsGetLoadImbalance();
Pair<int, double> domainsFindEwaldLevel(double theta, size_t key = 1,
    Range<double> box = unitBox<double>(), int depth = 0, double rb = -1.0);

struct DomainT {
    Range<double> box;
    double midx;
    int rank;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & box;
        arc & midx;
        arc & rank;
    }
};

struct DomainGlobal {
    Range<double> box;
    Pair<int> processRange;
    size_t totalCount;
    size_t loCount;
    double midhi;
    double midlo;
    size_t key;
};

struct DomainLocal {
    Range<double> box;
    Pair<int> processRange;
    Pair<ParticleInteger> particleRange;
    int depth;
};

#endif /* DOMAIN_HPP_ */
