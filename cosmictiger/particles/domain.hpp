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
