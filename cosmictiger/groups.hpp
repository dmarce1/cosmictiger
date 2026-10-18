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

#ifndef GROUPS_HPP_
#define GROUPS_HPP_

#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/safe_io.hpp>

struct ParticleData {
    Array<Fixed32, NDIM> x;
    Array<float, NDIM> v;
    GroupInteger lastGroup;
    GroupInteger identity = NO_GROUP;
    GroupInteger lastHost = NO_GROUP;
    ParticleInteger index;
    int rank;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & x;
        arc & v;
        arc & lastGroup;
        arc & identity;
        arc & lastHost;
        arc & index;
        arc & rank;
    }
};

void groupsAddParticles(int wave, double scale);
void groupsPrepareHistory(int number);
void groupsReduce(double);
std::pair<size_t, size_t> groupsSave(int number, double scale, double time);
void groupsCull();

#endif /* GROUPS_HPP_ */
