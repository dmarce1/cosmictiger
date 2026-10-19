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

#include <functional>
#include <unordered_map>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

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

namespace detail {
// Read access to the finder's local tables for the catalogue tests
// (tests/groups_catalogue.cpp): the group entries of the current snapshot, the
// membership updates queued for the next cull, the halo hosts restored from the
// previous snapshot, and a visit over every candidate group's collected
// particles.
Vector<GroupEntry> const& groupsLocalEntries();
Vector<Pair<ParticleInteger, GroupInteger>> const& groupsPendingAssignments();
std::unordered_map<GroupInteger, GroupInteger> const& groupsPreviousHaloHosts();
void groupsResetHistory();
void groupsForEachCandidate(
    std::function<void(GroupInteger, Vector<ParticleData> const&)> const& visit);
}    // namespace detail

#endif /* GROUPS_HPP_ */
