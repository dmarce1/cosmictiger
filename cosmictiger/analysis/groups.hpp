/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
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
    Array<Position, kDimensions> x;
    Array<float, kDimensions> v;
    GroupInteger lastGroup;
    GroupInteger identity = kNoParticleGroup;
    GroupInteger lastHost = kNoParticleGroup;
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
