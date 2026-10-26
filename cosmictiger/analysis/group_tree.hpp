/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#pragma once

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

using GroupRange = Range<double>;

struct GroupTreeNode {
    GroupRange box;
    Array<TreeId, kChildCount> children;
    Pair<ParticleInteger> particleRange;
    Pair<int> processRange;
    bool localRoot;
    bool active;
    bool lastActive;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & active;
        arc & lastActive;
        arc & box;
        arc & children;
        arc & particleRange;
        arc & processRange;
        arc & localRoot;
    }
    inline ParticleGlobalRange globalParticleRange() const {
        ParticleGlobalRange r;
        r.proc = processRange.first;
        r.range = particleRange;
        return r;
    }
};

struct GroupTreeReturn {
    TreeId id;
    GroupRange box;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & id;
        arc & box;
    }
};

GroupTreeReturn groupTreeCreate(size_t key = 1, Pair<int, int> processRange = {0, hpxSize()},
    Pair<ParticleInteger> particleRange = {-1, -1}, GroupRange box = unitBox<double>(),
    int depth = 0, bool localRoot = hpxSize() == 1);
void groupTreeDestroy();
GroupTreeNode const* groupTreeGetNode(TreeId id);
void groupTreeSetActive(TreeId, bool);
void groupTreeIncCacheEpoch();
