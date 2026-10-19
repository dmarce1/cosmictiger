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

#pragma once

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/particles/particles.hpp>

using GroupRange = Range<double>;

struct GroupTreeNode {
    GroupRange box;
    Array<TreeId, NCHILD> children;
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
