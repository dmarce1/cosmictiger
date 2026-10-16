/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#ifndef KICK_WORKSPACE_HPP_
#define KICK_WORKSPACE_HPP_

#include <set>
#include <unordered_map>

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/semaphore.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/unordered_set_ts.hpp>

struct KickWorkspaceTreeIdHash {
    inline size_t operator()(TreeId id) const {
        return id.index * hpxSize() + id.proc;
    }
};

#ifndef __CUDACC__

class KickWorkspace {
    MutexType mutex_;
    Vector<KickWorkitem> workitems_;
    ParticleInteger totalParticles_;
    ParticleInteger numberProcessedParticles_;
    KickParameters params_;
    std::unordered_set<TreeId, KickWorkspaceTreeIdHash> treeIds_;

public:
    static void clearBuffers();
    KickWorkspace() = default;
    KickWorkspace(KickParameters, ParticleInteger);
    KickWorkspace(KickWorkspace const&) = delete;
    KickWorkspace(KickWorkspace&&) = delete;
    KickWorkspace& operator=(KickWorkspace const&) = delete;
    KickWorkspace& operator=(KickWorkspace&&) = delete;
    ~KickWorkspace();
    template <class A>
    void serialize(A&&, unsigned) {}
    void addWork(std::shared_ptr<KickWorkspace> ptr, Expansion<float> l, Array<Fixed32, NDIM> pos,
        TreeId self, Vector<TreeId>&& dchecklist, Vector<TreeId>&& echecklist);
    void addParts(std::shared_ptr<KickWorkspace> ptr, ParticleInteger n);
    void toGpu();
};

#endif
#endif /* KICK_WORKSPACE_HPP_ */
