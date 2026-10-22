/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef KICK_WORKSPACE_HPP_
#define KICK_WORKSPACE_HPP_

#include <set>
#include <unordered_map>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/semaphore.hpp>
#include <cosmictiger/core/unordered_set_ts.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/tree.hpp>

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
    ParticleInteger numberOfProcessedParticles_;
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
    void addWork(std::shared_ptr<KickWorkspace> ptr, ExpansionBuffer const& l,
        Array<Position, NDIM> pos, TreeId self, Vector<TreeId>&& dchecklist,
        Vector<TreeId>&& echecklist);
    void recordProcessedParticles(std::shared_ptr<KickWorkspace> ptr, ParticleInteger n);
    void toGpu();
};

#endif
#endif /* KICK_WORKSPACE_HPP_ */
