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
#include <unordered_set>

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

using KickGpuExecutor = KickReturn (*)(KickParameters, Vector<KickWorkitem>,
    std::unordered_set<TreeId, KickWorkspaceTreeIdHash> const&, detail::FmmForceOutput);

#ifndef __CUDACC__

class KickWorkspace {
    MutexType mutex_;
    Vector<KickWorkitem> workitems_;
    ParticleInteger totalParticles_;
    ParticleInteger numberOfProcessedParticles_;
    KickParameters params_;
    std::unordered_set<TreeId, KickWorkspaceTreeIdHash> treeIds_;
    bool batch_ = false;
    KickGpuExecutor gpuExecutor_ = nullptr;
    detail::FmmForceOutput output_{};
    KickReturn batchReturn_{};
    KickInteractionCounts interactionCounts_{};

public:
    KickWorkspace() = default;
    KickWorkspace(KickParameters, ParticleInteger, KickGpuExecutor);
    KickWorkspace(KickParameters, ParticleInteger, detail::FmmForceOutput, KickGpuExecutor);
    KickWorkspace(KickWorkspace const&) = delete;
    KickWorkspace(KickWorkspace&&) = delete;
    KickWorkspace& operator=(KickWorkspace const&) = delete;
    KickWorkspace& operator=(KickWorkspace&&) = delete;
    ~KickWorkspace();
    template <class A>
    void serialize(A&&, unsigned) {}
    void addWork(std::shared_ptr<KickWorkspace> ptr, ExpansionBuffer const& l,
        Array<Position, NDIM> pos, TreeId self, Vector<TreeId>&& dchecklist,
        Vector<TreeId>&& echecklist, FmmGeometry const& geometry, ParticleInteger outputOffset = 0);
    void recordProcessedParticles(std::shared_ptr<KickWorkspace> ptr, ParticleInteger n);
    KickReturn toGpu();
    bool isBatch() const {
        return batch_;
    }
    detail::FmmForceOutput output() const {
        return output_;
    }
    void accumulateBatchReturn(KickReturn const& result);
    void addInteractionCounts(KickInteractionCounts const& counts);
    KickInteractionCounts interactionCounts();
    KickReturn finishBatch();
};

#endif
#endif /* KICK_WORKSPACE_HPP_ */
