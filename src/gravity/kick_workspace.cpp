/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <mutex>
#include <set>

#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/particles/particles.hpp>

KickWorkspace::KickWorkspace(
    KickParameters p, ParticleInteger totalParticlesArgument, KickGpuExecutor executor) {
    gpuExecutor_ = executor;
    totalParticles_ = totalParticlesArgument;
    params_ = p;
    numberOfProcessedParticles_ = 0;
}

KickWorkspace::KickWorkspace(KickParameters p, ParticleInteger totalParticlesArgument,
    detail::FmmForceOutput output, KickGpuExecutor executor)
  : KickWorkspace(p, totalParticlesArgument, executor) {
    batch_ = true;
    output_ = output;
}

KickWorkspace::~KickWorkspace() {}

void KickWorkspace::accumulateBatchReturn(KickReturn const& result) {
    std::lock_guard<MutexType> lock(mutex_);
    batchReturn_ += result;
}

void KickWorkspace::addInteractionCounts(KickInteractionCounts const& counts) {
    std::lock_guard<MutexType> lock(mutex_);
    interactionCounts_.acceptanceTests += counts.acceptanceTests;
    interactionCounts_.cellCellDirect += counts.cellCellDirect;
    interactionCounts_.cellCellEwald += counts.cellCellEwald;
    interactionCounts_.cellParticle += counts.cellParticle;
    interactionCounts_.cellParticleAtLeafSinks += counts.cellParticleAtLeafSinks;
    interactionCounts_.particleCell += counts.particleCell;
    interactionCounts_.particleCellFromLeafSources += counts.particleCellFromLeafSources;
    interactionCounts_.particleParticleLeafPairs += counts.particleParticleLeafPairs;
    interactionCounts_.particleParticlePairs += counts.particleParticlePairs;
    interactionCounts_.sourceOpenings += counts.sourceOpenings;
    interactionCounts_.deferredSources += counts.deferredSources;
}

KickInteractionCounts KickWorkspace::interactionCounts() {
    std::lock_guard<MutexType> lock(mutex_);
    return interactionCounts_;
}

KickReturn KickWorkspace::finishBatch() {
    ALWAYS_ASSERT(batch_);
    {
        std::lock_guard<MutexType> lock(mutex_);
        ALWAYS_ASSERT(numberOfProcessedParticles_ == totalParticles_);
    }
    // The caller has joined every CPU producer; the work packet is immutable.
    if (!workitems_.empty()) {
        accumulateBatchReturn(toGpu());
    }
    return batchReturn_;
}

KickReturn KickWorkspace::toGpu() {
    ALWAYS_ASSERT(gpuExecutor_ != nullptr);
    return gpuExecutor_(params_, std::move(workitems_), treeIds_, output_);
}

void KickWorkspace::recordProcessedParticles(
    std::shared_ptr<KickWorkspace> ptr, ParticleInteger n) {
    bool doWork = false;
    std::unique_lock<MutexType> lock(mutex_);
    numberOfProcessedParticles_ += n;
    ALWAYS_ASSERT(numberOfProcessedParticles_ <= totalParticles_);
    if (!batch_ && numberOfProcessedParticles_ == totalParticles_) {
        doWork = true;
    }
    lock.unlock();
    if (doWork) {
        ptr->toGpu();
    }
}

void KickWorkspace::addWork(std::shared_ptr<KickWorkspace> ptr, ExpansionBuffer const& l,
    Array<Position, NDIM> pos, TreeId self, Vector<TreeId>&& dchecks, Vector<TreeId>&& echecks,
    FmmGeometry const& geometry, ParticleInteger outputOffset) {
    KickWorkitem item;
    item.geometry = geometry;
    item.outputOffset = outputOffset;
    item.l = l;
    item.pos = pos;
    item.self = self;
    item.dchecklist = std::move(dchecks);
    item.echecklist = std::move(echecks);
    bool doWork = false;
    {
        ParticleInteger const numberOfNodeParticles = treeGetNode(self)->numberOfParticles();
        std::lock_guard<MutexType> lock(mutex_);
        for (auto const& id : item.dchecklist) {
            ALWAYS_ASSERT(treeGetNode(id)->isLocal());
            treeIds_.insert(id);
        }
        for (auto const& id : item.echecklist) {
            ALWAYS_ASSERT(treeGetNode(id)->isLocal());
            treeIds_.insert(id);
        }
        // Publish the work before another producer can observe completion.
        workitems_.push_back(std::move(item));
        numberOfProcessedParticles_ += numberOfNodeParticles;
        ALWAYS_ASSERT(numberOfProcessedParticles_ <= totalParticles_);
        doWork = !batch_ && numberOfProcessedParticles_ == totalParticles_;
    }
    if (doWork) {
        ptr->toGpu();
    }
}
