/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <mutex>
#include <set>

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

/// Initialize a kick workspace with its parameters, total particle goal, and GPU executor.
KickWorkspace::KickWorkspace(
    KickParameters p, ParticleInteger totalParticlesArgument, KickGpuExecutor executor) {
    gpuExecutor_ = executor;
    totalParticles_ = totalParticlesArgument;
    params_ = p;
    numberOfProcessedParticles_ = 0;
}

/// Initialize batch-mode kick storage with the caller-provided force-output view.
KickWorkspace::KickWorkspace(KickParameters p, ParticleInteger totalParticlesArgument,
    detail::FmmForceOutput output, KickGpuExecutor executor)
  : KickWorkspace(p, totalParticlesArgument, executor) {
    batch_ = true;
    output_ = output;
}

/// Destroy the workspace and release its owned containers through member destruction.
KickWorkspace::~KickWorkspace() {}

/// Merge a completed batch's kick diagnostics under the workspace mutex.
void KickWorkspace::accumulateBatchReturn(KickReturn const& result) {
    std::lock_guard<MutexType> lock(mutex_);
    batchReturn_ += result;
}

/// Accumulate each interaction category under the workspace mutex.
void KickWorkspace::addInteractionCounts(KickInteractionCounts const& counts) {
    std::lock_guard<MutexType> lock(mutex_);
    interactionCounts_.acceptanceTests += counts.acceptanceTests;
    interactionCounts_.cellCellDirect += counts.cellCellDirect;
    interactionCounts_.cellParticle += counts.cellParticle;
    interactionCounts_.cellParticleAtLeafSinks += counts.cellParticleAtLeafSinks;
    interactionCounts_.particleCell += counts.particleCell;
    interactionCounts_.particleCellFromLeafSources += counts.particleCellFromLeafSources;
    interactionCounts_.particleParticleLeafPairs += counts.particleParticleLeafPairs;
    interactionCounts_.particleParticlePairs += counts.particleParticlePairs;
    interactionCounts_.sourceOpenings += counts.sourceOpenings;
    interactionCounts_.deferredSources += counts.deferredSources;
}

/// Return a synchronized snapshot of accumulated interaction counts.
KickInteractionCounts KickWorkspace::interactionCounts() {
    std::lock_guard<MutexType> lock(mutex_);
    return interactionCounts_;
}

/// Require all producers to finish, execute queued GPU work, and return combined batch
/// diagnostics.
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

/// Pass the queued work items and output view to the configured GPU executor.
KickReturn KickWorkspace::toGpu() {
    ALWAYS_ASSERT(gpuExecutor_ != nullptr);
    return gpuExecutor_(params_, std::move(workitems_), treeIds_, output_);
}

/// Record completed CPU particles and trigger deferred GPU execution when all producers
/// finish.
void KickWorkspace::recordProcessedParticles(
    std::shared_ptr<KickWorkspace> ptr, ParticleInteger n) {
    profile::ScopedWork work(profile::WorkPhase::KickGpuQueue,
#if COSMICTIGER_PROFILE
        hpxRank() == 0
#else
        false
#endif
    );
    bool doWork = false;
    std::unique_lock<MutexType> lock(mutex_);
    numberOfProcessedParticles_ += n;
    ALWAYS_ASSERT(numberOfProcessedParticles_ <= totalParticles_);
    if (!batch_ && numberOfProcessedParticles_ == totalParticles_) {
        doWork = true;
    }
    lock.unlock();
    work.stop();
    if (doWork) {
        ptr->toGpu();
    }
}

/// Queue one GPU sink work item and trigger execution when the complete particle goal is
/// accounted for.
void KickWorkspace::addWork(std::shared_ptr<KickWorkspace> ptr, ExpansionBuffer const& l,
    Vector3<Position> pos, TreeId self, Vector<TreeId>&& dchecks, Vector<TreeId>&& echecks,
    ParticleInteger outputOffset) {
    profile::ScopedWork work(profile::WorkPhase::KickGpuQueue,
#if COSMICTIGER_PROFILE
        hpxRank() == 0
#else
        false
#endif
    );
    KickWorkitem item;
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
    work.stop();
    if (doWork) {
        ptr->toGpu();
    }
}
