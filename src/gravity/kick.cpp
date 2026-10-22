/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <atomic>
#include <memory>
#include <mutex>
#include <stack>
#include <utility>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_interaction_counts.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/particles/particles.hpp>

HPX_PLAIN_ACTION(kick, KickAction);

namespace detail {
double kickBackgroundSecondMoment(Array<Position, NDIM> center);
}

HPX_PLAIN_ACTION(detail::kickBackgroundSecondMoment, KickBackgroundSecondMomentAction);

namespace detail {
double kickBackgroundSecondMoment(Array<Position, NDIM> center) {
    Vector<hpx::future<double>> children;
    for (auto const& locality : hpxChildren()) {
        children.push_back(hpx::async<KickBackgroundSecondMomentAction>(locality, center));
    }
    double secondMoment = 0.0;
    auto const range = particlesCurrentRange();
    for (ParticleInteger i = range.first; i < range.second; i++) {
        double radiusSquared = 0.0;
        for (int dim = 0; dim < NDIM; dim++) {
            double const displacement = unwrappedDoubleDistance(particlesPos(dim, i), center[dim]);
            radiusSquared += displacement * displacement;
        }
        secondMoment += double(particlesMass(i)) * radiusSquared;
    }
    for (auto& child : children) {
        secondMoment += child.get();
    }
    addCpuFlops(FlopCount(range.second - range.first) * 17 + children.size());
    return secondMoment;
}
}    // namespace detail

namespace detail {

static thread_local std::stack<KickLists> localLists;

KickLists kickAcquireLists() {
    if (localLists.empty()) {
        localLists.push(KickLists());
    }
    KickLists lists = std::move(localLists.top());
    localLists.pop();
    return std::move(lists);
}

void kickReleaseLists(KickLists&& lists) {
    lists.cclist.resize(0);
    lists.cplist.resize(0);
    lists.pclist.resize(0);
    lists.nextlist.resize(0);
    lists.leaflist.resize(0);
    localLists.push(std::move(lists));
}

bool kickSourcesAreSingleRank(Vector<TreeId> const& checklist) {
    for (auto const& source : checklist) {
        if (!treeGetNode(source)->isLocal()) {
            return false;
        }
    }
    return true;
}

}    // namespace detail

static std::atomic<size_t> particlesCovered;
static hpx::promise<KickReturn> localReturnPromise;
static hpx::future<KickReturn> localReturnFuture;
static KickReturn localReturn;
static MutexType localReturnMutex;
static size_t localParticleGoal;
static bool localReturnSent;

namespace detail {

void kickBeginLocalReturn(ParticleInteger particleGoal) {
    particlesCovered = 0;
    localParticleGoal = particleGoal;
    localReturnSent = false;
    localReturn = KickReturn();
    localReturnPromise = decltype(localReturnPromise)();
    localReturnFuture = localReturnPromise.get_future();
}

void kickCoverParticles(ParticleInteger count) {
    particlesCovered += count;
}

void kickAccumulateLeafReturn(KickReturn const& kr, ParticleInteger count, bool allowPublish) {
    KickReturn completed;
    bool publish = false;
    {
        std::lock_guard<MutexType> lock(localReturnMutex);
        particlesCovered += count;
        localReturn += kr;
        if (allowPublish && particlesCovered == localParticleGoal && !localReturnSent) {
            completed = localReturn;
            localReturnSent = true;
            publish = true;
        }
    }
    if (publish) {
        localReturnPromise.set_value(completed);
    }
}

KickReturn kickWaitLocalReturn() {
    return localReturnFuture.get();
}

}    // namespace detail

hpx::future<KickReturn> kickFork(KickParameters params, ExpansionBuffer const& l,
    Array<Position, NDIM> pos, TreeId self, Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace, bool threadme) {
    static std::atomic<int> numberOfThreads(1);
    hpx::future<KickReturn> rc;
    TreeNode const* selfPtr = treeGetNode(self);
    bool remote = false;
    bool allLocal = true;
    for (auto const& i : dchecklist) {
        if (i.proc != hpxRank()) {
            allLocal = false;
            break;
        }
    }
    if (self.proc != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme =
            selfPtr->particleRange.second - selfPtr->particleRange.first > MIN_KICK_THREAD_PARTS;
        int const maximumThreads = std::max(1,
            hpxSize() == 1 ? (int) hpxHardwareConcurrency() * 2 :
                             (int) hpxHardwareConcurrency() / 2);
        if (threadme) {
            if (numberOfThreads++ < maximumThreads || !selfPtr->isLocal()) {
                threadme = true;
            } else {
                threadme = false;
                numberOfThreads--;
            }
        }
    }
    if (!threadme) {
        if (allLocal) {
            hpxYield();
        }
        rc = hpx::make_ready_future(kick(params, l, pos, self, std::move(dchecklist),
            std::move(echecklist), std::move(cudaWorkspace)));
    } else if (remote) {
        rc = hpx::async<KickAction>(hpxLocalities()[selfPtr->processRange.first], params, l, pos,
            self, std::move(dchecklist), std::move(echecklist), nullptr);
    } else {
        auto run = [params, self, l, pos, cudaWorkspace](
                       Vector<TreeId> dchecklist, Vector<TreeId> echecklist) {
            struct ThreadCountGuard {
                std::atomic<int>& count;
                ~ThreadCountGuard() {
                    --count;
                }
            } guard{numberOfThreads};
            return kick(
                params, l, pos, self, std::move(dchecklist), std::move(echecklist), cudaWorkspace);
        };
        if (allLocal) {
            rc = hpx::async(std::move(run), std::move(dchecklist), std::move(echecklist));
        } else {
            rc = hpx::async(
                HPX_PRIORITY_HI, std::move(run), std::move(dchecklist), std::move(echecklist));
        }
    }
    return rc;
}

// The walk itself is order-templated (kick_walk.hpp) and instantiated once per
// order by the instance translation units; select the runtime order's walk.
KickReturn kick(KickParameters params, ExpansionBuffer l, Array<Position, NDIM> pos, TreeId self,
    Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace) {
    params.openLargerCell = getOptions().gravity.openLargerCell;
    return fmmDispatch([&](auto orderTag) {
        return kickWalk<decltype(orderTag)::value>(params, std::move(l), pos, self,
            std::move(dchecklist), std::move(echecklist), std::move(cudaWorkspace));
    });
}

void kickSetRc(KickReturn kr) {
    KickReturn completed;
    {
        std::lock_guard<MutexType> lock(localReturnMutex);
        localReturn += kr;
        completed = localReturn;
        localReturnSent = true;
    }
    localReturnPromise.set_value(completed);
}

namespace {

struct KickInteractionCounters {
    std::atomic<long long> acceptanceTests{0};
    std::atomic<long long> cellCellDirect{0};
    std::atomic<long long> cellCellEwald{0};
    std::atomic<long long> cellParticle{0};
    std::atomic<long long> cellParticleAtLeafSinks{0};
    std::atomic<long long> particleCell{0};
    std::atomic<long long> particleCellFromLeafSources{0};
    std::atomic<long long> particleParticleLeafPairs{0};
    std::atomic<long long> particleParticlePairs{0};
    std::atomic<long long> sourceOpenings{0};
    std::atomic<long long> deferredSources{0};
};

KickInteractionCounters interactionCounters;

}    // namespace

void kickResetInteractionCounts() {
    interactionCounters.acceptanceTests = 0;
    interactionCounters.cellCellDirect = 0;
    interactionCounters.cellCellEwald = 0;
    interactionCounters.cellParticle = 0;
    interactionCounters.cellParticleAtLeafSinks = 0;
    interactionCounters.particleCell = 0;
    interactionCounters.particleCellFromLeafSources = 0;
    interactionCounters.particleParticleLeafPairs = 0;
    interactionCounters.particleParticlePairs = 0;
    interactionCounters.sourceOpenings = 0;
    interactionCounters.deferredSources = 0;
}

void kickAddInteractionCounts(KickInteractionCounts const& counts) {
    interactionCounters.acceptanceTests += counts.acceptanceTests;
    interactionCounters.cellCellDirect += counts.cellCellDirect;
    interactionCounters.cellCellEwald += counts.cellCellEwald;
    interactionCounters.cellParticle += counts.cellParticle;
    interactionCounters.cellParticleAtLeafSinks += counts.cellParticleAtLeafSinks;
    interactionCounters.particleCell += counts.particleCell;
    interactionCounters.particleCellFromLeafSources += counts.particleCellFromLeafSources;
    interactionCounters.particleParticleLeafPairs += counts.particleParticleLeafPairs;
    interactionCounters.particleParticlePairs += counts.particleParticlePairs;
    interactionCounters.sourceOpenings += counts.sourceOpenings;
    interactionCounters.deferredSources += counts.deferredSources;
}

KickInteractionCounts kickInteractionCounts() {
    KickInteractionCounts counts;
    counts.acceptanceTests = interactionCounters.acceptanceTests;
    counts.cellCellDirect = interactionCounters.cellCellDirect;
    counts.cellCellEwald = interactionCounters.cellCellEwald;
    counts.cellParticle = interactionCounters.cellParticle;
    counts.cellParticleAtLeafSinks = interactionCounters.cellParticleAtLeafSinks;
    counts.particleCell = interactionCounters.particleCell;
    counts.particleCellFromLeafSources = interactionCounters.particleCellFromLeafSources;
    counts.particleParticleLeafPairs = interactionCounters.particleParticleLeafPairs;
    counts.particleParticlePairs = interactionCounters.particleParticlePairs;
    counts.sourceOpenings = interactionCounters.sourceOpenings;
    counts.deferredSources = interactionCounters.deferredSources;
    return counts;
}
