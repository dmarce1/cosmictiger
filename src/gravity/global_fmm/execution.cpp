/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <stack>
#include <utility>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/global_fmm/gpu_staging.hpp>
#include <cosmictiger/gravity/global_fmm/particle_exchange.hpp>
#include <cosmictiger/gravity/global_fmm/periodic.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_interaction_counts.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/source_access.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

HPX_PLAIN_ACTION(kick, KickAction);
namespace detail {
double kickBackgroundSecondMoment(Vector3<Position> center);
}

HPX_PLAIN_ACTION(detail::kickBackgroundSecondMoment, KickBackgroundSecondMomentAction);

namespace detail {
double kickBackgroundSecondMoment(Vector3<Position> center) {
    Vector<hpx::future<double>> children;
    for (auto const& locality : hpxChildren()) {
        children.push_back(hpx::async<KickBackgroundSecondMomentAction>(locality, center));
    }
    double secondMoment = 0.0;
    auto const range = particlesCurrentRange();
    for (ParticleInteger i = range.first; i < range.second; i++) {
        double radiusSquared = 0.0;
        for (int dim = 0; dim < kDimensions; dim++) {
            double const displacement = unwrappedDoubleDistance(particlesPos(dim, i), center[dim]);
            radiusSquared += displacement * displacement;
        }
        secondMoment += double(particlesMass(i)) * radiusSquared;
    }
    for (auto& child : children) {
        secondMoment += child.get();
    }
    addCpuFlops(
        FlopCount(range.second - range.first) * 17 + children.size(), FlopPrecision::Double);
    return secondMoment;
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
    Vector3<Position> pos, TreeId self, Vector<TreeId> dchecklist, Vector<TreeId> echecklist,
    std::shared_ptr<KickWorkspace> cudaWorkspace, bool threadme) {
#if COSMICTIGER_PROFILE
    bool const profileWork = hpxRank() == 0;
#else
    bool const profileWork = false;
#endif
    profile::ScopedWork scheduling(profile::WorkPhase::KickCpuSchedule, profileWork);
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
            selfPtr->particleRange.second - selfPtr->particleRange.first > kMinKickThreadParts;
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
        scheduling.stop();
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
            rc = hpx::async(hpx::annotated_function(std::move(run), "gravity.kickWalk"),
                std::move(dchecklist), std::move(echecklist));
        } else {
            rc = hpx::async(HPX_PRIORITY_HI,
                hpx::annotated_function(std::move(run), "gravity.kickWalk"), std::move(dchecklist),
                std::move(echecklist));
        }
    }
    return rc;
}

KickReturn kick(KickParameters params, ExpansionBuffer l, Vector3<Position> pos, TreeId self,
    Vector<TreeId> direct, Vector<TreeId> periodic, std::shared_ptr<KickWorkspace> workspace) {
#if COSMICTIGER_PROFILE
    bool const coordinator = params.profileCoordinator && self.proc == 0 && self.index == 0 &&
        !params.forceOnly && hpxRank() == 0 && treeGetNode(self)->depth == 0;
    profile::ScopedPhase phase("kick.walk", coordinator);
#endif
    params.openLargerCell = getOptions().gravity.openLargerCell;
    gravitySetRemoteSourceReader(globalFmmReadSourcePositions);
    KickContext context;
    context.fork = kickFork;
    context.background = globalFmmBackground;
    context.periodic = globalFmmPeriodic;
    context.beginReturn = detail::kickBeginLocalReturn;
    context.coverParticles = detail::kickCoverParticles;
    context.accumulateReturn = detail::kickAccumulateLeafReturn;
    context.waitReturn = detail::kickWaitLocalReturn;
    context.publishReturn = kickSetRc;
    context.gpuExecutor = globalFmmStageGpu;
#if COSMICTIGER_PROFILE
    if (coordinator) {
        // kickFork constructs a fresh context in every descendant. Only this
        // global-root context may nest phases while concurrent workers run.
        context.background = [](int order, KickParameters const& parameters,
                                 ExpansionBuffer& expansion, Vector3<Position>& position,
                                 TreeId node, Vector<TreeId> const& checks) -> FlopCounts {
            profile::ScopedPhase phase("kick.background");
            return globalFmmBackground(order, parameters, expansion, position, node, checks);
        };
        context.waitReturn = []() -> KickReturn {
            // On one locality this is the remaining local/GPU completion wait.
            // A distributed global root waits for child results inside kick.walk.
            profile::ScopedPhase phase("kick.waitLocalResult");
            return detail::kickWaitLocalReturn();
        };
    }
#endif
    return kickWalkDispatch(params, std::move(l), pos, self, std::move(direct), std::move(periodic),
        std::move(workspace), context);
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

HPX_PLAIN_ACTION(kickClearBuffers, ClearBuffersAction);
void kickClearBuffers() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ClearBuffersAction>(c));
    }
    hpx::wait_all(futs.begin(), futs.end());
}
