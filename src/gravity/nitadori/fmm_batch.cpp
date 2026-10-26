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

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_interaction_counts.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/nitadori/fmm_batch.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {
KickReturn localBatchGpu(KickParameters params, Vector<KickWorkitem> workitems,
    std::unordered_set<TreeId, KickWorkspaceTreeIdHash> const&, FmmForceOutput output) {
#ifdef USE_CUDA
    cudaSetDevice();
    std::sort(workitems.begin(), workitems.end(), [](auto const& left, auto const& right) {
        return treeGetNode(left.self)->particleRange.first <
            treeGetNode(right.self)->particleRange.first;
    });
    return cudaExecuteKicks(params, &particlesPos(kXDimension, 0), &particlesPos(kYDimension, 0),
        &particlesPos(kZDimension, 0), treeData(), std::move(workitems), output);
#else
    THROW_ERROR("GPU gravity requested in a build without CUDA\n");
    return {};
#endif
}
hpx::future<KickReturn> localBatchFork(KickParameters params, ExpansionBuffer const& l,
    Array<Position, kDimensions> pos, TreeId self, Vector<TreeId> direct, Vector<TreeId> periodic,
    std::shared_ptr<KickWorkspace> workspace, bool threadme) {
    ALWAYS_ASSERT(self.proc == hpxRank() && periodic.empty());
    auto run = [params, l, pos, self, workspace](Vector<TreeId> direct) {
        KickContext context;
        context.fork = localBatchFork;
        context.gpuExecutor = localBatchGpu;
        return kickWalkDispatch(params, l, pos, self, std::move(direct), {}, workspace, context);
    };
    static std::atomic<int> threads{1};
    bool const eligible = threadme && treeGetNode(self)->numberOfParticles() > kMinKickThreadParts;
    int const maximumThreads = std::max(
        1, hpxSize() == 1 ? int(hpxHardwareConcurrency()) * 2 : int(hpxHardwareConcurrency()) / 2);
    if (eligible) {
        if (threads.fetch_add(1) < maximumThreads) {
            try {
                return hpx::async(
                    [run = std::move(run)](Vector<TreeId> direct) mutable {
                        struct ThreadCountGuard {
                            ~ThreadCountGuard() {
                                --threads;
                            }
                        } guard;
                        return run(std::move(direct));
                    },
                    std::move(direct));
            } catch (...) {
                --threads;
                throw;
            }
        }
        --threads;
    }
    hpxYield();
    return hpx::make_ready_future(run(std::move(direct)));
}
}    // namespace detail
FmmForceResult kickCellBatch(Vector<KickWorkitem> workitems, bool doPotential) {
    ParticleInteger const ownedCount = particlesOwnedSize();
    std::sort(workitems.begin(), workitems.end(), [](auto const& left, auto const& right) {
        return treeGetNode(left.self)->particleRange.first <
            treeGetNode(right.self)->particleRange.first;
    });
    ParticleInteger particleGoal = 0;
    ParticleInteger previousEnd = 0;
    FmmForceResult result;
    for (auto& item : workitems) {
        auto const* node = treeGetNode(item.self);
        ALWAYS_ASSERT(item.self.proc == hpxRank() && node->isLocalHere());
        ALWAYS_ASSERT(node->sourceStore == FmmParticleStore::Owned);
        auto const range = node->particleRange;
        ALWAYS_ASSERT(range.first >= previousEnd && range.second <= ownedCount);
        ALWAYS_ASSERT(range.second >= range.first);
        ALWAYS_ASSERT(item.echecklist.empty() && item.geometry.meshSize > 0);
        for (auto const source : item.dchecklist) {
            ALWAYS_ASSERT(source.proc == hpxRank() && treeGetNode(source)->isLocalHere());
        }
        item.outputOffset = particleGoal - range.first;
        particleGoal += node->numberOfParticles();
        previousEnd = range.second;
        for (ParticleInteger index = range.first; index < range.second; ++index) {
            result.particleIndices.push_back(index);
        }
    }
    size_t const size = particleGoal;
    result.potential.resize(size);
    for (int dim = 0; dim < kDimensions; ++dim) {
        result.acceleration[dim].resize(size);
    }
    if (particleGoal == 0) {
        return result;
    }
    std::unique_ptr<float, decltype(&freeManagedMemory)> storage(
        static_cast<float*>(allocateManagedMemory(4 * size * sizeof(float))), freeManagedMemory);
    std::fill_n(storage.get(), 4 * size, 0.0f);
    detail::FmmForceOutput output;
    output.potential = storage.get();
    for (int dim = 0; dim < kDimensions; ++dim) {
        output.acceleration[dim] = storage.get() + (dim + 1) * size;
    }
    KickParameters params;
    params.forceOnly = true;
    params.htime = false;
    params.minLevel = params.minRung = 0;
    params.a = params.t0 = params.gm = 1.0f;
    params.theta = getOptions().gravity.openingAngle;
    params.openLargerCell = getOptions().gravity.openLargerCell;
    params.gravitySupportRadius = getOptions().gravity.supportRadius;
    params.eta = params.nodeLoad = 0.0f;
    params.saveForce = false;
    params.firstCall = true;
    params.gpu = getOptions().compute.cuda;
    params.ascending = params.descending = params.top = false;
    params.doPhi = doPotential;
    auto workspace =
        std::make_shared<KickWorkspace>(params, particleGoal, output, detail::localBatchGpu);
    Vector<hpx::future<KickReturn>> futures;
    futures.reserve(workitems.size());
    try {
        for (auto& item : workitems) {
            params.geometry = item.geometry;
            params.outputOffset = item.outputOffset;
            futures.push_back(detail::localBatchFork(params, item.l, item.pos, item.self,
                std::move(item.dchecklist), std::move(item.echecklist), workspace, true));
        }
        hpx::wait_all(futures.begin(), futures.end());
        for (auto& future : futures) {
            future.get();
        }
        // Every CPU producer is finished before the packet is sealed. This
        // batch can launch while other cells still await their source trees.
        workspace->finishBatch();
    } catch (...) {
        // No worker may retain scratch or tree references past this scope.
        for (auto& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
        throw;
    }
    std::copy_n(output.potential, size, result.potential.begin());
    for (int dim = 0; dim < kDimensions; ++dim) {
        std::copy_n(output.acceleration[dim], size, result.acceleration[dim].begin());
    }
    result.interactionCounts = workspace->interactionCounts();
    return result;
}

FmmForceResult kickCellForest(Vector<KickWorkitem> workitems, bool doPotential) {
    auto result = kickCellBatch(std::move(workitems), doPotential);
    ALWAYS_ASSERT(result.particleIndices.size() == size_t(particlesOwnedSize()));
    for (size_t index = 0; index < result.particleIndices.size(); ++index) {
        ALWAYS_ASSERT(result.particleIndices[index] == ParticleInteger(index));
    }
    return result;
}
