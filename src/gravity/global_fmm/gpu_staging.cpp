/*
 Assemble local and imported source trees and particle data for CUDA gravity.

 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/global_fmm/gpu_staging.hpp>
#include <cosmictiger/gravity/global_fmm/measurement.hpp>
#include <cosmictiger/gravity/global_fmm/source_forest.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

static void addTreeNode(std::vector<TreeId>& trees, TreeId id) {
    TreeNode const* node = treeGetNode(id);
    trees.push_back(id);
    if (node->children[kLeftChild].index != -1) {
        addTreeNode(trees, node->children[kLeftChild]);
        addTreeNode(trees, node->children[kRightChild]);
    }
}

bool mortonCompare(Vector3<Position> a, Vector3<Position> b) {
    Vector3<PositionInteger> abits;
    Vector3<PositionInteger> bbits;
    for (int dim = 0; dim < kDimensions; dim++) {
        abits[dim] = a[dim].raw();
        bbits[dim] = b[dim].raw();
    }
    for (int i = 0; i < kPositionBits; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            bool const abit = (abits[dim] >> (kPositionBits - 1)) != 0;
            bool const bbit = (bbits[dim] >> (kPositionBits - 1)) != 0;
            if (bbit && !abit) {
                return true;
            } else if (!bbit && abit) {
                return false;
            }
            abits[dim] <<= 1;
            bbits[dim] <<= 1;
        }
    }
    return false;
}

KickReturn globalFmmStageGpu(KickParameters params, Vector<KickWorkitem> workitems,
    std::unordered_set<TreeId, KickWorkspaceTreeIdHash> const& treeIds, detail::FmmForceOutput) {
#ifdef USE_CUDA
    FmmMeasurementScope measurementTimer(FmmMeasurementPhase::GpuStaging);
    cudaSetDevice();
    if (workitems.size() == 0) {
        kickSetRc(KickReturn());
        return KickReturn();
    }
#if COSMICTIGER_PROFILE
    bool const profileEnabled = hpxRank() == 0;
#else
    bool const profileEnabled = false;
#endif
    profile::ScopedWork gpuPhase(profile::WorkPhase::KickGpuGroupSources, profileEnabled);

    struct GlobalParticleRange {
        int rank;                       // Locality owning the source particles.
        Pair<ParticleInteger> range;    // Half-open range in that owner's particle array.
        bool operator==(GlobalParticleRange other) const {
            return rank == other.rank && range == other.range;
        }
    };

    struct GlobalParticleRangeHash {
        size_t operator()(GlobalParticleRange g) const {
            return (size_t) hpxSize() * g.range.first + g.rank;
        }
    };
    MutexType mutex;
    Vector<hpx::shared_future<void>> futs1;
    Vector<TreeId> treeIdsVector(treeIds.begin(), treeIds.end());
    Vector<Vector<TreeId>> idsByDepth(kMaxDepth);
    int numberOfThreads = hpxSize() == 1 ? 1 : std::max(1, (int) hpxHardwareConcurrency() / 2);
    futs1.reserve(numberOfThreads);
    std::exception_ptr groupingFailure;
    try {
        for (int proc = 0; proc < numberOfThreads; proc++) {
            futs1.push_back(hpx::async(
                hpx::annotated_function(
                    [proc, numberOfThreads, &treeIdsVector, &idsByDepth, &mutex]() {
                        std::unique_lock<MutexType> lock(mutex);
                        int const b = (size_t) proc * treeIdsVector.size() / numberOfThreads;
                        int const e = (size_t) (proc + 1) * treeIdsVector.size() / numberOfThreads;
                        for (int i = b; i < e; i++) {
                            lock.unlock();
                            TreeNode const* ptr = treeGetNode(treeIdsVector[i]);
                            lock.lock();
                            idsByDepth[ptr->depth].push_back(treeIdsVector[i]);
                        }
                    },
                    "gravity.gpuStage.groupTreeNodes"))
                    .share());
        }
        hpx::wait_all(futs1.begin(), futs1.end());
        for (auto& future : futs1) {
            future.get();
        }
    } catch (...) {
        groupingFailure = std::current_exception();
    }
    if (groupingFailure) {
        for (auto& future : futs1) {
            if (future.valid()) {
                future.wait();
            }
        }
        std::rethrow_exception(groupingFailure);
    }
    gpuPhase.next(profile::WorkPhase::KickGpuCollectSources);
    struct ParticleRequest {
        Vector<Vector<Pair<ParticleInteger>>> data;    // Batches of particle-bearing leaf ranges.
        size_t count;    // Particles in the last batch, bounded except for an oversized leaf.
        ParticleRequest() {
            count = 0;
            data.resize(1);
        }
    };
    std::unordered_map<TreeId, int, KickWorkspaceTreeIdHash> treeMap;
    std::unordered_map<int, ParticleRequest> particleRequests;
    std::unordered_map<GlobalParticleRange, Pair<ParticleInteger>, GlobalParticleRangeHash>
        particleMap;
    ParticleInteger particleIndex = particlesSize();
    Vector<hpx::future<void>> futs2;
    size_t const originalParticleCount = particlesSize();
    size_t const maximumParticlesPerRequest = 64 * 1024 * 1024;
    Vector<hpx::future<Vector<TreeId>>> futs0;
    Vector<TreeId> trees;
    std::unique_ptr<TreeStagingArena> stagedTree;
    TreeNode* treeNodes = nullptr;
    hpx::future<void> sortFuture;
    futs0.reserve(numberOfThreads);
    futs2.reserve(std::max(numberOfThreads, int(hpxHardwareConcurrency())));
    std::exception_ptr stagingFailure;
    try {
        for (int depth = 0; depth < kMaxDepth; depth++) {
            auto const& ids = idsByDepth[depth];
            for (int proc = 0; proc < numberOfThreads; proc++) {
                futs0.push_back(hpx::async(hpx::annotated_function(
                    [proc, numberOfThreads, &ids, &treeMap]() {
                        Vector<TreeId> trees;
                        int const b = (size_t) proc * ids.size() / numberOfThreads;
                        int const e = (size_t) (proc + 1) * ids.size() / numberOfThreads;
                        for (int i = b; i < e; i++) {
                            if (treeMap.find(ids[i]) == treeMap.end()) {
                                addTreeNode(trees, ids[i]);
                            }
                        }
                        return trees;
                    },
                    "gravity.gpuStage.collectSources")));
            }
            hpx::wait_all(futs0.begin(), futs0.end());
            auto const start = trees.size();
            for (auto& f : futs0) {
                auto tmp = f.get();
                trees.reserve(trees.size() + tmp.size());
                for (auto const& entry : tmp) {
                    trees.push_back(entry);
                }
            }
            for (int i = start; i < trees.size(); i++) {
                if (trees[i].proc == hpxRank()) {
                    treeMap[trees[i]] = trees[i].index;
                } else {
                    treeMap[trees[i]] = -1;
                }
            }
            futs0.resize(0);
        }
        // A collapsed terminal represents a source count and moments, with no
        // coordinate payload. Stage each true remote leaf independently so a
        // parent containing both leaf data and terminals need not be contiguous
        // in the staging particle array. Internal/terminal ranges keep their
        // owner indices for counts only; CUDA never dereferences those ranges.
        for (auto const id : trees) {
            if (id.proc == hpxRank()) {
                continue;
            }
            TreeNode const& node = *treeGetNode(id);
            if (!node.leaf || node.sourceKind == kTreeSourceMultipoleTerminal ||
                node.numberOfParticles() == 0) {
                continue;
            }
            GlobalParticleRange const globalRange{node.processRange.first, node.particleRange};
            if (particleMap.contains(globalRange)) {
                continue;
            }
            auto& request = particleRequests[globalRange.rank];
            ParticleInteger const count = node.numberOfParticles();
            if (request.count != 0 && request.count + count > maximumParticlesPerRequest) {
                request.data.emplace_back();
                request.count = 0;
            }
            request.data.back().push_back(node.particleRange);
            request.count += count;
            particleMap.emplace(
                globalRange, Pair<ParticleInteger>{particleIndex, particleIndex + count});
            particleIndex += count;
        }
        {
            std::unique_lock<SharedMutexType> lock(particlesSharedMutex());
            particlesResize(particleIndex, false);
        }
        gpuPhase.next(profile::WorkPhase::KickGpuRemap);
        size_t const remoteNodeCount = std::count_if(
            trees.begin(), trees.end(), [](TreeId id) { return id.proc != hpxRank(); });
        stagedTree = std::make_unique<TreeStagingArena>(remoteNodeCount);
        treeNodes = stagedTree->data();
        auto const treeNodesRemoteBegin = stagedTree->remoteBegin();
        for (int i = 0; i < trees.size(); i++) {
            if (trees[i].proc != hpxRank()) {
                treeMap[trees[i]] = stagedTree->appendRemote(*treeGetNode(trees[i]));
            }
        }
        auto const treeNodesRemoteEnd = stagedTree->size();
        futs1.resize(0);
        numberOfThreads = std::max(1, (int) hpxHardwareConcurrency());
        futs1.reserve(particleRequests.size());
        // Continuations copy this closure: it leaves scope before failure cleanup
        // drains them, while its referenced particleMap remains alive there.
        auto const stageImported = [&particleMap](
                                       int rank, Vector<Pair<ParticleInteger>> const& ranges) {
            auto const& opts = getOptions();
            for (auto const& range : ranges) {
                GlobalParticleRange const globalRange{rank, range};
                auto const localRange = particleMap.at(globalRange);
                TreeNode source{};
                source.processRange = {rank, rank + 1};
                source.particleRange = range;
                if (!treeSourceForestReadParticleRange(source, &particlesPos(kXi, localRange.first),
                        &particlesPos(kYi, localRange.first), &particlesPos(kZi, localRange.first),
                        opts.hydro.enabled ? &particlesType(localRange.first) : nullptr, 0)) {
                    return false;
                }
            }
            return true;
        };
        for (auto i = particleRequests.begin(); i != particleRequests.end(); i++) {
            int const rank = i->first;
            // Retain the predecessor if continuation attachment throws. A plain
            // future::then invalidates its input even on that exception path.
            hpx::shared_future<void> fut = hpx::make_ready_future().share();
            std::exception_ptr continuationFailure;
            try {
                for (int j = 0; j < i->second.data.size(); j++) {
                    size_t const count = i->second.data[j].size();
                    if (count) {
                        fut = fut.then(hpx::annotated_function(
                                           [rank, count, &particleMap, stageImported, i, j](
                                               hpx::shared_future<void> fut) {
                                               auto const& opts = getOptions();
                                               fut.get();
                                               auto const& ranges = i->second.data[j];
                                               if (treeSourceForestImportsEnabled()) {
                                                   if (stageImported(rank, ranges)) {
                                                       return;
                                                   }
                                                   treeSourceForestCountFallback();
                                               }
                                               auto const data = particlesGet(rank, ranges).get();
                                               ParticleInteger index = 0;
                                               for (int k = 0; k < ranges.size(); k++) {
                                                   index += ranges[k].second - ranges[k].first;
                                               }
                                               size_t sz = index;
                                               index = 0;
                                               for (int k = 0; k < ranges.size(); k++) {
                                                   GlobalParticleRange gpr;
                                                   gpr.rank = rank;
                                                   gpr.range = ranges[k];
                                                   auto const localRange = particleMap.at(gpr);
                                                   auto const numberOfParticles =
                                                       localRange.second - localRange.first;
                                                   for (int dim = 0; dim < kDimensions; dim++) {
                                                       memcpy(&particlesPos(dim, localRange.first),
                                                           &data[sz * dim + index],
                                                           numberOfParticles * sizeof(Position));
                                                   }
                                                   if (opts.hydro.enabled) {
                                                       Vector<Position> x(numberOfParticles),
                                                           y(numberOfParticles),
                                                           z(numberOfParticles);
                                                       Vector<char> types(numberOfParticles);
                                                       ParticleGlobalRange source;
                                                       source.proc = rank;
                                                       source.range = ranges[k];
                                                       particlesGlobalReadPos(source, x.data(),
                                                           y.data(), z.data(), types.data(), 0);
                                                       std::memcpy(&particlesType(localRange.first),
                                                           types.data(),
                                                           numberOfParticles * sizeof(char));
                                                   }
                                                   index += numberOfParticles;
                                               }
                                           },
                                           "gravity.gpuStage.fetchParticles"))
                                  .share();
                    }
                }
                futs1.push_back(std::move(fut));
            } catch (...) {
                continuationFailure = std::current_exception();
            }
            if (continuationFailure) {
                if (fut.valid()) {
                    fut.wait();
                }
                std::rethrow_exception(continuationFailure);
            }
        }
        for (int proc = 0; proc < numberOfThreads; proc++) {
            futs2.push_back(hpx::async(HPX_PRIORITY_HI,
                hpx::annotated_function(
                    [proc, numberOfThreads, &treeMap, &particleMap, &treeNodes,
                        treeNodesRemoteBegin, treeNodesRemoteEnd]() {
                        for (size_t i = treeNodesRemoteBegin + proc; i < treeNodesRemoteEnd;
                            i += numberOfThreads) {
                            if (treeNodes[i].children[kLeftChild].index != -1) {
                                ASSERT(treeMap.find(treeNodes[i].children[kLeftChild]) !=
                                    treeMap.end());
                                ASSERT(treeMap.find(treeNodes[i].children[kRightChild]) !=
                                    treeMap.end());
                                auto& left = treeNodes[i].children[kLeftChild];
                                auto& right = treeNodes[i].children[kRightChild];
                                left.index = treeMap[left];
                                right.index = treeMap[right];
                            }
                            auto& node = treeNodes[i];
                            if (node.leaf && node.sourceKind != kTreeSourceMultipoleTerminal &&
                                node.numberOfParticles() != 0) {
                                GlobalParticleRange const globalRange{
                                    node.processRange.first, node.particleRange};
                                node.particleRange = particleMap.at(globalRange);
                            }
                        }
                    },
                    "gravity.gpuStage.remapTreeNodes")));
        }
        hpx::wait_all(futs2.begin(), futs2.end());
        for (auto& future : futs2) {
            future.get();
        }
        futs2.resize(0);
        for (int proc = 0; proc < numberOfThreads; proc++) {
            futs2.push_back(hpx::async(HPX_PRIORITY_HI,
                hpx::annotated_function(
                    [&workitems, proc, numberOfThreads, &treeMap]() {
                        for (int i = proc; i < workitems.size(); i += numberOfThreads) {
                            for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
                                if (workitems[i].dchecklist[j].proc != hpxRank()) {
                                    auto const iter = treeMap.find(workitems[i].dchecklist[j]);
                                    ASSERT(iter != treeMap.end());
                                    workitems[i].dchecklist[j].index = iter->second;
                                }
                            }
                            for (int j = 0; j < workitems[i].echecklist.size(); j++) {
                                if (workitems[i].echecklist[j].proc != hpxRank()) {
                                    auto const iter = treeMap.find(workitems[i].echecklist[j]);
                                    ASSERT(iter != treeMap.end());
                                    workitems[i].echecklist[j].index = iter->second;
                                }
                            }
                        }
                    },
                    "gravity.gpuStage.remapWorkitems")));
        }
        hpx::wait_all(futs2.begin(), futs2.end());
        for (auto& future : futs2) {
            future.get();
        }
        futs2.resize(0);
        gpuPhase.next(profile::WorkPhase::KickGpuSortFetchWait);
        sortFuture = hpx::sort(annotatedParallelPolicy("gravity.gpuStage.sortWorkitems"),
            workitems.begin(), workitems.end(), [](KickWorkitem const& a, KickWorkitem const& b) {
                return treeGetNode(a.self)->particleRange.first <
                    treeGetNode(b.self)->particleRange.first;
            });
        hpx::wait_all(futs1.begin(), futs1.end());
        for (auto& future : futs1) {
            future.get();
        }
        sortFuture.get();
        gpuPhase.next(profile::WorkPhase::KickGpuMemoryAdvice);
        stagedTree->adviseGpu();
        particlesMemadviseGpu();
        gpuPhase.stop();
        measurementTimer.stop();
        auto const kr = cudaExecuteKicks(params, &particlesPos(kXi, 0), &particlesPos(kYi, 0),
            &particlesPos(kZi, 0), treeNodes, std::move(workitems), {});
        profile::ScopedWork cleanupPhase(profile::WorkPhase::KickGpuCleanup, profileEnabled);
        tree2Cpu();
        particlesResize(originalParticleCount);
        kickSetRc(kr);
        return kr;
    } catch (...) {
        stagingFailure = std::current_exception();
    }
    gpuPhase.stop();
    profile::ScopedWork cleanupPhase(profile::WorkPhase::KickGpuCleanup, profileEnabled);
    // All referents remain alive. Wait outside the catch handler because HPX
    // may resume this task on a different native thread after suspension.
    auto const join = [](auto& futures) {
        for (auto& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
    };
    join(futs0);
    join(futs1);
    join(futs2);
    if (sortFuture.valid()) {
        sortFuture.wait();
    }
    particlesResize(originalParticleCount);
    std::rethrow_exception(stagingFailure);
#else
    throw std::runtime_error("GPU gravity requested in a build without CUDA");
    return {};
#endif
}
