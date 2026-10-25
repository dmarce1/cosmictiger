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
#include <cosmictiger/gravity/dehnen/execution.hpp>
#include <cosmictiger/gravity/dehnen/gpu_staging.hpp>
#include <cosmictiger/gravity/kick_workspace.hpp>
#include <cosmictiger/particles/particles.hpp>

static void addTreeNode(std::vector<TreeId>& trees, TreeId id) {
    TreeNode const* node = treeGetNode(id);
    trees.push_back(id);
    if (node->children[LEFT].index != -1) {
        addTreeNode(trees, node->children[LEFT]);
        addTreeNode(trees, node->children[RIGHT]);
    }
}

bool mortonCompare(Array<Position, NDIM> a, Array<Position, NDIM> b) {
    Array<PositionInteger, NDIM> abits;
    Array<PositionInteger, NDIM> bbits;
    for (int dim = 0; dim < NDIM; dim++) {
        abits[dim] = a[dim].raw();
        bbits[dim] = b[dim].raw();
    }
    for (int i = 0; i < kPositionBits; i++) {
        for (int dim = 0; dim < NDIM; dim++) {
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

KickReturn dehnenStageGpu(KickParameters params, Vector<KickWorkitem> workitems,
    std::unordered_set<TreeId, KickWorkspaceTreeIdHash> const& treeIds, detail::FmmForceOutput) {
#ifdef USE_CUDA
    cudaSetDevice();
    //	PRINT("Preparing gpu send on %i\n", hpx_rank());

    Timer tm;
    tm.start();
    auto const treeNodesRemoteBegin = treeNodesNextIndex();
    if (workitems.size() == 0) {
        kickSetRc(KickReturn());
        return KickReturn();
    }

    struct GlobalParticleRange {
        int rank;
        Pair<ParticleInteger> range;
        bool operator==(GlobalParticleRange other) const {
            return rank == other.rank && range == other.range;
        }
    };

    struct GlobalParticleRangeHash {
        size_t operator()(GlobalParticleRange g) const {
            return (size_t) hpxSize() * g.range.first + g.rank;
        }
    };
    Timer tm2;
    Timer tm3;
    Timer tm4;
    tm2.start();
    MutexType mutex;
    Vector<hpx::future<void>> futs1;
    Vector<TreeId> treeIdsVector(treeIds.begin(), treeIds.end());
    Vector<Vector<TreeId>> idsByDepth(MAX_DEPTH);
    int numberOfThreads = hpxSize() == 1 ? 1 : std::max(1, (int) hpxHardwareConcurrency() / 2);
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs1.push_back(hpx::async([proc, numberOfThreads, &treeIdsVector, &idsByDepth, &mutex]() {
            std::unique_lock<MutexType> lock(mutex);
            int const b = (size_t) proc * treeIdsVector.size() / numberOfThreads;
            int const e = (size_t) (proc + 1) * treeIdsVector.size() / numberOfThreads;
            for (int i = b; i < e; i++) {
                lock.unlock();
                TreeNode const* ptr = treeGetNode(treeIdsVector[i]);
                lock.lock();
                idsByDepth[ptr->depth].push_back(treeIdsVector[i]);
            }
        }));
    }
    hpx::wait_all(futs1.begin(), futs1.end());
    tm2.stop();
    tm3.start();
    struct ParticleRequest {
        Vector<Vector<Pair<ParticleInteger>>> data;
        size_t count;
        ParticleRequest() {
            count = 0;
            data.resize(1);
        }
    };
    std::unordered_map<TreeId, int, KickWorkspaceTreeIdHash> treeMap;
    std::unordered_map<int, ParticleRequest> particleRequests;
    std::unordered_map<GlobalParticleRange, Pair<ParticleInteger>, GlobalParticleRangeHash>
        particleMap;
    std::set<TreeId> remoteRoots;
    ParticleInteger particleIndex = particlesSize();
    Vector<hpx::future<void>> futs2;
    Vector<hpx::future<void>> futs3;
    size_t const originalParticleCount = particlesSize();
    size_t const maximumParticlesPerRequest = 64 * 1024 * 1024;
    Vector<hpx::future<Vector<TreeId>>> futs0;
    Vector<TreeId> trees;
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        auto const& ids = idsByDepth[depth];
        for (int proc = 0; proc < numberOfThreads; proc++) {
            futs0.push_back(
                hpx::async([proc, numberOfThreads, &ids, &treeMap, &mutex, &particleRequests,
                               &particleMap, &particleIndex, &remoteRoots]() {
                    Vector<TreeId> trees;
                    int const b = (size_t) proc * ids.size() / numberOfThreads;
                    int const e = (size_t) (proc + 1) * ids.size() / numberOfThreads;
                    for (int i = b; i < e; i++) {
                        if (treeMap.find(ids[i]) == treeMap.end()) {
                            TreeNode const* node = treeGetNode(ids[i]);
                            addTreeNode(trees, ids[i]);
                            int const rank = node->processRange.first;
                            if (rank != hpxRank()) {
                                auto const range = node->particleRange;
                                auto const thisCount = range.second - range.first;
                                GlobalParticleRange gpr;
                                gpr.rank = rank;
                                gpr.range = range;
                                size_t const numberOfParticles = node->numberOfParticles();
                                std::unique_lock<SharedMutexType> lock(particlesSharedMutex());
                                auto& entry = particleRequests[rank];
                                entry.data.back().push_back(range);
                                entry.count += thisCount;
                                particleMap[gpr].first = particleIndex;
                                particleMap[gpr].second = particleIndex + numberOfParticles;
                                particleIndex += numberOfParticles;
                                particlesResize(particleIndex, false);
                                remoteRoots.insert(ids[i]);
                                if (entry.count > maximumParticlesPerRequest) {
                                    size_t count = entry.count;
                                    ParticleInteger index = 0;
                                    entry.count = 0;
                                    entry.data.resize(entry.data.size() + 1);
                                }
                            }
                        }
                    }
                    return trees;
                }));
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
    tm3.stop();
    tm4.start();
    for (int i = 0; i < trees.size(); i++) {
        if (trees[i].proc != hpxRank()) {
            treeMap[trees[i]] = treeAddRemote(*treeGetNode(trees[i]));
        }
    }
    tm4.stop();

    //	PRINT("%e / %e %e %e to load tree nodes\n", tm2.read() + tm3.read() + tm4.read(),
    // tm2.read(), tm3.read(), tm4.read());
    tm2.reset();
    futs1.resize(0);
    numberOfThreads = std::max(1, (int) hpxHardwareConcurrency());
    for (auto i = particleRequests.begin(); i != particleRequests.end(); i++) {
        int const rank = i->first;
        hpx::future<void> fut = hpx::make_ready_future();
        for (int j = 0; j < i->second.data.size(); j++) {
            size_t const count = i->second.data[j].size();
            if (count) {
                fut = fut.then([rank, count, &particleMap, i, j](hpx::future<void> fut) {
                    fut.get();
                    auto const& ranges = i->second.data[j];
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
                        auto localRange = particleMap[gpr];
                        auto const numberOfParticles = localRange.second - localRange.first;
                        for (int dim = 0; dim < NDIM; dim++) {
                            memcpy(&particlesPos(dim, localRange.first), &data[sz * dim + index],
                                numberOfParticles * sizeof(Position));
                        }
                        if (getOptions().hydro.enabled) {
                            Vector<Position> x(numberOfParticles), y(numberOfParticles),
                                z(numberOfParticles);
                            Vector<char> types(numberOfParticles);
                            ParticleGlobalRange source;
                            source.proc = rank;
                            source.range = ranges[k];
                            particlesGlobalReadPos(
                                source, x.data(), y.data(), z.data(), types.data(), 0);
                            std::memcpy(&particlesType(localRange.first), types.data(),
                                numberOfParticles * sizeof(char));
                        }
                        index += numberOfParticles;
                    }
                });
            }
        }
        futs1.push_back(std::move(fut));
    }
    auto* treeNodes = treeData();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs2.push_back(hpx::async(
            HPX_PRIORITY_HI, [proc, numberOfThreads, &treeMap, &treeNodes, treeNodesRemoteBegin]() {
                for (int i = treeNodesRemoteBegin + proc; i < treeNodesNextIndex();
                    i += numberOfThreads) {
                    if (treeNodes[i].children[LEFT].index != -1) {
                        ASSERT(treeMap.find(treeNodes[i].children[LEFT]) != treeMap.end());
                        ASSERT(treeMap.find(treeNodes[i].children[RIGHT]) != treeMap.end());
                        auto& left = treeNodes[i].children[LEFT];
                        auto& right = treeNodes[i].children[RIGHT];
                        left.index = treeMap[left];
                        right.index = treeMap[right];
                    }
                }
            }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    futs2.resize(0);
    std::function<void(int, ParticleInteger)> const adjustParticleReferences =
        [&adjustParticleReferences, &treeNodes](int index, ParticleInteger offset) -> void {
        auto& node = treeNodes[index];
        node.particleRange.first += offset;
        node.particleRange.second += offset;
        ASSERT(node.particleRange.first >= 0);
        ASSERT(node.particleRange.second <= particlesSize());
        if (node.children[LEFT].index != -1) {
            adjustParticleReferences(node.children[LEFT].index, offset);
            adjustParticleReferences(node.children[RIGHT].index, offset);
        }
    };
    for (auto i = remoteRoots.begin(); i != remoteRoots.end(); i++) {
        futs3.push_back(hpx::async(HPX_PRIORITY_HI,
            [i, &treeMap, &particleMap, adjustParticleReferences, &treeNodes,
                treeNodesRemoteBegin]() {
                GlobalParticleRange gpr;
                ASSERT(treeMap.find(*i) != treeMap.end());
                int index = treeMap[*i];
                ASSERT(index >= treeNodesRemoteBegin);
                auto& node = treeNodes[index];
                gpr.rank = node.processRange.first;
                gpr.range = node.particleRange;
                ASSERT(particleMap.find(gpr) != particleMap.end());
                auto localRange = particleMap[gpr];
                ParticleInteger offset = localRange.first - gpr.range.first;
                ASSERT(treeNodes[index].particleRange.first == gpr.range.first);
                adjustParticleReferences(index, offset);
            }));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs2.push_back(
            hpx::async(HPX_PRIORITY_HI, [&workitems, proc, numberOfThreads, &treeMap]() {
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
            }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    futs2.resize(0);
    auto sfut = hpx::sort(PAR_EXECUTION_POLICY, workitems.begin(), workitems.end(),
        [](KickWorkitem const& a, KickWorkitem const& b) {
            return treeGetNode(a.self)->particleRange.first <
                treeGetNode(b.self)->particleRange.first;
        });
    hpx::wait_all(futs3.begin(), futs3.end());
    hpx::wait_all(futs1.begin(), futs1.end());
    sfut.get();
    tm.stop();
    //	PRINT("Took %e seconds to prepare gpu send\n", tm.read());
    tm.reset();
    tm.start();
    tree2Gpu();
    particlesMemadviseGpu();
    auto const kr = cudaExecuteKicks(params, &particlesPos(XDIM, 0), &particlesPos(YDIM, 0),
        &particlesPos(ZDIM, 0), treeNodes, std::move(workitems));
    tree2Cpu();
    tm.stop();
    particlesResize(originalParticleCount);
    kickSetRc(kr);
    return kr;
    //	PRINT("GPU took %e seconds\n", tm.read());
#else
    THROW_ERROR("GPU gravity requested in a build without CUDA\n");
    return {};
#endif
}
